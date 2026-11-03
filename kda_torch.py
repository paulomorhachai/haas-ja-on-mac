# MIT License(LICENSE 参照)。計算式は flash-linear-attention(MIT)のソースを読んで PyTorch で書き直したもの。
# 出典と著作権表示は THIRD_PARTY_NOTICES.md を参照。
"""flash-linear-attention 0.5.2 の KimiDeltaAttention を Triton なしの素の PyTorch で置き換える。

目的は haas_asr の予測ネットワーク(prednet.py)が使う範囲を macOS(MPS/CPU)で動かすこと。
パラメータ名・形は fla のレイヤーと一致させてあるので、公式の .nemo の重みがそのまま載る。
式の出典(すべて fla 0.5.2 のソースを読んで写した。fla 自体は import しない):
  layers/kda.py                 レイヤー全体(射影・短畳み込み・ゲート・出力ノルム)
  ops/kda/fused_recurrent.py    1ステップの更新式(l2norm の eps=1e-6、state は V-first [B,HV,V,K])
  ops/kda/gate.py               g = -exp(A_log) * softplus(g + dt_bias)
  modules/fused_norm_gate.py    RMSNorm * weight * sigmoid(gate)
  modules/conv/short_conv.py    短畳み込みのキャッシュ = 直近 W 個の入力(古い順)
対応範囲: 推論のみ。chunk カーネルは持たず、全長を逐次再帰で処理する(結果は同じ、長い系列では遅い)。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ShortConv(nn.Conv1d):
    """fla.modules.ShortConvolution と同じ state_dict キー(weight: [D,1,W])を持つ因果深さ方向畳み込み。"""

    def __init__(self, hidden_size: int, kernel_size: int, bias: bool = False):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=0,
        )
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, cache: torch.Tensor | None = None):
        """x: [B,T,D]。cache: [B,D,W](直近W個の入力、古い順)または None。(y, 新cache) を返す。

        1トークンずつ進める fla の step は「cache を左へ1つずらして末尾に x を入れ、
        sum(cache * weight, -1) に silu」。履歴として使うのは cache の末尾 W-1 個。
        """
        w = self.kernel_size[0]
        b, _, d = x.shape
        xt = x.transpose(1, 2)  # [B,D,T]
        if cache is None:
            hist = xt.new_zeros(b, d, w - 1)
        else:
            hist = cache[:, :, -(w - 1) :].to(xt.dtype) if w > 1 else xt.new_zeros(b, d, 0)
        xc = torch.cat([hist, xt], dim=-1)  # [B,D,W-1+T]
        y = F.conv1d(xc, self.weight, self.bias, groups=d)  # [B,D,T]
        y = F.silu(y).transpose(1, 2)
        return y, xc[:, :, -w:].contiguous()


class _RMSNormGated(nn.Module):
    """fla.modules.FusedRMSNormGated(activation="sigmoid") : rmsnorm(x) * weight * sigmoid(g)。"""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        xf, gf = x.float(), g.float()
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        y = xf * rstd * self.weight.float() * torch.sigmoid(gf)
        return y.to(x.dtype)


def kda_recurrent(q, k, v, g_raw, beta_raw, A_log, dt_bias, state=None):
    """fused_recurrent_kda(use_qk_l2norm/gate/beta_sigmoid_in_kernel=True, state_v_first=True) の素の PyTorch 版。

    q,k: [B,T,H,K]  v: [B,T,HV,V]  g_raw: [B,T,HV,K]  beta_raw: [B,T,HV]
    state: [B,HV,V,K] または None。(o [B,T,HV,V] float32, 新state [B,HV,V,K] float32) を返す。
    """
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    grp = HV // H
    q, k, v, g_raw, beta_raw = (x.float() for x in (q, k, v, g_raw, beta_raw))
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * (K**-0.5)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    if grp > 1:
        q, k = q.repeat_interleave(grp, dim=2), k.repeat_interleave(grp, dim=2)
    gk = -A_log.float().exp().view(HV, 1) * F.softplus(g_raw + dt_bias.float().view(HV, K))
    beta = torch.sigmoid(beta_raw)
    S = q.new_zeros(B, HV, V, K) if state is None else state.float().clone()
    outs = []
    for t in range(T):
        S = S * gk[:, t].exp().unsqueeze(-2)  # K 軸に減衰
        u = (v[:, t] - (S * k[:, t].unsqueeze(-2)).sum(-1)) * beta[:, t].unsqueeze(-1)  # [B,HV,V]
        S = S + u.unsqueeze(-1) * k[:, t].unsqueeze(-2)
        outs.append((S * q[:, t].unsqueeze(-2)).sum(-1))
    return torch.stack(outs, dim=1), S


class KimiDeltaAttentionTorch(nn.Module):
    """fla.layers.kda.KimiDeltaAttention の推論専用・素の PyTorch 版(prednet が使う引数だけ対応)。"""

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        # prednet が渡さない・使わない分岐は黙って無視せず落とす(別物の計算をしないため)。
        if not use_short_conv:
            raise NotImplementedError("use_short_conv=False")
        if allow_neg_eigval or safe_gate or lower_bound is not None:
            raise NotImplementedError("allow_neg_eigval / safe_gate / lower_bound")
        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.conv_size = conv_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx
        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(f"expand_v={expand_v} gives non-integer head_v_dim")
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError("num_v_heads must be divisible by num_heads")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.q_conv1d = _ShortConv(self.key_dim, conv_size, bias=conv_bias)
        self.k_conv1d = _ShortConv(self.key_dim, conv_size, bias=conv_bias)
        self.v_conv1d = _ShortConv(self.value_dim, conv_size, bias=conv_bias)

        self.gate_dim = int(self.num_v_heads * self.head_k_dim)
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.gate_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(self.gate_dim, dtype=torch.float32))
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = _RMSNormGated(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ):
        if attention_mask is not None or kwargs.get("cu_seqlens") is not None:
            raise NotImplementedError("attention_mask / cu_seqlens")
        if self.training:
            raise NotImplementedError("推論専用")
        x = hidden_states
        last = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last = past_key_values[self.layer_idx]
        cq = ck = cv = None
        if last is not None and last.get("conv_state") is not None:
            cq, ck, cv = last["conv_state"]
        q, cq = self.q_conv1d(self.q_proj(x), cq)
        k, ck = self.k_conv1d(self.k_proj(x), ck)
        v, cv = self.v_conv1d(self.v_proj(x), cv)

        g = self.f_proj(x)
        beta = self.b_proj(x)
        hk, hv = self.head_k_dim, self.head_v_dim
        q = q.unflatten(-1, (self.num_heads, hk))
        k = k.unflatten(-1, (self.num_heads, hk))
        g = g.unflatten(-1, (self.num_v_heads, hk))
        v = v.unflatten(-1, (self.num_v_heads, hv))

        state = last["recurrent_state"] if last is not None else None
        o, state = kda_recurrent(q, k, v, g, beta, self.A_log, self.dt_bias, state)
        o = o.to(x.dtype)

        if past_key_values is not None:
            past_key_values.update(
                layer_idx=self.layer_idx, recurrent_state=state, conv_state=(cq, ck, cv), offset=x.shape[1]
            )
        gate = self.g_proj(x).unflatten(-1, (self.num_v_heads, hv))
        o = self.o_norm(o, gate).flatten(-2)
        return self.o_proj(o), None, past_key_values


def install() -> None:
    """haas_asr の fla 依存点(_fla.load_kda_class と prednet が import 済みの同名)を差し替える。"""
    import haas_asr._fla as _fla

    _fla.load_kda_class = lambda: KimiDeltaAttentionTorch
    import haas_asr.prednet as prednet

    prednet.load_kda_class = _fla.load_kda_class
