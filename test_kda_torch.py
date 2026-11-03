"""検品: kda_torch が fla の参照実装と(同じ式の写しではなく)独立に一致するかを確かめる。

1. 再帰本体: fla の ops/kda/naive.py(純PyTorch、triton 不要。ファイルを直接読み込む)の
   naive_recurrent_kda と、kda_recurrent を、前処理だけ揃えて比較する(state は K/V 転置)
2. 短畳み込み: 全長一括 = 1トークンずつ(キャッシュ持ち回り) = 手書きの左ゼロ詰め因果畳み込み
3. レイヤー全体: 全長一括 = 1トークンずつ(キャッシュ持ち回り)

通ることの意味: 1 は再帰の取り違え、2 はキャッシュの意味の取り違え、3 は両者の接続の取り違えを検出する。
fla の Triton カーネル出力との直接比較ではない(それは CUDA 機が要る)。前処理(l2norm/ゲート/beta)と
出力ノルムは fla のソースを読んで写した式で、このテストでは独立検証になっていない。
"""

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from kda_torch import KimiDeltaAttentionTorch, _ShortConv, kda_recurrent  # noqa: E402

# fla-core 0.5.2 の naive.py(純PyTorch)。setup_env.sh が fla_core/ に展開する。別の場所なら FLA_NAIVE_KDA で指定
NAIVE = Path(os.environ.get("FLA_NAIVE_KDA", HERE / "fla_core/fla/ops/kda/naive.py"))
spec = importlib.util.spec_from_file_location("fla_naive_kda", NAIVE)
naive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(naive)

torch.manual_seed(0)
fails = []


def check(name, a, b, tol=1e-5):
    err = (a - b).abs().max().item()
    ok = err < tol
    print(f"{'OK  ' if ok else 'FAIL'} {name}: max|diff|={err:.3e}")
    if not ok:
        fails.append(name)


# 1. 再帰本体 --------------------------------------------------------------
for (B, T, H, HV, K, V) in [(2, 37, 5, 5, 64, 64), (2, 9, 2, 4, 16, 8)]:
    q, k = torch.randn(B, T, H, K), torch.randn(B, T, H, K)
    v = torch.randn(B, T, HV, V)
    g_raw, beta_raw = torch.randn(B, T, HV, K), torch.randn(B, T, HV)
    A_log, dt_bias = torch.randn(HV) * 0.5, torch.randn(HV * K) * 0.1
    s0 = torch.randn(B, HV, V, K) * 0.1

    o, S = kda_recurrent(q, k, v, g_raw, beta_raw, A_log, dt_bias, s0)

    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    gk = -A_log.exp().view(HV, 1) * F.softplus(g_raw + dt_bias.view(HV, K))
    o_ref, S_ref = naive.naive_recurrent_kda(
        qn, kn, v, gk, torch.sigmoid(beta_raw), initial_state=s0.transpose(-1, -2), output_final_state=True
    )
    check(f"recurrent o  (H={H},HV={HV},K={K},V={V})", o, o_ref)
    check(f"recurrent S  (H={H},HV={HV},K={K},V={V})", S, S_ref.transpose(-1, -2))

# 2. 短畳み込み ------------------------------------------------------------
D, W, T = 24, 4, 11
conv = _ShortConv(D, W)
torch.nn.init.normal_(conv.weight)
x = torch.randn(2, T, D)
y_all, c_all = conv(x)
cache, ys = None, []
for t in range(T):
    y, cache = conv(x[:, t : t + 1], cache)
    ys.append(y)
y_step = torch.cat(ys, 1)
xp = F.pad(x.transpose(1, 2), (W - 1, 0))
y_hand = sum(conv.weight[:, 0, i].view(1, D, 1) * xp[:, :, i : i + T] for i in range(W))
y_hand = F.silu(y_hand).transpose(1, 2)
check("conv 一括 vs 手書き因果畳み込み", y_all, y_hand)
check("conv 1トークンずつ vs 一括", y_step, y_all)
check("conv 最終キャッシュ(一括 vs 1トークンずつ)", cache, c_all)
# 途中までを一括 → 続きを1トークンずつ、でも同じになること
y1, c1 = conv(x[:, :6])
y2, _ = conv(x[:, 6:], c1)
check("conv 前半一括+後半続き vs 一括", torch.cat([y1, y2], 1), y_all)


# 3. レイヤー全体 ----------------------------------------------------------
class Cache:
    def __init__(self):
        self.states = []

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        return self.states[i]

    def update(self, layer_idx=0, offset=None, **state):
        while len(self.states) <= layer_idx:
            self.states.append({})
        self.states[layer_idx].update(state)
        return self.states[layer_idx]


layer = KimiDeltaAttentionTorch(
    hidden_size=640, num_heads=5, head_dim=64, expand_v=1.0, conv_size=4, layer_idx=0
).eval()
with torch.no_grad():
    for p in layer.parameters():
        p.add_(torch.randn_like(p) * 0.05)
    h = torch.randn(3, 17, 640)
    full, _, _ = layer(h, past_key_values=Cache(), use_cache=True)
    cache, outs = Cache(), []
    for t in range(17):
        o, _, _ = layer(h[:, t : t + 1], past_key_values=cache, use_cache=True)
        outs.append(o)
    check("layer 1トークンずつ vs 一括", torch.cat(outs, 1), full, tol=1e-4)

print("\nALL OK" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
