# haas-ja-on-mac

日本語音声認識モデル [Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) を、Apple Silicon の Mac(MPS / CPU)で動かすための実験記録とスクリプト。

*English summary: Haas-ja-v1.0 officially requires Linux + an NVIDIA GPU because its prediction network (Kimi Delta Attention, KDA) runs on Triton/CUDA kernels from flash-linear-attention. This repo replaces only that layer with a plain-PyTorch, inference-only implementation (`kda_torch.py`) and runs the official `haas_asr` code on Apple Silicon. On FLEURS (ja_jp, 650 utterances) it gives CER 3.69 / 3.02 / 2.38 (strict / flex / kana) vs. the published 3.67 / 2.98 / 2.34. Unofficial and unaffiliated; see "Not verified" below.*

**初めての方は、コピー＆ペーストで進められる [手順書（GUIDE.ja.md）](GUIDE.ja.md) からどうぞ。**

## 何をしたか

公式は「Linux + NVIDIA GPU のみ。CPU と macOS は非対応」としています。理由は、デコーダの予測ネットワークが使う
KDA(Kimi Delta Attention)が、flash-linear-attention の Triton カーネルに依存しているためです。

公式コードを読むと、fla への依存は `_fla.py` と `prednet.py` に集まっていて、差し替えの継ぎ目は `load_kda_class()` でした。
そこで、KDA 層だけを素の PyTorch で書き直し(`kda_torch.py`、推論専用)、実行時に差し替えます(公式のコードは書き換えません)。

```python
import kda_torch
kda_torch.install()   # モデルを読み込む前に呼ぶ(run_haas.py と同じ順: import haas_asr → import kda_torch → install())
```

パラメータの名前と形は fla のレイヤーと同じにしてあり、公式の重みがそのまま載ります。

## 結果(FLEURS ja_jp test、650 件・2.36 時間)

| | cer_strict | cer_flex | cer_kana |
|---|---|---|---|
| このリポジトリ(Apple M1 Max・MPS・fp32・バッチ16) | 3.69 | 3.02 | 2.38 |
| 公称(モデルカードの表。H200・bf16・バッチ512) | 3.67 | 2.98 | 2.34 |

- 集計値は [`results/fleurs_summary.json`](results/fleurs_summary.json)。評価器は jacer 1.2.0(公式固定コミット、v1.2)で、採点環境は固定版と一致(`pinned: true`)。
- 650 件の推論は 63.5 秒(RTFx 134)。**公称の RTFx 3,236 は H200・bf16・バッチ512 の値で、比較になりません。**
- 公式の評価スクリプトの関数(`read_subset` / `transcribe` / `write_hypotheses` / `score_line`)をそのまま呼んでいます。違いは、デバイス(MPS)・精度(fp32)・バッチ(16。MPS のメモリ保護のため)・KDA の差し替えです。

## 動作確認した環境

Apple M1 Max(32GB)、macOS 26.3.2、Python 3.12、PyTorch 2.15.0、NeMo 2.7.3、Haas-ASR `acc614f`、jacer 1.2.0。
**他の環境(M4 など)は未確認です。**

## 使い方

前提: [uv](https://docs.astral.sh/uv/)、git。

```bash
./setup_env.sh          # 公式リポジトリの clone、venv、NeMo 2.7、fla-core の参照実装の展開
.venv/bin/python test_kda_torch.py                         # KDA の検品(CPU、数秒)
.venv/bin/python run_haas.py --device mps                  # 合成音声3文の煙テスト(初回は HF から重みを取得)
```

重みのダウンロードが進まないときは `HF_HUB_DISABLE_XET=1` を付けてみてください。

### 公式ベンチ(FLEURS だけ)

asr-ja-bench は音声を含みません。FLEURS は HF から自動で取得されます(約450MB)。

```bash
git clone https://huggingface.co/datasets/third-intelligence/asr-ja-bench bench_repo
uv pip install --python .venv/bin/python pyarrow "jacer @ git+https://github.com/third-intelligence/jacer.git@b861959fe0a8f67f47c848d19c1a9cb8a32e1c2e"
.venv/bin/python bench_repo/rebuild.py --subset fleurs --out bench_data    # 650/650 件の書き起こしをハッシュ照合
.venv/bin/python test_jacer.py                                             # 評価器の検品
.venv/bin/python run_bench.py --subset fleurs                              # 結果は bench_out/
```

他のサブセット(JSUT・JVNV・TEDxJP-10K・SpeechBSD)は、取得に手間がかかる(配布元の規約、HF の gated、YouTube からの再構成など)ため、試していません。

### 自分の音声で

`run_clips.py` と `run_film.py` は `film_test/*.wav`(16kHz モノラル)を読みます。mp4 などからは、macOS 標準の afconvert で作れます。

```bash
mkdir -p film_test && afconvert -f WAVE -d LEI16@16000 -c 1 input.mp4 film_test/input.wav
.venv/bin/python run_film.py     # 約30秒ごとに静かな箇所で切って文字起こし。時刻付きで film_out/ に出力
```

`run_film.py` は公式の `transcribe_longform` と同じ処理(静かな箇所での分割 → 文字起こし)ですが、断片ごとの時刻を残します。

## 検品

- `test_kda_torch.py`: fla-core の参照実装(`naive_recurrent_kda`)との一致(状態・出力の差 1e-7 未満)、短畳み込みのキャッシュ(一括処理と1トークンずつの処理の一致)、レイヤー全体の一致。
- `test_jacer.py`: 手計算した6ケースで、評価器が期待値どおりの CER を返すか。
- `check_empty.py`: 「出力なし」の断片が、無音なのか、音があるのに出力がないのかを音量で仕分ける。

## 未検証・言えないこと(Not verified)

- fla の **Triton カーネルとの直接比較はしていません**(CUDA 機が必要)。前処理(l2norm、ゲート式、beta の sigmoid)と出力ノルムは、fla のソースを読んで写した式で、参照実装(naive)には含まれず、独立には検証できていません。FLEURS の CER が公称値とほぼ一致したことは傍証にとどまります。
- **FLEURS 以外の4サブセットは未測定**です。公称の全体値(6.12)とは比べられません。特に自然発話に近い TEDxJP-10K・SpeechBSD での挙動は分かりません。
- 公称との差(+0.02〜+0.04 ポイント)が、fp32 と bf16 の違いか、バッチや演算順の誤差かは、切り分けていません。bf16 は未試験です。
- 推論専用です。chunk カーネルは持たず、全長を逐次の再帰で処理するため、長い系列では遅くなりえます。`attention_mask` などの引数は未対応(`NotImplementedError`)です。
- 映画の音声での試験は、**参照テキストが無いので精度を測れていません**。「出力なし」の断片が、音楽・効果音なのか、台詞の取りこぼしなのかは、音量だけでは区別できませんでした。

## 同梱していないもの

モデルの重み、公式の推論コード(Haas-ASR)、ベンチの音声と書き起こし、映画などの音声データとその書き起こし。
第三者の著作物については [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。

非公式の個人の実験で、Third Intelligence とは関係がありません。

## ライセンス

[MIT](LICENSE)。第三者のコードとデータは、それぞれのライセンスに従います。
