# haas-ja-on-mac

日本語音声認識モデル [Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) を、Apple Silicon の Mac(MPS / CPU)で動かすためのスクリプトと実験記録。

*English summary: Haas-ja-v1.0 officially requires Linux + an NVIDIA GPU because its prediction network (Kimi Delta Attention, KDA) runs on Triton/CUDA kernels from flash-linear-attention. This repo replaces only that layer with a plain-PyTorch, inference-only implementation (`kda_torch.py`) and runs the official `haas_asr` code on Apple Silicon. On FLEURS (ja_jp, 650 utterances) it gives CER 3.69 / 3.02 / 2.38 (strict / flex / kana) vs. the published 3.67 / 2.98 / 2.34. Unofficial and unaffiliated; see "Not verified" in [docs/verification.md](docs/verification.md).*

**初めての方は、コピー＆ペーストで進められる [手順書（GUIDE.ja.md）](GUIDE.ja.md) からどうぞ。**

## 使い方

前提: [uv](https://docs.astral.sh/uv/)、git。

```bash
./setup_env.sh                                 # 公式リポジトリの clone、venv、NeMo 2.7、互換修正(脆弱性対策の版指定つき)
.venv/bin/python test_kda_torch.py             # KDA の検品(CPU、数秒)
.venv/bin/python run_haas.py --device mps      # 合成音声3文の煙テスト(初回は HF から重みを取得)
```

自分の音声(16kHz モノラルの wav)を文字にするには、`film_test/` に置いて `.venv/bin/python run_film.py` を実行します。約30秒ごとに静かな箇所で切り、時刻つきで `film_out/` に出力します。変換方法は [手順書](GUIDE.ja.md) の STEP 7 にあります。

重みのダウンロードが進まないときは `HF_HUB_DISABLE_XET=1` を付けてみてください。

## 何をしたか

公式は「Linux + NVIDIA GPU のみ。CPU と macOS は非対応」としています。理由は、デコーダの予測ネットワークが使う KDA が、flash-linear-attention の Triton カーネルに依存しているためです。

公式コードの fla への依存は `_fla.py` と `prednet.py` に集まっていて、差し替えの継ぎ目は `load_kda_class()` でした。そこで KDA 層だけを素の PyTorch で書き直し(`kda_torch.py`、推論専用)、実行時に差し替えます(公式のコードは書き換えません)。

```python
import kda_torch
kda_torch.install()   # モデルを読み込む前に呼ぶ(run_haas.py と同じ順: import haas_asr → import kda_torch → install())
```

パラメータの名前と形は fla のレイヤーと同じで、公式の重みがそのまま載ります。

## 結果(FLEURS ja_jp test、650 件・2.36 時間)

| | cer_strict | cer_flex | cer_kana |
|---|---|---|---|
| このリポジトリ(Apple M1 Max・MPS・fp32・バッチ16) | 3.69 | 3.02 | 2.38 |
| 公称(モデルカードの表。H200・bf16・バッチ512) | 3.67 | 2.98 | 2.34 |

- 集計値は [`results/fleurs_summary.json`](results/fleurs_summary.json)。評価器は jacer 1.2.0(公式固定コミット、v1.2)で、採点環境は固定版と一致(`pinned: true`)。
- 650 件の推論は 63.5 秒(RTFx 134)。**公称の RTFx 3,236 は H200・bf16・バッチ512 の値で、比較になりません。**
- 公式の評価スクリプトの関数をそのまま呼んでいます。違いは、デバイス(MPS)・精度(fp32)・バッチ(16。MPS のメモリ保護のため)・KDA の差し替えです。
- 動作確認した環境: Apple M1 Max(32GB)、macOS 26.3.2、Python 3.12、PyTorch 2.15.0、NeMo 2.7.3、lightning 2.6.6、Haas-ASR `acc614f`。**他の環境は未確認です。**

## 公式の機能は動くか(Apple M1 Max・MPS・fp32)

公式のサンプルとデモは、そのままでは Mac で動きません。公式のコードを書き換えずに動かす起動用の包みが `run_official.py` です。詳細は [docs/verification.md](docs/verification.md)、実行手順は [docs/guide-optional.md](docs/guide-optional.md) にあります。

| 機能 | 結果 |
|---|---|
| クイックスタート | 動く(合成音声3文が期待文と一致) |
| 長尺(`transcribe_longform`) | 動く(精度は測っていない) |
| 語彙の指定 | **動くが、効き方は音声しだい**(ブーストの強さを上げると誤爆する。CUDA 機と同じ出力かは未確認) |
| 評価 | 公式スクリプトは cuda・bf16 前提でそのままは動かない。関数を流用して FLEURS 650 件を測定 |
| ブラウザデモ 長尺 | 動く |
| ブラウザデモ リアルタイム | ファイルの再生で動く。**マイクは未確認** |
| 話者ラベル | **未確認**(別環境と大きな取得が要るため、見送った) |

## 制限・未検証

- fla の Triton カーネルとの**直接比較はしていません**(CUDA 機が必要)。FLEURS の CER が公称値とほぼ一致したことは傍証にとどまります。
- FLEURS 以外の4サブセットは未測定で、公称の全体値(6.12)とは比べられません。
- 推論専用です。学習モードで呼ぶと止まります。長い系列は、全長を逐次処理するため遅くなりえます。
- 映画の音声での試験は、参照テキストが無いので精度を測れていません。

全文は [docs/verification.md](docs/verification.md) の「未検証・言えないこと」にあります。

## セキュリティ

依存パッケージは、既知の脆弱性を避ける版に固定しています。2026-11-03 時点で pip-audit は **0 件**でした(安全の保証ではありません)。`./audit_deps.sh` で再確認できます。lightning だけは公式と違う版にしているため、互換修正(`patch_compat.py`)を当てています。経緯と注意は [docs/security.md](docs/security.md) にあります。

## ファイル構成

```text
haas-ja-on-mac/
├── README.md                  この文書
├── GUIDE.ja.md                はじめての方向けの手順書(STEP 0〜7)
├── docs/
│   ├── guide-optional.md      任意の手順(ベンチ・脆弱性確認・公式デモ)
│   ├── verification.md        テスト・公式機能の確認結果・未検証の一覧
│   └── security.md            依存パッケージの脆弱性対策の経緯と注意
│
├── kda_torch.py               核心。KDA 層を素の PyTorch で書いた差し替え(推論専用)
├── setup_env.sh               環境づくり(公式コードの取得、venv、依存の導入、互換修正)
├── overrides.txt              脆弱性を避けるための版の固定
├── patch_compat.py            lightning 2.6.6 との互換修正(第三者ファイル2か所)
├── audit_deps.sh              入れたパッケージの脆弱性を照合
│
├── run_haas.py                合成音声3文の煙テスト(動くかの確認)
├── run_film.py                長い音声を約30秒ごとに切って文字起こし(時刻つき)
├── run_clips.py               実録音から25秒ずつ切り出して試す定性試験
├── run_bench.py               FLEURS ベンチの測定(公式の関数を流用)
├── run_official.py            公式のサンプル・デモを、書き換えずに Mac で動かす包み
├── check_boosting_alpha.py    語彙ブーストの強さ(alpha)の切り分け
│
├── test_kda_torch.py          KDA の検品(fla の参照実装との一致)
├── test_jacer.py              評価器 jacer の検品(手計算した6ケース)
├── check_empty.py             「出力なし」が無音か取りこぼしかを音量で仕分け
│
├── results/fleurs_summary.json  FLEURS の集計値
├── LICENSE                    MIT
└── THIRD_PARTY_NOTICES.md     第三者のコード・データの表示
```

セットアップや実行で作られるフォルダ(`.venv/`、`upstream/`、`fla_core/`、`bench_repo/`、`bench_data/`、`out/`、`film_test/`、`film_out/` など)は、`.gitignore` で管理の対象外です。モデルの重みは、リポジトリの外の `~/.cache/huggingface/` に置かれます。

## 同梱していないもの

モデルの重み、公式の推論コード(Haas-ASR)、ベンチの音声と書き起こし、映画などの音声データとその書き起こし。第三者の著作物については [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。

非公式の個人の実験で、Third Intelligence とは関係がありません。

## ライセンス

[MIT](LICENSE)。第三者のコードとデータは、それぞれのライセンスに従います。
