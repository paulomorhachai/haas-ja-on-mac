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

Apple M1 Max(32GB)、macOS 26.3.2、Python 3.12、PyTorch 2.15.0、NeMo 2.7.3、lightning 2.6.6、transformers 5.22.1、Haas-ASR `acc614f`、jacer 1.2.0(依存の版の固定については、下の「セキュリティ」を参照)。
**他の環境は未確認です。**

## 使い方

前提: [uv](https://docs.astral.sh/uv/)、git。

```bash
./setup_env.sh          # 公式リポジトリの clone、venv、NeMo 2.7(脆弱性対策の版指定つき)、互換修正、fla-core の参照実装の展開
.venv/bin/python test_kda_torch.py                         # KDA の検品(CPU、数秒)
.venv/bin/python run_haas.py --device mps                  # 合成音声3文の煙テスト(初回は HF から重みを取得)
```

重みのダウンロードが進まないときは `HF_HUB_DISABLE_XET=1` を付けてみてください。

### 公式ベンチ(FLEURS だけ)

asr-ja-bench は音声を含みません。FLEURS は HF から自動で取得されます(約450MB)。

```bash
git clone https://huggingface.co/datasets/third-intelligence/asr-ja-bench bench_repo
uv pip install --python .venv/bin/python --override overrides.txt pyarrow "jacer @ git+https://github.com/third-intelligence/jacer.git@b861959fe0a8f67f47c848d19c1a9cb8a32e1c2e"
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

## 公式の機能の確認結果(Apple M1 Max・MPS・fp32)

公式 Haas-ASR の `examples/` と `demos/` は、`load_model` の既定がcudaで、KDA も fla に頼るため、そのままでは Mac で動きません。**`run_official.py`** は、公式のコードを書き換えずに、(1) KDA の差し替えと、(2) `load_model` の既定のデバイスを MPS にすること、だけを行ってから、公式のサンプルやデモを動かす起動用の包みです。

```bash
.venv/bin/python run_official.py upstream/examples/quickstart.py --model third-intelligence/Haas-ja-v1.0 audio/say_0.wav
.venv/bin/python run_official.py -m demos.app --device mps     # ブラウザデモ(下の注意を参照)
```

| 機能 | 結果 | 確認した内容 |
|---|---|---|
| クイックスタート(`examples/quickstart.py`) | 動く | 合成音声3文が、期待文と一致 |
| 長尺(`examples/transcribe_longform.py`) | 動く | 実音声(映画の断片)2分が4つに分割され、454 文字が出た(デモでも同じ 454 文字)。精度は測っていない |
| 語彙の指定(`examples/transcribe_with_vocabulary.py`) | **動くが、効き方は音声しだい** | 下の「語彙の指定について」を参照 |
| 評価(`examples/evaluate_asr_ja_bench.py`) | 関数を流用 | 公式のスクリプトは cuda・bf16 前提で、そのままは動かない。`run_bench.py` が、その関数をそのまま呼ぶ。FLEURS 650 件の CER は上の表のとおり |
| ブラウザデモ 長尺(`/longform`) | 動く | 画面に音声ファイルを入れて、2分の音声が4つの断片で、0.92 秒で文字になり、波形と時刻が表示された。通信(WebSocket)も直接試して、`progress → plan → chunks → done` の順で返った |
| ブラウザデモ リアルタイム(`/realtime`) | **ファイルの再生で動く。マイクは未確認** | 字幕が段階的に伸び、確定した。遅延(エンドツーエンドの中央値)は、再生中に測った2回で 0.5〜0.7 秒(モデル処理は約 105〜190 ms)。**ほかに1回だけ、再生後の画面に約 4 秒の表示が出た**(原因は不明。測り直していない) |
| 話者ラベル(`examples/speakers/`) | **未確認(確かめないことにした。2026-11-03)** | 別の Python 環境と、NeMo の main ブランチ(GitHub から取得)、NVIDIA の別のモデルが要る。公式の指定のままだと、既知の脆弱性のある版が戻る |

**ブラウザデモの注意**

- デモ用の依存は、`uv pip install --python .venv/bin/python --override overrides.txt "fastapi>=0.110" "uvicorn>=0.29" "wsproto>=1.2"` で入れます(入れたあとの環境で、脆弱性 0 件を確認済み)。
- 画面の確認は、アプリ内のブラウザ 1 種類だけで行いました。**マイク入力と、ほかのブラウザ(Safari・Chrome など)は、確かめていません。**
- 公式の説明どおり、デモは試用のためのもので、本番用ではありません。サーバーは `127.0.0.1` のみで待ち受けます。

**語彙の指定について**(`check_boosting_alpha.py` で再現できます)

macOS の `say` で合成した、固有名詞を含む文で調べました。

| 条件 | 結果 |
|---|---|
| 語彙なし(基準) | 「サードインテリジェンス」「**first**」「**mc**」(モデルは、合成音声の「ハース」と「Mac」を、そう聞き取った) |
| 語彙あり、読みの候補を追加(`Haas=ハース,first,ファースト` など) | 3語とも指定の書き方に置き換わった(「Third Intelligence」「Haas」「Mac」) |
| ブーストの強さ `alpha` = 2.0(既定) | 基準と同じ(この音声では変化なし) |
| `alpha` = 5.0 / 10.0 | 「mc」→「Mac」(正しい)が出る一方、「サード」→「Haas」(誤爆)や「first」→「Mac」(誤り)も出た |
| `alpha` = 30 以上 | 出力が崩れる |

- **動いている部分**: 見つかった読みを、指定の書き方に置き換える処理は動きます。ブースト(デコード中に語彙へ寄せる処理。NeMo の GPU フレーズブースト)も、Triton が無い Mac では、代わりの経路に切り替わって動き(NeMo のログに「Triton is disabled」と出る)、`alpha` を上げると、出力に影響します。
- **気をつけること**: 公式の説明のとおり、語彙を足すほど、誤爆の可能性が増えます。**読みの候補には、モデルが実際にそう書いてしまう綴り(上の例なら「first」)も入れるのが、確実でした。**
- **確かめられていないこと**: CUDA 機での出力と同じかどうか。この結果は、合成音声1文での観察で、語彙指定の一般的な効果の評価ではありません。

## セキュリティ(依存パッケージの脆弱性)

2026-11-03 時点で、このリポジトリの手順で作った環境(セットアップ直後は 188 パッケージ、ベンチの道具を入れたあとは 194 パッケージ)を、[pip-audit](https://github.com/pypa/pip-audit) で照合すると、**既知の脆弱性は 0 件**でした。`./audit_deps.sh` で、いつでも再確認できます(新しい脆弱性は日々見つかるので、定期的に実行してください)。

**何をしているか**

- NeMo 2.7.3 などが課す古い上限を、`overrides.txt` で上書きして、修正版以上に固定しています(fsspec、hydra-core、packaging、transformers、lightning)。最初の4つは、公式 Haas-ASR の `override-dependencies` と同じです。公式の設定は、公式のリポジトリで `uv sync` したときだけ効くので、ここでは `uv pip install --override` で同じ効果を得ています。
- **lightning だけは、公式と違います。** 公式は、NeMo 2.7.3 との互換のため lightning 2.4.0 のままにしていますが、2.4.0 には既知の脆弱性(悪意のあるチェックポイントの読み込みでコードが実行される問題など)が残ります。ここでは 2.6.6 に上げて、`patch_compat.py` で互換を取っています。

**`patch_compat.py` がしていること**

インストールされた第三者のファイル2か所を書き換えます。どちらも、宣言や読み込み先を直すだけで、学習時にしか使わない部分です(文字起こしには使いません)。

1. nv-one-logger 2.3.1 の `save_checkpoint` の型宣言を、`bool` から `Optional[bool]` に直す(lightning 2.6.6 の宣言に合わせる)。
2. NeMo の `exp_manager.py` が `NeptuneLogger` を読み込む場所を、`lightning.pytorch.loggers.neptune` に直す(2.6.6 では、`loggers` の入口から公開されなくなったため)。

対象の版が確認済みのものでない、または該当箇所がちょうど1つでない場合は、書き換えずに止まります(そのときは `setup_env.sh` が失敗して終わります)。NVIDIA 側や NeMo 側で直った版が出れば、不要になります。

**確認したこと**: 対策前と対策後で、FLEURS 650 件の CER は小数4桁まで同じで(strict 3.6886、flex 3.0182、kana 2.3829)、650 件すべてのサンプルで、モデルの出力文字列が一致しました。KDA の検品テストも通っています。

**注意(照合できていないもの)**

- 公式のコード(Haas-ASR)と jacer は PyPI に無いため、照合の対象外です。OS や uv などの道具も、対象外です。
- 「既知の脆弱性が 0 件」は、安全の保証ではありません。未知の脆弱性は、見つかりません。
- 依存パッケージとは別に、**信頼できない配布元のモデルの重みやチェックポイントを読み込むと、危険があります**。重みは、信頼できる配布元のものだけを使ってください。

## 未検証・言えないこと(Not verified)

- fla の **Triton カーネルとの直接比較はしていません**(CUDA 機が必要)。前処理(l2norm、ゲート式、beta の sigmoid)と出力ノルムは、fla のソースを読んで写した式で、参照実装(naive)には含まれず、独立には検証できていません。FLEURS の CER が公称値とほぼ一致したことは傍証にとどまります。
- **FLEURS 以外の4サブセットは未測定**です。公称の全体値(6.12)とは比べられません。特に自然発話に近い TEDxJP-10K・SpeechBSD での挙動は分かりません。
- 公称との差(+0.02〜+0.04 ポイント)が、fp32 と bf16 の違いか、バッチや演算順の誤差かは、切り分けていません。bf16 は未試験です。
- 推論専用です。chunk カーネルは持たず、全長を逐次の再帰で処理するため、長い系列では遅くなりえます。`attention_mask` などの引数は未対応(`NotImplementedError`)です。
- 映画の音声での試験は、**参照テキストが無いので精度を測れていません**。「出力なし」の断片が、音楽・効果音なのか、台詞の取りこぼしなのかは、音量だけでは区別できませんでした。

- **話者ラベル**(大きな取得と別環境が要るため、意図的に見送り)、デモの**マイク入力**、デモのアプリ内ブラウザ以外での動作は未確認です(上の「公式の機能の確認結果」を参照)。
- **学習(ファインチューニング)**は、このリポジトリの対象外です。公式のコードは推論用で、`kda_torch.py` は学習モードで呼ぶと止まります。互換修正(`patch_compat.py`)の2か所は学習時にしか使わない部分なので、学習する場合は、別に確認が必要です。

## 同梱していないもの

モデルの重み、公式の推論コード(Haas-ASR)、ベンチの音声と書き起こし、映画などの音声データとその書き起こし。
第三者の著作物については [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。

非公式の個人の実験で、Third Intelligence とは関係がありません。

## ライセンス

[MIT](LICENSE)。第三者のコードとデータは、それぞれのライセンスに従います。
