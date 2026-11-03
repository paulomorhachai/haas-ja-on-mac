# 手順書: Mac で日本語音声認識 Haas-ja-v1.0 を動かす（はじめての方向け）

この手順書は、コマンドをコピー＆ペーストするだけで、Haas-ja-v1.0（日本語の音声を文字にするモデル）を Mac で動かせるように書いています。
専門用語は、いちばん下の「用語の説明」にまとめました。

**この手順書でできること**

1. 動くかどうかの確認（合成した日本語の音声で）
2. 自分の音声ファイルの文字起こし（時刻つき）
3. （任意）公式ベンチマークでの精度の測定

**動作確認した環境**: Apple M1 Max（メモリ 32GB）、macOS 26.3.2。STEP 4 以降のコマンドは、この環境で、白紙のフォルダから実行して確かめました。**STEP 2〜3（uv の導入、GitHub からの取得）は、手元にすでに uv と複製があったため、書いたとおりには実行していません。** ほかの環境（M2 以降、Intel Mac など）も、確かめていません。

---

## STEP 0: 必要なものを確認する

| 項目 | 目安 |
|---|---|
| Mac | Apple シリコン（M1 以降）。下の確認コマンドで `arm64` と出ること |
| 空き容量 | 約 **5GB**（実測: 環境 約2.2GB、モデルの重み 約1.1GB。ベンチマークをやる場合は、さらに約 0.6GB） |
| メモリ | 実行中のピークは、2〜3.4GB でした（32GB の機械で測定）。|
| ネットワーク | 数 GB のダウンロードがあります |

```bash
uname -m
```

`arm64` と出れば Apple シリコンです。

## STEP 1: ターミナルを開く

Spotlight（`⌘` + `スペース`）で「ターミナル」と入力して開きます。以降のコマンドは、すべてここに貼り付けます。

## STEP 2: 道具を入れる（git と uv）

**git** は、たいていの Mac に入っています。次を実行して、バージョンが出れば OK です（入っていない場合は、macOS が「コマンドラインデベロッパツール」のインストールを案内することがあるので、その場合は従ってください）。

```bash
git --version
```

**uv**（Python の環境を作る道具）を入れます。Homebrew を使っている場合は、これで入ります。

```bash
brew install uv
```

Homebrew を使っていない場合は、uv の公式のインストーラを使います（詳しくは https://docs.astral.sh/uv/ ）。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

入れたら、**ターミナルを一度閉じて、開き直して**から、確認します。

```bash
uv --version
```

バージョンが出れば OK です。Python 本体は、次の STEP で uv が自動で用意します（別に入れる必要はありません）。

## STEP 3: このリポジトリを取ってくる

`<ユーザー名>` は、このリポジトリの持ち主の GitHub ユーザー名に置き換えてください。リポジトリが非公開の間は、アクセス権のある GitHub アカウントでの認証が必要です（その場合は、持ち主にご相談ください）。

```bash
git clone https://github.com/paulomorhachai/haas-ja-on-mac.git
```

```bash
cd haas-ja-on-mac
```

## STEP 4: セットアップ（初回だけ）

```bash
./setup_env.sh
```

公式のコードの取得、Python 3.12 の環境づくり、NeMo などのインストールを、まとめて行います。ダウンロードが多いので、回線によっては、時間がかかります。
**最後に `SETUP_DONE` と出れば成功です。**

途中で `[修正] …` という行が出ることがありますが、正常です。既知の脆弱性を避けるため、一部のパッケージを修正版以上に固定し（`overrides.txt`）、その組み合わせのための小さな互換修正（`patch_compat.py`）を当てているためです。詳しくは [README](README.md) の「セキュリティ」を読んでください。

もし `Permission denied` と出たら、こちらで実行してください。

```bash
zsh setup_env.sh
```

## STEP 5: 計算の検品テスト（数秒）

差し替えた計算（`kda_torch.py`）が、正しく動くかを確かめるテストです。

```bash
.venv/bin/python test_kda_torch.py
```

**最後に `ALL OK` と出れば成功です**（私の環境では 3 秒弱でした）。

## STEP 6: 合成音声で、動くかを確認する

macOS の `say` で日本語の文を 3 つ読み上げた音声を作り、モデルで文字にします。

```bash
.venv/bin/python run_haas.py --device mps
```

- **初回は、モデルの重み（約 1.1GB）の自動ダウンロードが入るので、時間がかかります**（私の環境の初回は約 3 分でした）。2 回目以降は、数秒で読み込めます。
- 成功すると、期待した文と、モデルが書いた文が並びます。`"exact_match"` が、3 つとも `true` なら、3 文とも一致しています。結果は `out/result_mps.json` にも保存されます。
- CPU で試したいときは、`--device mps` を `--device cpu` に変えます（遅くなります）。
- **日本語の声 `Kyoko` が必要です。** `say: Voice ... not found` のように出たら、システム設定の「アクセシビリティ」にある読み上げの設定で、日本語の声を追加してください（メニュー名は、macOS のバージョンで違うことがあります）。声を追加せずに、STEP 7 に進んでもかまいません。

## STEP 7: 自分の音声を文字起こしする

**1. 音声を、16kHz・モノラルの wav にして、`film_test` フォルダに置きます。** `入力ファイル.mp4` と `名前.wav` を、ご自分のものに置き換えてください。`afconvert` は macOS に標準で入っています（mp4 で動作を確認しました）。

```bash
mkdir -p film_test
```

```bash
afconvert -f WAVE -d LEI16@16000 -c 1 入力ファイル.mp4 film_test/名前.wav
```

**2. 文字起こしを実行します。**

```bash
.venv/bin/python run_film.py
```

**3. 結果は `film_out/名前.txt` に出ます。** 次のような形です（4 秒の音声で確認しました）。

```
# 名前  全長 00:00:04  断片 1 個(約30秒ごと、静かな箇所で分割)
# [開始 - 終了] 本文 / 出力なしの断片は (出力なし)

[00:00:00 - 00:00:04] （ここに文字起こしの本文）
```

- 長い音声は、静かな箇所で、**約 30 秒ごと**に区切って処理され、1 行が 1 つの区切りです。時刻つきなので、元の音声と見比べて確認できます。
- `film_test` フォルダの中の **wav ファイルは、すべて**処理されます（フォルダ名の `film` は、作った経緯によるもので、映画に限りません）。
- 速さの目安: 私の環境（M1 Max）では、約 100 分の音声が、30〜35 秒でした。
- **「(出力なし)」の行**: 無音や音楽のこともありますが、音があるのに文字にならなかった（取りこぼし）可能性もあります。音量だけでは区別できないので、気になる区間は、耳で確かめてください。
- **著作権にご注意ください。** 他人の音声・映画などの文字起こしを、配布・公開することは、権利に触れる場合があります。

## STEP 8（任意）: 公式ベンチマークで精度を測る

公式のベンチマーク asr-ja-bench のうち、手続きなしで取れる **FLEURS**（日本語・650 件）で、文字の誤り率（CER）を測ります。FLEURS の音声（約 440MB）は、自動でダウンロードされます。

**1. 準備（ベンチマークの定義と、評価器 jacer の取得）**

```bash
git clone https://huggingface.co/datasets/third-intelligence/asr-ja-bench bench_repo
```

```bash
uv pip install --python .venv/bin/python --override overrides.txt pyarrow "jacer @ git+https://github.com/third-intelligence/jacer.git@b861959fe0a8f67f47c848d19c1a9cb8a32e1c2e"
```

**2. 音声の取り直し**

```bash
.venv/bin/python bench_repo/rebuild.py --subset fleurs --out bench_data
```

`650/650 samples rebuilt, all transcripts verified` と出れば成功です。

**3. 評価器の検品（数秒）**

```bash
.venv/bin/python test_jacer.py
```

`ALL OK` と出れば成功です。

**4. 測定（M1 Max で約 1 分）**

```bash
.venv/bin/python run_bench.py --subset fleurs
```

最後のほうに、次のような行が出ます。**私の環境の結果は `cer_strict 3.69  cer_flex 3.02  cer_kana 2.38`** です（公式の公称値は 3.67 / 2.98 / 2.34）。数字が小さいほど、誤りが少ないことを表します。

```
fleurs   650 samples  cer_strict  3.69  cer_flex  3.02  cer_kana  2.38   RTFx 135
```

`RTFx` は、音声の長さが処理時間の何倍かを表します（ここでは約 135 倍速）。**公称の RTFx（3,236）は、別の機械・別の設定の値で、比べられません。**
結果の詳しい読み方と、できていないことは、[README](README.md) の「未検証・言えないこと」を読んでください。

---

## STEP 9（任意）: 入れたパッケージに、既知の脆弱性がないか確かめる

```bash
./audit_deps.sh
```

最後に `No known vulnerabilities found` と出れば、**照合した時点で**、既知の脆弱性はありません（私の環境では、2026-11-03 に、セットアップ直後の 188 パッケージと、STEP 8 の道具を入れたあとの 194 パッケージの、どちらでも 0 件でした）。新しい脆弱性は日々見つかるので、定期的に実行してください。脆弱性が見つかった場合は、パッケージ名と修正版が表示されます。

照合できないものがあります。公式のコード（Haas-ASR）と jacer は PyPI にないため、対象外です。「0 件」は、安全の保証ではありません。詳しくは [README](README.md) の「セキュリティ」を読んでください。

---

## STEP 10（任意）: 公式のサンプルと、ブラウザデモを動かす

公式 Haas-ASR のサンプルとデモは、そのままでは Mac で動きません。公式のコードを書き換えずに動かすための包み、`run_official.py` を使います。

**公式のサンプル（クイックスタート）**: STEP 6 で作った音声を、文字にします。

```bash
.venv/bin/python run_official.py upstream/examples/quickstart.py --model third-intelligence/Haas-ja-v1.0 audio/say_0.wav
```

**語彙の指定**: 専門用語や人名を、指定した書き方で出させる機能です。`--words` の `=` の左が出したい書き方、右が、モデルが書きそうな綴り（カンマ区切り）です。まず、固有名詞を含む音声を作ります。

```bash
mkdir -p feature_in
```

```bash
say -v Kyoko -o feature_in/vocab.aiff "サードインテリジェンスが公開した、ハースという音声認識モデルを、Macで動かしてみました。"
```

```bash
afconvert -f WAVE -d LEI16@16000 -c 1 feature_in/vocab.aiff feature_in/vocab.wav
```

語彙なしで文字にすると、私の環境では「サードインテリジェンスが公開した**first**という音声認識モデルを**mc**で動かしてみました。」になりました（モデルが「ハース」と「Mac」を、そう聞き取ります）。

```bash
.venv/bin/python run_official.py upstream/examples/quickstart.py --model third-intelligence/Haas-ja-v1.0 feature_in/vocab.wav
```

次に、語彙を指定します。モデルが実際に書いてしまった綴り（`first`、`mc`）も、読みの候補に入れるのがコツです。

```bash
.venv/bin/python run_official.py upstream/examples/transcribe_with_vocabulary.py --model third-intelligence/Haas-ja-v1.0 feature_in/vocab.wav --words "Haas=ハース,first,ファースト" "Third Intelligence=サードインテリジェンス" "Mac=マック,mc"
```

私の環境では、「Third Intelligenceが公開したHaasという音声認識モデルをMacで動かしてみました。」と出ました。効き方は音声しだいです。詳しくは [README](README.md) の「語彙の指定について」を読んでください。

**ブラウザデモ**: まず、デモ用の部品を入れます（1回だけ）。

```bash
uv pip install --python .venv/bin/python --override overrides.txt "fastapi>=0.110" "uvicorn>=0.29" "wsproto>=1.2"
```

デモのサーバーを起動します（最初は、モデルの読み込みなどで、少し待ちます）。

```bash
.venv/bin/python run_official.py -m demos.app --device mps
```

ブラウザで http://localhost:7860 を開きます。「長時間音声」の画面では、「音声ファイルを選択…」から、音声を選ぶと、文字になります。「リアルタイム」の画面では、「音声ファイルを再生…」で、字幕が段階的に出ます。サーバーを止めるときは、ターミナルで `Ctrl` + `C` を押します。

- 私の環境では、画面の動作を、**アプリ内のブラウザ 1 種類**で確かめました。**マイク入力（「マイクで話す」）と、Safari や Chrome での動作は、確かめていません。**
- サーバーは、自分の Mac（`127.0.0.1`）からしか見えません。公式の説明どおり、試用のためのもので、本番用ではありません。
- 話者ラベルの機能（`examples/speakers/`）は、別の環境が必要で、確かめていません。

---

## つまずいたとき

| 症状 | 対処 |
|---|---|
| `uv: command not found` | STEP 2 の後に、ターミナルを開き直しましたか。それでも出る場合は、uv の入れ直しをしてください |
| `./setup_env.sh` が `Permission denied` | `zsh setup_env.sh` で実行します |
| ダウンロードがずっと進まない | `HF_HUB_DISABLE_XET=1` を付けて、やり直します。例: `HF_HUB_DISABLE_XET=1 .venv/bin/python run_haas.py --device mps` |
| STEP 8 で `ModuleNotFoundError: jacer`（または `pyarrow`） | STEP 8 の「1. 準備」の 2 つ目のコマンドを、実行してください |
| 動作が遅い・メモリが足りない | ほかのアプリを閉じてください。STEP 8 の測定は、`--batch-size 8` を付けると、使うメモリが減ります（例: `.venv/bin/python run_bench.py --subset fleurs --batch-size 8`） |
| `say` が日本語の声を見つけられない | STEP 6 の注意を見てください |
| `Triton` や `CUDA` に関するエラー | 付属のスクリプト（`run_*.py`）を使わず、ご自分のコードで呼んだ場合は、モデルを読み込む前に `kda_torch.install()` を呼んでいないことが原因かもしれません |
| 「(出力なし)」が多い | STEP 7 の説明を見てください |
| `setup_env.sh` が「未確認の版です」または「想定と違います」と出て止まる | 依存パッケージの版が、確認した版から変わった可能性があります。`patch_compat.py` が、確認していない版を書き換えないように、止めています。リポジトリの持ち主に連絡してください |
| `./audit_deps.sh` が脆弱性を報告した | 表示されたパッケージの修正版を、`overrides.txt` に加えて、`setup_env.sh` をやり直すのが基本です。ただし、版が上がると動かなくなることがあります。持ち主に連絡してください |

※ このうち、`uv` の件、`Permission denied` の件、`ModuleNotFoundError` の件、`Triton` の件、`--batch-size 8` で使うメモリが減るという点、最後の2行（版が変わって止まる件、脆弱性が報告された件）は、起こりうる原因や理屈から書いたもので、実際にその症状を再現したり、効果を測ったりして確かめたものではありません。

## 後片付け（不要になったとき）

このリポジトリで作ったものは、次のとおりです。フォルダごと消せば、元に戻ります。

- リポジトリのフォルダそのもの（`haas-ja-on-mac`）。中に、環境（`.venv`）、公式のコード（`upstream`）、ベンチマークの取得物（`bench_repo`、`bench_data`）、結果（`out`、`bench_out`、`film_test`、`film_out`）が入っています。
- **モデルの重み（約 1.1GB）と、FLEURS の音声（約 0.4GB）**は、リポジトリの外の、`~/.cache/huggingface/` にあります。不要なら、ここから、削除してください。

## 用語の説明

- **音声認識（ASR）**: 音声を、文字にすること。
- **CER（文字誤り率）**: 文字起こしの間違いの割合。小さいほどよい。`strict` は表記の違い（漢字とひらがななど）も間違いに数える厳しい指標、`flex` は表記ゆれをある程度許す指標、`kana` は読みで比べる指標。
- **MPS**: Apple シリコンの GPU を、PyTorch から使う仕組み。`--device mps` で使い、`--device cpu` なら使いません。
- **fp32**: 数値の精度の一種（32 ビットの浮動小数点数）。この手順書の測定は、すべてこれです。公式の公称値は、別の精度（bf16）で測られています。
- **KDA（Kimi Delta Attention）**: このモデルの、デコーダの一部の計算の仕組み。公式の実装は、NVIDIA GPU 専用の部品（Triton）に頼っていて、Mac では動きません。このリポジトリは、この部分だけを、普通の PyTorch で書き直しています（`kda_torch.py`）。
- **RTFx**: 音声の長さが、処理時間の何倍か。大きいほど速い。
- **jacer**: 日本語の文字起こしの誤り率を計算する、公式が固定した評価器。
- **脆弱性**: ソフトウェアの弱点で、悪用されると、意図しない動作（例: 悪意のあるファイルを読み込んだときに、コードが実行される）が起きうるもの。**pip-audit** は、入れたパッケージを、既知の脆弱性のデータベースと照合する道具です。

---

このリポジトリは、非公式の個人の実験です。内容の正しさは保証できません。ライセンスは [MIT](LICENSE) です。
