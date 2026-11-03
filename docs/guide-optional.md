[← 手順書](../GUIDE.ja.md)

# 任意の手順: ベンチマーク・脆弱性確認・公式デモ

[手順書](../GUIDE.ja.md) の STEP 7 まで終わっていることが前提です。コマンドは、リポジトリのフォルダ(`haas-ja-on-mac`)で実行します。必要なところだけ、お読みください。

## A.  公式ベンチマークで精度を測る

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
結果の詳しい読み方と、できていないことは、[確認結果の詳細](verification.md) の「未検証・言えないこと」を読んでください。

FLEURS の音声（約 0.4GB）は、リポジトリの外の `~/.cache/huggingface/` に置かれます。不要なら、ここから削除してください。

## B.  入れたパッケージに、既知の脆弱性がないか確かめる

```bash
./audit_deps.sh
```

最後に `No known vulnerabilities found` と出れば、**照合した時点で**、既知の脆弱性はありません（私の環境では、2026-11-03 に、セットアップ直後の 188 パッケージと、A の道具を入れたあとの 194 パッケージの、どちらでも 0 件でした）。新しい脆弱性は日々見つかるので、定期的に実行してください。脆弱性が見つかった場合は、パッケージ名と修正版が表示されます。

照合できないものがあります。公式のコード（Haas-ASR）と jacer は PyPI にないため、対象外です。「0 件」は、安全の保証ではありません。詳しくは [セキュリティ](security.md)を読んでください。

## C.  公式のサンプルと、ブラウザデモを動かす

公式 Haas-ASR のサンプルとデモは、そのままでは Mac で動きません。公式のコードを書き換えずに動かすための包み、`run_official.py` を使います。

**公式のサンプル（クイックスタート）**: 手順書の STEP 6 で作った音声を、文字にします。

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

私の環境では、「Third Intelligenceが公開したHaasという音声認識モデルをMacで動かしてみました。」と出ました。効き方は音声しだいです。詳しくは [確認結果の詳細](verification.md) の「語彙の指定について」を読んでください。

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
