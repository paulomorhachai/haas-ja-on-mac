# 第三者の著作物について

このリポジトリは、以下を**同梱していません**(取得手順だけを README に書いています)。
使う場合は、それぞれのライセンスに従ってください。

## flash-linear-attention / fla-core（MIT）

`kda_torch.py` の計算式は、flash-linear-attention 0.5.2 / fla-core 0.5.2 のソース
(`fla/layers/kda.py`、`fla/ops/kda/`、`fla/modules/`)を読んで、素の PyTorch で書き直したものです。
ソースのコピーではありませんが、出典を明らかにし、同ライセンスの著作権表示を以下に掲げます。
`test_kda_torch.py` は、検品のために fla-core の `fla/ops/kda/naive.py`(純 PyTorch の参照実装)を
ファイルとして読み込みます(`setup_env.sh` が `fla_core/` に展開します。このリポジトリには含みません)。

https://github.com/fla-org/flash-linear-attention

```
MIT License

Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Haas-ASR / Haas-ja-v1.0（Apache-2.0）

推論コードとモデルは Third Intelligence のものです。このリポジトリには含めず、
`setup_env.sh` が公式リポジトリを clone します(動作確認したコミット `acc614f`)。
`run_*.py` は、公式の `haas_asr` を import して使います。

- https://github.com/third-intelligence/Haas-ASR
- https://huggingface.co/third-intelligence/Haas-ja-v1.0

## asr-ja-bench と jacer

ベンチのメタデータと評価器です。音声と書き起こしは、ベンチの配布元の方針で同梱されておらず、
このリポジトリにも含めていません。`results/fleurs_summary.json` には、FLEURS の集計値だけを入れています。

- https://huggingface.co/datasets/third-intelligence/asr-ja-bench （サブセットごとにライセンスが異なります）
- https://github.com/third-intelligence/jacer

## FLEURS（CC BY 4.0）

評価に使ったコーパスです(音声と書き起こしは同梱していません)。

Conneau, A., Ma, M., Khanuja, S., Zhang, Y., Axelrod, V., Dalmia, S., Riesa, J., Rivera, C., & Bapna, A. (2022).
FLEURS: Few-shot learning evaluation of universal representations of speech. arXiv:2205.12446.
https://huggingface.co/datasets/google/fleurs
