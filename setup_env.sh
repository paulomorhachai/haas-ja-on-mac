#!/bin/zsh
# Haas-ja-v1.0 を macOS で動かすための環境。flash-linear-attention は入れない(Triton が無いため)。
set -e
cd "$(dirname "$0")"
[ -d upstream ] || git clone https://github.com/third-intelligence/Haas-ASR.git upstream
git -C upstream checkout -q acc614f   # 動作確認したコミット
uv venv --python 3.12 .venv
# nemo は上限を公式と同じ <2.8 に固定。haas-asr 本体は依存解決を避けて --no-deps で入れる。
# overrides.txt: 既知の脆弱性のある版を避けるため、一部のパッケージを修正版以上に固定する(README の「セキュリティ」参照)。
uv pip install --python .venv/bin/python --override overrides.txt "nemo-toolkit[asr]>=2.7.3,<2.8" numpy soundfile librosa einops
uv pip install --python .venv/bin/python --no-deps -e upstream
# lightning 2.6.6 と NeMo 2.7.3 を組み合わせるための、2か所の小さな互換修正(版が想定と違えば、書き換えずに止まる)
.venv/bin/python patch_compat.py
# test_kda_torch.py が参照実装(fla-core の naive.py)を読み込むために使う。展開するだけで import はしない
uv pip install --no-deps --target fla_core "fla-core==0.5.2"
echo SETUP_DONE
