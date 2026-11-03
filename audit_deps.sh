#!/bin/zsh
# インストール済みのパッケージを、既知の脆弱性データベースと照合する(pip-audit を一時的に使う)。
# 新しい脆弱性は日々見つかるので、定期的に実行してください。脆弱性が見つかると、終了コード 1 で終わります。
set -e
cd "$(dirname "$0")"
REQ="$(mktemp)"
trap 'rm -f "$REQ"' EXIT
# 公式のコード(upstream)と jacer は PyPI に無いので、照合の対象から外す
uv pip freeze --python .venv/bin/python | grep -v -E ' @ |^-e' > "$REQ"
echo "対象: $(wc -l < "$REQ" | tr -d ' ') パッケージ（公式のコードと jacer は PyPI に無いため、対象外）"
uvx pip-audit -r "$REQ" --no-deps --disable-pip --progress-spinner off
