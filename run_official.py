"""公式 Haas-ASR のサンプルやデモを、中身を変えずに Mac で動かすための起動用の包み。

公式のサンプルは `haas_asr.load_model(model)` を既定のデバイス(cuda)で呼び、KDA は fla(Triton)に頼る。
この包みは、サンプルを動かす前に、(1) KDA を kda_torch に差し替え、(2) load_model の既定のデバイスを
環境変数 HAAS_DEVICE(既定 mps)にするだけで、公式のコードは書き換えない。

使い方(リポジトリの直下から):
    .venv/bin/python run_official.py upstream/examples/quickstart.py --model third-intelligence/Haas-ja-v1.0 a.wav
    .venv/bin/python run_official.py -m demos.app --device mps       (デモ。upstream/ が import できる位置で起動される)
"""

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "upstream"))  # demos を import できるようにする

import haas_asr  # noqa: E402
import kda_torch  # noqa: E402

kda_torch.install()

_original_load_model = haas_asr.load_model


def _load_model(model, *, device=None, dtype=haas_asr.model.AUTO):
    return _original_load_model(model, device=device or os.environ.get("HAAS_DEVICE", "mps"), dtype=dtype)


haas_asr.load_model = _load_model


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "-m":
        module = sys.argv[2]
        sys.argv = [module] + sys.argv[3:]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    else:
        script = sys.argv[1]
        sys.argv = [script] + sys.argv[2:]
        runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
