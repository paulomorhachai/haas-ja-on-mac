"""lightning 2.6.6 を使うための、NeMo 2.7.3 まわりの小さな互換修正(セットアップ時に1回だけ実行する)。

背景: lightning 2.4.0 には既知の脆弱性がある(チェックポイントの読み込みでコードが実行される問題など)。
修正版は 2.6.6 だが、NeMo 2.7.3 は 2.4.0 までを想定しており、そのままでは import に失敗する。
食い違いは次の2か所だけ(NeMo が lightning から読み込む 116 種類の名前を全部調べ、実在しないのは1件だけだった)。

1. nv-one-logger 2.3.1(NeMo が import 時に読み込む)の `save_checkpoint` が `weights_only: bool` と宣言しているが、
   lightning 2.6.6 は `Optional[bool]`。`overrides` ライブラリが import 時にこの食い違いを TypeError にする。
   nv-one-logger の最新版も 2.3.1 で、直した版はまだ無い(2026-11-03 時点)。型の宣言を直すだけ。
2. NeMo の `utils/exp_manager.py` が `lightning.pytorch.loggers` から `NeptuneLogger` を読み込むが、2.6.6 では
   その入口から公開されなくなった(実体は `lightning.pytorch.loggers.neptune` にある)。読み込み先を直すだけ。

どちらも、学習時にしか使わない部分で、文字起こしの動作には影響しない。

安全のための決まり(各修正について): 対象パッケージが確認済みの版でない、該当箇所がちょうど1つでない、の
どちらかなら、書き換えずに止まる。すでに直っていれば何もしない。
"""

import importlib.metadata as md
import importlib.util
import sys
from pathlib import Path

PATCHES = [
    {
        "name": "nv-one-logger の型宣言を lightning 2.6.6 に合わせる",
        "package": "nv-one-logger-pytorch-lightning-integration",
        "verified": "2.3.1",
        "module": "nv_one_logger",
        "relpath": "training_telemetry/integration/pytorch_lightning.py",
        "old": "weights_only: bool = False",
        "new": "weights_only: Optional[bool] = False",
    },
    {
        "name": "NeMo の NeptuneLogger の読み込み先を直す",
        "package": "nemo-toolkit",
        "verified": "2.7.3",
        "module": "nemo",
        "relpath": "utils/exp_manager.py",
        "old": "from lightning.pytorch.loggers import MLFlowLogger, NeptuneLogger, TensorBoardLogger, WandbLogger",
        "new": (
            "from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger, WandbLogger\n"
            "from lightning.pytorch.loggers.neptune import NeptuneLogger"
        ),
    },
]


def apply(p: dict) -> None:
    spec = importlib.util.find_spec(p["module"])  # import せずに場所だけ調べる(import すると失敗するため)
    if spec is None or not spec.submodule_search_locations:
        sys.exit(f"{p['module']} が見つかりません")
    target = Path(list(spec.submodule_search_locations)[0]) / p["relpath"]
    text = target.read_text()
    if p["old"] not in text and p["new"] in text:
        print(f"[不要] {p['name']}(すでに直っています)")
        return
    version = md.version(p["package"])
    if version != p["verified"]:
        sys.exit(f"{p['package']} {version} は未確認の版です(確認済み: {p['verified']})。書き換えません")
    n = text.count(p["old"])
    if n != 1:
        sys.exit(f"想定と違います: 該当箇所が {n} 個({target.name})。書き換えません")
    target.write_text(text.replace(p["old"], p["new"]))
    print(f"[修正] {p['name']}: {target}")


def main() -> None:
    for p in PATCHES:
        apply(p)


if __name__ == "__main__":
    main()
