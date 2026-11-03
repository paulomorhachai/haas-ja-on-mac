"""asr-ja-bench(取り直し済みのサブセット)を MPS・fp32 で評価する。

公式 examples/evaluate_asr_ja_bench.py の関数(read_subset / transcribe / write_hypotheses / score_line)を
そのまま import して使う。公式との違いは次の3点だけ:
  1. モデルは device=mps, dtype=float32 で読み、KDA を kda_torch に差し替える(公式は cuda/bf16)
  2. バッチは公式の 512 ではなく小さくする(MPS は形状ごとにキャッシュを溜めるため)。バッチごとに
     torch.mps.empty_cache() を呼ぶ。バッチの切り出し順は公式と同じ(ベンチの順)
  3. 全サブセットの重み付き平均は、取り直したサブセットだけで計算する(公式の全体値とは別物)
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def load_official():
    spec = importlib.util.spec_from_file_location("official_eval", HERE / "upstream/examples/evaluate_asr_ja_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", action="append", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 件だけ(煙テスト用、0=全件)")
    ap.add_argument("--out", type=Path, default=HERE / "bench_out")
    args = ap.parse_args()

    import jacer
    import resource
    import torch

    ev = load_official()
    import haas_asr
    import kda_torch

    kda_torch.install()
    bench = HERE / "bench_data"
    ev.check_bench(bench, tuple(args.subset))

    model = haas_asr.load_model("third-intelligence/Haas-ja-v1.0", device="mps", dtype=torch.float32)
    print("model loaded; metrics", jacer.METRICS_VERSION, flush=True)

    results = {}
    for subset in args.subset:
        rows = ev.read_subset(bench, subset)
        if args.limit:
            rows = rows[: args.limit]
        hyps, t0 = [], time.time()
        for i in range(0, len(rows), args.batch_size):
            with torch.inference_mode():
                hyps += ev.transcribe(model, bench, rows[i : i + args.batch_size], args.batch_size)
            torch.mps.empty_cache()
            if (i // args.batch_size) % 5 == 0:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
                print(f"{subset} {len(hyps)}/{len(rows)}  {time.time() - t0:.0f}s  maxrss={rss:.2f}GB", flush=True)
        dt = time.time() - t0
        refs = [r["text"] for r in rows]
        cer = jacer.cer(refs, hyps, jacer.METRICS)
        hours = sum(r["duration"] for r in rows) / 3600
        results[subset] = {"samples": len(rows), "hours": hours, "wall_sec": dt, "rtfx": hours * 3600 / dt, **cer}
        print(ev.score_line(subset, len(rows), cer), f"  RTFx {hours * 3600 / dt:.0f}", flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        ev.write_hypotheses(args.out / f"{subset}.jsonl", rows, hyps)

    summary = {
        "device": "mps", "dtype": "float32", "batch_size": args.batch_size, "limit": args.limit,
        "subsets": results, "metrics_version": jacer.METRICS_VERSION, "scoring": jacer.scoring_environment(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("summary written", flush=True)


if __name__ == "__main__":
    main()
