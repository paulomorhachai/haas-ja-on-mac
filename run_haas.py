"""Haas-ja-v1.0 を macOS(CPU/MPS)で動かす。KDA は kda_torch に差し替え。

使い方: run_haas.py --device cpu|mps
出力は out/ に JSON で残す。
音声は macOS の say(Kyoko) で合成した既知文。合成音声は録音より易しいので、
これは「動くか」の煙テストであって精度評価ではない。
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

SENTENCES = [
    "今日は天気が良いので、公園へ散歩に行きました。",
    "会議の資料は明日の朝までに送ってください。",
    "この音声認識モデルは日本語の文字起こしに特化しています。",
]


def make_audio(i: int, text: str) -> Path:
    wav = HERE / "audio" / f"say_{i}.wav"
    if not wav.exists():
        wav.parent.mkdir(exist_ok=True)
        aiff = wav.with_suffix(".aiff")
        subprocess.run(["say", "-v", "Kyoko", "-o", str(aiff), text], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True
        )
        aiff.unlink()
    return wav


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "mps"], required=True)
    ap.add_argument("--model", default="third-intelligence/Haas-ja-v1.0")
    args = ap.parse_args()

    import torch

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS が使えない")

    import haas_asr  # noqa: F401  NeMo にカスタムデコーダを登録する(先に import する必要がある)
    import kda_torch

    kda_torch.install()

    from haas_asr.audio import load_audio

    audios = [load_audio(str(make_audio(i, t))) for i, t in enumerate(SENTENCES)]
    print("audio seconds:", [round(len(a) / 16000, 2) for a in audios], flush=True)

    t0 = time.time()
    model = haas_asr.load_model(args.model, device=args.device, dtype=torch.float32)
    print(f"load {time.time() - t0:.1f}s  params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    print("decoder class:", type(model.decoder).__name__, "/ prednet layer:", type(model.decoder.prediction["dec_rnn"].layers[0]).__name__, flush=True)

    texts, times = None, []
    for rep in range(2):  # 1回目は初期化込み、2回目を定常として見る
        t0 = time.time()
        with torch.inference_mode():
            texts = haas_asr.transcribe(model, audios, batch_size=8)
        if args.device == "mps":
            torch.mps.synchronize()
        times.append(round(time.time() - t0, 2))
    secs = sum(len(a) for a in audios) / 16000
    result = {
        "device": args.device,
        "texts": texts,
        "expected": SENTENCES,
        "exact_match": [a == b for a, b in zip(texts, SENTENCES)],
        "seconds_per_run": times,
        "audio_seconds": round(secs, 2),
    }
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    (out / f"result_{args.device}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
