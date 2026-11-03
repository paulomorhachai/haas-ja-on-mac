"""日本映画を通しで文字起こしする(MPS・fp32)。断片ごとの時刻付き。

公式 haas_asr.transcribe_longform と同じ処理(split_at_quiet_points で約30秒ごとに静かな箇所で切り、
haas_asr.transcribe で文字起こし)だが、本文だけでなく各断片の開始・終了時刻を残す。
film_test/*.wav(16kHz mono、事前に afconvert で作成)を読む。台詞はログに出さない(件数と時間だけ)。
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
BATCH = 16


def hms(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def main() -> None:
    import soundfile as sf
    import torch

    import haas_asr
    import kda_torch
    from haas_asr.longform import split_at_quiet_points

    kda_torch.install()
    out = HERE / "film_out"
    out.mkdir(exist_ok=True)

    t0 = time.time()
    model = haas_asr.load_model("third-intelligence/Haas-ja-v1.0", device="mps", dtype=torch.float32)
    print(f"model loaded {time.time() - t0:.1f}s", flush=True)

    summary = []
    for wav in sorted((HERE / "film_test").glob("*.wav")):
        audio, sr = sf.read(wav, dtype="float32")
        assert sr == 16000 and audio.ndim == 1, (sr, audio.shape)
        bounds = split_at_quiet_points(audio)
        chunks = [audio[a:b] for a, b in bounds]
        t0 = time.time()
        texts = []
        for i in range(0, len(chunks), BATCH):
            with torch.inference_mode():
                texts += haas_asr.transcribe(model, chunks[i : i + BATCH], batch_size=BATCH)
            torch.mps.empty_cache()
            if (i // BATCH) % 10 == 0:
                print(f"{wav.stem[:20]} {len(texts)}/{len(chunks)} {time.time() - t0:.0f}s", flush=True)
        wall = time.time() - t0
        dur = len(audio) / sr
        rows = [{"start": a / sr, "end": b / sr, "text": t} for (a, b), t in zip(bounds, texts)]
        (out / f"{wav.stem}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        with open(out / f"{wav.stem}.txt", "w", encoding="utf-8") as f:
            f.write(f"# {wav.stem}  全長 {hms(dur)}  断片 {len(rows)} 個(約30秒ごと、静かな箇所で分割)\n")
            f.write("# [開始 - 終了] 本文 / 出力なしの断片は (出力なし)\n\n")
            for r in rows:
                f.write(f"[{hms(r['start'])} - {hms(r['end'])}] {r['text'] or '(出力なし)'}\n")
        empty = sum(1 for r in rows if not r["text"])
        chars = sum(len(r["text"]) for r in rows)
        summary.append({"film": wav.stem, "audio_sec": round(dur, 1), "wall_sec": round(wall, 1),
                        "rtfx": round(dur / wall, 1), "chunks": len(rows), "empty_chunks": empty, "chars": chars})
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("all done", flush=True)


if __name__ == "__main__":
    main()
