"""手元の古い日本映画の音声(実録音・雑音あり)で Haas-ja-v1.0 を試す(MPS, fp32)。

参照テキストは無いので、精度は測れない(読んで妥当性を見る定性試験)。
film_test/*.wav(事前に afconvert で 16kHz mono 化)の 25% と 60% 地点から 25 秒ずつ切り出す。git管理外。
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
CLIP_SEC, POINTS = 25, (0.25, 0.60)


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch

    import haas_asr
    import kda_torch

    kda_torch.install()
    out = HERE / "film_test"
    out.mkdir(exist_ok=True)

    clips, meta = [], []
    # wav は afconvert で事前に 16kHz mono にしておく(README 参照。映画の音声は同梱しない)
    for wav in sorted(out.glob("*.wav")):
        x, sr = sf.read(wav, dtype="float32")
        assert sr == 16000, sr
        for p in POINTS:
            s = int(len(x) * p)
            clips.append(np.ascontiguousarray(x[s : s + CLIP_SEC * sr]))
            meta.append({"film": wav.stem, "start_sec": round(s / sr), "total_sec": round(len(x) / sr)})
        print(wav.stem, "total", round(len(x) / sr), "s", flush=True)

    model = haas_asr.load_model("third-intelligence/Haas-ja-v1.0", device="mps", dtype=torch.float32)
    t0 = time.time()
    with torch.inference_mode():
        texts = haas_asr.transcribe(model, clips, batch_size=4)
    torch.mps.synchronize()
    dt = time.time() - t0
    for m, t in zip(meta, texts):
        m["text"] = t
    (out / "result.json").write_text(json.dumps({"seconds": round(dt, 2), "clips": meta}, ensure_ascii=False, indent=1))
    print("done", round(dt, 2), "s for", len(clips), "clips", flush=True)


if __name__ == "__main__":
    main()
