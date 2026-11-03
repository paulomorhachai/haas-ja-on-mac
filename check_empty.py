"""検品: 「出力なし」の断片が本当に無音か、音があるのに取りこぼしたのかを音量で仕分ける。"""

import glob
import json

import numpy as np
import soundfile as sf

lines = []
for f in sorted(glob.glob("film_out/*.json")):
    name = f.split("/")[-1][:-5]
    if name == "summary":
        continue
    rows = json.load(open(f))
    x, sr = sf.read(f"film_test/{name}.wav", dtype="float32")
    n_empty = sum(1 for r in rows if not r["text"])
    quiet, suspects = 0, []
    for r in rows:
        if r["text"]:
            continue
        c = x[int(r["start"] * sr) : int(r["end"] * sr)]
        w = c[: len(c) // sr * sr].reshape(-1, sr)
        db = 20 * np.log10(np.sqrt((w**2).mean(1)) + 1e-9)
        silent_frac = float((db < -50).mean())
        if silent_frac >= 0.8:
            quiet += 1
        else:
            suspects.append((int(r["start"]), round(float(np.median(db))), round(silent_frac, 2)))
    short = sum(1 for r in rows if 0 < len(r["text"]) < 10)
    lines.append(f"## {name}")
    print(f"{name[:26]:26s} 出力なし{n_empty}個 = 無音(80%以上の窓が-50dB未満){quiet}個 + 音あり{len(suspects)}個 ; 10文字未満の断片 {short}個")
    for s in suspects:
        print(f"    音ありで出力なし: 開始 {s[0] // 60}分{s[0] % 60:02d}秒  中央値{s[1]}dB  無音窓率{s[2]}")
        lines.append(f"- {s[0] // 3600:02d}:{s[0] % 3600 // 60:02d}:{s[0] % 60:02d} から約30秒  音量(中央値){s[1]}dB  無音の窓の割合{s[2]}")
    lines.append("")
open("film_out/出力なし断片の一覧.md", "w", encoding="utf-8").write("# 出力なしだが音がある断片(耳で確認してほしい箇所)\n\n音量が小さいほど(-50dBに近いほど)無音に近い。-20dB前後は音楽や効果音の可能性が高いが、判別は人の耳で。\n\n" + "\n".join(lines) + "\n")
