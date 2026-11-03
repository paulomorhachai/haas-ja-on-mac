"""語彙のブースト(NeMo の GPU フレーズブースト)が Mac で効くかを、強さ alpha を変えて切り分ける。

macOS の say で、固有名詞(サードインテリジェンス・ハース・Mac)を含む文を合成して使う。まず語彙なしの基準を出し、
次に alpha を変えて、語彙(読みは「ハース」「マック」だけ)を与えたときの出力が、基準からどう変わるかを見る。
guard=False はブーストの生の効果、guard=True は公式の誤爆防止(語彙の語を含む変化だけを採用)を通した結果。
結果の解釈は README の「公式の機能の確認結果」を参照。
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch

import haas_asr
import kda_torch

kda_torch.install()

wav = HERE / "feature_in" / "vocab.wav"
if not wav.exists():
    wav.parent.mkdir(exist_ok=True)
    aiff = wav.with_suffix(".aiff")
    text = "サードインテリジェンスが公開した、ハースという音声認識モデルを、Macで動かしてみました。"
    subprocess.run(["say", "-v", "Kyoko", "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True)

model = haas_asr.load_model("third-intelligence/Haas-ja-v1.0", device="mps", dtype=torch.float32)
audio = [haas_asr.load_audio(str(wav))]

print("基準(語彙なし)             :", haas_asr.transcribe(model, audio)[0])
for alpha in (2.0, 5.0, 10.0, 30.0, 100.0):
    vocab = haas_asr.Vocabulary({"Haas": ["ハース"], "Mac": ["マック"]}, alpha=alpha)
    for guard in (False, True):
        try:
            out = haas_asr.transcribe(model, audio, vocabulary=vocab, guard=guard)[0]
        except Exception as e:
            out = "ERROR %s: %s" % (type(e).__name__, str(e)[:200])
        print("alpha=%-5s guard=%-5s: %s" % (alpha, guard, out))
