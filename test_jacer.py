"""検品: jacer(v1.2、公式固定コミット)が、答えの分かる入力で期待どおりの CER を返すか。

期待値は jacer の実装を読まずに、文字数・読みの文字数から手計算したもの。
(初版は「strict>=flex>=kana」を検査したが誤った前提で、kana は読みが分母のため外した)
"""

import sys

import jacer

print("METRICS =", jacer.METRICS, " version =", jacer.METRICS_VERSION)
fails = []


def run(name, refs, hyps, expect):
    got = jacer.cer(refs, hyps, jacer.METRICS)
    ok = all(abs(got[m] - e) < 1e-9 for m, e in expect.items())
    mono = True  # 「strict>=flex>=kana」は誤った前提だった: kana は読みの文字数が分母なので strict より大きくなりうる
    print(("OK  " if ok and mono else "FAIL"), name, {m: round(v, 4) for m, v in got.items()}, "期待", expect)
    if not (ok and mono):
        fails.append(name)


# 1. 完全一致は 0
run("完全一致", ["今日は天気が良いです"], ["今日は天気が良いです"], {"cer_strict": 0.0, "cer_flex": 0.0, "cer_kana": 0.0})
# 2. 空の仮説は 100%(全部脱落)
run("空の仮説", ["今日は天気が良いです"], [""], {"cer_strict": 1.0, "cer_flex": 1.0, "cer_kana": 1.0})
# 3. 5 文字の参照に 5 文字の挿入 = 100%
run("挿入5/5", ["ありがとう"], ["ありがとうございます"], {"cer_strict": 1.0})
# 4. 1 文字の置換(良→悪、読みも違う)。参照 8 文字なので strict は 1/8
run("置換1/8(良→悪)", ["今日は天気が良い"], ["今日は天気が悪い"], {"cer_strict": 1 / 8, "cer_flex": 1 / 8, "cer_kana": 2 / 10})  # 読み キョウハテンキガヨイ(10)->…ワルイ: 置換1+挿入1
# 5. micro 平均: (1 誤り + 0 誤り) / (8 + 2 文字) = 0.1。平均の平均(0.0625)ではない
run("micro平均", ["今日は天気が良い", "はい"], ["今日は天気が悪い", "はい"], {"cer_strict": 1 / 10, "cer_kana": 2 / 12})  # 読み 10+2=12 文字に誤り2

# 6. 漢字⇔かなの表記ゆれ: strict は誤りと数えるが、kana(読みで比べる)は 0 になるはず
got = jacer.cer(["今日は天気が良い"], ["きょうはてんきがよい"], jacer.METRICS)
ok = got["cer_strict"] > 0 and got["cer_kana"] == 0
print(("OK  " if ok else "FAIL"), "表記ゆれ(漢字→かな)", {m: round(v, 4) for m, v in got.items()}, "期待: strict>0, kana==0")
if not ok:
    fails.append("表記ゆれ")

print("\nALL OK" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
