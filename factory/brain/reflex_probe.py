"""작업대 반사 측정 — 게임 규칙·문턱·속도를 정하기 전의 실측. 뇌 모델 환경.

  ~/flybrain/shiu-brain-model/.venv/bin/python factory/brain/reflex_probe.py [--sigma 0]

1. 불빛 분류대: 밝은 상자·어두운 상자·빈 벨트 × 씨앗 6 → 다가가기(oDN1·P9)·거대섬유·방향 틀기
2. 경비 초소: 침입자(루밍)·구름·빈 하늘 × 씨앗 6 → 거대섬유 50ms 최고
3. 설탕대: 당 감각뉴런 0·25·50·100·150·200Hz × 씨앗 3 → MN9
4. 속도: 판단 창 0.3·0.5·1.0초 계산 시간, 프로세스 최대 메모리
5. 성격: σ(jitter_basis.json)로 초파리 4마리 × 분류대(어두운/밝은)·경비(침입자) 반응 차이
산출: factory/brain/reflex_probe.json
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stimuli  # noqa: E402
from fbrain import FactoryBrain, log  # noqa: E402


def stat(xs):
    a = np.asarray(xs, float)
    return {"mean": round(float(a.mean()), 3), "sd": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 3), "values": [round(float(x), 3) for x in a]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=None)
    a = ap.parse_args()
    brain = FactoryBrain(fps=10.0, sigma=a.sigma)
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "build_s": brain.build_s, "sigma": brain.sigma}

    res = {}
    for kind in ("bright", "dark", "none"):
        runs = [brain.decide(stimuli.sorter(kind, 0.5, 10, seed=s), seed=s) for s in range(6)]
        res[kind] = {k: stat([r["values"][k] for r in runs]) for k in ("approach_hz", "GF_peak50ms_hz", "turn_L_minus_R_hz", "PPL1_mean_hz")}
        res[kind]["wall_s"] = stat([r["wall_s"] for r in runs])
        log("SORTER", kind, json.dumps({k: v["mean"] for k, v in res[kind].items()}))
    out["sorter"] = res

    res = {}
    for kind in ("intruder", "clouds", "none"):
        runs = [brain.decide(stimuli.guard(kind, 0.5, 10, seed=s), seed=s) for s in range(6)]
        res[kind] = {k: stat([r["values"][k] for r in runs]) for k in ("GF_peak50ms_hz", "approach_hz", "MDN_mean_hz")}
        log("GUARD", kind, json.dumps({k: v["mean"] for k, v in res[kind].items()}), "GF", res[kind]["GF_peak50ms_hz"]["values"])
    out["guard"] = res

    res = {}
    for hz in (0, 25, 50, 100, 150, 200):
        runs = [brain.decide(stimuli.sugar(0.5, 10), sugar_hz=hz, seed=s) for s in range(3)]
        res[str(hz)] = {"MN9_mean_hz": stat([r["values"]["MN9_mean_hz"] for r in runs]), "PAM_mean_hz": stat([r["values"]["PAM_mean_hz"] for r in runs])}
        log("SUGAR", hz, json.dumps({k: v["mean"] for k, v in res[str(hz)].items()}))
    out["sugar"] = res

    res = {}
    for win in (0.3, 0.5, 1.0):
        runs = [brain.decide(stimuli.sorter("dark", win, 10, seed=s), seed=s) for s in range(3)]
        res[str(win)] = stat([r["wall_s"] for r in runs])
        log("TIMING", win, res[str(win)])
    out["timing_wall_s"] = res
    out["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)   # macOS는 바이트 단위
    log("MEMORY_MB", out["max_rss_mb"])

    if brain.sigma > 0:
        res = {}
        for fly in range(4):
            dark = [brain.decide(stimuli.sorter("dark", 0.5, 10, seed=s), fly_seed=100 + fly, seed=s)["values"]["approach_hz"] for s in range(4)]
            bright = [brain.decide(stimuli.sorter("bright", 0.5, 10, seed=s), fly_seed=100 + fly, seed=s)["values"]["approach_hz"] for s in range(4)]
            loom = [brain.decide(stimuli.guard("intruder", 0.5, 10, seed=s), fly_seed=100 + fly, seed=s)["values"]["GF_peak50ms_hz"] for s in range(4)]
            res[f"fly{fly}"] = {"sorter_dark_approach": stat(dark), "sorter_bright_approach": stat(bright), "guard_intruder_GF": stat(loom)}
            log("FLY", fly, json.dumps({k: v["mean"] for k, v in res[f"fly{fly}"].items()}))
        out["individuals"] = res
        brain.set_fly(None)

    (HERE / "reflex_probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log("PROBE_WROTE", str(HERE / "reflex_probe.json"))


if __name__ == "__main__":
    main()
