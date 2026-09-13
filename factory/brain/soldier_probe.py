"""병정 초파리 반사 측정 — 괴물과 싸우는 규칙을 정하기 전의 실측. 뇌 모델 환경.

  ~/flybrain/shiu-brain-model/.venv/bin/python factory/brain/soldier_probe.py [--seeds 5]

stimuli.soldier: 좀비 개미(작게 움직임)·거미 괴물(루밍) × 왼쪽·오른쪽, 빈 바닥 × 씨앗
→ 다가가기(oDN1·P9)·거대섬유 50ms 최고·방향 틀기(DNa02 왼쪽−오른쪽)·후진(MDN) + 눈별 시각 특징
확인할 것:
  1. 개미에 다가가기가 빈 바닥보다 확실히 켜지나 → 달려들기
  2. 거미에 거대섬유가 문턱(60Hz)을 넘나, 개미에는 안 넘나 → 도망
  3. 왼쪽·오른쪽 괴물에 방향 틀기 부호가 갈리나 → 몸 돌릴 쪽(안 갈리면 방향은 게임이 정하지 않고 쓰지 않는다)
산출: factory/brain/soldier_probe.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stimuli  # noqa: E402
from fbrain import FactoryBrain, log  # noqa: E402

KEYS = ("approach_hz", "GF_peak50ms_hz", "turn_L_minus_R_hz", "MDN_mean_hz")


def stat(xs):
    a = np.asarray(xs, float)
    return {"mean": round(float(a.mean()), 3), "sd": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 3),
            "values": [round(float(x), 3) for x in a]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    brain = FactoryBrain(fps=10.0)
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "build_s": brain.build_s, "sigma": brain.sigma, "window_s": 0.5,
           "conditions": {}}
    conds = [("ant", "left"), ("ant", "right"), ("spider", "left"), ("spider", "right"), ("none", "left")]
    for kind, side in conds:
        runs = [brain.decide(stimuli.soldier(kind, side, 0.5, 10, seed=s), seed=s) for s in range(a.seeds)]
        name = kind if kind == "none" else f"{kind}_{side}"
        c = {k: stat([r["values"][k] for r in runs]) for k in KEYS}
        c["wall_s"] = stat([r["wall_s"] for r in runs])
        c["features_seed0"] = runs[0].get("features")
        c["vpn_rate_hz_seed0"] = runs[0].get("vpn_rate_hz")
        out["conditions"][name] = c
        log("SOLDIER", name, json.dumps({k: c[k]["mean"] for k in KEYS}), "GF", c["GF_peak50ms_hz"]["values"],
            "turn", c["turn_L_minus_R_hz"]["values"])
    (HERE / "soldier_probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    log("PROBE_WROTE", str(HERE / "soldier_probe.json"))


if __name__ == "__main__":
    main()
