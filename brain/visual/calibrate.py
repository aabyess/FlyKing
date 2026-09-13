"""도파민 뉴런 문턱 보정 — 회색 정지 화면 대조군. 뇌 모델 환경에서 실행한다.

  ~/flybrain/shiu-brain-model/.venv/bin/python brain/visual/calibrate.py [--n 8] [--seconds 1.0]

문턱 = 대조군 N회(포아송 씨앗만 다름) 평균 + K·표준편차(K=3).
  「아무 변화 없는 화면을 볼 때의 흔들림 범위를 뚜렷이 넘으면 반응했다」로 본다. 문턱 숫자를 손으로 고르지 않기 위한 방식.
  표준편차가 0이면(발화가 거의 없으면) 최소 검출 단위(무리 전체에서 스파이크 1개)로 바닥을 둔다.
거대섬유 도주 문턱은 보정하지 않고 embodied brain_body_bridge 값(50ms 창 정규화 0.3 × 200Hz = 60Hz)을 그대로 쓴다.
루밍·줄무늬·깜빡임 결과도 함께 적어, 시각 입력이 도파민 뉴런·거대섬유까지 전달되는지 확인한다.

산출: brain/visual/calibration.json (git에 넣는다 — 허브가 읽는 문턱)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import eye
from brain_eval import CHANGE_REF, RATE_SUSTAINED, RATE_TRANSIENT, Brain, brief, log

HERE = Path(__file__).resolve().parent
K = 3.0
GF_THRESHOLD_HZ = 60.0


def stats(values, floor):
    v = np.asarray(values, float)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return {"mean": round(float(v.mean()), 4), "sd": round(sd, 4), "sd_used": round(max(sd, floor), 4),
            "values": [round(float(x), 4) for x in v]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=10.0)
    a = ap.parse_args()
    brain = Brain(fps=a.fps)
    gray = eye.synthetic("gray", a.seconds, a.fps)
    runs = []
    for s in range(a.n):
        r = brain.run(gray, a.fps, seed=1000 + s)
        runs.append(r)
        log("CONTROL", s, json.dumps(brief(r)))
    floor = {name: 1.0 / (a.seconds * len(brain.dan[name]["ids"])) for name in ("PAM", "PPL1")}
    control = {name: stats([r["dopamine"][name]["mean_hz"] for r in runs], floor[name]) for name in ("PAM", "PPL1")}
    control["GF_peak50ms_hz"] = stats([r["groups"]["GF"]["peak50ms_hz"] for r in runs], 0.0)
    threshold = {name: round(control[name]["mean"] + K * control[name]["sd_used"], 4) for name in ("PAM", "PPL1")}
    threshold["GF_peak50ms_hz"] = GF_THRESHOLD_HZ
    sanity = {}
    for kind in ("loom", "bars", "flicker"):
        r = brain.run(eye.synthetic(kind, a.seconds, a.fps), a.fps, seed=7)
        sanity[kind] = brief(r)
        log("SANITY", kind, json.dumps(sanity[kind]))
    out = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": f"회색 정지 화면 {a.n}회 평균 + {K}·표준편차(바닥: 무리 전체 스파이크 1개). GF는 brain_body_bridge 0.3×200Hz.",
        "model": "Shiu et al. 2024, FlyWire v783, model.py default_params",
        "stimulus": {"seconds": a.seconds, "fps": a.fps},
        "input_params": {"RATE_SUSTAINED": RATE_SUSTAINED, "RATE_TRANSIENT": RATE_TRANSIENT, "CHANGE_REF": CHANGE_REF},
        "n_neurons": {name: len(brain.dan[name]["ids"]) for name in ("PAM", "PPL1")},
        "control": control, "threshold": threshold, "sanity": sanity,
    }
    (HERE / "calibration.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log("CALIBRATION_WROTE", json.dumps(threshold))


if __name__ == "__main__":
    main()
