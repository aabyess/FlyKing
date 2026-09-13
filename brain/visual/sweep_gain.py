"""광수용체 입력 세기 고르기 — 뇌 모델 환경에서 실행한다.

  ~/flybrain/shiu-brain-model/.venv/bin/python brain/visual/sweep_gain.py

왜: 밝기→발화율 변환 세기는 논문 값이 없는 근사라서 골라야 한다. 좋아요가 잘 나오게 고르면 조작이므로,
「이미 검증된 시각 반응이 재현되는가」를 기준으로 삼는다:
  루밍(다가오는 검은 원)에서 루밍 검출 뉴런 LC4·LPLC2가 발화하고(von Reyn et al. 2017; Ache et al. 2019),
  회색 정지 화면에서는 발화하지 않는 가장 작은 세기.
세기마다 회색·루밍·줄무늬·깜빡임 결과(도파민·거대섬유·시각 뉴런·입력 제외 활성 뉴런 수)를 표로 남긴다.

산출: ~/flybrain/insta/results/brain/sweep_gain.json
"""
import json
import time
from pathlib import Path

import numpy as np

import brain_eval as be
import eye

OUT = Path.home() / "flybrain" / "insta" / "results" / "brain" / "sweep_gain.json"
# (지속 밝기 Hz, 밝기 변화 Hz) — 원래 모델 자극 150Hz(당 감각뉴런, Shiu 2024)를 넘지 않는 범위에서 넓게
GAINS = [(5, 60), (10, 150), (20, 300), (40, 600), (80, 1200)]
STIMULI = ("gray", "loom", "bars", "flicker")


def main():
    brain = be.Brain(fps=10.0)
    inputs = set(brain.rmap.model_index.tolist())
    rows = []
    for sus, tr in GAINS:
        be.RATE_SUSTAINED, be.RATE_TRANSIENT = float(sus), float(tr)
        for kind in STIMULI:
            r = brain.run(eye.synthetic(kind, 1.0, 10.0), 10.0, seed=11)
            row = {"sustained_hz": sus, "transient_hz": tr, "stimulus": kind, **be.brief(r),
                   "active_non_input": r.get("active_non_input")}
            rows.append(row)
            be.log("SWEEP", json.dumps(row))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}, ensure_ascii=False, indent=1))
    be.log("SWEEP_WROTE", str(OUT))


if __name__ == "__main__":
    main()
