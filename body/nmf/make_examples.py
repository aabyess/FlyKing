"""apply_joint_angles.py 시험용 예시 데이터 — ~/flybrain/shiu-brain-model/.venv/bin/python make_examples.py
  walk_step_flygym_v1.csv/.npz : flygym_demo/complex_terrain/assets/single_steps_untethered.pkl 그대로(v1 이름 42열, 45줄 = 한 걸음, 여섯 다리 같은 위상)
  walk_tripod_flygym_v1.csv    : 같은 한 걸음 조각을 삼각 보행 위상으로 민 것(LF·RM·LH 0, RF·LM·RH 반 주기) — flygym 원본이 아니라 조각을 밀어 붙인 예시
  wingbeat_cpg.csv             : 날갯짓 발생기 열만(0~0.05초, 200Hz, 진폭 120→150°, 비대칭 0→0.3) — 합성 모드 시험용"""
import csv
import pickle
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "examples"
SRC = Path.home() / "flybrain/flygym/src/flygym_demo/complex_terrain/assets/single_steps_untethered.pkl"
d = pickle.load(open(SRC, "rb"))
names = [k for k in d if k.startswith("joint_")]
table = np.stack([np.asarray(d[n], dtype=float) for n in names], 1)


def write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f"{x:.6f}" for x in r])


write(OUT / "walk_step_flygym_v1.csv", names, table)
np.savez(OUT / "walk_step_flygym_v1.npz", joint_names=np.array(names), angles=table)
half = len(table) // 2
shift = np.array([half if n.split("_")[1][:2] in ("RF", "LM", "RH") else 0 for n in names])
tripod = np.stack([np.roll(table[:, i], -shift[i]) for i in range(len(names))], 1)
write(OUT / "walk_tripod_flygym_v1.csv", names, np.vstack([tripod, tripod[:1]]))
t = np.linspace(0.0, 0.05, 11)
write(OUT / "wingbeat_cpg.csv", ["time", "wingbeat_freq", "wingbeat_amp", "wingbeat_asym"],
      np.stack([t, np.full_like(t, 200.0), np.linspace(120, 150, len(t)), np.linspace(0, 0.3, len(t))], 1))
print("wrote", sorted(p.name for p in OUT.iterdir()), "rows", table.shape)
