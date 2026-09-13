"""FlyWire 광수용체 뉴런 → 겹눈 낱눈 배정(근사). 뇌 모델 환경(pandas)에서 쓴다.

근거(2026-09-13 조사, ~/flybrain/embodied-fly-brain/data/flywire_annotations.tsv):
- cell_class == "visual", cell_type ∈ {R1-6, R7, R8}. 모델(Completeness_783.csv)에 있는 수:
  R1-6 왼 4042·오 3888, R7 왼 629·오 629, R8 왼 623·오 614. 등쪽 가장자리(DRA) 유형은 편광 전용이라 뺀다.
- 좌우 눈 = 주석 `side`(확실).
- ⚠ 어느 뉴런이 어느 낱눈인지 FlyWire 주석에는 없다(Male CNS의 assignedOlHex는 R 세포에 비어 있음).
  그래서 뉴런 대표점(pos_x/y/z)으로 망막 위치를 근사한다:
    고각(위아래) ← FlyWire y(배쪽으로 커짐)의 순위, 방위각(앞뒤) ← z의 순위.
    R1-6은 라미나, R7·R8은 수질에서 끝나고 그 사이 첫 시각 교차에서 앞뒤가 뒤집히므로 R7·R8은 부호를 반대로 둔다.
  대표점이 축삭 어디에 찍혔는지 알 수 없어 방향·부호는 검증되지 않은 근사다. 판단에 쓰는 하강뉴런 반응은 대부분 시야 전체를
  모으므로 이 근사의 영향은 좌우 눈 구분보다 작다고 본다.
- 채널: R1-6은 넓은 파장(Rh1) → pale·yellow 평균, R7(자외선) → pale, R8(청·녹) → yellow.
  실제로는 R7·R8 각각 p/y 아형이 섞여 있지만 FlyWire에 아형 주석이 없어 이렇게 나눈다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import eye

ANN = Path.home() / "flybrain" / "embodied-fly-brain" / "data" / "flywire_annotations.tsv"
COMP = Path.home() / "flybrain" / "shiu-brain-model" / "Completeness_783.csv"
TYPES = ("R1-6", "R7", "R8")
CHANNEL = {"R1-6": "mean", "R7": "pale", "R8": "yellow"}


def load_annotations(columns=("root_id", "pos_x", "pos_y", "pos_z", "cell_type", "cell_class", "super_class", "side")):
    return pd.read_csv(ANN, sep="\t", usecols=list(columns), dtype={"root_id": "int64"}, low_memory=False)


def model_index():
    comp = pd.read_csv(COMP, index_col=0)
    return {int(rid): i for i, rid in enumerate(comp.index)}


def _rank(v):
    """순위를 -1~1로 고르게 편다."""
    r = pd.Series(v).rank(method="first").to_numpy()
    return (r - 1) / max(len(r) - 1, 1) * 2 - 1


def build(ann=None, idx=None):
    """DataFrame: root_id, model_index, eye(0 왼/1 오), cell_type, channel, ommatidium."""
    ann = load_annotations() if ann is None else ann
    idx = model_index() if idx is None else idx
    lat = eye.hex_lattice()
    lat_n = np.stack([lat[:, 0] / np.abs(lat[:, 0]).max(), lat[:, 1] / np.abs(lat[:, 1]).max()], axis=1)
    rows = []
    pr = ann[(ann.cell_class == "visual") & ann.cell_type.isin(TYPES)]
    pr = pr[pr.root_id.isin(idx.keys())]
    for e, side in enumerate(("left", "right")):
        for ct in TYPES:
            d = pr[(pr.side == side) & (pr.cell_type == ct)]
            if d.empty:
                continue
            el = _rank(-d.pos_y.to_numpy())                     # 등쪽(작은 y) = 위
            front = _rank(-d.pos_z.to_numpy())                  # 작은 z = 앞쪽으로 둔다
            if ct != "R1-6":
                front = -front                                  # 첫 시각 교차에서 앞뒤 반전
            x = -front                                          # 격자 x는 바깥쪽(뒤)이 +
            pts = np.stack([x, el], axis=1)
            dist = ((pts[:, None, :] - lat_n[None, :, :]) ** 2).sum(-1)
            omm = dist.argmin(1)
            for rid, o in zip(d.root_id.to_numpy(), omm):
                rows.append((int(rid), idx[int(rid)], e, ct, CHANNEL[ct], int(o)))
    return pd.DataFrame(rows, columns=["root_id", "model_index", "eye", "cell_type", "channel", "ommatidium"])


if __name__ == "__main__":
    m = build()
    print(m.groupby(["eye", "cell_type"]).size())
    print("낱눈당 광수용체 수(중앙값):", int(m.groupby(["eye", "ommatidium"]).size().median()))
