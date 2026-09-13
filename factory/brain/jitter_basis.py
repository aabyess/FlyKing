"""개체 성격 흔들림 크기의 근거 — FlyWire 좌·우 반구에서 같은 세포 유형 쌍의 연결 세기가 얼마나 다른가. 뇌 모델 환경.

  ~/flybrain/shiu-brain-model/.venv/bin/python factory/brain/jitter_basis.py

방법
- 같은 반구 안 연결만 본다(왼→왼, 오→오). 유형 쌍(앞 cell_type, 뒤 cell_type)마다 시냅스 수를 반구별로 합한다.
- 양쪽 모두 시냅스가 있는 쌍에서 log(왼/오)를 구하고, 반구 전체 치우침(재구성 차이)을 빼기 위해 중앙값을 뺀다.
- 한쪽 반구의 흔들림 σ = 표준편차(log 비) / √2. 시냅스 수가 적으면 세는 잡음이 커서, 합이 많은 쌍 구간별로 따로 낸다.
- 게임에 쓰는 σ = 합 200개 이상인 쌍에서 잰 값(세는 잡음이 작은 구간). 이상치에 덜 흔들리게 IQR/1.349로도 같이 적는다.
⚠ 한 개체 안 좌우 차이다. 개체 사이 차이는 이 데이터로 직접 재지 않았다.
산출: factory/brain/jitter_basis.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANN = Path.home() / "flybrain" / "embodied-fly-brain" / "data" / "flywire_annotations.tsv"
CON = Path.home() / "flybrain" / "shiu-brain-model" / "Connectivity_783.parquet"


def main():
    ann = pd.read_csv(ANN, sep="\t", usecols=["root_id", "cell_type", "side"], dtype={"root_id": "int64"}, low_memory=False)
    ann = ann[ann.cell_type.notna() & ann.side.isin(["left", "right"])].set_index("root_id")
    con = pd.read_parquet(CON, columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
    con["pre_t"] = con.Presynaptic_ID.map(ann.cell_type)
    con["post_t"] = con.Postsynaptic_ID.map(ann.cell_type)
    con["pre_s"] = con.Presynaptic_ID.map(ann.side)
    con["post_s"] = con.Postsynaptic_ID.map(ann.side)
    con = con[con.pre_t.notna() & con.post_t.notna() & (con.pre_s == con.post_s)]
    g = con.groupby(["pre_t", "post_t", "pre_s"]).Connectivity.sum().unstack("pre_s").fillna(0)
    both = g[(g["left"] > 0) & (g["right"] > 0)].copy()
    both["total"] = both["left"] + both["right"]
    lr = np.log(both["left"] / both["right"])
    bias = float(lr.median())
    both["lr"] = lr - bias
    bins = [(10, 50), (50, 200), (200, 1000), (1000, None)]
    table = []
    for lo, hi in bins:
        sel = both[(both.total >= lo) & ((both.total < hi) if hi else True)]
        if len(sel) < 20:
            continue
        sd = float(sel.lr.std()) / np.sqrt(2)
        q75, q25 = np.percentile(sel.lr, [75, 25])
        robust = float((q75 - q25) / 1.349) / np.sqrt(2)
        row = {"total_synapses": f"{lo}~{hi or ''}", "pairs": int(len(sel)), "sigma_sd": round(sd, 4), "sigma_iqr": round(robust, 4)}
        table.append(row)
        print("BIN", json.dumps(row), flush=True)
    # 약한 쌍일수록 시냅스 세는 잡음이 섞여 σ가 부풀려진다(2026-09-13: 10~50개 0.58 → 1000개 이상 0.135).
    # 그래서 세는 잡음이 가장 작은 1000개 이상 구간을 생물학적 흔들림의 보수적 추정으로 쓴다.
    strong = both[both.total >= 1000]
    q75, q25 = np.percentile(strong.lr, [75, 25])
    sigma = round(float((q75 - q25) / 1.349) / np.sqrt(2), 4)
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "sigma": sigma,
           "sigma_note": "합 1000 시냅스 이상 유형 쌍의 좌우 log 비 IQR/1.349/√2 — 한 반구(한 개체의 한쪽)의 유형 쌍 연결 세기 흔들림",
           "hemisphere_bias_log": round(bias, 4), "pairs_both_sides": int(len(both)), "by_strength": table,
           "method": "FlyWire v783 같은 반구 안 연결, cell_type 쌍별 시냅스 합, log(왼/오) − 중앙값"}
    (HERE / "jitter_basis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("JITTER_SIGMA", sigma, "pairs", len(both), flush=True)


if __name__ == "__main__":
    main()
