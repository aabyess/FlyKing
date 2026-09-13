"""연결 데이터에서 「시각 → 버섯체·도파민 뉴런」 경로를 찾는다(추측 대신 커넥톰 조회). 뇌 모델 환경.

  ~/flybrain/shiu-brain-model/.venv/bin/python brain/visual/vpn_paths.py

출력(표준출력 + ~/flybrain/insta/results/brain/vpn_paths.json):
  1. super_class 값 목록
  2. 시각 투사 뉴런(super_class=visual_projection) 유형 중 Kenyon 세포에 직접 시냅스가 많은 순
  3. PAM·PPL1 도파민 뉴런에 직접 입력이 많은 유형(전체 / 시각 투사 뉴런만)
  4. 시각 투사 뉴런 유형 → (한 단계 거쳐) → PAM·PPL1 두 단계 경로 세기(흥분 부호 곱 포함) 상위
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import brain_eval as be
import retina_map

OUT = Path.home() / "flybrain" / "insta" / "results" / "brain" / "vpn_paths.json"


def main():
    ann = retina_map.load_annotations(("root_id", "cell_type", "cell_class", "super_class", "side"))
    ann["cell_type"] = ann.cell_type.fillna("?")
    con = pd.read_parquet(be.SHIU / "Connectivity_783.parquet",
                          columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory"])
    t = ann.set_index("root_id")
    con["pre_type"] = con.Presynaptic_ID.map(t.cell_type)
    con["post_type"] = con.Postsynaptic_ID.map(t.cell_type)
    con["pre_super"] = con.Presynaptic_ID.map(t.super_class)
    con["post_class"] = con.Postsynaptic_ID.map(t.cell_class)
    report = {"super_class": ann.super_class.value_counts(dropna=False).to_dict()}
    print("SUPER_CLASS", report["super_class"])

    vpn = con[con.pre_super == "visual_projection"]
    kc = vpn[vpn.post_class.fillna("").str.contains("Kenyon", case=False)]
    top_kc = kc.groupby("pre_type").Connectivity.sum().sort_values(ascending=False).head(20)
    report["vpn_to_kenyon"] = top_kc.to_dict()
    print("VPN_TO_KENYON", json.dumps(report["vpn_to_kenyon"], ensure_ascii=False))

    dan_pat = {"PAM": r"^PAM\d{2}$", "PPL1": r"^PPL1\d{2}$"}
    report["inputs"] = {}
    report["two_hop_vpn"] = {}
    for name, pat in dan_pat.items():
        dan_ids = set(ann[ann.cell_type.str.match(pat)].root_id)
        into = con[con.Postsynaptic_ID.isin(dan_ids)]
        top_all = into.groupby(["pre_type", "pre_super"]).Connectivity.sum().sort_values(ascending=False).head(20)
        top_vpn = into[into.pre_super == "visual_projection"].groupby("pre_type").Connectivity.sum().sort_values(ascending=False).head(15)
        report["inputs"][name] = {"all": {f"{a}|{b}": int(v) for (a, b), v in top_all.items()}, "vpn": top_vpn.astype(int).to_dict()}
        print("INPUT", name, "ALL", json.dumps(report["inputs"][name]["all"], ensure_ascii=False))
        print("INPUT", name, "VPN", json.dumps(report["inputs"][name]["vpn"], ensure_ascii=False))
        # 두 단계: VPN → 중간 뉴런 → DAN. 세기 = Σ (VPN→중간 부호·시냅스) × (중간→DAN 부호·시냅스)
        mid_to_dan = into.assign(w=into.Connectivity * into.Excitatory).groupby("Presynaptic_ID").w.sum()
        v2m = vpn[vpn.Postsynaptic_ID.isin(mid_to_dan.index)]
        v2m = v2m.assign(w=v2m.Connectivity * v2m.Excitatory * v2m.Postsynaptic_ID.map(mid_to_dan))
        two = v2m.groupby("pre_type").w.sum().sort_values(ascending=False)
        report["two_hop_vpn"][name] = {"top_positive": two.head(15).astype(float).round(0).to_dict(),
                                       "top_negative": two.tail(5).astype(float).round(0).to_dict()}
        print("TWO_HOP", name, json.dumps(report["two_hop_vpn"][name], ensure_ascii=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print("PATHS_WROTE", OUT)


if __name__ == "__main__":
    main()
