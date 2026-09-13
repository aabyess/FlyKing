"""시각 투사 뉴런(VPN)을 직접 자극하면 도파민 뉴런·거대섬유·하강뉴런까지 닿는지 확인한다. 뇌 모델 환경.

  ~/flybrain/shiu-brain-model/.venv/bin/python brain/visual/vpn_probe.py

왜: sweep_gain.py 결과(2026-09-13) — 광수용체를 80/1200Hz까지 올려도 신호가 라미나(L1~L3)와 수질 입구(Mi1 소수)에서 멈추고
루밍에 LC4·LPLC2가 반응하지 않았다. 앞단 시각 회로는 연속 전위·억제 해제로 신호를 전하는데 이 LIF 모델(모든 뉴런이
쉬다가 스파이크만 냄)로는 재현되지 않는다. 그래서 문헌으로 알려진 시각 특징을 계산해 VPN에 넣는 방식을 검토한다.
여기서는 VPN 무리마다 150Hz(Shiu et al. 2024가 감각뉴런 자극에 쓴 값)로 1초 자극해 아래쪽 반응을 잰다.

산출: ~/flybrain/insta/results/brain/vpn_probe.json
"""
import json
import os
import time
from pathlib import Path

import numpy as np

import brain_eval as be
import retina_map

OUT = Path.home() / "flybrain" / "insta" / "results" / "brain" / "vpn_probe.json"
VPN = {  # 이름: (cell_type 정규식, 문헌 역할)
    "LC4": (r"^LC4$", "루밍(다가옴) — von Reyn et al. 2017 Neuron 94:1190"),
    "LPLC2": (r"^LPLC2$", "루밍·방사 방향 움직임 — Klapoetke et al. 2017 Nature 551:237"),
    "LC11": (r"^LC11$", "작은 움직이는 물체 — Keleş & Frye 2017 Curr Biol 27:680"),
    "LC10": (r"^LC10[a-e]?$", "작은 물체 추적(짝 따라가기) — Ribeiro et al. 2018 Cell 174:607"),
    "HS": (r"^HS[NES]$", "넓은 시야 가로 움직임 — Borst, Haag & Reiff 2010 Annu Rev Neurosci 33:49"),
    "VS": (r"^VS\d*$", "넓은 시야 세로 움직임 — Borst, Haag & Reiff 2010"),
    "LC_all": (r"^LC\d+[a-z]?$", "모든 LC 유형(대조)"),
}
RATE_HZ = 150.0
SECONDS = 1.0


def main():
    from brian2 import Hz, Network, PoissonGroup, Synapses, mV, ms, second
    from brian2 import seed as b2seed
    from model import create_model, default_params

    t0 = time.time()
    cwd = os.getcwd()
    os.chdir(be.SHIU)
    ann = retina_map.load_annotations()
    idx = retina_map.model_index()
    params = dict(default_params)
    neu, syn, mon = create_model("./Completeness_783.csv", "./Connectivity_783.parquet", params)
    os.chdir(cwd)
    ct = ann.cell_type.fillna("")
    groups = {}
    for name, (pat, role) in VPN.items():
        sel = ann[ct.str.match(pat)]
        ids = [int(r) for r in sel.root_id if int(r) in idx]
        groups[name] = {"ids": ids, "role": role, "types": sorted(sel.cell_type.unique().tolist())[:12]}
        be.log("VPN", name, len(ids), groups[name]["types"])
    readout = {}
    for name, g in be.DAN_GROUPS.items():
        readout[name] = [int(r) for r in ann[ct.str.match(g["pattern"])].root_id if int(r) in idx]
    for name in ("GF", "P9", "oDN1", "MN9", "MDN"):
        readout[name] = be.READOUT[name]["ids"]
    all_targets = sorted({idx[i] for g in groups.values() for i in g["ids"]})
    pg = PoissonGroup(len(all_targets), rates=0 * Hz)
    pos = {j: k for k, j in enumerate(all_targets)}
    s_in = Synapses(pg, neu, on_pre="v += w_in", namespace={"w_in": params["w_syn"] * params["f_poi"]})
    s_in.connect(i=np.arange(len(all_targets)), j=np.array(all_targets))
    neu.rfc[np.array(all_targets)] = 0 * ms
    net = Network(neu, syn, mon, pg, s_in)
    be.log("BUILT", round(time.time() - t0, 1))

    results = {}
    for name, g in groups.items():
        rates = np.zeros(len(all_targets))
        rates[[pos[idx[i]] for i in g["ids"]]] = RATE_HZ
        pg.rates_ = rates
        neu.v = params["v_0"]
        neu.g = 0 * mV
        n0 = len(mon.t)
        start = float(net.t / second)
        b2seed(3)
        net.run(SECONDS * second)
        ii = np.asarray(mon.i)[n0:]
        out = {"stimulated": len(g["ids"]), "active_total": int(len(np.unique(ii)))}
        for rname, rids in readout.items():
            ri = np.array([idx[r] for r in rids])
            m = np.isin(ii, ri)
            out[rname] = {"mean_hz": round(float(m.sum()) / SECONDS / max(len(ri), 1), 3), "active": int(len(np.unique(ii[m]))), "n": len(ri)}
        results[name] = out
        be.log("PROBE", name, json.dumps({k: (v["mean_hz"], v["active"]) if isinstance(v, dict) else v for k, v in out.items()}))
        pg.rates_ = np.zeros(len(all_targets))
        net.run(0.2 * second)                      # 다음 자극 전에 가라앉힘
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "rate_hz": RATE_HZ, "seconds": SECONDS,
                               "groups": {k: {kk: vv for kk, vv in v.items() if kk != "ids"} | {"n": len(v["ids"])} for k, v in groups.items()},
                               "results": results}, ensure_ascii=False, indent=1))
    be.log("PROBE_WROTE", str(OUT))


if __name__ == "__main__":
    main()
