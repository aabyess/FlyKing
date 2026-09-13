"""도파민 뉴런(PAM·PPL1)이 이 모델에서 켜질 수 있는 뉴런인지 확인한다. 뇌 모델 환경.

  ~/flybrain/shiu-brain-model/.venv/bin/python brain/visual/dan_probe.py

왜: 시각 투사 뉴런 입력(2026-09-13 vpn_test)에서 PAM이 거의 0이었다. 원래 켜지지 않는 뉴런인지,
시각 경로로만 안 켜지는지 구분하려고 문헌상 PAM을 켜는 당 자극과 비교한다.
- sugar: 당 감각뉴런 23개(brain/shiu/sugar_test.py — Shiu et al. 2024 실험 목록) 150Hz 1초.
  PAM은 당 보상에 반응(Liu et al. 2012 Nature 488:512; Burke et al. 2012 Nature 492:433).
- mb_vpn: 버섯체 시각 투사 뉴런(brain_eval.MB_VPN_TYPES) 150Hz 1초.
- both: 둘 다.
산출: ~/flybrain/insta/results/brain/dan_probe.json
"""
import json
import os
import time
from pathlib import Path

import numpy as np

import brain_eval as be
import retina_map

OUT = Path.home() / "flybrain" / "insta" / "results" / "brain" / "dan_probe.json"
SUGAR = [720575940616885538, 720575940630233916, 720575940639332736, 720575940632889389, 720575940617000768, 720575940632425919,
         720575940637568838, 720575940629176663, 720575940621502051, 720575940638202345, 720575940612670570, 720575940611875570,
         720575940621754367, 720575940633143833, 720575940613601698, 720575940630797113, 720575940639198653, 720575940639259967,
         720575940624963786, 720575940640649691, 720575940610788069, 720575940623172843, 720575940628853239]


def main():
    from brian2 import Hz, Network, PoissonGroup, Synapses, mV, ms, second
    from brian2 import seed as b2seed
    from model import create_model, default_params

    cwd = os.getcwd()
    os.chdir(be.SHIU)
    ann = retina_map.load_annotations()
    idx = retina_map.model_index()
    params = dict(default_params)
    neu, syn, mon = create_model("./Completeness_783.csv", "./Connectivity_783.parquet", params)
    os.chdir(cwd)
    ct = ann.cell_type.fillna("")
    mb_vpn = [int(r) for r in ann[ct.isin(be.MB_VPN_TYPES)].root_id if int(r) in idx]
    sugar = [r for r in SUGAR if r in idx]
    stims = {"sugar": sugar, "mb_vpn": mb_vpn, "both": sugar + mb_vpn}
    dan = {name: [int(r) for r in ann[ct.str.match(g["pattern"])].root_id if int(r) in idx] for name, g in be.DAN_GROUPS.items()}
    sub = {int(r): c for r, c in zip(ann.root_id, ct) if c.startswith("PAM") or c.startswith("PPL1")}
    all_t = sorted({idx[i] for v in stims.values() for i in v})
    pos = {j: k for k, j in enumerate(all_t)}
    pg = PoissonGroup(len(all_t), rates=0 * Hz)
    s_in = Synapses(pg, neu, on_pre="v += w_in", namespace={"w_in": params["w_syn"] * params["f_poi"]})
    s_in.connect(i=np.arange(len(all_t)), j=np.array(all_t))
    neu.rfc[np.array(all_t)] = 0 * ms
    net = Network(neu, syn, mon, pg, s_in)
    out = {"n": {k: len(v) for k, v in stims.items()}, "results": {}}
    for name, ids in stims.items():
        r = np.zeros(len(all_t))
        r[[pos[idx[i]] for i in ids]] = 150.0
        pg.rates_ = r
        neu.v = params["v_0"]
        neu.g = 0 * mV
        n0 = len(mon.t)
        b2seed(5)
        net.run(1.0 * second)
        ii = np.asarray(mon.i)[n0:]
        res = {}
        for dname, dids in dan.items():
            di = np.array([idx[d] for d in dids])
            m = np.isin(ii, di)
            per_sub = {}
            for d in dids:
                per_sub.setdefault(sub.get(d, "?"), []).append(int((ii == idx[d]).sum()))
            res[dname] = {"mean_hz": round(float(m.sum()) / len(di), 3), "active": int(len(np.unique(ii[m]))), "n": len(di),
                          "subtypes_mean_hz": {k: round(float(np.mean(v)), 2) for k, v in sorted(per_sub.items())}}
        mn9 = np.isin(ii, [idx[i] for i in be.READOUT["MN9"]["ids"]]).sum() / 2
        res["MN9_mean_hz"] = round(float(mn9), 2)
        out["results"][name] = res
        be.log("DAN_PROBE", name, json.dumps({k: (v["mean_hz"], v["active"]) if isinstance(v, dict) else v for k, v in res.items()}))
        pg.rates_ = np.zeros(len(all_t))
        net.run(0.2 * second)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    be.log("DAN_PROBE_WROTE", str(OUT))


if __name__ == "__main__":
    main()
