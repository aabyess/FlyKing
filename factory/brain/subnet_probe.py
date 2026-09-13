"""부분망 시험 — 작업대에 필요한 경로 뉴런만 떼어 돌리면 얼마나 빨라지고, 판정이 전뇌와 얼마나 같은가. 뇌 모델 환경.

  ~/flybrain/shiu-brain-model/.venv/bin/python factory/brain/subnet_probe.py [--hops 2 3 4] [--seeds 4]

부분망 고르는 법
- 입력 뉴런(시각 투사 뉴런 1,603 + 당 감각뉴런 23)에서 시냅스를 따라 k단계 안에 닿는 뉴런 집합 F
- 판정 뉴런(oDN1·P9·거대섬유·MN9)에서 거꾸로 k단계 안에 닿는 뉴런 집합 B
- 부분망 = (F ∩ B) ∪ 입력 ∪ 판정 뉴런. 그 안의 연결만 남긴다(연결 세기·부호·모델 변수는 전뇌와 같다).
⚠ 경로 밖 되먹임(억제 포함)이 빠지므로 발화가 달라질 수 있다. 그래서 판정 일치율을 전뇌와 같은 자극·같은 씨앗으로 잰다.
판정: 분류대 다가가기 ≥ 21.3Hz, 경비 거대섬유 ≥ 60Hz, 설탕 MN9 ÷ 71Hz 속도 차이.
산출: factory/brain/subnet_probe.json
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stimuli  # noqa: E402
from fbrain import FactoryBrain, be, log  # noqa: E402

SORTER_THR, GF_THR, MN9_REF = 21.3, 60.0, 71.0
STIMULI = [("sorter", "bright"), ("sorter", "dark"), ("sorter", "none"), ("guard", "intruder"), ("guard", "clouds"),
           ("sugar", 100), ("sugar", 150), ("sugar", 200)]


def run_set(brain, seeds):
    out = []
    for st, arg in STIMULI:
        for s in range(seeds):
            if st == "sorter":
                r = brain.decide(stimuli.sorter(arg, 0.5, 10, seed=s), seed=s)
            elif st == "guard":
                r = brain.decide(stimuli.guard(arg, 0.5, 10, seed=s), seed=s)
            else:
                r = brain.decide(stimuli.sugar(0.5, 10), sugar_hz=arg, seed=s)
            v = r["values"]
            verdict = (v["approach_hz"] >= SORTER_THR) if st == "sorter" else (v["GF_peak50ms_hz"] >= GF_THR) if st == "guard" \
                else round(min(1.5, v["MN9_mean_hz"] / MN9_REF), 3)
            out.append({"station": st, "arg": arg, "seed": s, "values": v, "verdict": verdict, "wall_s": r["wall_s"]})
    return out


def reach(con, seeds, hops, forward):
    src, dst = ("Presynaptic_Index", "Postsynaptic_Index") if forward else ("Postsynaptic_Index", "Presynaptic_Index")
    seen, cur = set(seeds), set(seeds)
    for _ in range(hops):
        nxt = set(con.loc[con[src].isin(cur), dst].unique()) - seen
        seen |= nxt
        cur = nxt
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", type=int, nargs="*", default=[2, 3, 4])
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()

    full = FactoryBrain(fps=10.0, sigma=0.0)
    inputs = set(np.concatenate([full.vpn_index, full.sugar_index]).tolist())
    readouts = set(np.concatenate([full.readout[k] for k in ("oDN1", "P9", "GF", "MN9")]).tolist())
    log("FULL_BUILT", full.build_s, "neurons", full.n_neurons, "synapses", len(full.base_w))
    ref = run_set(full, a.seeds)
    full_wall = float(np.mean([x["wall_s"] for x in ref]))
    log("FULL_DONE mean_wall", round(full_wall, 3))
    report = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": a.seeds, "full": {"neurons": full.n_neurons,
              "synapses": len(full.base_w), "build_s": full.build_s, "mean_wall_s": round(full_wall, 3)}, "subnets": []}
    del full

    comp = pd.read_csv(be.SHIU / "Completeness_783.csv", index_col=0)
    con = pd.read_parquet(be.SHIU / "Connectivity_783.parquet")
    tmp = Path(tempfile.mkdtemp(prefix="flyfactory_subnet_"))
    for k in a.hops:
        keep = (reach(con, inputs, k, True) & reach(con, readouts, k, False)) | inputs | readouts
        keep_sorted = np.array(sorted(keep))
        new_index = {old: new for new, old in enumerate(keep_sorted)}
        sub_comp = comp.iloc[keep_sorted]
        sub_con = con[con.Presynaptic_Index.isin(keep) & con.Postsynaptic_Index.isin(keep)].copy()
        sub_con["Presynaptic_Index"] = sub_con.Presynaptic_Index.map(new_index)
        sub_con["Postsynaptic_Index"] = sub_con.Postsynaptic_Index.map(new_index)
        cp, pp = tmp / f"comp_k{k}.csv", tmp / f"con_k{k}.parquet"
        sub_comp.to_csv(cp)
        sub_con.to_parquet(pp)
        log("SUBNET", k, "neurons", len(sub_comp), "synapses", len(sub_con))
        sub = FactoryBrain(fps=10.0, sigma=0.0, completeness=cp, connectivity=pp)
        res = run_set(sub, a.seeds)
        agree = {}
        for st in ("sorter", "guard", "sugar"):
            pairs = [(x, y) for x, y in zip(ref, res) if x["station"] == st]
            if st == "sugar":
                diffs = [abs(x["verdict"] - y["verdict"]) for x, y in pairs]
                agree[st] = {"mean_speed_abs_diff": round(float(np.mean(diffs)), 3), "within_0.1": round(float(np.mean([d <= 0.1 for d in diffs])), 3)}
            else:
                agree[st] = {"verdict_agreement": round(float(np.mean([x["verdict"] == y["verdict"] for x, y in pairs])), 3)}
        wall = float(np.mean([x["wall_s"] for x in res]))
        row = {"hops": k, "neurons": len(sub_comp), "synapses": len(sub_con), "build_s": sub.build_s, "mean_wall_s": round(wall, 3),
               "speedup": round(full_wall / wall, 2), "agreement": agree,
               "example_values": {f"{x['station']}:{x['arg']}": {"full": x["values"], "sub": y["values"]} for x, y in zip(ref, res) if x["seed"] == 0}}
        report["subnets"].append(row)
        log("SUBNET_RESULT", json.dumps({kk: row[kk] for kk in ("hops", "neurons", "synapses", "mean_wall_s", "speedup", "agreement")}, ensure_ascii=False))
        del sub
    (HERE / "subnet_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    log("SUBNET_WROTE", str(HERE / "subnet_probe.json"))


if __name__ == "__main__":
    main()
