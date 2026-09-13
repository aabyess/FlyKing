"""릴스 프레임 → 겹눈 → Shiu 전뇌 LIF 모델 → 도파민 뉴런·하강뉴런 발화율. 뇌 모델 환경에서 실행한다.

  PY=~/flybrain/shiu-brain-model/.venv/bin/python
  $PY brain/visual/brain_eval.py --synthetic gray,loom,flicker,bars --seconds 1.0
  $PY brain/visual/brain_eval.py --clips ~/flybrain/insta/results/clips/clip_00.npz ...
  $PY brain/visual/brain_eval.py --serve          # 표준입력 JSON 줄 {"id":..., "npz": 경로} → 표준출력 JSON 줄

모델: Shiu et al. 2024 Nature, FlyWire v783 (~/flybrain/shiu-brain-model/model.py create_model, default_params 그대로).
네트워크는 한 번만 만들고 자극마다 막전위만 초기화해 다시 돌린다(원래 코드는 시행마다 새로 만듦).

입력(광수용체 → 포아송 발화):
  초파리 뇌에 시각 정보가 들어가는 입구는 광수용체뿐이라, 릴스는 반드시 겹눈(eye.py)을 거쳐 들어간다.
  원래 모델의 자극 방식과 같게 사건 하나가 w_syn·f_poi = 68.75 mV를 더해 곧바로 발화시킨다(model.py poi()).
  광수용체는 실제로는 스파이크가 아닌 연속 전위라서, 밝기를 발화율로 바꾸는 것은 근사다.
  rate = RATE_SUSTAINED·밝기 + RATE_TRANSIENT·min(1, |프레임 사이 밝기 변화| / CHANGE_REF)
  → 가만히 있는 밝은 화면보다 밝기가 바뀌는 순간(움직임·깜빡임)에 훨씬 세게 발화한다(광수용체 순응 근사).

출력(판단에 쓰는 뉴런):
  ⚠ 이 모델에서 도파민은 화학물질 확산·학습으로 표현되지 않는다. 「도파민 뉴런이 얼마나 발화했나」만 읽는다.
  - 보상 지수: PAM 도파민 뉴런 전체 평균 발화율 — PAM 무리는 보상 신호(당·먹이)를 버섯체로 전달
    (Liu et al. 2012 Nature 488:512; Burke et al. 2012 Nature 492:433). 유형 이름 PAM01~15 = hemibrain(Li et al. 2020 eLife 9:e62576).
  - 처벌 지수: PPL1 도파민 뉴런 전체 평균 발화율 — PPL1 무리는 처벌 신호(전기 충격 등)를 전달
    (Aso et al. 2010 Curr Biol 20:1445; Aso et al. 2012 PLoS Genet 8:e1002768).
  - 도주: 거대섬유 DNp01 50ms 창 최고 발화율 — 루밍에 도약 도주(von Reyn et al. 2014 Nat Neurosci 17:962).
  - 하강뉴런(P9·oDN1·MN9·MDN 등)과 루밍 시각 뉴런(LC4·LPLC2)은 비교·검증용으로 함께 기록한다.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHIU = Path.home() / "flybrain" / "shiu-brain-model"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SHIU))

import eye  # noqa: E402
import retina_map  # noqa: E402

RATE_SUSTAINED = 5.0     # Hz, 밝기 1일 때
RATE_TRANSIENT = 60.0    # Hz, 밝기 변화가 CHANGE_REF 이상일 때
CHANGE_REF = 0.25        # 프레임(0.1초) 사이 밝기 변화(0~1)
WINDOW_S = 0.05          # embodied brain_body_bridge DNRateDecoder(window_ms=50)와 같은 창

# 도파민 뉴런 무리 — flywire_annotations.tsv cell_type으로 모은다(2026-09-13: PAM01~15 307개, PPL101~108 16개, 전부 모델 안).
DAN_GROUPS = {
    "PAM": dict(pattern=r"^PAM\d{2}$", role="reward",
                source="Liu et al. 2012 Nature 488:512; Burke et al. 2012 Nature 492:433 — PAM 도파민 뉴런 = 보상 신호"),
    "PPL1": dict(pattern=r"^PPL1\d{2}$", role="punish",
                 source="Aso et al. 2010 Curr Biol 20:1445; Aso et al. 2012 PLoS Genet 8:e1002768 — PPL1 도파민 뉴런 = 처벌 신호"),
}

# 하강뉴런 등. ID는 FlyWire v783 root_id(flywire_annotations.tsv의 cell_type·side로 확인, 2026-09-13).
READOUT = {
    "GF": dict(cell_type="DNp01", role="escape", ids=[720575940622838154, 720575940632499757],
               source="von Reyn et al. 2014 Nat Neurosci 17:962 — 거대섬유, 루밍 자극에 도약 도주. embodied bridge escape 그룹"),
    "P9": dict(cell_type="DNp09", role="record", ids=[720575940635872101, 720575940627652358],
               source="Bidaye et al. 2020 Neuron 108:469 — P9 전진 보행·조향. bridge forward 그룹"),
    "oDN1": dict(cell_type="DNg97", role="record", ids=[720575940626730883, 720575940620300308],
                 source="Bidaye et al. 2020 Neuron 108:469 — oDN1 전진 보행. bridge forward 그룹"),
    "MN9": dict(cell_type="CB0701(MN9)", role="record", ids=[720575940618238523, 720575940660219265],
                source="Shiu et al. 2024 Nature — 당 감각 → MN9 주둥이 내밀기. bridge feed 그룹"),
    "MDN": dict(cell_type="MDN", role="record",
                ids=[720575940616026939, 720575940631082808, 720575940640331472, 720575940610236514],
                source="Bidaye et al. 2014 Science 344:97 — moonwalker, 뒷걸음. bridge backward 그룹"),
    "aDN1": dict(cell_type="DNg62", role="record", ids=[720575940624319124, 720575940616185531],
                 source="Hampel et al. 2015 eLife 4:e08758 — 더듬이 몸단장. bridge groom 그룹"),
    "DNa01": dict(cell_type="DNa01", role="record", ids=[720575940627787609, 720575940644438551],
                  source="Namiki et al. 2018 eLife 7:e34272 DN 지도 — 방향 틀기"),
    "DNa02": dict(cell_type="DNa02", role="record", ids=[720575940629327659, 720575940604737708],
                  source="Namiki et al. 2018 eLife 7:e34272 DN 지도 — 방향 틀기"),
}
VISUAL_PATH = {"L1": r"^L1$", "L2": r"^L2$", "L3": r"^L3$", "Mi1": r"^Mi1$", "Tm3": r"^Tm3$",
               "T4": r"^T4[a-d]$", "T5": r"^T5[a-d]$", "LC4": r"^LC4$", "LPLC2": r"^LPLC2$"}
VISUAL_RECORD = {   # 루밍 검출 시각 투사 뉴런 — 입력이 뇌 안쪽까지 전달되는지 확인용(기록만)
    "LC4": "von Reyn et al. 2017 Neuron 94:1190 — 루밍 → 거대섬유 입력",
    "LPLC2": "Ache et al. 2019 Curr Biol 29:1073 — 루밍 크기·속도 → 거대섬유 입력",
}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Brain:
    def __init__(self, fps=10.0):
        from brian2 import Hz, Network, PoissonGroup, Synapses, ms, network_operation
        from model import create_model, default_params

        self.fps = fps
        t0 = time.time()
        cwd = os.getcwd()
        os.chdir(SHIU)
        try:
            ann = retina_map.load_annotations()
            idx = retina_map.model_index()
            self.rmap = retina_map.build(ann, idx)
            ct = ann.cell_type.fillna("")
            self.visual_ids = {k: [int(r) for r in ann[ct == k].root_id if int(r) in idx] for k in VISUAL_RECORD}
            # 시각 경로 층별 진단(신호가 어디까지 가는지): 라미나 → 수질 → 운동 검출(T4·T5) → 루밍 검출(LC4·LPLC2)
            self.path_ids = {k: [int(r) for r in ann[ct.str.match(p)].root_id if int(r) in idx] for k, p in VISUAL_PATH.items()}
            self.dan = {}
            for name, g in DAN_GROUPS.items():
                sel = ann[ct.str.match(g["pattern"])]
                sel = sel[sel.root_id.isin(idx.keys())]
                self.dan[name] = {"ids": [int(r) for r in sel.root_id],
                                  "subtype": {int(r): c for r, c in zip(sel.root_id, sel.cell_type)}}
            self.idx = idx
            self.params = dict(default_params)
            neu, syn, mon = create_model("./Completeness_783.csv", "./Connectivity_783.parquet", self.params)
        finally:
            os.chdir(cwd)
        self.neu, self.syn, self.mon = neu, syn, mon
        n_in = len(self.rmap)
        self.pg = PoissonGroup(n_in, rates=0 * Hz)
        target = self.rmap.model_index.to_numpy()
        self.s_in = Synapses(self.pg, neu, on_pre="v += w_in", namespace={"w_in": self.params["w_syn"] * self.params["f_poi"]})
        self.s_in.connect(i=np.arange(n_in), j=target)
        neu.rfc[target] = 0 * ms                         # model.py poi()와 같게 입력 뉴런은 불응기 0
        self.state = {"rates": np.zeros((1, n_in)), "t0": 0.0}
        frame_dt = 1.0 / fps

        @network_operation(dt=frame_dt * 1000 * ms)
        def feed():
            k = int(round((float(self.net.t_) - self.state["t0"]) / frame_dt))
            r = self.state["rates"]
            self.pg.rates_ = r[min(max(k, 0), len(r) - 1)]

        self.net = Network(neu, syn, mon, self.pg, self.s_in, feed)
        self.build_s = time.time() - t0
        self.input_index_set = np.array(sorted(set(self.rmap.model_index.tolist())), dtype=np.int64)
        self.eye_i = self.rmap.eye.to_numpy()
        self.omm_i = self.rmap.ommatidium.to_numpy()
        self.chan = self.rmap.channel.to_numpy()
        self.ctype = self.rmap.cell_type.to_numpy()
        log(f"BRAIN_BUILT {self.build_s:.1f}s inputs={n_in} PAM={len(self.dan['PAM']['ids'])} PPL1={len(self.dan['PPL1']['ids'])}")

    def input_rates(self, eyes):
        """eyes (T, 2 눈, 2 유형, 721) → 광수용체별 발화율 (T, n_in) Hz."""
        change = np.zeros_like(eyes)
        change[1:] = np.abs(np.diff(eyes, axis=0))
        rates = np.zeros((len(eyes), len(self.rmap)), np.float32)
        for c, name in enumerate(("pale", "yellow", "mean")):
            sel = self.chan == name
            if not sel.any():
                continue
            if name == "mean":
                lum, chg = eyes.mean(2), change.mean(2)
            else:
                lum, chg = eyes[:, :, c], change[:, :, c]
            e, o = self.eye_i[sel], self.omm_i[sel]
            rates[:, sel] = RATE_SUSTAINED * lum[:, e, o] + RATE_TRANSIENT * np.minimum(1.0, chg[:, e, o] / CHANGE_REF)
        return rates

    def run(self, frames, fps, seed=0):
        from brian2 import mV, second
        from brian2 import seed as b2seed

        t_wall = time.time()
        eyes = eye.reel_to_eyes(frames)
        rates = self.input_rates(eyes)
        sim_s = len(frames) / fps
        self.neu.v = self.params["v_0"]
        self.neu.g = 0 * mV
        t0 = float(self.net.t / second)
        self.state.update(rates=rates, t0=t0)
        n_before = len(self.mon.t)
        b2seed(seed)
        np.random.seed(seed)
        self.net.run(sim_s * second)
        ts = np.asarray(self.mon.t / second)[n_before:] - t0
        ii = np.asarray(self.mon.i)[n_before:]
        res = {"sim_s": round(sim_s, 3), "fps": fps, "seed": seed, "input": eye.summarize(eyes, fps),
               "input_rate_hz": {c: round(float(rates[:, self.ctype == c].mean()), 2) for c in retina_map.TYPES},
               "input_params": {"RATE_SUSTAINED": RATE_SUSTAINED, "RATE_TRANSIENT": RATE_TRANSIENT, "CHANGE_REF": CHANGE_REF},
               "active_neurons": int(len(np.unique(ii))), "total_spikes": int(len(ii)), "dopamine": {}, "groups": {}}
        for name, g in DAN_GROUPS.items():
            d = self.dan[name]
            stat = self._group(d["ids"], ii, ts, sim_s, per_neuron=False)
            sub = {}
            for rid, st in d["subtype"].items():
                sub.setdefault(st, []).append(rid)
            stat["subtypes_mean_hz"] = {st: self._group(ids, ii, ts, sim_s, per_neuron=False)["mean_hz"] for st, ids in sorted(sub.items())}
            stat.update(role=g["role"], n=len(d["ids"]))
            res["dopamine"][name] = stat
        for name, g in READOUT.items():
            res["groups"][name] = self._group(g["ids"], ii, ts, sim_s) | {"cell_type": g["cell_type"], "role": g["role"]}
        for name, ids in self.visual_ids.items():
            res["groups"][name] = self._group(ids, ii, ts, sim_s, per_neuron=False) | {"cell_type": name, "role": "record", "n": len(ids)}
        input_set = self.input_index_set
        uniq = np.unique(ii)
        res["active_non_input"] = int(len(uniq) - np.isin(uniq, input_set).sum())
        res["visual_path"] = {}
        for name, ids in self.path_ids.items():
            s = self._group(ids, ii, ts, sim_s, per_neuron=False)
            res["visual_path"][name] = {"n": len(ids), "active": s["active"], "mean_hz": s["mean_hz"]}
        res["wall_s"] = round(time.time() - t_wall, 2)
        return res

    def _group(self, ids, ii, ts, sim_s, per_neuron=True):
        idxs = np.array([self.idx[i] for i in ids if i in self.idx], dtype=np.int64)
        mask = np.isin(ii, idxs)
        t = ts[mask]
        n = max(len(idxs), 1)
        out = {"mean_hz": round(float(mask.sum()) / sim_s / n, 3), "active": int(len(np.unique(ii[mask])))}
        peak = 0.0
        if len(t):
            counts, _ = np.histogram(t, bins=np.arange(0, sim_s + 1e-9, 0.01))
            win = int(round(WINDOW_S / 0.01))
            slide = np.convolve(counts, np.ones(win), mode="valid") if len(counts) >= win else counts
            peak = float(slide.max()) / WINDOW_S / n
        out["peak50ms_hz"] = round(peak, 2)
        out["first_spike_s"] = round(float(t.min()), 3) if len(t) else None
        if per_neuron:
            out["per_neuron_hz"] = {str(rid): round(float((ii[mask] == self.idx[rid]).sum()) / sim_s, 2) for rid in ids if rid in self.idx}
        return out


def brief(res):
    d = res["dopamine"]
    g = res["groups"]
    return {"PAM": d["PAM"]["mean_hz"], "PPL1": d["PPL1"]["mean_hz"], "GF_peak": g["GF"]["peak50ms_hz"],
            "P9": g["P9"]["mean_hz"], "MDN": g["MDN"]["mean_hz"], "LC4": g["LC4"]["mean_hz"], "LPLC2": g["LPLC2"]["mean_hz"],
            "active": res["active_neurons"], "active_non_input": res.get("active_non_input"),
            "in_R16": res["input_rate_hz"]["R1-6"], "wall_s": res["wall_s"],
            "path": {k: f'{v["active"]}/{v["n"]}@{v["mean_hz"]}' for k, v in res.get("visual_path", {}).items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default="")
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    brain = Brain(fps=a.fps)
    results = []
    for kind in [k for k in a.synthetic.split(",") if k]:
        for r in range(a.repeats):
            res = brain.run(eye.synthetic(kind, a.seconds, a.fps), a.fps, seed=r) | {"stimulus": f"synthetic:{kind}", "repeat": r}
            results.append(res)
            log("RESULT", res["stimulus"], r, json.dumps(brief(res)))
    for path in a.clips:
        z = np.load(path)
        frames = z["frames"][: int(round(a.seconds * a.fps))]
        meta = json.loads(str(z["meta"]))
        if len(frames) == 0:
            log("SKIP_EMPTY", path)
            continue
        res = brain.run(frames, a.fps, seed=0) | {"stimulus": f"clip:{Path(path).name}", "meta": meta}
        results.append(res)
        log("RESULT", res["stimulus"], meta.get("author"), json.dumps(brief(res)))
    if a.out:
        Path(a.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).expanduser().write_text(json.dumps({"build_s": brain.build_s, "results": results}, ensure_ascii=False, indent=1))
        log("WROTE", a.out)
    if a.serve:
        print(json.dumps({"type": "ready", "build_s": round(brain.build_s, 1)}), flush=True)
        for line in sys.stdin:
            req = None
            try:
                req = json.loads(line)
                z = np.load(req["npz"])
                res = brain.run(z["frames"], float(z["fps"]) if "fps" in z.files else a.fps, seed=int(req.get("seed", 0)))
                print(json.dumps({"type": "result", "id": req.get("id"), **res}, ensure_ascii=False), flush=True)
            except Exception as e:  # noqa: BLE001
                print(json.dumps({"type": "error", "id": req.get("id") if isinstance(req, dict) else None, "error": repr(e)}), flush=True)


if __name__ == "__main__":
    main()
