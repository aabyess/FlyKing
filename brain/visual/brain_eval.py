"""릴스 프레임 → 겹눈 → 시각 특징 → Shiu 전뇌 LIF 모델 → 도파민 뉴런·거대섬유·하강뉴런 발화율. 뇌 모델 환경에서 실행한다.

  PY=~/flybrain/shiu-brain-model/.venv/bin/python
  $PY brain/visual/brain_eval.py --synthetic gray,loom,bars,vbars,dot,flicker --seconds 1.0
  $PY brain/visual/brain_eval.py --clips ~/flybrain/insta/results/clips/clip_00.npz ...
  $PY brain/visual/brain_eval.py --serve          # 표준입력 JSON 줄 {"id":..., "npz": 경로} → 표준출력 JSON 줄
  $PY brain/visual/brain_eval.py --input photoreceptor ...   # 비교용: 광수용체 직접 자극(신호가 수질에서 멈춤)

모델: Shiu et al. 2024 Nature, FlyWire v783 (~/flybrain/shiu-brain-model/model.py create_model, default_params 그대로).
네트워크는 한 번만 만들고 자극마다 막전위만 초기화해 다시 돌린다(원래 코드는 시행마다 새로 만듦).
입력 사건 하나 = w_syn·f_poi = 68.75 mV → 받은 뉴런이 곧바로 발화(model.py poi()와 같은 방식).

입력 방식
- vpn(기본): 겹눈(eye.py) → 시각 특징(features.py) → 시각 투사 뉴런 무리(VPN_INPUT)에 포아송 발화.
  2026-09-13 확인: LC4·LPLC2 150Hz 자극 → 거대섬유 112·162Hz(루밍 도주 경로 재현), 버섯체 VPN이 PAM으로 가는 두 단계 경로 최강.
- photoreceptor: 광수용체 R1-6·R7·R8 10,580개 직접 자극. sweep_gain.py에서 신호가 수질 입구에서 멈춰 판단에 못 씀 — 기록용.

출력(판단에 쓰는 뉴런)
  ⚠ 이 모델에서 도파민은 화학물질 확산·학습으로 표현되지 않는다. 「도파민 뉴런이 얼마나 발화했나」만 읽는다.
  - 보상: PAM 도파민 뉴런 전체 평균 발화율 (Liu et al. 2012 Nature 488:512; Burke et al. 2012 Nature 492:433). PAM01~15 = hemibrain 이름(Li et al. 2020 eLife 9:e62576).
  - 처벌: PPL1 도파민 뉴런 전체 평균 발화율 (Aso et al. 2010 Curr Biol 20:1445; Aso et al. 2012 PLoS Genet 8:e1002768).
  - 도주: 거대섬유 DNp01 50ms 창 최고 발화율 (von Reyn et al. 2014 Nat Neurosci 17:962).
  - 하강뉴런(P9·oDN1·MN9·MDN 등)과 시각 경로 층은 비교·검증용으로 함께 기록.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHIU = Path.home() / "flybrain" / "shiu-brain-model"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SHIU))

import eye  # noqa: E402
import features  # noqa: E402
import retina_map  # noqa: E402

# 광수용체 모드(비교용) 세기
RATE_SUSTAINED = 5.0
RATE_TRANSIENT = 60.0
CHANGE_REF = 0.25
WINDOW_S = 0.05          # embodied brain_body_bridge DNRateDecoder(window_ms=50)와 같은 창

# 버섯체로 가는 시각 투사 뉴런 유형 — vpn_paths.py FlyWire 연결 조회(2026-09-13): Kenyon 세포에 시냅스 ≥150개인 visual_projection 유형.
MB_VPN_TYPES = ["aMe12", "MTe32", "MTe30", "LTe25", "MTe40", "aMe20", "aMe26", "LTe16", "LTe02", "MTe38",
                "LTe51", "MTe28", "LTe72", "LTe40", "MTe35", "MTe22", "MTe26", "LTe04", "LTe73", "LTe50"]

VPN_INPUT = {
    "LC4": dict(pattern=r"^LC4$", weights={"loom_dark": 1.0}, source="von Reyn et al. 2017 Neuron 94:1190 — 루밍 → 거대섬유"),
    "LC6": dict(pattern=r"^LC6$", weights={"loom_dark": 1.0}, source="Wu et al. 2016 eLife 5:e21022 — LC6 루밍 반응"),
    "LC16": dict(pattern=r"^LC16$", weights={"loom_dark": 1.0}, source="Wu et al. 2016 eLife 5:e21022 — LC16 활성화 시 뒷걸음·회피"),
    "LPLC2": dict(pattern=r"^LPLC2$", weights={"loom_any": 1.0}, source="Klapoetke et al. 2017 Nature 551:237 — 루밍·방사 방향 확장"),
    "LC11": dict(pattern=r"^LC11$", weights={"small": 1.0}, source="Keleş & Frye 2017 Curr Biol 27:680 — 작은 움직이는 물체"),
    "LC10": dict(pattern=r"^LC10[a-e]$", weights={"small": 1.0}, source="Ribeiro et al. 2018 Cell 174:607 — 작은 물체 추적"),
    "HS": dict(pattern=r"^HS[NES]$", weights={"wide_h": 1.0}, source="Borst, Haag & Reiff 2010 Annu Rev Neurosci 33:49 — 가로 넓은 시야 움직임"),
    "VS": dict(pattern=r"^VS\d+$", weights={"wide_v": 1.0}, source="Borst, Haag & Reiff 2010 — 세로 넓은 시야 움직임"),
    "MB_VPN": dict(types=MB_VPN_TYPES, weights={"lum": 0.5, "lum_change": 0.5},
                   source="Vogt et al. 2016 eLife 5:e14009 — 버섯체로 가는 시각 투사 뉴런(밝기·색). 유형 = 연결 조회 Kenyon 시냅스 ≥150"),
}

DAN_GROUPS = {
    "PAM": dict(pattern=r"^PAM\d{2}$", role="reward",
                source="Liu et al. 2012 Nature 488:512; Burke et al. 2012 Nature 492:433 — PAM 도파민 뉴런 = 보상 신호"),
    "PPL1": dict(pattern=r"^PPL1\d{2}$", role="punish",
                 source="Aso et al. 2010 Curr Biol 20:1445; Aso et al. 2012 PLoS Genet 8:e1002768 — PPL1 도파민 뉴런 = 처벌 신호"),
}

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
VISUAL_PATH = {"L1": r"^L1$", "L2": r"^L2$", "Mi1": r"^Mi1$", "T4": r"^T4[a-d]$", "T5": r"^T5[a-d]$",
               "LC4": r"^LC4$", "LPLC2": r"^LPLC2$", "KC": r"^KC"}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Brain:
    def __init__(self, fps=10.0, mode="vpn"):
        from brian2 import Hz, Network, PoissonGroup, Synapses, ms, network_operation
        from model import create_model, default_params

        self.fps, self.mode = fps, mode
        t0 = time.time()
        cwd = os.getcwd()
        os.chdir(SHIU)
        try:
            ann = retina_map.load_annotations()
            idx = retina_map.model_index()
            ct = ann.cell_type.fillna("")
            self.idx = idx
            self.dan = {}
            for name, g in DAN_GROUPS.items():
                sel = ann[ct.str.match(g["pattern"])]
                sel = sel[sel.root_id.isin(idx.keys())]
                self.dan[name] = {"ids": [int(r) for r in sel.root_id],
                                  "subtype": {int(r): c for r, c in zip(sel.root_id, sel.cell_type)}}
            self.path_ids = {k: [int(r) for r in ann[ct.str.match(p)].root_id if int(r) in idx] for k, p in VISUAL_PATH.items()}
            if mode == "photoreceptor":
                self.rmap = retina_map.build(ann, idx)
                target = self.rmap.model_index.to_numpy()
                self.eye_i = self.rmap.eye.to_numpy()
                self.omm_i = self.rmap.ommatidium.to_numpy()
                self.chan = self.rmap.channel.to_numpy()
                self.ctype = self.rmap.cell_type.to_numpy()
            else:
                rows = []
                for gi, (name, g) in enumerate(VPN_INPUT.items()):
                    sel = ann[ct.isin(g["types"])] if "types" in g else ann[ct.str.match(g["pattern"])]
                    for rid, side in zip(sel.root_id, sel.side.fillna("")):
                        if int(rid) in idx:
                            rows.append((idx[int(rid)], gi, {"left": 0, "right": 1}.get(side, 2)))
                arr = np.array(rows, dtype=np.int64)
                target, self.vpn_group, self.vpn_eye = arr[:, 0], arr[:, 1], arr[:, 2]
                self.vpn_names = list(VPN_INPUT)
                self.vpn_counts = {n: int((self.vpn_group == i).sum()) for i, n in enumerate(self.vpn_names)}
                self.feature_ref = features.reference(fps)
            self.params = dict(default_params)
            neu, syn, mon = create_model("./Completeness_783.csv", "./Connectivity_783.parquet", self.params)
        finally:
            os.chdir(cwd)
        self.neu, self.syn, self.mon = neu, syn, mon
        n_in = len(target)
        self.input_index_set = np.array(sorted(set(target.tolist())), dtype=np.int64)
        self.pg = PoissonGroup(n_in, rates=0 * Hz)
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
        extra = f"VPN={self.vpn_counts} ref={self.feature_ref}" if mode == "vpn" else ""
        log(f"BRAIN_BUILT {self.build_s:.1f}s mode={mode} inputs={n_in} PAM={len(self.dan['PAM']['ids'])} PPL1={len(self.dan['PPL1']['ids'])} {extra}")

    def photoreceptor_rates(self, eyes):
        change = np.zeros_like(eyes)
        change[1:] = np.abs(np.diff(eyes, axis=0))
        rates = np.zeros((len(eyes), len(self.eye_i)), np.float32)
        for c, name in enumerate(("pale", "yellow", "mean")):
            sel = self.chan == name
            if not sel.any():
                continue
            lum, chg = (eyes.mean(2), change.mean(2)) if name == "mean" else (eyes[:, :, c], change[:, :, c])
            e, o = self.eye_i[sel], self.omm_i[sel]
            rates[:, sel] = RATE_SUSTAINED * lum[:, e, o] + RATE_TRANSIENT * np.minimum(1.0, chg[:, e, o] / CHANGE_REF)
        return rates

    def vpn_rates(self, feats):
        rates = np.zeros((len(next(iter(feats.values()))), len(self.vpn_group)), np.float32)
        per_group = {}
        for gi, name in enumerate(self.vpn_names):
            g = features.group_rates(feats, self.feature_ref, VPN_INPUT[name]["weights"])      # (T, 2)
            both = g.mean(1)
            sel = self.vpn_group == gi
            eyes_sel = self.vpn_eye[sel]
            cols = np.where(eyes_sel[None, :] == 2, both[:, None], g[:, np.minimum(eyes_sel, 1)])
            rates[:, sel] = cols
            per_group[name] = round(float(g.mean()), 2)
        return rates, per_group

    def run(self, frames, fps, seed=0):
        from brian2 import mV, second
        from brian2 import seed as b2seed

        t_wall = time.time()
        eyes = eye.reel_to_eyes(frames)
        res = {"input_mode": self.mode, "fps": fps, "seed": seed, "input": eye.summarize(eyes, fps)}
        if self.mode == "photoreceptor":
            rates = self.photoreceptor_rates(eyes)
            res["input_rate_hz"] = {c: round(float(rates[:, self.ctype == c].mean()), 2) for c in retina_map.TYPES}
        else:
            feats = features.compute(eyes, fps)
            rates, per_group = self.vpn_rates(feats)
            res["features"] = features.summarize(feats, self.feature_ref)
            res["vpn_rate_hz"] = per_group
        sim_s = len(frames) / fps
        res["sim_s"] = round(sim_s, 3)
        res["washout_spikes"] = self.washout()
        t0 = float(self.net.t / second)
        self.state.update(rates=rates, t0=t0)
        n_before = len(self.mon.t)
        b2seed(seed)
        np.random.seed(seed)
        self.net.run(sim_s * second)
        ts = np.asarray(self.mon.t / second)[n_before:] - t0
        ii = np.asarray(self.mon.i)[n_before:]
        uniq = np.unique(ii)
        res.update(active_neurons=int(len(uniq)), active_non_input=int(len(uniq) - np.isin(uniq, self.input_index_set).sum()),
                   total_spikes=int(len(ii)), dopamine={}, groups={}, visual_path={})
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
        for name, ids in self.path_ids.items():
            s = self._group(ids, ii, ts, sim_s, per_neuron=False)
            res["visual_path"][name] = {"n": len(ids), "active": s["active"], "mean_hz": s["mean_hz"]}
        res["wall_s"] = round(time.time() - t_wall, 2)
        return res

    def washout(self, rounds=3, step_ms=3.0):
        """자극 사이 초기화. 막전위만 되돌리면 시냅스 지연(1.8ms) 안에 대기 중인 스파이크가 되먹임 회로를 다시 켠다
        (2026-09-13 보정 중 발견: 한 릴스에서 켜진 PAM 20Hz·PPL1 110Hz 상태가 다음 릴스들로 이어짐).
        그래서 입력을 끄고 → 모든 뉴런을 불응 상태로 만든 채 대기 스파이크를 흘려보내고 → 막전위·시냅스 전류를 0으로, 를 반복한다.
        마지막 라운드에서 발화한 스파이크 수를 돌려준다(0이어야 깨끗함)."""
        from brian2 import mV, ms

        self.state["rates"] = np.zeros((1, self.pg.N))
        self.pg.rates_ = 0.0
        t_rfc = self.params["t_rfc"]
        self.neu.rfc[self.input_index_set] = t_rfc
        last = 0
        for _ in range(rounds):
            self.neu.v = self.params["v_0"]
            self.neu.g = 0 * mV
            self.neu.lastspike = self.net.t
            n0 = len(self.mon.t)
            self.net.run(step_ms * ms)
            last = len(self.mon.t) - n0
        self.neu.rfc[self.input_index_set] = 0 * ms
        self.neu.v = self.params["v_0"]
        self.neu.g = 0 * mV
        return int(last)

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
    d, g = res["dopamine"], res["groups"]
    out = {"PAM": d["PAM"]["mean_hz"], "PPL1": d["PPL1"]["mean_hz"], "GF_peak": g["GF"]["peak50ms_hz"],
           "P9": g["P9"]["mean_hz"], "oDN1": g["oDN1"]["mean_hz"], "MDN": g["MDN"]["mean_hz"],
           "active_non_input": res.get("active_non_input"), "wall_s": res["wall_s"],
           "path": {k: f'{v["active"]}/{v["n"]}' for k, v in res.get("visual_path", {}).items()}}
    if "vpn_rate_hz" in res:
        out["vpn_hz"] = res["vpn_rate_hz"]
    if "input_rate_hz" in res:
        out["in_R16"] = res["input_rate_hz"]["R1-6"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["vpn", "photoreceptor"], default="vpn")
    ap.add_argument("--synthetic", default="")
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    brain = Brain(fps=a.fps, mode=a.input)
    results = []
    for kind in [k for k in a.synthetic.split(",") if k]:
        for r in range(a.repeats):
            res = brain.run(eye.synthetic(kind, a.seconds, a.fps), a.fps, seed=r) | {"stimulus": f"synthetic:{kind}", "repeat": r}
            results.append(res)
            log("RESULT", res["stimulus"], r, json.dumps(brief(res), ensure_ascii=False))
    for path in a.clips:
        z = np.load(path)
        frames = z["frames"][: int(round(a.seconds * a.fps))]
        meta = json.loads(str(z["meta"]))
        if len(frames) == 0:
            log("SKIP_EMPTY", path)
            continue
        res = brain.run(frames, a.fps, seed=0) | {"stimulus": f"clip:{Path(path).name}", "meta": meta}
        results.append(res)
        log("RESULT", res["stimulus"], meta.get("author"), json.dumps(brief(res), ensure_ascii=False))
    if a.out:
        Path(a.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).expanduser().write_text(json.dumps({"build_s": brain.build_s, "mode": a.input, "results": results}, ensure_ascii=False, indent=1))
        log("WROTE", a.out)
    if a.serve:
        print(json.dumps({"type": "ready", "build_s": round(brain.build_s, 1), "mode": a.input}), flush=True)
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
