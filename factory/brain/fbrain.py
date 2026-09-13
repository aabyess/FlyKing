"""공장 게임용 초파리 뇌 — Shiu 2024 전뇌 LIF(FlyWire v783) + 시각 투사 뉴런 입력 + 당 감각뉴런 입력 + 개체 성격.

릴스 뇌 모듈(brain/visual/brain_eval.py)의 입력 무리·특징 계산·기준 뉴런 정의를 **읽어서만** 쓴다(릴스 코드는 수정하지 않음).

작업대가 쓰는 반사(릴스 작업에서 모델로 확인된 것만)
- 다가가기: oDN1·P9 전진 명령 뉴런 평균 발화(Bidaye et al. 2020 Neuron 108:469)          → 불빛 분류대 레버
- 도주:     거대섬유 DNp01 50ms 창 최고 발화(von Reyn et al. 2014 Nat Neurosci 17:962)  → 경비 초소 경보
- 섭식:     당 감각뉴런 → MN9 주둥이 운동뉴런 평균 발화(Shiu et al. 2024 Nature)          → 설탕 배달 속도
보상 도파민 PAM은 이 모델에서 당·시각 어느 쪽으로도 켜지지 않아(brain/visual/dan_probe.py) 게임에 쓰지 않고 기록만 한다.

개체 성격(시냅스 무게 흔들림)
- 초파리마다 시드로, 같은 (시냅스 앞 세포 유형, 뒤 세포 유형) 쌍의 모든 시냅스에 같은 배수 exp(N(0, σ))를 곱한다.
  시냅스 하나하나에 따로 주면 수백만 개가 평균돼 행동 차이가 사라지므로, 유형 쌍 단위로 준다.
- σ는 factory/brain/jitter_basis.py가 FlyWire 좌·우 반구의 같은 유형 쌍 연결 세기 차이에서 잰 값(jitter_basis.json)을 쓴다.
- 부호(흥분·억제)는 바꾸지 않는다. 가소성(학습)은 없다.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "brain" / "visual"))

import brain_eval as be  # noqa: E402  (읽기만)
import eye  # noqa: E402
import features  # noqa: E402
import retina_map  # noqa: E402

# 당 감각뉴런 23개 — brain/shiu/sugar_test.py(Shiu et al. 2024 실험 목록)
SUGAR_IDS = [720575940616885538, 720575940630233916, 720575940639332736, 720575940632889389, 720575940617000768,
             720575940632425919, 720575940637568838, 720575940629176663, 720575940621502051, 720575940638202345,
             720575940612670570, 720575940611875570, 720575940621754367, 720575940633143833, 720575940613601698,
             720575940630797113, 720575940639198653, 720575940639259967, 720575940624963786, 720575940640649691,
             720575940610788069, 720575940623172843, 720575940628853239]
WINDOW_S = be.WINDOW_S
JITTER_FILE = HERE / "jitter_basis.json"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class FactoryBrain:
    def __init__(self, fps=10.0, sigma=None):
        from brian2 import Hz, Network, PoissonGroup, Synapses, ms, network_operation
        from model import create_model, default_params

        t0 = time.time()
        self.fps = fps
        cwd = os.getcwd()
        os.chdir(be.SHIU)
        try:
            ann = retina_map.load_annotations(("root_id", "cell_type", "cell_class", "super_class", "side"))
            idx = retina_map.model_index()
            ct = ann.cell_type.fillna("")
            self.idx = idx
            rows = []
            for gi, (name, g) in enumerate(be.VPN_INPUT.items()):
                sel = ann[ct.isin(g["types"])] if "types" in g else ann[ct.str.match(g["pattern"])]
                for rid, side in zip(sel.root_id, sel.side.fillna("")):
                    if int(rid) in idx:
                        rows.append((idx[int(rid)], gi, {"left": 0, "right": 1}.get(side, 2)))
            self.vpn_names = list(be.VPN_INPUT)
            self.vpn_index = np.array([r[0] for r in rows])
            self.vpn_group = np.array([r[1] for r in rows])
            self.vpn_eye = np.array([r[2] for r in rows])
            self.sugar_index = np.array([idx[r] for r in SUGAR_IDS if r in idx])
            self.dan = {name: np.array([idx[int(r)] for r in ann[ct.str.match(g["pattern"])].root_id if int(r) in idx])
                        for name, g in be.DAN_GROUPS.items()}
            self.readout = {name: np.array([idx[r] for r in g["ids"] if r in idx]) for name, g in be.READOUT.items()}
            # 유형 쌍 id(성격 흔들림 단위). 유형 없는 뉴런은 자기만의 유형으로 둔다.
            comp_ids = list(idx.keys())
            type_of = dict(zip(ann.root_id.astype("int64"), ct))
            names = [type_of.get(r, "") or f"_{r}" for r in comp_ids]
            _, self.neuron_type = np.unique(np.array(names), return_inverse=True)
            self.params = dict(default_params)
            neu, syn, mon = create_model("./Completeness_783.csv", "./Connectivity_783.parquet", self.params)
        finally:
            os.chdir(cwd)
        self.neu, self.syn, self.mon = neu, syn, mon
        target = np.concatenate([self.vpn_index, self.sugar_index])
        self.input_index_set = np.array(sorted(set(target.tolist())), dtype=np.int64)
        self.pg = PoissonGroup(len(target), rates=0 * Hz)
        self.s_in = Synapses(self.pg, neu, on_pre="v += w_in", namespace={"w_in": self.params["w_syn"] * self.params["f_poi"]})
        self.s_in.connect(i=np.arange(len(target)), j=target)
        neu.rfc[target] = 0 * ms
        self.state = {"rates": np.zeros((1, len(target))), "t0": 0.0}
        frame_dt = 1.0 / fps

        @network_operation(dt=frame_dt * 1000 * ms)
        def feed():
            k = int(round((float(self.net.t_) - self.state["t0"]) / frame_dt))
            r = self.state["rates"]
            self.pg.rates_ = r[min(max(k, 0), len(r) - 1)]

        self.net = Network(neu, syn, mon, self.pg, self.s_in, feed)
        self.base_w = np.asarray(syn.w_[:]).copy()
        pre, post = np.asarray(syn.i[:]), np.asarray(syn.j[:])
        n_types = int(self.neuron_type.max()) + 1
        pair_key = self.neuron_type[pre].astype(np.int64) * n_types + self.neuron_type[post]
        _, self.pair_id = np.unique(pair_key, return_inverse=True)
        self.n_pairs = int(self.pair_id.max()) + 1
        if sigma is None:
            sigma = json.loads(JITTER_FILE.read_text())["sigma"] if JITTER_FILE.exists() else 0.0
        self.sigma = float(sigma)
        self.current_fly = None
        self.feature_ref = features.reference(fps)
        self.build_s = round(time.time() - t0, 1)
        log(f"FACTORY_BRAIN_BUILT {self.build_s}s vpn={len(self.vpn_index)} sugar={len(self.sugar_index)} "
            f"synapses={len(self.base_w)} type_pairs={self.n_pairs} sigma={self.sigma}")

    def set_fly(self, fly_seed):
        """초파리 개체 선택. 같은 시드면 항상 같은 성격."""
        if fly_seed == self.current_fly:
            return
        if fly_seed is None or self.sigma == 0:
            self.syn.w_ = self.base_w
        else:
            rng = np.random.default_rng(int(fly_seed))
            factor = np.exp(rng.normal(0.0, self.sigma, self.n_pairs))
            self.syn.w_ = self.base_w * factor[self.pair_id]
        self.current_fly = fly_seed

    def washout(self, rounds=3, step_ms=3.0):
        """자극 사이 초기화(brain_eval.Brain.washout와 같은 방식 — 버섯체 되먹임이 다음 판단으로 넘어가지 않게)."""
        from brian2 import mV, ms

        self.state["rates"] = np.zeros((1, self.pg.N))
        self.pg.rates_ = 0.0
        self.neu.rfc[self.input_index_set] = self.params["t_rfc"]
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

    def decide(self, frames, sugar_hz=0.0, fly_seed=None, seed=0):
        """한 번의 판단 계산. frames (T,256,144,3), sugar_hz = 당 감각뉴런 발화율. 반사 뉴런 발화율과 입력 요약을 돌려준다."""
        from brian2 import second
        from brian2 import seed as b2seed

        t_wall = time.time()
        self.set_fly(fly_seed)
        eyes = eye.reel_to_eyes(frames)
        feats = features.compute(eyes, self.fps)
        n = len(frames)
        rates = np.zeros((n, self.pg.N), np.float32)
        per_group = {}
        for gi, name in enumerate(self.vpn_names):
            g = features.group_rates(feats, self.feature_ref, be.VPN_INPUT[name]["weights"])
            sel = self.vpn_group == gi
            e = self.vpn_eye[sel]
            rates[:, np.nonzero(sel)[0]] = np.where(e[None, :] == 2, g.mean(1)[:, None], g[:, np.minimum(e, 1)])
            per_group[name] = round(float(g.mean()), 2)
        rates[:, len(self.vpn_index):] = sugar_hz
        washout_spikes = self.washout()
        sim_s = n / self.fps
        t0 = float(self.net.t / second)
        self.state.update(rates=rates, t0=t0)
        n0 = len(self.mon.t)
        b2seed(seed)
        self.net.run(sim_s * second)
        ts = np.asarray(self.mon.t / second)[n0:] - t0
        ii = np.asarray(self.mon.i)[n0:]

        def mean_hz(idxs):
            return round(float(np.isin(ii, idxs).sum()) / sim_s / max(len(idxs), 1), 3)

        def peak_hz(idxs):
            t = ts[np.isin(ii, idxs)]
            if not len(t):
                return 0.0
            counts, _ = np.histogram(t, bins=np.arange(0, sim_s + 1e-9, 0.01))
            win = int(round(WINDOW_S / 0.01))
            slide = np.convolve(counts, np.ones(win), mode="valid") if len(counts) >= win else counts
            return round(float(slide.max()) / WINDOW_S / max(len(idxs), 1), 2)

        r = self.readout
        values = {
            "approach_hz": round((mean_hz(r["oDN1"]) + mean_hz(r["P9"])) / 2, 3),
            "GF_peak50ms_hz": peak_hz(r["GF"]),
            "MN9_mean_hz": mean_hz(r["MN9"]),
            "MDN_mean_hz": mean_hz(r["MDN"]),
            "turn_L_minus_R_hz": round(float(np.isin(ii, r["DNa02"][:1]).sum() - np.isin(ii, r["DNa02"][1:]).sum()) / sim_s, 2),
            "PAM_mean_hz": mean_hz(self.dan["PAM"]),
            "PPL1_mean_hz": mean_hz(self.dan["PPL1"]),
        }
        return {"values": values, "sim_s": sim_s, "fps": self.fps, "sugar_hz": sugar_hz, "fly_seed": fly_seed, "seed": seed,
                "sigma": self.sigma, "features": features.summarize(feats, self.feature_ref), "vpn_rate_hz": per_group,
                "input": eye.summarize(eyes, self.fps), "washout_spikes": washout_spikes,
                "active_neurons": int(len(np.unique(ii))), "wall_s": round(time.time() - t_wall, 3)}
