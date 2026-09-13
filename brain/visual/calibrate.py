"""판정 문턱 보정 — 회색 대조군 + 기준 릴스 묶음. 뇌 모델 환경에서 실행한다.

  ~/flybrain/shiu-brain-model/.venv/bin/python brain/visual/calibrate.py [--n 8] [--seconds 1.0]

문턱을 손으로 고르지 않기 위해 두 가지 데이터로 정한다.
1) 회색 정지 화면 대조군 N회(포아송 씨앗만 다름) → PAM 보상 문턱 = 평균 + 3·표준편차
   (표준편차가 0이면 무리 전체 1초 스파이크 1개를 바닥으로). 「변화 없는 화면의 흔들림을 뚜렷이 넘으면 보상 반응」.
2) 기준 릴스 묶음(~/flybrain/insta/results/clips/*.npz, 실제 릴스 첫 1초) → 분포 백분위
   - 다가가기(oDN1·P9 평균) 좋아요 문턱 = 기준 릴스 상위 1/3(67번째 백분위), 대조군 문턱보다 낮으면 대조군 문턱
   - 좋아요일 때 PPL1 허용 상한 = 기준 릴스 67번째 백분위
   - 처벌로 넘기기 PPL1 문턱 = 기준 릴스 90번째 백분위
   ⚠ 그래서 「좋아요」는 절대 기준이 아니라 「이 기준 릴스들 중 상대적으로 다가가기 명령이 강하게 켜짐」이라는 뜻이다.
3) 거대섬유 도주 문턱 = embodied brain_body_bridge 값(50ms 창 정규화 0.3 × 200Hz = 60Hz), 보정 안 함.
검증 자극(루밍·줄무늬·세로 줄무늬·작은 점·깜빡임) 결과도 함께 적는다 — 루밍에 거대섬유가 반응해야 입력 경로가 맞다.

산출: brain/visual/calibration.json (git에 넣는다 — 허브가 읽는 문턱)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import eye
import features
from brain_eval import VPN_INPUT, Brain, brief, log

HERE = Path(__file__).resolve().parent
CLIPS = Path.home() / "flybrain" / "insta" / "results" / "clips"
K = 3.0
GF_THRESHOLD_HZ = 60.0


def approach_hz(res):
    g = res["groups"]
    return round((g["oDN1"]["mean_hz"] + g["P9"]["mean_hz"]) / 2, 3)


def stats(values, floor):
    v = np.asarray(values, float)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return {"mean": round(float(v.mean()), 4), "sd": round(sd, 4), "sd_used": round(max(sd, floor), 4),
            "values": [round(float(x), 4) for x in v]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=10.0)
    a = ap.parse_args()
    brain = Brain(fps=a.fps, mode="vpn")
    gray = eye.synthetic("gray", a.seconds, a.fps)
    runs = []
    for s in range(a.n):
        r = brain.run(gray, a.fps, seed=1000 + s)
        runs.append(r)
        log("CONTROL", s, json.dumps(brief(r), ensure_ascii=False))
    floor = {name: 1.0 / (a.seconds * len(brain.dan[name]["ids"])) for name in ("PAM", "PPL1")}
    control = {name: stats([r["dopamine"][name]["mean_hz"] for r in runs], floor[name]) for name in ("PAM", "PPL1")}
    control["approach_hz"] = stats([approach_hz(r) for r in runs], 1.0 / (a.seconds * 4))
    control["GF_peak50ms_hz"] = stats([r["groups"]["GF"]["peak50ms_hz"] for r in runs], 0.0)

    reels = []
    n_frames = int(round(a.seconds * a.fps))
    for path in sorted(CLIPS.glob("clip_*.npz")):
        z = np.load(path)
        frames = z["frames"][:n_frames]
        if len(frames) < n_frames:
            continue
        meta = json.loads(str(z["meta"]))
        r = brain.run(frames, a.fps, seed=0)
        row = {"clip": path.name, "author": meta.get("author"), "url": meta.get("url"), "PAM": r["dopamine"]["PAM"]["mean_hz"],
               "PPL1": r["dopamine"]["PPL1"]["mean_hz"], "approach_hz": approach_hz(r), "GF_peak": r["groups"]["GF"]["peak50ms_hz"]}
        reels.append(row)
        log("REFERENCE", json.dumps(row, ensure_ascii=False))
    if not reels:
        raise SystemExit("기준 릴스가 없다 — brain/visual/capture_clips.py로 먼저 모은다")
    # 초기화 검사: 릴스를 다 돌린 뒤 회색 화면을 다시 넣어 대조군과 같은지 본다(되먹임 상태가 남으면 여기서 드러남)
    post = [brain.run(gray, a.fps, seed=2000 + s) for s in range(3)]
    post_control = {"PAM": [p["dopamine"]["PAM"]["mean_hz"] for p in post], "PPL1": [p["dopamine"]["PPL1"]["mean_hz"] for p in post],
                    "approach_hz": [approach_hz(p) for p in post], "washout_spikes": [p["washout_spikes"] for p in post]}
    log("POST_CONTROL", json.dumps(post_control))
    appr = np.array([x["approach_hz"] for x in reels])
    ppl1 = np.array([x["PPL1"] for x in reels])
    ctrl_appr_thr = control["approach_hz"]["mean"] + K * control["approach_hz"]["sd_used"]
    threshold = {
        "PAM": round(control["PAM"]["mean"] + K * control["PAM"]["sd_used"], 4),
        "approach_like": round(max(float(np.percentile(appr, 67)), ctrl_appr_thr), 3),
        "PPL1_like_max": round(max(float(np.percentile(ppl1, 67)), control["PPL1"]["mean"] + K * control["PPL1"]["sd_used"]), 4),
        "PPL1_avoid": round(max(float(np.percentile(ppl1, 90)), control["PPL1"]["mean"] + K * control["PPL1"]["sd_used"]), 4),
        "GF_peak50ms_hz": GF_THRESHOLD_HZ,
    }
    sanity = {}
    for kind in ("loom", "bars", "vbars", "dot", "flicker"):
        r = brain.run(eye.synthetic(kind, a.seconds, a.fps), a.fps, seed=7)
        sanity[kind] = brief(r) | {"approach_hz": approach_hz(r)}
        log("SANITY", kind, json.dumps(sanity[kind], ensure_ascii=False))
    out = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": (f"PAM: 회색 대조군 {a.n}회 평균+{K}SD. 다가가기(oDN1·P9) 좋아요: 기준 릴스 {len(reels)}개 67백분위(대조군 문턱 이상). "
                   f"PPL1 좋아요 상한: 67백분위, 처벌 넘기기: 90백분위. GF: brain_body_bridge 0.3×200Hz."),
        "model": "Shiu et al. 2024, FlyWire v783, model.py default_params",
        "input": {"mode": "vpn", "rate_max_hz": features.RATE_MAX, "feature_ref": brain.feature_ref,
                  "vpn_groups": {k: {"n": brain.vpn_counts[k], "weights": v["weights"], "source": v["source"]} for k, v in VPN_INPUT.items()}},
        "stimulus": {"seconds": a.seconds, "fps": a.fps},
        "n_neurons": {name: len(brain.dan[name]["ids"]) for name in ("PAM", "PPL1")},
        "control": control, "post_control": post_control, "reference_reels": reels, "threshold": threshold, "sanity": sanity,
    }
    (HERE / "calibration.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log("CALIBRATION_WROTE", json.dumps(threshold))


if __name__ == "__main__":
    main()
