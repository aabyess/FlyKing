"""겹눈 신호 → 초파리 시각계가 뽑는다고 알려진 특징 → 시각 투사 뉴런(VPN) 발화율. numpy만 쓴다.

왜 이 단계가 있나(2026-09-13 sweep_gain.py 결과):
  광수용체를 직접 자극하면 신호가 라미나·수질 입구에서 멈춘다(광수용체 80/1200Hz에서도 LC4·LPLC2 루밍 반응 없음).
  앞단 시각 회로는 연속 전위·억제 해제로 신호를 전하는데 Shiu LIF 모델(모든 뉴런이 쉬다가 스파이크만 냄)로는 재현되지 않는다.
  그래서 앞단 계산만 아래 근사로 대신하고, VPN부터 뒤(중심 뇌·버섯체·도파민 뉴런·하강뉴런)는 FlyWire 실제 배선 그대로 둔다.
  ⚠ 여기 특징 계산은 뉴런 시뮬레이션이 아니라 문헌 모델을 흉내 낸 근사다.

특징(눈마다, 프레임 0.1초마다). C = 밝기 ÷ 화면 평균 − 1 (대비), 이웃 = 육각 격자 이웃 낱눈.
- wide_h, wide_v: 하센슈타인–라이하르트 상관 검출기(Hassenstein & Reichardt 1956; 초파리 T4·T5 — Borst 2014 Nat Rev Neurosci)
    R = C_i(t−1)·C_j(t) − C_j(t−1)·C_i(t). wide_h = 가로 이웃쌍 R 평균(+ = 그 눈의 앞→뒤), wide_v = 세로 이웃쌍 R 평균의 크기.
- loom_dark: 가장 큰 「어두운(C < −DARK)」 덩어리가 넓이·가로 폭·세로 폭 모두 두 프레임 연속 커질 때, 넓이 증가(작은 쪽)×fps.
    한 덩어리가 사방으로 커지는 것만 잡는다 → 장면 전환·깜빡임(한 프레임에 확 바뀜), 화면으로 들어오는 줄무늬(한 방향만 커짐)는 걸러짐.
    (루밍 → LC4·LC6·LC16 — von Reyn et al. 2017; Wu et al. 2016)
- loom_any: 프레임 사이 크게 바뀐 낱눈(|ΔL|/평균 > DARK)의 가장 큰 덩어리에 같은 규칙 — 밝기와 무관한 확장(LPLC2, Klapoetke 2017 근사).
- small: 국소 운동 크기 상위 TOPK쌍 평균 − 전체 평균 − 넓은 운동 크기, 그리고 확장(loom_any 지수)만큼 줄임.
    작은 물체만 움직일 때 크고 화면 전체가 흐르거나 덩어리가 커질 때는 작다(LC11은 넓은 움직임에 억제 — Keleş & Frye 2017).
- lum: 화면을 보는 낱눈 평균 밝기(선형 0~1). lum_change: 프레임 사이 평균 밝기 변화량×fps.
발화율: 무리마다 가중합 Σ w·min(1, 특징/기준값) → RATE_MAX·min(1, 합). 기준값 = 각 특징의 대표 인공 자극 최댓값
(lum은 흰색 235의 선형 밝기). RATE_MAX 150Hz = Shiu 2024가 감각뉴런 자극에 쓴 값.
"""
import numpy as np

import eye

RATE_MAX = 150.0
DARK = 0.3
TOPK = 6
FEATURES = ("loom_dark", "loom_any", "small", "wide_h", "wide_v", "lum", "lum_change")
CANONICAL = {"loom_dark": "loom", "loom_any": "loom", "small": "dot", "wide_h": "bars", "wide_v": "vbars", "lum_change": "flicker"}
LUM_REF = (235 / 255) ** 2.2
LOOM_ANY_SUPPRESS_REF = 0.4     # small 억제에 쓰는 loom_any 규모(대표 루밍 최댓값 근처)


def _lattice():
    lat = eye.hex_lattice()
    d = eye.DPHI
    q = np.round(lat[:, 0] / (d / 2)).astype(int)
    r = np.round(lat[:, 1] / (d * np.sqrt(3) / 2)).astype(int)
    key = {(a, b): k for k, (a, b) in enumerate(zip(q, r))}
    hor, ver = [], []
    for k, (a, b) in enumerate(zip(q, r)):
        if (a + 2, b) in key:
            hor.append((k, key[(a + 2, b)]))
        for dx in (-1, 1):
            if (a + dx, b + 1) in key:
                ver.append((k, key[(a + dx, b + 1)]))
    return np.array(hor), np.array(ver), q, r


HOR, VER, QCOL, RROW = _lattice()
EDGES = np.concatenate([HOR, VER])


def _largest_blob(mask):
    """육각 격자 위 True 낱눈의 가장 큰 연결 덩어리 → (크기, 가로 폭, 세로 폭)."""
    if not mask.any():
        return 0, 0, 0
    parent = np.arange(len(mask))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in EDGES[mask[EDGES[:, 0]] & mask[EDGES[:, 1]]]:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    members = np.nonzero(mask)[0]
    roots = np.array([find(i) for i in members])
    big = np.bincount(roots).argmax()
    sel = members[roots == big]
    return len(sel), int(QCOL[sel].max() - QCOL[sel].min()), int(RROW[sel].max() - RROW[sel].min())


def _expansion(blobs, n_on, fps):
    """blobs (T, 3) → 넓이·가로 폭·세로 폭이 두 프레임 연속 모두 커진 경우의 넓이 증가율."""
    out = np.zeros(len(blobs))
    for k in range(2, len(blobs)):
        d1, d0 = blobs[k] - blobs[k - 1], blobs[k - 1] - blobs[k - 2]
        if (d1 > 0).all() and (d0 > 0).all():
            out[k] = min(d1[0], d0[0]) / n_on * fps
    return out


def compute(eyes, fps):
    """eyes (T, 2, 2, 721) → dict 특징명 → (T, 2) 원시값(눈 0 왼쪽, 1 오른쪽)."""
    _, _, _, _, frac = eye._boxes(256, 144)
    on = frac > 0.5
    lum = eyes[:, :, 1].astype(np.float64)
    t = len(lum)
    out = {f: np.zeros((t, 2)) for f in FEATURES}
    for e in range(2):
        m = on[e]
        n_on = int(m.sum())
        if n_on < 3:
            continue
        L = lum[:, e]
        mean = L[:, m].mean(1) + 1e-3
        C = L / mean[:, None] - 1.0
        hor = HOR[m[HOR[:, 0]] & m[HOR[:, 1]]]
        ver = VER[m[VER[:, 0]] & m[VER[:, 1]]]
        dark_b, chg_b = np.zeros((t, 3)), np.zeros((t, 3))
        small_raw = np.zeros(t)
        for k in range(t):
            dark_b[k] = _largest_blob((C[k] < -DARK) & m)
            out["lum"][k, e] = float(L[k, m].mean())
            if k == 0:
                continue
            chg_b[k] = _largest_blob((np.abs(L[k] - L[k - 1]) / mean[k] > DARK) & m)
            out["lum_change"][k, e] = float(np.abs(L[k, m] - L[k - 1, m]).mean()) * fps
            a, b = C[k - 1], C[k]
            rh = a[hor[:, 0]] * b[hor[:, 1]] - a[hor[:, 1]] * b[hor[:, 0]] if len(hor) else np.zeros(1)
            rv = a[ver[:, 0]] * b[ver[:, 1]] - a[ver[:, 1]] * b[ver[:, 0]] if len(ver) else np.zeros(1)
            wh, wv = float(rh.mean()), float(rv.mean())
            out["wide_h"][k, e] = max(wh, 0.0)
            out["wide_v"][k, e] = abs(wv)
            loc = np.abs(np.concatenate([rh, rv]))
            top = float(np.sort(loc)[-TOPK:].mean()) if len(loc) >= TOPK else float(loc.mean())
            small_raw[k] = max(top - float(loc.mean()) - abs(wh) - abs(wv), 0.0)
        out["loom_dark"][:, e] = _expansion(dark_b, n_on, fps)
        out["loom_any"][:, e] = _expansion(chg_b, n_on, fps)
        out["small"][:, e] = small_raw * (1.0 - np.clip(out["loom_any"][:, e] / LOOM_ANY_SUPPRESS_REF, 0.0, 1.0))
    return out


def reference(fps=10.0, seconds=1.0):
    """각 특징의 대표 인공 자극 최댓값(= 발화율 150Hz)."""
    ref = {"lum": round(LUM_REF, 6)}
    cache = {}
    for f, kind in CANONICAL.items():
        if kind not in cache:
            cache[kind] = compute(eye.reel_to_eyes(eye.synthetic(kind, seconds, fps)), fps)
        ref[f] = round(float(cache[kind][f].max()), 6)
    return ref


def group_rates(feats, ref, weights):
    """무리 가중치 {특징: w} → (T, 2) 발화율 Hz."""
    total = None
    for f, w in weights.items():
        v = w * np.clip(feats[f] / ref[f], 0.0, 1.0) if ref.get(f) else np.zeros_like(feats[f])
        total = v if total is None else total + v
    return RATE_MAX * np.clip(total, 0.0, 1.0)


def summarize(feats, ref):
    """기록용: 특징마다 최댓값·평균과 기준값 대비 지수."""
    s = {}
    for f, v in feats.items():
        r = ref.get(f) or 0
        s[f] = {"max": round(float(v.max()), 5), "mean": round(float(v.mean()), 5), "ref": r,
                "max_index": round(float(v.max() / r), 3) if r else None}
    return s
