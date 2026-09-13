"""초파리 겹눈 근사 — 릴스 프레임 → 눈당 낱눈 721개, 광수용체 두 유형의 밝기 시계열.

numpy만 쓴다(인스타 환경·뇌 모델 환경 둘 다에서 import 가능).

근거와 근사
- 낱눈 721개/눈, 한 눈 시야 157°: flygym NeuroMechFly v2 `assets/model/neuromechfly/vision.yaml`
  (num_ommatidia_per_eye: 721, fovy_per_eye: 157). 721 = 1 + 3·15·16 → 반지름 15 고리 육각 격자.
  낱눈 사이 각 Δφ = 157 / 31 ≈ 5.1° (노랑초파리 실측 약 4.5~5°).
- 광수용체 유형 두 가지: pale(R7p·R8p, 자외선·청색 쪽)과 yellow(R7y·R8y, 자외선·녹색 쪽). flygym도 두 채널로 나눈다.
  ⚠ RGB 영상에는 자외선이 없어서 pale ≈ 0.75·B + 0.25·G, yellow ≈ 0.65·G + 0.35·R 로 근사한다. 스펙트럼 민감도 재현이 아니다.
- 화면이 차지하는 시야: 3D 뷰어에서 몸을 세운 초파리 눈과 폰 화면 거리로 잰 대략값 — 가로 70°, 세로 110°, 가운데 위 15°.
  화면 밖은 어두운 방(밝기 ROOM).
- 두 눈 겹침: 가운데에서 각 눈이 반대쪽으로 10°씩(앞쪽 겹침 20°).
- 낱눈 하나는 한 변 ACCEPT_DEG(≈Δφ) 사각 창의 평균 밝기를 받는다(수용각 근사).
- 방위각 +는 초파리 오른쪽, 고각 +는 위. 초파리가 화면을 마주 보므로 영상 오른쪽 = 초파리 오른쪽.
"""
from functools import lru_cache

import numpy as np

N_RING = 15
N_OMM = 1 + 3 * N_RING * (N_RING + 1)          # 721
FOV_DEG = 157.0
DPHI = FOV_DEG / (2 * N_RING + 1)
OVERLAP_DEG = 10.0
SCREEN_W_DEG, SCREEN_H_DEG, SCREEN_EL_DEG = 70.0, 110.0, 15.0
ACCEPT_DEG = DPHI
ROOM = 0.03
EYES = ("L", "R")
TYPES = ("pale", "yellow")


def hex_lattice(n=N_RING):
    """눈 안쪽 좌표(가로 x: 바깥쪽 +, 세로 y: 위 +, 도)."""
    pts = []
    for q in range(-n, n + 1):
        for r in range(max(-n, -q - n), min(n, -q + n) + 1):
            pts.append((DPHI * (q + r / 2.0), DPHI * r * np.sqrt(3) / 2.0))
    return np.array(pts, dtype=np.float64)


def ommatidia_angles():
    """(2, 721, 2): 눈별 낱눈 (방위각, 고각). 눈 0 = 왼쪽, 1 = 오른쪽."""
    lat = hex_lattice()
    center = N_RING * DPHI - OVERLAP_DEG            # 가장 안쪽 낱눈이 반대쪽으로 OVERLAP만큼 넘어가게
    right = np.stack([center + lat[:, 0], lat[:, 1]], axis=1)
    left = np.stack([-(center + lat[:, 0]), lat[:, 1]], axis=1)
    return np.stack([left, right])


@lru_cache(maxsize=8)
def _boxes(hp, wp):
    ang = ommatidia_angles()
    top = SCREEN_EL_DEG + SCREEN_H_DEG / 2
    u = (ang[..., 0] + SCREEN_W_DEG / 2) / SCREEN_W_DEG * wp
    v = (top - ang[..., 1]) / SCREEN_H_DEG * hp
    hx = ACCEPT_DEG / 2 / SCREEN_W_DEG * wp
    hy = ACCEPT_DEG / 2 / SCREEN_H_DEG * hp
    x0, x1, y0, y1 = u - hx, u + hx, v - hy, v + hy
    full = (x1 - x0) * (y1 - y0)
    cx0, cx1 = np.clip(np.floor(x0), 0, wp).astype(int), np.clip(np.ceil(x1), 0, wp).astype(int)
    cy0, cy1 = np.clip(np.floor(y0), 0, hp).astype(int), np.clip(np.ceil(y1), 0, hp).astype(int)
    clipped = np.clip(x1, 0, wp) - np.clip(x0, 0, wp)
    clipped = clipped * (np.clip(y1, 0, hp) - np.clip(y0, 0, hp))
    frac = np.clip(clipped / full, 0.0, 1.0)
    cx1 = np.maximum(cx1, cx0 + (frac > 0))
    cy1 = np.maximum(cy1, cy0 + (frac > 0))
    return cx0, cx1, cy0, cy1, frac


def reel_to_eyes(frames):
    """frames uint8 (T, H, W, 3) → float32 (T, 2 눈, 2 유형, 721) 밝기 0~1(선형)."""
    frames = np.asarray(frames)
    t, hp, wp, _ = frames.shape
    lin = (frames.astype(np.float32) / 255.0) ** 2.2
    chans = (0.75 * lin[..., 2] + 0.25 * lin[..., 1], 0.65 * lin[..., 1] + 0.35 * lin[..., 0])
    x0, x1, y0, y1, frac = _boxes(hp, wp)
    out = np.empty((t, 2, 2, N_OMM), np.float32)
    for c, img in enumerate(chans):
        ii = np.zeros((t, hp + 1, wp + 1), np.float64)
        ii[:, 1:, 1:] = img.cumsum(1).cumsum(2)
        for e in range(2):
            s = ii[:, y1[e], x1[e]] - ii[:, y0[e], x1[e]] - ii[:, y1[e], x0[e]] + ii[:, y0[e], x0[e]]
            area = np.maximum((x1[e] - x0[e]) * (y1[e] - y0[e]), 1)
            out[:, e, c] = frac[e] * (s / area) + (1 - frac[e]) * ROOM
    return out


def summarize(eyes, fps):
    """기록용 입력 요약: 화면을 보는 낱눈 수, 평균 밝기, 프레임 사이 밝기 변화(움직임·깜빡임)."""
    _, _, _, _, frac = _boxes(256, 144)
    on = frac > 0.5
    lum = eyes[:, :, 1]                              # yellow 채널(녹색 쪽)을 밝기 대표로
    d = np.abs(np.diff(lum, axis=0)) * fps if len(lum) > 1 else np.zeros_like(lum[:1])
    return {
        "ommatidia_on_screen": [int(on[0].sum()), int(on[1].sum())],
        "mean_luminance": round(float(lum[:, on].mean()), 4) if on.any() else 0.0,
        "mean_change_per_s": round(float(d[:, on].mean()), 4) if on.any() else 0.0,
        "peak_change_per_s": round(float(d[:, on].max()), 4) if on.any() else 0.0,
        "frames": int(len(eyes)), "fps": fps,
    }


def synthetic(kind, seconds=2.0, fps=10.0, h=256, w=144):
    """검증용 인공 자극(uint8 프레임). gray: 정지 회색, flicker: 전체 깜빡임 5Hz,
    loom: 흰 바탕에 검은 원이 가운데로 다가옴(루밍 — 도망 회로 검증), bars: 오른쪽으로 흐르는 줄무늬,
    vbars: 아래로 흐르는 줄무늬, dot: 작은 검은 점이 가로질러 감."""
    n = int(round(seconds * fps))
    yy, xx = np.mgrid[0:h, 0:w]
    frames = np.empty((n, h, w, 3), np.uint8)
    for k in range(n):
        t = k / fps
        if kind == "gray":
            img = np.full((h, w), 128.0)
        elif kind == "flicker":
            img = np.full((h, w), 230.0 if int(t * 10) % 2 == 0 else 25.0)
        elif kind == "loom":
            # 크기 l/v = 40ms 물체가 t=seconds에 충돌하는 루밍(시야각 2·atan(l/(v·(T−t))))
            ttc = max(seconds + 0.02 - t, 0.02)
            ang = np.degrees(2 * np.arctan(0.04 / ttc))
            r = min(ang, 180) / SCREEN_W_DEG * w / 2
            img = np.where((xx - w / 2) ** 2 + (yy - h / 2) ** 2 < r * r, 10.0, 235.0)
        elif kind == "bars":
            img = np.where(((xx + t * 60) // 16) % 2 == 0, 230.0, 25.0)
        elif kind == "vbars":
            img = np.where(((yy + t * 60) // 16) % 2 == 0, 230.0, 25.0)
        elif kind == "dot":
            # 흰 바탕에 작은 검은 점(지름 약 7°)이 왼쪽→오른쪽으로 지나감 — 작은 물체 움직임 기준
            cx = (t / seconds) * w
            img = np.where((xx - cx) ** 2 + (yy - h * 0.45) ** 2 < (w * 0.05) ** 2, 10.0, 235.0)
        else:
            raise ValueError(kind)
        frames[k] = np.repeat(img[..., None], 3, axis=2).astype(np.uint8)
    return frames
