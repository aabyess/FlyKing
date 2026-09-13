"""공장 작업대 자극 — 초파리 겹눈 입력용 프레임 uint8 (T, 256, 144, 3).

화면 좌표 144×256은 초파리 정면 시야 창이다(brain/visual/eye.py가 가로 70°·세로 110°로 매핑).
작업대 앞에 선 초파리가 정면으로 보는 장면을 이 창에 그린다. 밝기 값은 0~255 sRGB.

- sorter(kind): 회색 벨트(128) 위로 상자가 왼쪽 → 오른쪽으로 지나감. kind = bright(235) · dark(20) · none(빈 벨트).
  상자 한 변 26px(가로 약 13°), 속도 124px/s(약 60°/s). 벨트 뒤 벽은 90.
- guard(kind): 하늘(200). intruder = 다가오는 검은 원(루밍, l/v = 40ms, 창 끝 무렵 충돌),
  clouds = 천천히 흐르는 큰 흐린 얼룩(방해 움직임), none = 빈 하늘.
- sugar(): 설탕대는 시각 변화 없는 회색 화면. 설탕 세기는 뇌 입력(당 감각뉴런 발화율)으로 따로 준다.
seed는 위치·시각을 조금씩 바꿔 매 판단 장면이 똑같지 않게 한다(판정은 여전히 뇌 계산).
"""
import numpy as np

H, W = 256, 144
BELT, WALL, BRIGHT, DARK, SKY = 128.0, 90.0, 235.0, 20.0, 200.0


def _rgb(gray):
    return np.repeat(np.clip(gray, 0, 255)[..., None], 3, axis=-1).astype(np.uint8)


def sorter(kind, seconds=0.5, fps=10.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(round(seconds * fps))
    yy, xx = np.mgrid[0:H, 0:W]
    y0 = H * 0.58 + rng.uniform(-6, 6)
    x_start = rng.uniform(-8, 18)
    size, speed = 26.0, 124.0
    frames = np.full((n, H, W), BELT)
    for k in range(n):
        frames[k, : int(H * 0.35)] = WALL
        if kind in ("bright", "dark"):
            cx = x_start + speed * (k / fps)
            box = (np.abs(xx - cx) < size / 2) & (np.abs(yy - y0) < size / 2)
            frames[k][box] = BRIGHT if kind == "bright" else DARK
    return _rgb(frames)


def guard(kind, seconds=0.5, fps=10.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(round(seconds * fps))
    yy, xx = np.mgrid[0:H, 0:W]
    frames = np.full((n, H, W), SKY)
    if kind == "intruder":
        cx, cy = W / 2 + rng.uniform(-15, 15), H * 0.45 + rng.uniform(-20, 20)
        t_hit = seconds + 0.02 + rng.uniform(-0.03, 0.05)
        for k in range(n):
            ttc = max(t_hit - k / fps, 0.02)
            ang = np.degrees(2 * np.arctan(0.04 / ttc))
            r = min(ang, 170) / 70.0 * W / 2
            frames[k][(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = DARK
    elif kind == "clouds":
        blobs = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(25, 45), rng.uniform(-25, 25)) for _ in range(3)]
        for k in range(n):
            t = k / fps
            img = np.full((H, W), SKY)
            for bx, by, rad, vx in blobs:
                d2 = (xx - (bx + vx * t)) ** 2 + (yy - by) ** 2
                img -= 55.0 * np.exp(-d2 / (2 * rad * rad))
            frames[k] = img
    return _rgb(frames)


def sugar(seconds=0.5, fps=10.0, seed=0):
    return _rgb(np.full((int(round(seconds * fps)), H, W), BELT))


FLOOR, DARK_WALL, MONSTER = 110.0, 70.0, 15.0


def soldier(kind, side="left", seconds=0.5, fps=10.0, seed=0):
    """병정 초파리 앞 공장 바닥(110)·어두운 벽(70).

    ant    = 좀비 개미: 작은 검은 몸(가로 22px ≈ 11°) + 다리 6개가 side 가장자리에서 가운데 쪽으로 걸어 들어온다(약 110px/s).
             분류대 상자(26px, 124px/s)와 같은 「작게 움직이는 것」 — 다가가기(oDN1·P9) 확인용.
    spider = 거미 괴물: side 쪽에서 다리 달린 검은 원이 덮치듯 커진다(루밍 l/v 40ms, 창 끝 무렵 충돌) — 거대섬유 확인용.
    none   = 빈 바닥.
    side: left(화면 왼쪽 절반) · right(오른쪽 절반).
    """
    rng = np.random.default_rng(seed)
    n = int(round(seconds * fps))
    yy, xx = np.mgrid[0:H, 0:W]
    sgn = -1.0 if side == "left" else 1.0
    frames = np.full((n, H, W), FLOOR)
    frames[:, : int(H * 0.35)] = DARK_WALL
    if kind == "ant":
        y0 = H * 0.62 + rng.uniform(-8, 8)
        x_edge = W / 2 + sgn * (W / 2 + 6)
        for k in range(n):
            t = k / fps
            cx = x_edge - sgn * 110.0 * t
            body = ((xx - cx) / 11.0) ** 2 + ((yy - y0) / 7.0) ** 2 < 1.0
            frames[k][body] = MONSTER
            phase = 1.0 if k % 2 == 0 else -1.0                                   # 다리가 번갈아 앞뒤로(깜빡이는 움직임)
            for j, dy in enumerate((-6.0, 0.0, 6.0)):
                for s in (-1.0, 1.0):
                    lx = cx + (dy + phase * (2.0 if j % 2 == 0 else -2.0))
                    leg = (np.abs(xx - lx) < 1.2) & (np.abs(yy - (y0 + s * 11.0)) < 4.5)
                    frames[k][leg] = MONSTER
    elif kind == "spider":
        cx, cy = W / 2 + sgn * W * 0.25 + rng.uniform(-6, 6), H * 0.5 + rng.uniform(-15, 15)
        t_hit = seconds + 0.02 + rng.uniform(-0.03, 0.05)
        ang_img = np.arctan2(yy - cy, xx - cx)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        for k in range(n):
            ttc = max(t_hit - k / fps, 0.02)
            ang = np.degrees(2 * np.arctan(0.04 / ttc))
            r = min(ang, 170) / 70.0 * W / 2
            frames[k][dist < r] = MONSTER
            for a in np.linspace(-np.pi, np.pi, 8, endpoint=False) + np.pi / 8:   # 다리 8개(몸 반지름의 1.8배까지)
                d_ang = np.abs(np.angle(np.exp(1j * (ang_img - a))))
                frames[k][(dist < r * 1.8) & (dist * d_ang < max(1.5, r * 0.12))] = MONSTER
    return _rgb(frames)
