"""날갯짓 운동학 — 한 박동(위상 0→1)의 날개 판 방향을 몸 좌표계(+X 앞·+Y 왼·Z 위)로 만들고, fly_nmf로 NMF v2 날개 관절각
(c_thorax-l_wing-yaw/pitch/roll)으로 푼다. 뼈대는 그 각을 fly_nmf.Driver로 받는다(걷기 데이터와 같은 길).
매개변수(기본값):
  freq 200Hz · amplitude 150°(스트로크 앞뒤 폭) — PM 지시값
  deviation ±10°(박동당 두 번 → 끝이 8자), rotation ±45°(스트로크 중간 받음각, 반전에서 뒤집힘), stroke_plane 30°(스트로크 면 앞쪽이 몸 긴축보다 아래),
  stroke_bias −16°(스트로크를 뒤로 — 경첩 옆 가슴 벽 회피, 2026-09-13 실측. 원래 0°) — 초파리 자유비행 운동학에서 흔히 보고되는 규모의 대략값(예: Fry, Sayaman & Dickinson 2003).
  flygym 파일에서 온 값이 아니므로 조절용 매개변수로 둔다.
날개 판 기준(φ=θ=α=0): 끝(span) = 옆(+Y 왼), 앞 가장자리 = +X, 등 = +Z. 오른쪽은 XZ 면 거울상(S·R·S)."""
import math

from mathutils import Matrix, Vector

DEFAULTS = dict(freq=200.0, amplitude=150.0, deviation=10.0, rotation=45.0, stroke_plane=30.0, stroke_bias=-16.0, asym=0.0)
# 🔴 한쪽 날개 폭 상한(2026-09-13 실측, 경첩 y 0.52·bias −16°, 위상 24칸): 폭 150°·비대칭 0이면 경첩 0.25mm 밖 날개 정점이 몸(가슴·머리·배) 안에 0.
#   한쪽 폭이 더 크면 날개 뿌리 쪽이 가슴 벽을 조금 스친다 — 비대칭 0.1(157.5°) 1점 · 0.2(165°) 3점 · 0.3(172.5°) 4점 · 폭 165° 6점, 0.5mm 밖은 모두 0.
#   경첩 0.25mm 안은 경첩 구역(실제로도 관절로 이어진 부위)이라 검사에서 뺀다(PM 기준). apply_joint_angles가 넘으면 경고만 한다(데이터는 그대로).
SAFE_AMPLITUDE = 150.0


def _sign_smooth(s, k=3.0):
    return math.tanh(k * s) / math.tanh(k)


def angles_at(phase, p):
    """위상(박동 단위, 0~1 되풀이) → (stroke φ, deviation θ, rotation α) 라디안. φ>0 = 앞."""
    t = phase % 1.0
    phi = math.radians(p["stroke_bias"]) - math.radians(p["amplitude"]) / 2 * math.cos(math.tau * t)   # t=0 뒤 끝 → 0.5 앞 끝
    theta = math.radians(p["deviation"]) * math.sin(2 * math.tau * t)
    alpha = -math.radians(p["rotation"]) * _sign_smooth(math.sin(math.tau * t))                        # 앞으로 갈 땐 앞 가장자리가 들린다
    return phi, theta, alpha


def wing_frame(side, phase, p):
    """(끝 방향, 앞 가장자리 방향) 뼈대 공간 단위벡터."""
    phi, theta, alpha = angles_at(phase, p)
    r = (Matrix.Rotation(math.radians(p["stroke_plane"]), 3, "Y") @ Matrix.Rotation(-phi, 3, "Z")
         @ Matrix.Rotation(theta, 3, "X") @ Matrix.Rotation(alpha, 3, "Y"))
    span, lead = r @ Vector((0.0, 1.0, 0.0)), r @ Vector((1.0, 0.0, 0.0))
    if side == "R":
        span.y, lead.y = -span.y, -lead.y
    return span, lead


def side_params(p, side):
    q = dict(p)
    scale = 1.0 + (0.5 if side == "L" else -0.5) * p.get("asym", 0.0)              # asym>0 → 왼쪽 진폭 큼
    q["amplitude"] = p["amplitude"] * scale
    return q


def solve_angles(nmf, side, span, lead, warm, axis_order=None):
    """날개 판 방향 → NMF v2 날개 3각(이전 해에서 이어 풀어 가지가 튀지 않게)."""
    import numpy as np
    from fly_nmf import AXES, _lm
    bone = side + "Wing"
    span_l, lead_l = nmf.wing_axes(side)
    keys = [(bone, a) for a in AXES]

    def residual(x):
        q = nmf.local("v2", bone, dict(zip(keys, x)), axis_order)
        return np.array([*(q @ span_l - span), *(q @ lead_l - lead)])

    x, cost = _lm(residual, warm)
    return dict(zip(keys, x)), x, math.sqrt(cost)


def beat_angles(nmf, frames, p=None, axis_order=None):
    """한 박동을 frames 칸으로 — 프레임마다 {(뼈, 축): 각}. 첫 칸 = 끝 칸이 되게 마지막 해를 첫 해와 같은 가지로 맞춘다."""
    p = dict(DEFAULTS, **(p or {}))
    out, warm, worst = [], {"L": [0.0, 0.0, 0.0], "R": [0.0, 0.0, 0.0]}, 0.0
    for f in range(frames + 1):
        angles = {}
        for side in "LR":
            span, lead = wing_frame(side, f / frames, side_params(p, side))
            a, warm[side], err = solve_angles(nmf, side, span, lead, warm[side], axis_order)
            angles.update(a)
            worst = max(worst, err)
        out.append(angles)
    return out, worst
