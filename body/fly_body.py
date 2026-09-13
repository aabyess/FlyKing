"""초파리(노랑초파리 암컷) 몸통 — 머리(겹눈·홑눈·더듬이·주둥이)·가슴(순판·소순판·강모)·배(마디 띠)·평균곤.
좌표: +X 앞(머리), +Y 왼쪽, Z 위, 원점 = 가슴 아래 바닥. Blender 단위 1 = 1mm(NeuroMechFly와 같은 축)."""
import math
import random

from mathutils import Vector

from fly_mesh import bristle, dome, ellipsoid, loft, tube

X, Y, Z = Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0))
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
LEG_SEGS = ("Coxa", "Femur", "Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5")
BONES = ["Thorax", "Head", "LAntenna", "RAntenna", "LWing", "RWing", "LHaltere", "RHaltere",
         "A1A2", "A3", "A4", "A5", "A6"] + [leg + seg for leg in LEGS for seg in LEG_SEGS]

M_BODY, M_LEG, M_EYE, M_OCELLUS = "초파리_몸", "초파리_다리", "초파리_겹눈", "초파리_홑눈"
M_HAIR, M_HALTERE, M_WING = "초파리_강모", "초파리_평균곤", "초파리_날개_막"


def srgb(r, g, b):
    return tuple(((c / 255 + 0.055) / 1.055) ** 2.4 for c in (r, g, b))


TAN = srgb(152, 106, 58)          # 등판 황갈
PLEURA = srgb(176, 142, 100)      # 옆판(더 옅은 회황갈)
FRONS = srgb(178, 120, 66)        # 이마 주황빛(1차 190,118,52는 너무 주황)
DARK = srgb(62, 44, 30)           # 홑눈 삼각 짙은 갈색
BAND = srgb(86, 60, 38)           # 배 마디 띠
PALE = srgb(198, 170, 122)        # 배 바탕 옅은 황갈(2차 206,172,108은 벌처럼 노랬다)
CREAM = srgb(226, 212, 178)       # 배 아래(복판)
LEG = srgb(192, 150, 92)          # 다리 옅은 황갈
EYE_RED = srgb(150, 20, 14)


def mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def smooth(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


# ──────────────────────────────────────────────────────────── 단면 표(뒤 → 앞): (x, 중심 z, 반폭, 위 반높이, 아래 반높이)

# 🔴 2차: 머리통 반폭 0.37에 눈을 바깥에 붙였더니 정면에서 둥근 공 양옆에 붉은 원판을 단 것처럼 보였다 —
# 눈 높이에서 머리통을 좁히고(0.32) 눈을 안쪽·앞쪽으로 당겨 머리 윤곽을 감싸게 한다.
HEAD = [(0.70, 0.80, 0.18, 0.16, 0.16), (0.73, 0.80, 0.28, 0.26, 0.25), (0.78, 0.80, 0.32, 0.31, 0.30),
        (0.86, 0.795, 0.33, 0.33, 0.32), (0.94, 0.785, 0.32, 0.32, 0.31), (1.01, 0.775, 0.28, 0.28, 0.29),
        (1.06, 0.77, 0.21, 0.22, 0.24), (1.09, 0.765, 0.12, 0.14, 0.16)]
HEAD_SQ = 2.4
THORAX = [(-0.18, 0.68, 0.16, 0.14, 0.18), (-0.13, 0.70, 0.30, 0.24, 0.26), (-0.05, 0.72, 0.40, 0.32, 0.32),
          (0.08, 0.74, 0.46, 0.37, 0.36), (0.25, 0.76, 0.48, 0.39, 0.38), (0.42, 0.77, 0.46, 0.38, 0.38),
          (0.55, 0.77, 0.40, 0.34, 0.36), (0.64, 0.76, 0.30, 0.26, 0.30), (0.70, 0.74, 0.16, 0.16, 0.18)]
THORAX_SQ = 2.2


def _catmull(vals, t):
    n = len(vals) - 1
    f = min(max(t * n, 0.0), n - 1e-9)
    i = int(f)
    u = f - i
    p0, p1, p2, p3 = vals[max(i - 1, 0)], vals[i], vals[i + 1], vals[min(i + 2, n)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)


def resample(sections, count):
    cols = list(zip(*sections))
    return [tuple(_catmull(c, i / (count - 1)) for c in cols) for i in range(count)]


def section_at(sections, x):
    xs = [s[0] for s in sections]
    x = min(max(x, xs[0]), xs[-1])
    for a, b in zip(sections, sections[1:]):
        if a[0] <= x <= b[0]:
            t = (x - a[0]) / (b[0] - a[0])
            return tuple(p + (q - p) * t for p, q in zip(a, b))
    return sections[-1]


def top_point(sections, sq, x, y):
    _, zc, hw, hu, _ = section_at(sections, x)
    q = min(abs(y) / hw, 0.995)
    return Vector((x, y, zc + hu * (1 - q ** sq) ** (1 / sq)))


def top_normal(sections, sq, x, y, h=0.004):
    p = top_point(sections, sq, x, y)
    n = (top_point(sections, sq, x + h, y) - p).cross(top_point(sections, sq, x, y + h) - p).normalized()
    return n if n.z > 0 else -n


def _body_loft(mesh, sections, sq, sides, mat, bone, color, count):
    rs = resample(sections, count)
    pts = [Vector((s[0], 0.0, s[1])) for s in rs]
    radii = [(s[2], s[3], s[4]) for s in rs]
    poles = (pts[0] - X * 0.015, pts[-1] + X * 0.015)
    return loft(mesh, pts, radii, mat, bone, color, sides, Z, sq, poles)


# ──────────────────────────────────────────────────────────── 머리

def _head_color(i, k, co, th):
    c = TAN
    c = mix(c, FRONS, smooth(0.97, 1.05, co.x) * smooth(0.26, 0.12, abs(co.y)))
    c = mix(c, PLEURA, smooth(0.62, 0.52, co.z))
    c = mix(c, DARK, smooth(0.13, 0.06, math.hypot(co.x - 0.89, co.y)) * smooth(1.04, 1.10, co.z))
    return c


def build_eye(mesh, side):
    """겹눈 — 타원체 바탕(짙은 적갈) 위에 육각 격자 낱눈 돔. 줄마다 반 칸 어긋나고, 가장자리로 갈수록 좁아지는 간격에 맞춰 낱눈을 줄인다."""
    s = 1.0 if side == "L" else -1.0
    center = Vector((0.91, 0.25 * s, 0.81))
    n = Vector((0.30, s, 0.02)).normalized()
    up = (Z - n * Z.dot(n)).normalized()
    fwd = up.cross(n) if s > 0 else n.cross(up)
    a, b, c = 0.17, 0.27, 0.23
    # 🔴 1차: 바탕 UV를 틈 색(거의 검정)에 두고 낱눈을 74°까지만 덮었더니 눈 안쪽에 반들한 검은 초승달이 드러났다 → 짙은 빨강 자리, 84°까지
    ellipsoid(mesh, center, up * (b * 0.975), fwd * (c * 0.975), n * (a * 0.95), M_EYE, "Head", EYE_RED,
              rings=16, sides=24, uv_rect=(0.79, 0.49, 0.81, 0.51))
    spacing, cap = 0.019, math.cos(math.radians(84))
    d_beta = spacing * 0.866 / b
    rows = int(math.radians(84) / d_beta)
    count = 0
    for r in range(-rows, rows + 1):
        beta = r * d_beta
        d_alpha = spacing / (c * math.cos(beta))
        half = 0.5 if r % 2 else 0.0
        cols = int(math.radians(80) / d_alpha) + 1
        for q in range(-cols, cols + 1):
            alpha = (q + half) * d_alpha
            if math.cos(beta) * math.cos(alpha) < cap:
                continue
            dn, du, df = math.cos(beta) * math.cos(alpha), math.sin(beta), math.cos(beta) * math.sin(alpha)
            p = center + n * (a * dn) + up * (b * du) + fwd * (c * df)
            grad = (n * (dn / a) + up * (du / b) + fwd * (df / c)).normalized()
            step_a = (-n * (a * math.cos(beta) * math.sin(alpha)) + fwd * (c * math.cos(beta) * math.cos(alpha))).length * d_alpha
            step_b = (-n * (a * math.sin(beta) * math.cos(alpha)) + up * (b * math.cos(beta))
                      - fwd * (c * math.sin(beta) * math.sin(alpha))).length * d_beta / 0.866
            # 🔴 1차(반지름 0.96·높이 0.45)는 낱눈이 따로 박힌 구슬처럼 떠 보였다 — 서로 맞닿게, 낮게
            radius = 0.5 * min(step_a, step_b) * 1.03
            dome(mesh, p, grad, up, radius, radius * 0.28, M_EYE, "Head", EYE_RED)
            count += 1
    return count


def build_head(mesh):
    _body_loft(mesh, HEAD, HEAD_SQ, 32, M_BODY, "Head", _head_color, 22)
    tube(mesh, Vector((0.62, 0.0, 0.77)), Vector((0.76, 0.0, 0.80)), lambda t: (0.12, 0.11, 0.12), M_BODY, "Head", PLEURA, 16, 6)
    facets = {side: build_eye(mesh, side) for side in "LR"}
    # 홑눈 셋 — 정수리 짙은 삼각 위 작은 렌즈, 둘레에 홑눈 강모
    for x, y in ((0.945, 0.0), (0.875, 0.05), (0.875, -0.05)):
        p = top_point(HEAD, HEAD_SQ, x, y)
        nrm = top_normal(HEAD, HEAD_SQ, x, y)
        ellipsoid(mesh, p, X * 0.024, Y * 0.024, nrm * 0.016, M_OCELLUS, "Head", (0.3, 0.05, 0.02), rings=6, sides=12)
    for s in (1.0, -1.0):
        hb = lambda x, y, d, length, bend=(0, 0, -0.2): bristle(
            mesh, top_point(HEAD, HEAD_SQ, x, y * s), Vector((d[0], d[1] * s, d[2])), length, 0.008, M_HAIR, "Head",
            bend=Vector(bend), normal=top_normal(HEAD, HEAD_SQ, x, y * s))
        hb(0.905, 0.035, (0.8, 0.45, 0.5), 0.20, (0, 0, -0.15))        # 홑눈 강모(앞으로 벌어짐)
        hb(0.80, 0.06, (-0.6, -0.3, 0.6), 0.15)                        # 뒤정수리 강모(교차)
        hb(0.81, 0.19, (-0.7, -0.35, 0.5), 0.24)                       # 안쪽 정수리
        hb(0.80, 0.27, (-0.7, 0.45, 0.35), 0.22)                       # 바깥 정수리
        hb(0.99, 0.21, (-0.5, 0.2, 0.75), 0.17)                        # 눈둘레 셋
        hb(1.03, 0.215, (0.6, 0.15, 0.7), 0.16, (0, 0, -0.1))
        hb(0.95, 0.215, (-0.2, 0.25, 0.9), 0.12)
        for j, length in enumerate((0.17, 0.08, 0.06)):                # 입가 강모(vibrissa) 큰 것 하나 + 작은 둘
            bristle(mesh, Vector((1.07 - 0.03 * j, 0.13 * s, 0.56 - 0.02 * j)), Vector((0.8, 0.25 * s, -0.5)), length,
                    0.007 if j == 0 else 0.005, M_HAIR, "Head", bend=Vector((0, 0, -0.2)))
    build_antenna(mesh, "L")
    build_antenna(mesh, "R")
    # 주둥이 — 기부(rostrum)·가운데(haustellum)를 조금 내리고 끝에 입술판(labellum) 두 쪽
    tube(mesh, Vector((0.96, 0.0, 0.55)), Vector((1.01, 0.0, 0.40)), lambda t: (0.085, 0.075, 0.075), M_BODY, "Head", PLEURA, 14, 7)
    tube(mesh, Vector((1.00, 0.0, 0.42)), Vector((1.06, 0.0, 0.31)), lambda t: (0.05, 0.045, 0.045), M_BODY, "Head", LEG, 12, 7)
    for s in (1.0, -1.0):
        ellipsoid(mesh, Vector((1.09, 0.036 * s, 0.285)), X * 0.065, Y * 0.042, Z * 0.045, M_BODY, "Head", CREAM, rings=8, sides=14)
    mesh.joints["Head"] = (Vector((0.70, 0.0, 0.79)), Vector((1.10, 0.0, 0.79)), "Thorax")
    return facets


def build_antenna(mesh, side):
    """더듬이 — 1마디(scape)·2마디(pedicel)·3마디(funiculus, 아래로 늘어진 알꼴) + 깃털 아리스타(위 가지 6·아래 가지 3)."""
    s = 1.0 if side == "L" else -1.0
    bone = side + "Antenna"
    base = Vector((1.075, 0.075 * s, 0.885))
    p1 = base + Vector((0.035, 0.012 * s, 0.006))
    p2 = p1 + Vector((0.05, 0.03 * s, -0.035))
    p3 = p2 + Vector((0.03, 0.025 * s, -0.13))
    tube(mesh, base - X * 0.01, p1, lambda t: (0.024, 0.024, 0.024), M_BODY, bone, TAN, 10, 5)
    tube(mesh, p1, p2, lambda t: (0.026 + 0.012 * t, 0.026 + 0.012 * t, 0.026 + 0.012 * t), M_BODY, bone, TAN, 12, 6)
    axis = (p3 - p2).normalized()
    side_v = Vector((0.0, s, 0.0))
    # 🔴 1차(길이 0.18·폭 0.11)는 얼굴 앞에 주황 알 두 개가 매달린 것처럼 컸다
    ellipsoid(mesh, (p2 + p3) / 2, axis * 0.07, (side_v - axis * side_v.dot(axis)).normalized() * 0.036,
              X * 0.042, M_BODY, bone, mix(FRONS, TAN, 0.5), rings=10, sides=14)
    root = p2 + axis * 0.04 + Vector((0.0, 0.04 * s, 0.0))
    shaft = [root + Vector((0.08, 0.10 * s, 0.03)) * t * 3.0 + Vector((0.0, 0.0, 0.05)) * t * t for t in (0.0, 0.33, 0.66, 1.0)]
    for a, b, r0, r1 in zip(shaft, shaft[1:], (0.010, 0.006, 0.004), (0.006, 0.004, 0.0025)):
        tube(mesh, a, b, lambda t, r0=r0, r1=r1: (r0 + (r1 - r0) * t,) * 3, M_HAIR, bone, DARK, 5, 4)
    along = (shaft[-1] - shaft[0]).normalized()
    for j in range(6):
        t = 0.3 + 0.12 * j
        p = shaft[0] + (shaft[-1] - shaft[0]) * t
        bristle(mesh, p, along * 0.5 + Z * 0.9, 0.09 - 0.008 * j, 0.0025, M_HAIR, bone, DARK, Vector((0, 0, 0)), 3, 3)
    for j in range(3):
        p = shaft[0] + (shaft[-1] - shaft[0]) * (0.45 + 0.15 * j)
        bristle(mesh, p, along * 0.5 - Z * 0.9, 0.06 - 0.01 * j, 0.0025, M_HAIR, bone, DARK, Vector((0, 0, 0)), 3, 3)
    mesh.joints[bone] = (base, p3, "Head")


# ──────────────────────────────────────────────────────────── 가슴

def _thorax_color(i, k, co, th):
    _, zc, hw, hu, hd = section_at(THORAX, co.x)
    up = (co.z - zc) / hu
    c = mix(PLEURA, TAN, smooth(0.05, 0.45, up))
    c = mix(c, CREAM, smooth(-0.55, -0.85, up))
    stripe = smooth(0.05, 0.02, abs(abs(co.y) - 0.14)) * smooth(0.6, 0.8, up)
    return mix(c, DARK, 0.18 * stripe)


# (x, y, 방향, 길이) — 암컷 표준 강모 배치의 근사: 어깨 2·옆가슴 2·날개위 2·날개뒤 2·등가운데 2(쌍마다)
THORAX_BRISTLES = [(0.60, 0.33, (-0.3, 0.6, 0.7), 0.17), (0.56, 0.38, (-0.5, 0.7, 0.4), 0.14),
                   (0.45, 0.43, (-0.7, 0.5, 0.4), 0.20), (0.31, 0.45, (-0.8, 0.5, 0.3), 0.18),
                   (0.20, 0.40, (-0.8, 0.4, 0.45), 0.24), (0.08, 0.39, (-0.8, 0.3, 0.5), 0.20),
                   (-0.01, 0.33, (-0.9, 0.2, 0.4), 0.25), (-0.05, 0.27, (-0.9, 0.1, 0.4), 0.18),
                   (0.26, 0.16, (-0.9, 0.05, 0.45), 0.30), (0.06, 0.16, (-0.9, 0.05, 0.4), 0.36)]


def build_thorax(mesh):
    _body_loft(mesh, THORAX, THORAX_SQ, 36, M_BODY, "Thorax", _thorax_color, 26)
    ellipsoid(mesh, Vector((-0.12, 0.0, 0.985)), X * 0.17, Y * 0.22, Z * 0.095, M_BODY, "Thorax", TAN, rings=12, sides=24)
    # 어깨 굳은살(humeral callus)은 따로 붙이지 않는다 — 🔴 1차 공은 혹, 2차 납작 타원은 테두리 선 달린 렌즈처럼 튀었다. 어깨 강모만 둔다.
    bases = []
    for x, y, d, length in THORAX_BRISTLES:
        for s in (1.0, -1.0):
            p = top_point(THORAX, THORAX_SQ, x, y * s)
            bristle(mesh, p, Vector((d[0], d[1] * s, d[2])), length, 0.011, M_HAIR, "Thorax",
                    bend=Vector((-0.05, 0.0, -0.22)), normal=top_normal(THORAX, THORAX_SQ, x, y * s), sides=5)
            bases.append(p)
    for y, back, length in ((0.17, 0.0, 0.26), (0.07, -0.12, 0.38)):     # 소순판 강모 앞쌍·뒤쌍(뒤쌍은 길고 서로 엇갈림)
        for s in (1.0, -1.0):
            p = Vector((-0.07 + back, y * s, 1.06 - (0.03 if back else 0.0)))
            bristle(mesh, p, Vector((-0.9, -0.25 * s if back else 0.1 * s, 0.35)), length, 0.012, M_HAIR, "Thorax",
                    bend=Vector((0.0, 0.0, -0.2)), sides=5)
            bases.append(p)
    rng = random.Random(7)
    x = 0.08
    while x < 0.66:                                                     # 잔털(microchaetae) — 앞뒤 줄로
        _, _, hw, _, _ = section_at(THORAX, x)
        y = -0.34
        while y <= 0.34:
            yy, xx = y + rng.uniform(-0.012, 0.012), x + rng.uniform(-0.01, 0.01)
            p = top_point(THORAX, THORAX_SQ, xx, yy)
            if abs(yy) < hw * 0.78 and all((p - b).length > 0.05 for b in bases):
                nrm = top_normal(THORAX, THORAX_SQ, xx, yy)
                bristle(mesh, p, -X * 0.9 + nrm * 0.5, 0.055, 0.0035, M_HAIR, "Thorax", bend=Vector((0, 0, -0.1)),
                        normal=nrm, sides=3, rings=3)
            y += 0.052
        x += 0.046
    mesh.joints["Thorax"] = (Vector((-0.20, 0.0, 0.72)), Vector((0.70, 0.0, 0.76)), None)


def build_haltere(mesh, side):
    s = 1.0 if side == "L" else -1.0
    # 🔴 1차(길이 0.3, 옆으로)는 등에서 봐도 옆으로 삐죽 나온 핀이었다 — 실제는 날개 뿌리 뒤 아래에 숨은 0.2mm 곤봉
    base = Vector((0.0, 0.40 * s, 0.84))
    tip = base + Vector((-0.06, 0.05 * s, -0.10))
    tube(mesh, base, tip, lambda t: (0.014 + 0.008 * (1 - t),) * 3, M_HALTERE, side + "Haltere", CREAM, 8, 6, uv_rect=(0, 0, 1, 0.55))
    d = (tip - base).normalized()
    ellipsoid(mesh, tip + d * 0.03, d * 0.045, Y * 0.038, d.cross(Y).normalized() * 0.034, M_HALTERE, side + "Haltere", CREAM,
              8, 14, uv_rect=(0, 0.55, 1, 1))
    mesh.joints[side + "Haltere"] = (base, tip + d * 0.075, "Thorax")


# ──────────────────────────────────────────────────────────── 배

ABD_BONES = ("A1A2", "A1A2", "A3", "A4", "A5", "A6", "A6")
ABD_EDGES = (0.0, 0.11, 0.28, 0.45, 0.61, 0.76, 0.89, 1.0)
ABD_BAND = (1.1, 0.72, 0.62, 0.60, 0.55, 0.45, 0.15)     # 마디 뒤쪽 짙은 띠 시작(1 넘으면 띠 없음)


def abd_axis(s):
    x = -0.10 - 1.26 * s
    z = 0.70 - 0.16 * s ** 1.7
    w = 0.50 * math.sin(math.pi * (0.1 + 0.9 * s)) ** 0.55          # 암컷 배는 가슴만큼 넓다(1차 0.47은 좁았다)
    return x, z, w, 0.37 * w / 0.50, 0.36 * w / 0.50


def build_abdomen(mesh):
    for j, bone in enumerate(ABD_BONES):
        s0, s1 = ABD_EDGES[j] - (0.03 if j else 0.0), ABD_EDGES[j + 1]
        last = j == len(ABD_BONES) - 1
        ts = [i / 9 for i in range(10)]
        if last:
            ts = ts[:-1] + [0.97]
        pts, radii = [], []
        for t in ts:
            x, z, w, hu, hd = abd_axis(s0 + (s1 - s0) * t)
            f = 0.95 + 0.07 * t if not last else 0.95
            pts.append(Vector((x, 0.0, z)))
            radii.append((w * f, hu * f, hd * f))
        band = ABD_BAND[j]

        def color(i, k, co, th, ts=ts, band=band):
            t = ts[i]
            sn, cs = math.sin(th), abs(math.cos(th))
            c = mix(PALE, CREAM, smooth(-0.3, -0.7, sn))
            # 🔴 1차는 띠가 옆구리 아래까지 검은 줄로 내려가 벌 줄무늬 같았다 — 등쪽에서 넓고 옆으로 가며 좁아져 사라진다
            # 🔴 2차(옆으로 0.3씩 밀림)는 띠가 V자 물결로 휘었다 — 거의 곧게, 옆에서 옅어지기만
            dark = smooth(band + 0.1 * cs - 0.04, band + 0.1 * cs + 0.04, t) * smooth(0.0, 0.45, sn)
            return mix(c, BAND, dark)

        tip = abd_axis(1.0)
        poles = (pts[0] + X * 0.01, Vector((tip[0] - 0.02, 0.0, tip[1])) if last else pts[-1] - X * 0.01)
        loft(mesh, pts, radii, M_BODY, bone, color, 32, Z, 2.0, poles)
        if 1 <= j <= 5:                                                 # 마디 뒤 가장자리 털 줄
            x, z, w, hu, hd = abd_axis(s0 + (s1 - s0) * 0.9)
            for q in range(16):
                th = math.radians(28 + 124 * q / 15)
                p = Vector((x, w * 1.01 * math.cos(th), z + hu * 1.01 * math.sin(th)))
                nrm = Vector((0.0, math.cos(th) / w, math.sin(th) / hu)).normalized()
                bristle(mesh, p, -X * 0.85 + nrm * 0.35, 0.08, 0.004, M_HAIR, bone, bend=Vector((0, 0, -0.1)),
                        normal=nrm, sides=3, rings=3)
        x0, z0 = abd_axis(s0)[:2]
        x1, z1 = abd_axis(s1)[:2]
        if bone not in mesh.joints:
            mesh.joints[bone] = [Vector((x0, 0.0, z0)), Vector((x1, 0.0, z1)), "Thorax" if bone == "A1A2" else None]
        else:
            mesh.joints[bone][1] = Vector((x1, 0.0, z1))
    for a, b in zip(("A1A2", "A3", "A4", "A5"), ("A3", "A4", "A5", "A6")):
        mesh.joints[b][2] = a
