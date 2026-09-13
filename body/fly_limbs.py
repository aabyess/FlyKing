"""초파리 다리 6개(밑마디–도래마디–넓적다리–종아리–발목마디 5–발톱·욕반)와 날개 한 쌍(시맥 모양 표는 fly_tex와 같이 쓴다)."""
import math

from mathutils import Vector

from fly_body import CREAM, DARK, LEG, M_HAIR, M_LEG, M_WING, X, Y, Z, mix, smooth
from fly_mesh import bristle, ellipsoid, tube

# 왼쪽 기준(오른쪽은 y·방위각 부호만 뒤집는다). 길이 mm — 🔴 2026-09-13 NeuroMechFly v2 rigging.yaml 마디 길이로 맞춤(PM 결정):
#   문헌 근사값(전체 0.77배)은 flygym 걷기 관절각을 넣으면 뒷다리가 배를 뚫었다(최대 0.70mm). Femur 뼈 = tro + femur = NMF trochanterfemur→tibia.
#   Tarsus5(마지막 0.08)는 NMF에 길이가 없어 그대로. 옛 값: F coxa .30 femur .53 tibia .45 tarsi .14/.07/.05/.05 · M .18/.55/.50 · .20/.09/.06/.05 · H .20/.60/.55 · .22/.10/.07/.05
LEG_CFG = {
    "F": dict(base=(0.58, 0.13, 0.42), coxa_dir=(0.5, 0.2, -0.85), coxa=0.365, tro=0.08, femur=0.625, tibia=0.518,
              tarsi=(0.225, 0.154, 0.099, 0.087, 0.08), phi=30, elev=35, ankle=0.12, r=(0.062, 0.047, 0.031, 0.021)),
    "M": dict(base=(0.30, 0.20, 0.40), coxa_dir=(0.05, 0.45, -0.9), coxa=0.181, tro=0.08, femur=0.704, tibia=0.667,
              tarsi=(0.292, 0.160, 0.091, 0.064, 0.08), phi=85, elev=35, ankle=0.14, r=(0.042, 0.042, 0.028, 0.019)),
    "H": dict(base=(0.08, 0.20, 0.42), coxa_dir=(-0.2, 0.4, -0.9), coxa=0.199, tro=0.08, femur=0.756, tibia=0.684,
              tarsi=(0.353, 0.175, 0.097, 0.073, 0.08), phi=150, elev=40, ankle=0.15, r=(0.045, 0.045, 0.03, 0.02)),
}
# 🔴 뒷다리 밑마디 붙는 자리(PM 허용 2026-09-13): NMF 길이만으로는 flygym 걷기에서 뒷다리 넓적다리가 여전히 배를 지났다(13프레임).
#   NMF는 앞다리 밑마디 기준 뒷다리 밑마디가 0.27mm 낮다 — 붙는 자리를 내린다(안쪽 이동은 시험에서 오히려 나빠짐). 값은 시험 결과로 채운다.
HIND_BASE_SHIFT = (0.0, 0.0, -0.16)   # (x 앞, y 바깥, z 위) mm — 시험: 아래 −0.12 → 3프레임, −0.16·−0.20 → 0(가장 작은 −0.16). 안쪽 −0.07·뒤 −0.12는 나빠지거나 이득 없음
COXA_ROOT = {"F": 0.0, "M": 0.0, "H": 0.2}   # 밑마디 메시를 가슴 쪽으로 늘이는 길이 mm — 붙는 자리가 표면 아래로 내려가도 몸에 붙어 보이게(뼈 머리는 그대로)
# 🔴 1차 반지름(넓적다리 0.062 등)은 소시지 같은 굵은 다리였다 — 몸폭 0.96 대비 넓적다리 지름 약 0.1로 줄임


def leg_joints(side, key):
    """관절 점 — 가슴 붙는 곳(ThC) · 밑마디 끝(CTr) · 도래마디 끝 · 무릎(FTi) · 발목(TiTa) · 발목마디 경계 5 + 발끝."""
    cfg = LEG_CFG[key]
    s = 1.0 if side == "L" else -1.0
    base = Vector((cfg["base"][0], cfg["base"][1] * s, cfg["base"][2]))
    if key == "H":
        base += Vector((HIND_BASE_SHIFT[0], HIND_BASE_SHIFT[1] * s, HIND_BASE_SHIFT[2]))
    cd = Vector((cfg["coxa_dir"][0], cfg["coxa_dir"][1] * s, cfg["coxa_dir"][2])).normalized()
    ctr = base + cd * cfg["coxa"]
    phi, elev = math.radians(cfg["phi"]) * s, math.radians(cfg["elev"])
    hdir = Vector((math.cos(phi), math.sin(phi), 0.0))
    fdir = hdir * math.cos(elev) + Z * math.sin(elev)
    tro = ctr + fdir * cfg["tro"]
    knee = tro + fdir * cfg["femur"]
    dz = knee.z - cfg["ankle"]
    assert dz < cfg["tibia"], f"{side}{key} 종아리가 발목 높이에 못 닿는다"
    ankle = knee + hdir * math.sqrt(cfg["tibia"] ** 2 - dz ** 2) - Z * dz
    total = sum(cfg["tarsi"])
    drop = ankle.z - 0.048                      # 발끝 높이 0.048 — 아래로 휜 발톱 끝·욕반이 바닥(z≈0)에 닿는다
    tip = ankle + hdir * math.sqrt(total ** 2 - drop ** 2) - Z * drop
    tarsal, acc = [ankle], 0.0
    for length in cfg["tarsi"]:
        acc += length
        t = acc / total
        tarsal.append(ankle + (tip - ankle) * t + Z * (0.025 * math.sin(math.pi * t)))
    return dict(base=base, ctr=ctr, tro=tro, knee=knee, tarsal=tarsal, hdir=hdir, s=s)


def _leg_color(t_seg):
    return lambda i, k, co, th: mix(LEG, mix(LEG, DARK, 0.25), smooth(0.8, 1.0, t_seg))


def _row(mesh, p0, p1, radius_fn, count, bone, length, outward, t0=0.15, t1=0.9, radius=0.0045):
    axis = p1 - p0
    d = axis.normalized()
    out = (outward - d * outward.dot(d)).normalized()
    for j in range(count):
        t = t0 + (t1 - t0) * j / max(count - 1, 1)
        p = p0 + axis * t + out * radius_fn(t)
        bristle(mesh, p, d * 0.8 + out * 0.45, length, radius, M_HAIR, bone, bend=Vector((0, 0, 0)), normal=out, sides=3, rings=3)


def build_leg(mesh, side, key):
    j = leg_joints(side, key)
    cfg = LEG_CFG[key]
    name = side + key
    rc, rf, rt, rta = cfg["r"]
    up_out = Z + Y * j["s"] * 0.6
    # 🔴 3차(가운데 1.05rc로 부푼 통)는 옆에서 앞다리 밑마디가 풍선처럼 먼저 보였다(PM) — 약 25% 가늘게, 가슴 쪽이 좁은 원뿔형
    coxa_w = lambda t: rc * 0.8 * (0.62 + 0.38 * t + 0.15 * math.sin(math.pi * t))
    coxa_r = lambda t: (coxa_w(t), coxa_w(t) * 0.9, coxa_w(t) * 0.9)
    root = j["base"] - (j["ctr"] - j["base"]).normalized() * COXA_ROOT[key]
    tube(mesh, root, j["ctr"], coxa_r, M_LEG, name + "Coxa", LEG, 14, 9)
    tube(mesh, j["ctr"] - (j["tro"] - j["ctr"]) * 0.2, j["tro"], lambda t: (rf * 0.7,) * 3, M_LEG, name + "Femur", LEG, 12, 6)
    femur_r = lambda t: (rf * (0.72 + 0.35 * math.sin(math.pi * min(1.0, t * 1.25))), rf * (0.7 + 0.3 * math.sin(math.pi * t)),
                         rf * (0.7 + 0.3 * math.sin(math.pi * t)))
    tube(mesh, j["tro"] - (j["tro"] - j["ctr"]) * 0.4, j["knee"], femur_r, M_LEG, name + "Femur", _leg_color(0.2), 14, 12)
    tibia_r = lambda t: (rt * (0.78 + 0.25 * t),) * 3
    tube(mesh, j["knee"], j["tarsal"][0], tibia_r, M_LEG, name + "Tibia", _leg_color(0.4), 12, 12)
    _row(mesh, j["tro"], j["knee"], lambda t: femur_r(t)[0], 7, name + "Femur", 0.07, up_out)
    _row(mesh, j["tro"], j["knee"], lambda t: femur_r(t)[0], 5, name + "Femur", 0.05, -Z + X * 0.3, 0.3, 0.85)
    _row(mesh, j["knee"], j["tarsal"][0], lambda t: tibia_r(t)[0], 8, name + "Tibia", 0.06, up_out, 0.1, 0.95)
    bristle(mesh, j["knee"] + (j["tarsal"][0] - j["knee"]) * 0.93, (j["tarsal"][0] - j["knee"]).normalized() - Z * 0.3, 0.07,
            0.006, M_HAIR, name + "Tibia", bend=Vector((0, 0, 0)), sides=4)                           # 종아리 끝 가시
    pts = j["tarsal"]
    for q in range(5):
        r0 = rta * (1.0 - 0.1 * q)
        tube(mesh, pts[q] - (pts[q + 1] - pts[q]) * 0.08, pts[q + 1],
             lambda t, r0=r0: (r0 * (0.8 + 0.25 * t),) * 3, M_LEG, f"{name}Tarsus{q + 1}", _leg_color(0.6 + 0.1 * q), 10, 7)
        _row(mesh, pts[q], pts[q + 1], lambda t, r0=r0: r0 * (0.8 + 0.25 * t), 3, f"{name}Tarsus{q + 1}", 0.035,
             -Z + j["hdir"] * 0.2, 0.25, 0.85, 0.003)
    tip, d = pts[5], (pts[5] - pts[4]).normalized()
    lateral = Z.cross(d).normalized()
    for sgn in (1.0, -1.0):                                             # 발톱 한 쌍 + 욕반(pulvillus)
        bristle(mesh, tip + lateral * (0.012 * sgn), d + lateral * (0.35 * sgn), 0.05, 0.007, M_HAIR, f"{name}Tarsus5",
                DARK, bend=Vector((0, 0, -0.9)), sides=4)
        # 🔴 크림색 욕반은 털고르기 정면에서 더듬이 옆 발끝이 눈알처럼 튀었다 — 작게, 다리 색 쪽으로
        ellipsoid(mesh, tip + d * 0.01 + lateral * (0.01 * sgn) - Z * 0.01, d * 0.015, lateral * 0.009, Z * 0.006, M_LEG,
                  f"{name}Tarsus5", mix(LEG, CREAM, 0.35), 6, 10)
    parent = "Thorax"
    for seg, (h, t) in zip(("Coxa", "Femur", "Tibia"), ((j["base"], j["ctr"]), (j["ctr"], j["knee"]), (j["knee"], pts[0]))):
        mesh.joints[name + seg] = (h, t, parent)
        parent = name + seg
    for q in range(5):
        mesh.joints[f"{name}Tarsus{q + 1}"] = (pts[q], pts[q + 1], parent)
        parent = f"{name}Tarsus{q + 1}"
    return j


# ──────────────────────────────────────────────────────────── 날개 (s = 뿌리→끝 mm, w = 앞 가장자리 쪽 +)

WING_ANTERIOR = [(0.0, 0.0), (0.12, 0.10), (0.35, 0.22), (0.75, 0.33), (1.2, 0.41), (1.6, 0.43), (1.9, 0.39),
                 (2.1, 0.29), (2.21, 0.14), (2.24, 0.0)]
WING_POSTERIOR = [(0.0, 0.0), (0.07, -0.10), (0.16, -0.22), (0.26, -0.33), (0.36, -0.37), (0.44, -0.33), (0.50, -0.40),
                  (0.62, -0.47), (0.9, -0.56), (1.25, -0.59), (1.6, -0.56), (1.9, -0.46), (2.1, -0.32), (2.21, -0.16), (2.24, 0.0)]
VEINS = {  # 이름: (점열, 굵기 mm)
    "L1": ([(0.08, 0.06), (0.45, 0.22), (0.85, 0.33), (1.05, 0.375)], 0.012),
    "L2": ([(0.50, 0.12), (0.9, 0.22), (1.4, 0.33), (1.82, 0.40)], 0.010),
    "L3": ([(0.05, 0.02), (0.5, 0.10), (1.1, 0.12), (1.7, 0.12), (2.225, 0.09)], 0.010),
    "L4": ([(0.1, -0.03), (0.7, -0.08), (1.3, -0.17), (1.8, -0.25), (2.12, -0.29)], 0.010),
    "L5": ([(0.15, -0.07), (0.7, -0.25), (1.3, -0.42), (1.66, -0.54)], 0.010),
    "L6": ([(0.2, -0.11), (0.5, -0.28), (0.72, -0.38)], 0.007),
    "ACV": ([(0.86, 0.118), (0.88, -0.105)], 0.008),
    "PCV": ([(1.36, -0.18), (1.42, -0.43)], 0.008),
}
WING_LENGTH = 2.24
UV_BOX = (-0.06, -0.66, 2.32, 0.52)        # 텍스처가 덮는 (s0, w0, s1, w1) — 2:1


def smooth_line(points, sub=6):
    out = []
    n = len(points)
    for i in range(n - 1):
        p0, p1, p2, p3 = (points[max(i - 1, 0)], points[i], points[i + 1], points[min(i + 2, n - 1)])
        for k in range(sub):
            u = k / sub
            out.append(tuple(0.5 * (2 * b + (-a + c) * u + (2 * a - 5 * b + 4 * c - d) * u * u + (-a + 3 * b - 3 * c + d) * u ** 3)
                             for a, b, c, d in zip(p0, p1, p2, p3)))
    out.append(points[-1])
    return out


def outline():
    """닫힌 윤곽(앞 가장자리 뿌리→끝, 뒤 가장자리 끝→뿌리)."""
    return smooth_line(WING_ANTERIOR) + smooth_line(WING_POSTERIOR)[::-1][1:-1]


def _edge_w(line, s):
    for (s0, w0), (s1, w1) in zip(line, line[1:]):
        if s0 <= s <= s1 and s1 > s0:
            return w0 + (w1 - w0) * (s - s0) / (s1 - s0)
    return line[-1][1] if s > line[-1][0] else line[0][1]


def wing_frame(side, pose):
    s = 1.0 if side == "L" else -1.0
    # 🔴 2026-09-13 경첩 y 0.37 → 0.52(PM 허용): 기본 날갯짓 앞끝에서 날개 뿌리가 가슴 벽 속을 지났다(경첩 0.25mm 밖 48점).
    #   바깥 +0.15mm + fly_wing stroke_bias −16°에서 0(가슴 표면과 0.079mm, 렌더로 붙어 보임 확인). 위로 옮기면 오히려 나빠짐.
    hinge = Vector((0.12, 0.52 * s, 1.01))
    psi = math.radians(-12.0 if pose == "rest" else 80.0)
    u = Vector((-math.cos(psi), math.sin(psi) * s, 0.0))
    v = Vector((math.sin(psi), math.cos(psi) * s, 0.0))
    return hinge, u, v


def build_wing(mesh, side, pose="rest"):
    """날개 막 — s 줄마다 뒤 가장자리~앞 가장자리(+잔털 여유 0.035)를 9칸으로 나눈 띠. 쉬는 자세는 배 위로 겹쳐 접고
    (왼 날개가 위), 바깥 앞 가장자리가 살짝 처진다. 편 자세는 옆으로 80°."""
    hinge, u, v = wing_frame(side, pose)
    ant, post = smooth_line(WING_ANTERIOR), sorted(smooth_line(WING_POSTERIOR))
    s0, w0, s1, w1 = UV_BOX
    rows = [WING_LENGTH * (1 - math.cos(math.pi / 2 * i / 36)) for i in range(37)] + [WING_LENGTH + 0.035]
    grid = []
    bone = side + "Wing"
    for s in rows:
        wa = _edge_w(ant, min(s, WING_LENGTH)) + 0.035
        wp = _edge_w(post, min(s, WING_LENGTH)) - 0.035
        if s > WING_LENGTH:
            wa, wp = 0.03, -0.03
        row = []
        for c in range(10):
            w = wp + (wa - wp) * c / 9
            if pose == "rest":
                ramp = smooth(0.0, 0.4, s)
                lift = 0.09 * ramp - 0.10 * w * ramp + (0.012 if side == "L" else 0.0) * ramp
            else:
                lift = 0.05 * s
            p = hinge + u * s + v * w + Z * lift
            row.append((mesh.vert(p, (0.8, 0.8, 0.8), bone), ((s - s0) / (s1 - s0), (w - w0) / (w1 - w0))))
        grid.append(row)
    for i in range(len(grid) - 1):
        for c in range(9):
            quad = [grid[i][c], grid[i + 1][c], grid[i + 1][c + 1], grid[i][c + 1]]
            if side == "L":                                          # 면 법선을 위(+Z)로 — 뒷면 검사가 날개도 센다
                quad = quad[::-1]
            mesh.face([q[0] for q in quad], M_WING, [q[1] for q in quad])
    mesh.joints[bone] = (hinge, hinge + u * WING_LENGTH, "Thorax")
