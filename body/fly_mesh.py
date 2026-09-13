"""초파리 메시 도구 — Blender 단위 1 = 1mm. 부위마다 닫힌 껍질을 쌓고, 꼭짓점마다 뼈 번호(강체 가중치 1)와 색을 단다."""
import math

import bmesh
from mathutils import Vector

UP = Vector((0.0, 0.0, 1.0))


class FlyMesh:
    def __init__(self, bones):
        self.bm = bmesh.new()
        self.uv = self.bm.loops.layers.uv.new("UVMap")
        self.color = self.bm.verts.layers.float_color.new("색")
        self.bone = self.bm.verts.layers.int.new("뼈")
        self.bones = list(bones)
        self.mats = []
        self.joints = {}          # 뼈 이름 → [머리, 꼬리, 부모] (뼈대 단계에서 쓴다)

    def m(self, name):
        if name not in self.mats:
            self.mats.append(name)
        return self.mats.index(name)

    def vert(self, co, color, bone):
        v = self.bm.verts.new(co)
        v[self.color] = (color[0], color[1], color[2], 1.0)
        v[self.bone] = self.bones.index(bone)
        return v

    def face(self, verts, mat, uvs=None):
        f = self.bm.faces.new(verts)
        f.material_index = self.m(mat)
        if uvs:
            for loop, uv in zip(f.loops, uvs):
                loop[self.uv].uv = uv
        return f


def frame(tangent, hint):
    """u × v = T 가 되게 — 이 순서로 고리를 돌면 사각면 법선이 바깥을 향한다."""
    t = tangent.normalized()
    h = hint if abs(hint.normalized().dot(t)) < 0.97 else Vector((1.0, 0.0, 0.0))
    u = h.cross(t).normalized()
    return t, u, t.cross(u)


def _sq(c, e):
    return math.copysign(abs(c) ** e, c)


def loft(mesh, pts, radii, mat, bone, color, sides=12, hint=UP, sq=2.0, poles=(None, None), uv_rect=(0, 0, 1, 1)):
    """점열 `pts`를 따라 초타원 고리를 잇는다. radii[i] = (옆 반지름, 위 반지름, 아래 반지름).
    color: (r,g,b) 또는 fn(i, k, 점, 고리 각) → (r,g,b). poles: 양 끝을 닫는 꼭짓점(없으면 고리 중심)."""
    n = len(pts)
    e = 2.0 / sq
    rings, frames = [], []
    u_prev = None
    for i, p in enumerate(pts):
        t = (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized()
        if u_prev is None:
            t, u, v = frame(t, hint)
        else:
            u = (u_prev - t * u_prev.dot(t)).normalized()
            v = t.cross(u)
        u_prev = u
        frames.append(t)
        ru, rup, rdn = radii[i]
        ring = []
        for k in range(sides):
            th = math.tau * k / sides
            c, s = math.cos(th), math.sin(th)
            co = p + u * (ru * _sq(c, e)) + v * ((rup if s >= 0 else rdn) * _sq(s, e))
            col = color(i, k, co, th) if callable(color) else color
            ring.append(mesh.vert(co, col, bone))
        rings.append(ring)
    x0, y0, x1, y1 = uv_rect

    def uv(i, k):
        return (x0 + (x1 - x0) * k / sides, y0 + (y1 - y0) * i / max(n - 1, 1))

    faces = []
    for i in range(n - 1):
        for k in range(sides):
            k1 = (k + 1) % sides
            faces.append(mesh.face([rings[i][k], rings[i][k1], rings[i + 1][k1], rings[i + 1][k]], mat,
                                   [uv(i, k), uv(i, k + 1), uv(i + 1, k + 1), uv(i + 1, k)]))
    for end, i in ((0, 0), (1, n - 1)):
        pole = poles[end] if poles[end] is not None else pts[i]
        col = color(i, 0, pole, 0.0) if callable(color) else color
        pv = mesh.vert(pole, col, bone)
        for k in range(sides):
            k1 = (k + 1) % sides
            vs = [rings[i][k], rings[i][k1], pv] if end else [rings[i][k1], rings[i][k], pv]
            faces.append(mesh.face(vs, mat, [uv(i, k), uv(i, k + 1), (x0 + (x1 - x0) * (k + 0.5) / sides, y1 if end else y0)]))
    bmesh.ops.recalc_face_normals(mesh.bm, faces=faces)
    return rings


def tube(mesh, p0, p1, radius, mat, bone, color, sides=10, rings=9, hint=UP, sq=2.0, uv_rect=(0, 0, 1, 1)):
    """p0→p1 마디. radius(t) → (옆, 위, 아래). 끝은 둥글게 줄여 극점으로 닫는다."""
    axis = p1 - p0
    ts = [0.5 - 0.5 * math.cos(math.pi * (i + 0.5) / rings) for i in range(rings)]
    pts = [p0 + axis * t for t in ts]
    radii = []
    for t in ts:
        ru, rup, rdn = radius(t)
        cap = math.sqrt(max(0.0, 1.0 - (2.0 * t - 1.0) ** 6))
        radii.append((ru * cap, rup * cap, rdn * cap))
    return loft(mesh, pts, radii, mat, bone, color, sides, hint, sq, (p0, p1), uv_rect)


def ellipsoid(mesh, center, a0, a1, a2, mat, bone, color, rings=10, sides=16, uv_rect=(0, 0, 1, 1)):
    """반축 벡터 a0(고리 줄 방향)·a1·a2(서로 수직). a1 방향이 고리의 u가 되게 힌트를 맞춘다."""
    d0, d1, d2 = a0.normalized(), a1.normalized(), a2.normalized()
    if d1.cross(d2).dot(d0) < 0:
        d2 = -d2
    hint = d2
    phis = [math.pi * (i + 0.5) / rings for i in range(rings)]
    pts = [center - a0 * math.cos(ph) for ph in phis]
    radii = [(a1.length * math.sin(ph), a2.length * math.sin(ph), a2.length * math.sin(ph)) for ph in phis]
    return loft(mesh, pts, radii, mat, bone, color, sides, hint, 2.0, (center - a0, center + a0), uv_rect)


def bristle(mesh, base, direction, length, radius, mat, bone, color=(0.05, 0.03, 0.02), bend=None, sides=4, rings=4,
            normal=None):
    """강모 — 뿌리를 표면에 조금 묻고 끝으로 가며 가늘어지는 휜 원추. UV v = 뿌리 0 → 끝 1(강모 텍스처 결)."""
    d = direction.normalized()
    b = bend if bend is not None else Vector((0.0, 0.0, -0.25))
    start = base - (normal.normalized() if normal is not None else d) * radius * 1.5
    pts = [start + d * (length * t) + b * (length * t * t) for t in (i / rings for i in range(rings))]
    radii = [(radius * (1 - 0.85 * i / rings),) * 3 for i in range(rings)]
    tip = start + d * length + b * length
    return loft(mesh, pts, radii, mat, bone, color, sides, UP, 2.0, (start, tip))


def dome(mesh, center, normal, tangent, radius, height, mat, bone, color, sides=6, uv_c=(0.5, 0.5), uv_r=0.5):
    """열린 육각 돔(낱눈 하나) — 테두리는 표면, 가운데는 법선 방향으로 솟는다. 법선 쪽이 앞면."""
    n = normal.normalized()
    u = (tangent - n * tangent.dot(n)).normalized()
    v = n.cross(u)
    outer, inner = [], []
    for k in range(sides):
        th = math.tau * k / sides
        dirv = u * math.cos(th) + v * math.sin(th)
        outer.append((mesh.vert(center + dirv * radius, color, bone), (uv_c[0] + uv_r * math.cos(th), uv_c[1] + uv_r * math.sin(th))))
        inner.append((mesh.vert(center + dirv * radius * 0.55 + n * height * 0.75, color, bone),
                      (uv_c[0] + 0.55 * uv_r * math.cos(th), uv_c[1] + 0.55 * uv_r * math.sin(th))))
    top = mesh.vert(center + n * height, color, bone)
    for k in range(sides):
        k1 = (k + 1) % sides
        (a, ua), (b, ub) = outer[k], outer[k1]
        (c, uc), (d, ud) = inner[k1], inner[k]
        mesh.face([a, b, c, d], mat, [ua, ub, uc, ud])
        mesh.face([d, c, top], mat, [ud, uc, uv_c])
