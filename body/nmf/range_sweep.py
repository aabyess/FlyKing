"""관절 가동범위 스윕 검사 — 다리 42자유도(flygym 걷기 실측 범위)를 하나씩 최소·최대로(나머지는 범위 가운데), 접지 올린 뒤
   먼 다리 마디가 몸 표면을 뚫는지(BVH 광선, 가운데 자세 기준선 대비)·다른 다리와 최소 거리. 날개: 최대 비대칭 스트로크 위상 × 서 있는/비행 몸.
   blender -b 초파리.blend --python range_sweep.py -- 출력폴더 [--render]"""
import itertools
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

B = Path(__import__("os").environ.get("FLYBODY") or Path(__file__).resolve().parent.parent)  # 기본 = body/(이 파일은 body/nmf/), 시험 복사본은 FLYBODY로
sys.path.insert(0, str(B))
import fly_body as fb     # noqa: E402
import fly_limbs as fl    # noqa: E402
import fly_nmf            # noqa: E402
import fly_rig            # noqa: E402
import fly_wing           # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
out, do_render = Path(args[0]), "--render" in args
out.mkdir(parents=True, exist_ok=True)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o.name.startswith("초파리"))
mesh = next(o for o in arm.children if o.type == "MESH")
if arm.animation_data:
    arm.animation_data.action = None
for pb in arm.pose.bones:
    pb.rotation_mode = "QUATERNION"
nmf = fly_nmf.NMF()
rng = nmf.spec["ranges"]["walking_observed_v1"]
version, found, skipped = nmf.parse(list(rng))
assert version == "v1" and not skipped and len(found) == 42, (version, skipped, len(found))
poser = fly_rig.Poser(arm)
drv1 = fly_nmf.Driver(nmf, poser, arm, "v1", None, wing_frame_fn=fl.wing_frame)
drv2 = fly_nmf.Driver(nmf, poser, arm, "v2", None, wing_frame_fn=fl.wing_frame)
legs_driven = {b for b, _ in found.values()}
BODY = [b for b in fb.BONES if b in ("Thorax", "Head") or (b[0] == "A" and b[1:2].isdigit())]
print("S 몸 뼈", BODY)
gname = {g.index: g.name for g in mesh.vertex_groups}
member = {}
for v in mesh.data.vertices:
    for g in v.groups:
        member.setdefault(gname[g.group], []).append(v.index)
body_set = {i for b in BODY for i in member.get(b, ())}
body_polys = [tuple(p.vertices) for p in mesh.data.polygons if all(i in body_set for i in p.vertices)]
wing_idx = np.array(sorted({i for s in "LR" for i in member.get(s + "Wing", ())}))
leg_idx = np.array(sorted({i for l in fb.LEGS for b in fb.BONES if b.startswith(l) for i in member.get(b, ())}))
rest_tip = min(arm.data.bones[l + "Tarsus5"].tail_local.z for l in fb.LEGS)


def apply(desired, ground=True):
    lift = 0.0
    if ground:
        tip = min((desired[l + "Tarsus5"] @ poser.rest[l + "Tarsus5"].inverted() @ arm.data.bones[l + "Tarsus5"].tail_local.to_4d()).z
                  for l in fb.LEGS if l + "Tarsus5" in desired)
        lift = rest_tip - tip
        for m in desired.values():
            m.translation.z += lift
    for name, m in poser.basis(desired).items():
        arm.pose.bones[name].matrix_basis = m
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = mesh.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    mw = np.array(mesh.matrix_world)
    co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    bvh = BVHTree.FromPolygons([Vector(c) for c in co], body_polys)
    return lift, co, bvh


def segs(leg):
    out_ = []
    for b in [leg + "Femur", leg + "Tibia"] + [f"{leg}Tarsus{k}" for k in range(1, 6)]:
        pb = arm.pose.bones[b]
        h, t = arm.matrix_world @ pb.head, arm.matrix_world @ pb.tail
        if b.endswith("Femur"):
            h = h + (t - h) * 0.3                                     # 밑동은 몸에 붙어 있다 — 먼 70%만
        out_.append((b, h, t))
    return out_


def hits(bvh, leg):
    n = []
    for b, h, t in segs(leg):
        d = t - h
        if d.length > 1e-9 and bvh.ray_cast(h, d.normalized(), d.length)[0] is not None:
            n.append(b[2:])
        if d.length > 1e-9 and bvh.ray_cast(t, -d.normalized(), d.length)[0] is not None and b[2:] not in n:
            n.append(b[2:])
    return n


def seg_dist(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1.dot(d1), d2.dot(d2), d2.dot(r)
    c, b = d1.dot(r), d1.dot(d2)
    den = a * e - b * b
    s = min(max((b * f - c * e) / den, 0.0), 1.0) if den > 1e-12 else 0.0
    t = (b * s + f) / e if e > 1e-12 else 0.0
    if t < 0:
        t, s = 0.0, min(max(-c / a, 0.0), 1.0) if a > 1e-12 else 0.0
    elif t > 1:
        t, s = 1.0, min(max((b - c) / a, 0.0), 1.0) if a > 1e-12 else 0.0
    return ((p1 + d1 * s) - (p2 + d2 * t)).length


def nearest_leg(leg):
    mine = segs(leg)
    best = (1e9, None)
    for other in fb.LEGS:
        if other == leg:
            continue
        d = min(seg_dist(h1, t1, h2, t2) for _, h1, t1 in mine for _, h2, t2 in segs(other))
        best = min(best, (d, other))
    return best


def render(prefix, shot):
    import render_fly
    render_fly.OUT = str(out) + "/"
    render_fly.render([mesh.name], prefix, shots=(shot,), res=(640, 480))


mid = {found[n]: (r["min"] + r["max"]) / 2 for n, r in rng.items()}
_, _, bvh = apply(drv1.desired(dict(mid), legs_driven))
base = {l: hits(bvh, l) for l in fb.LEGS}
base_near = {l: nearest_leg(l) for l in fb.LEGS}
print("S 가운데 자세 기준선 — 몸 뚫는 마디", base, "다리 최소 거리", {l: (round(d, 3), o) for l, (d, o) in base_near.items()})
if do_render:
    render("mid", "three_q")
rows, flags = [], []
for n in sorted(rng):
    bone, axis = found[n]
    leg = bone[:2]
    for tag in ("min", "max"):
        ang = dict(mid)
        ang[(bone, axis)] = rng[n][tag]
        lift, co, bvh = apply(drv1.desired(ang, legs_driven))
        new = [h for h in hits(bvh, leg) if h not in base[leg]]
        d, other = nearest_leg(leg)
        row = (n.replace("joint_", ""), tag, round(math.degrees(rng[n][tag]), 1), round(lift, 3), new, round(d, 3), other)
        rows.append(row)
        if new or d < 0.1:
            flags.append(row)
        if do_render and leg[0] == "L":
            render(f"{n.replace('joint_', '')}_{tag}", "three_q")
print("S 스윕", len(rows), "자세  (관절, 끝, 각°, 접지 올림mm, 새로 몸 뚫는 마디, 다른 다리 최소 거리mm, 그 다리)")
for r in rows:
    print("S  ", r)
print("S 걸린 것(몸 뚫음 또는 다리 사이 0.1mm 미만)", len(flags))
for r in flags:
    print("S ⚑", r)

# 날개: 최대 폭(150°·비대칭 0.3 → 왼 172.5°) 위상 넷 × 서 있는/비행 몸. 몸 안쪽 판정 = 두 방향 광선 교차 수가 둘 다 홀수(겹친 껍질은 놓칠 수 있다),
#   날개 뿌리(경첩, 뼈 머리에서 0.25mm 안)는 원래 가슴에 붙어 있어 뺀다. 기준선 = 날개 접은 쉬는 자세.
# 판정 기준(PM 2026-09-13): 날개 경첩(날개 뼈 머리)에서 0.25mm 안은 「경첩 구역」 — 실제 초파리도 관절로 이어진 부위라 검사에서 뺀다.
#   그 밖 날개 정점은 몸(가슴·머리·배) 안에 0이어야 한다. 기본 날갯짓(폭 150°·비대칭 0, 경첩 y 0.52·bias −16°)에서 0.
#   한쪽 폭이 fly_wing.SAFE_AMPLITUDE(150°)를 넘으면 뿌리 쪽 몇 점이 남는다(주석 참고) — apply_joint_angles가 경고.
def inside(bvh, p):
    for direction in (Vector((0.0, 0.0, 1.0)), Vector((1.0, 0.0, 0.0))):
        count, origin = 0, p.copy()
        for _ in range(64):
            loc = bvh.ray_cast(origin, direction)[0]
            if loc is None:
                break
            count += 1
            origin = loc + direction * 1e-5
        if count % 2 == 0:
            return False
    return True


def wing_report(label, co, bvh):
    hinge = {s: arm.matrix_world @ arm.pose.bones[s + "Wing"].head for s in "LR"}
    pts = [Vector(co[i]) for i in wing_idx[::3]]
    pts = [q for q in pts if min((q - h).length for h in hinge.values()) > 0.25]
    n_in = sum(1 for q in pts if inside(bvh, q))
    wl, lg = co[wing_idx], co[leg_idx]
    gap = float(np.min(np.linalg.norm(wl[::6, None, :] - lg[None, ::6, :], axis=2)))
    print(f"S 날개 {label}: 날개 최저 z {wl[:, 2].min():+.3f}mm  메시 최저 z {co[:, 2].min():+.3f}mm  "
          f"몸 안쪽 날개 점 {n_in}/{len(pts)}  날개↔다리 최소 거리 {gap:.3f}mm")


for pb in arm.pose.bones:
    pb.matrix_basis.identity() if False else None
    pb.matrix_basis = __import__("mathutils").Matrix.Identity(4)
bpy.context.view_layer.update()
_, co, bvh = apply({}, ground=False)
wing_report("쉬는 자세(접음) 기준선", co, bvh)
p = dict(fly_wing.DEFAULTS, asym=0.3)
for mode in ("ground", "flight"):
    for ph in (0.0, 0.25, 0.5, 0.75):
        ang = {}
        for s in "LR":
            span, lead = fly_wing.wing_frame(s, ph, fly_wing.side_params(p, s))
            a, _, _ = fly_wing.solve_angles(nmf, s, span, lead, [0.0, 0.0, 0.0])
            ang.update(a)
        if mode == "flight":
            lift, co, bvh = apply(fly_rig.flight_pose(poser, drv2, ang), ground=False)
        else:
            lift, co, bvh = apply(drv2.desired(ang, {"LWing", "RWing"}), ground=False)
        wing_report(f"{mode} 위상 {ph:.2f}", co, bvh)
        if do_render:
            render(f"wing_{mode}_{ph:.2f}", "lateral")
if do_render:
    for r in flags:
        pass
