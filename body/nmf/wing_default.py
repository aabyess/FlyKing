"""기본 날갯짓(150°·비대칭 0) 위상 12개 × 서 있는/비행 몸 — 몸 안쪽 날개 점(두 방향 광선 홀수, 경첩 0.25mm 제외)·어느 몸 부위."""
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

# 기본 = body/(이 파일은 body/nmf/), 시험 복사본은 환경변수 FLYBODY로
sys.path.insert(0, str(Path(__import__("os").environ.get("FLYBODY") or Path(__file__).resolve().parent.parent)))
import fly_body as fb    # noqa: E402
import fly_limbs as fl   # noqa: E402
import fly_nmf           # noqa: E402
import fly_rig           # noqa: E402
import fly_wing          # noqa: E402

arm = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o.name.startswith("초파리"))
mesh = next(o for o in arm.children if o.type == "MESH")
if arm.animation_data:
    arm.animation_data.action = None
for pb in arm.pose.bones:
    pb.rotation_mode = "QUATERNION"
nmf = fly_nmf.NMF()
poser = fly_rig.Poser(arm)
drv = fly_nmf.Driver(nmf, poser, arm, "v2", None, wing_frame_fn=fl.wing_frame)
BODY = [b for b in fb.BONES if b in ("Thorax", "Head") or (b[0] == "A" and b[1:2].isdigit())]
gname = {g.index: g.name for g in mesh.vertex_groups}
vg = {v.index: next((gname[g.group] for g in v.groups), None) for v in mesh.data.vertices}
polys = [tuple(p.vertices) for p in mesh.data.polygons if all(vg[i] in BODY for i in p.vertices)]
part = [vg[p[0]] for p in polys]
wing_idx = [i for i, g in vg.items() if g and g.endswith("Wing")][::3]


# 판정 기준(PM 2026-09-13): 날개 경첩(날개 뼈 머리)에서 0.25mm 안은 「경첩 구역」 — 실제 초파리도 관절로 이어진 부위라 검사에서 뺀다.
#   그 밖 날개 정점은 몸(가슴·머리·배) 안에 0이어야 한다. 기본 날갯짓(폭 150°·비대칭 0, 경첩 y 0.52·bias −16°)에서 0.
#   한쪽 폭이 fly_wing.SAFE_AMPLITUDE(150°)를 넘으면 뿌리 쪽 몇 점이 남는다(주석 참고) — apply_joint_angles가 경고.
def inside(bvh, p):
    first = None
    for direction in (Vector((0.0, 0.0, 1.0)), Vector((1.0, 0.0, 0.0))):
        count, origin = 0, p.copy()
        for _ in range(64):
            loc, _, idx, _ = bvh.ray_cast(origin, direction)
            if loc is None:
                break
            if first is None:
                first = part[idx]
            count += 1
            origin = loc + direction * 1e-5
        if count % 2 == 0:
            return None
    return first


p = dict(fly_wing.DEFAULTS)
for mode in ("ground", "flight"):
    row = []
    for k in range(12):
        ph = k / 12
        ang = {}
        for s in "LR":
            span, lead = fly_wing.wing_frame(s, ph, fly_wing.side_params(p, s))
            ang.update(fly_wing.solve_angles(nmf, s, span, lead, [0.0, 0.0, 0.0])[0])
        des = fly_rig.flight_pose(poser, drv, ang) if mode == "flight" else drv.desired(ang, {"LWing", "RWing"})
        for name, m in poser.basis(des).items():
            arm.pose.bones[name].matrix_basis = m
        bpy.context.view_layer.update()
        ev = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.to_mesh()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        ev.to_mesh_clear()
        co = co.reshape(-1, 3)
        bvh = BVHTree.FromPolygons([Vector(c) for c in co], polys)
        hinge = [arm.pose.bones[s + "Wing"].head for s in "LR"]
        parts = {}
        for i in wing_idx:
            q = Vector(co[i])
            if min((q - h).length for h in hinge) <= 0.25:
                continue
            got = inside(bvh, q)
            if got:
                parts[got] = parts.get(got, 0) + 1
        row.append((round(ph, 2), parts or 0, round(float(co[[i for i in wing_idx], 2].min()), 2)))
    print("D", mode, "(위상, 몸 안쪽 날개 점 부위별, 날개 최저 z)", row)
