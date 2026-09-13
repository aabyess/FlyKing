"""뒷다리 관통 진단 — 최악 프레임에서 뚫린 몸 부위·깊이, 우리 대 NMF 다리 마디 길이·밑마디 위치."""
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# 기본 = body/(이 파일은 body/nmf/), 시험 복사본은 환경변수 FLYBODY로
sys.path.insert(0, str(Path(__import__("os").environ.get("FLYBODY") or Path(__file__).resolve().parent.parent)))
import fly_body as fb   # noqa: E402
import fly_nmf          # noqa: E402

sc = bpy.context.scene
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o.name.startswith("초파리"))
mesh = next(o for o in arm.children if o.type == "MESH")
BODY = [b for b in fb.BONES if b in ("Thorax", "Head") or (b[0] == "A" and b[1:2].isdigit())]
gname = {g.index: g.name for g in mesh.vertex_groups}
vgroup = {v.index: next((gname[g.group] for g in v.groups if gname[g.group] in BODY), None) for v in mesh.data.vertices}
polys = [p for p in mesh.data.polygons if all(vgroup[i] for i in p.vertices)]
poly_part = [vgroup[p.vertices[0]] for p in polys]
poly_verts = [tuple(p.vertices) for p in polys]


def evaluate(frame):
    sc.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = mesh.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    mw = np.array(mesh.matrix_world)
    co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    return BVHTree.FromPolygons([Vector(c) for c in co], poly_verts)


def crossings(bvh, h, t):
    d = t - h
    L = d.length
    u = d.normalized()
    hits, origin, travelled = [], h.copy(), 0.0
    while True:
        loc, nrm, idx, dist = bvh.ray_cast(origin, u, L - travelled)
        if loc is None:
            break
        travelled += dist + 1e-5
        hits.append((round(travelled, 3), poly_part[idx]))
        origin = loc + u * 1e-5
    return hits


act = arm.animation_data.action
f0, f1 = (int(round(x)) for x in act.frame_range)
worst = (0, None)
for f in range(f0, f1 + 1):
    bvh = evaluate(f)
    total = 0.0
    for b in ("LHFemur", "LHTibia"):
        pb = arm.pose.bones[b]
        h, t = arm.matrix_world @ pb.head, arm.matrix_world @ pb.tail
        if b.endswith("Femur"):
            h = h + (t - h) * 0.3
        c = crossings(bvh, h, t)
        # 교차점 쌍 사이 = 몸 안 구간(홀수면 끝이 안에 있음 → 끝까지)
        ds = [x[0] for x in c]
        inside = sum(ds[i + 1] - ds[i] for i in range(0, len(ds) - 1, 2)) + ((t - h).length - ds[-1] if len(ds) % 2 else 0.0)
        total += inside
    if total > worst[0]:
        worst = (total, f)
print("H 액션", act.name, "몸 안 길이 최대(넓적다리 먼 70%+종아리, mm)", round(worst[0], 3), "프레임", worst[1])
bvh = evaluate(worst[1])
for b in ("LHFemur", "LHTibia", "LHTarsus1"):
    pb = arm.pose.bones[b]
    h, t = arm.matrix_world @ pb.head, arm.matrix_world @ pb.tail
    print("H  ", b, "머리", tuple(round(c, 3) for c in h), "꼬리", tuple(round(c, 3) for c in t), "교차(거리mm, 부위)", crossings(bvh, h, t))
ab = [arm.matrix_world @ arm.pose.bones[b].head for b in BODY]
print("H 몸 뼈 머리 z", {b: round(p.z, 3) for b, p in zip(BODY, ab)})
nmf = fly_nmf.NMF()
rig = nmf.v2_rig
print("H 마디 길이 우리/NMF(mm):")
for seg, nxt in (("Coxa", "trochanterfemur"), ("Femur", "tibia"), ("Tibia", "tarsus1")):
    ours = arm.data.bones["LH" + seg].length
    key = f"lh_{nxt}"
    nm = Vector(rig[key]["pos"]).length if key in rig else None
    print("H   ", seg, round(ours, 3), round(nm, 3) if nm else (key, "없음"), "비율", round(ours / nm, 2) if nm else None)
coxa = arm.data.bones["LHCoxa"].head_local
nm_coxa = Vector(rig["lh_coxa"]["pos"]) if "lh_coxa" in rig else None
print("H 뒷다리 밑마디 위치 우리", tuple(round(c, 3) for c in coxa), "NMF(가슴 기준)", tuple(round(c, 3) for c in nm_coxa) if nm_coxa else None,
      "가슴 뼈 머리", tuple(round(c, 3) for c in arm.data.bones["Thorax"].head_local))
print("H rig 키 예", [k for k in rig if k.startswith("lh")])
