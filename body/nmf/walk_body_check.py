"""걷기 클립이 실제로 다리 마디를 몸에 박는지 — 매 프레임 먼 다리 마디(넓적다리 먼 70%·종아리·발목마디) 광선 vs 몸(가슴·머리·배) 표면.
    blender -b 파일.blend --python walk_body_check.py -- [액션이름 …](없으면 활성 액션)"""
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# 기본 = body/(이 파일은 body/nmf/), 시험 복사본은 환경변수 FLYBODY로
sys.path.insert(0, str(Path(__import__("os").environ.get("FLYBODY") or Path(__file__).resolve().parent.parent)))
import fly_body as fb  # noqa: E402

names = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
sc = bpy.context.scene
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o.name.startswith("초파리"))
mesh = next(o for o in arm.children if o.type == "MESH")
BODY = [b for b in fb.BONES if b in ("Thorax", "Head") or (b[0] == "A" and b[1:2].isdigit())]
gname = {g.index: g.name for g in mesh.vertex_groups}
body_set = {v.index for v in mesh.data.vertices if any(gname[g.group] in BODY for g in v.groups)}
body_polys = [tuple(p.vertices) for p in mesh.data.polygons if all(i in body_set for i in p.vertices)]
acts = [bpy.data.actions[n] for n in names] if names else [arm.animation_data.action]
for act in acts:
    arm.animation_data.action = act
    if len(act.slots):
        arm.animation_data.action_slot = act.slots[0]
    f0, f1 = (int(round(x)) for x in act.frame_range)
    found = {}
    for f in range(f0, f1 + 1):
        sc.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = mesh.evaluated_get(dg)
        me = ev.to_mesh()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        ev.to_mesh_clear()
        mw = np.array(mesh.matrix_world)
        co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        bvh = BVHTree.FromPolygons([Vector(c) for c in co], body_polys)
        for leg in fb.LEGS:
            for b in [leg + "Femur", leg + "Tibia"] + [f"{leg}Tarsus{k}" for k in range(1, 6)]:
                pb = arm.pose.bones[b]
                h, t = arm.matrix_world @ pb.head, arm.matrix_world @ pb.tail
                if b.endswith("Femur"):
                    h = h + (t - h) * 0.3
                d = t - h
                if d.length > 1e-9 and (bvh.ray_cast(h, d.normalized(), d.length)[0] is not None
                                        or bvh.ray_cast(t, -d.normalized(), d.length)[0] is not None):
                    found.setdefault(b, []).append(f)
    print("W", Path(bpy.data.filepath).name, repr(act.name), "프레임", (f0, f1), "몸 뚫는 마디(프레임)",
          {k: (v[0], v[-1], len(v)) for k, v in found.items()} or "없음")
