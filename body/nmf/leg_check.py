"""걷기 적용 결과 물리 검사: 프레임마다 발끝 최저(쉬는 자세 대비)·메시 최저 z·서로 다른 다리 마디 사이 최소 거리."""
import itertools
import sys
from pathlib import Path

import bpy
from mathutils import Vector

# 기본 = body/(이 파일은 body/nmf/), 시험 복사본은 환경변수 FLYBODY로
sys.path.insert(0, str(Path(__import__("os").environ.get("FLYBODY") or Path(__file__).resolve().parent.parent)))
import fly_body as fb  # noqa: E402

sc = bpy.context.scene
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o.name.startswith("초파리"))
mesh = next(o for o in arm.children if o.type == "MESH")
act = arm.animation_data.action
print("K 액션", act.name, "프레임", tuple(act.frame_range), "다리", fb.LEGS)


def chain(leg):
    names = [b.name for b in arm.data.bones if b.name.startswith(leg)]
    root = next(n for n in names if "Coxa" in n)
    out, cur = [], arm.data.bones[root]
    while cur is not None:
        out.append(cur.name)
        cur = next((c for c in cur.children if c.name.startswith(leg)), None)
    return out


chains = {leg: chain(leg) for leg in fb.LEGS}
print("K 사슬 예", chains[fb.LEGS[0]])


def seg_dist(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1.dot(d1), d2.dot(d2), d2.dot(r)
    if a < 1e-12 and e < 1e-12:
        return r.length
    if a < 1e-12:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = d1.dot(r)
        if e < 1e-12:
            t, s = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = d1.dot(d2)
            den = a * e - b * b
            s = min(max((b * f - c * e) / den, 0.0), 1.0) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    return ((p1 + d1 * s) - (p2 + d2 * t)).length


rest_tip = min((arm.matrix_world @ arm.data.bones[chains[l][-1]].tail_local).z for l in fb.LEGS)
f0, f1 = (int(round(x)) for x in act.frame_range)
worst_pair, worst_d, tip_lo, mesh_lo, tip_rows = None, 1e9, 1e9, 1e9, []
for f in range(f0, f1 + 1):
    sc.frame_set(f)
    segs = {}
    for leg, names in chains.items():
        segs[leg] = [(arm.matrix_world @ arm.pose.bones[n].head, arm.matrix_world @ arm.pose.bones[n].tail) for n in names[1:]]  # 넓적다리부터(밑마디끼리는 원래 붙어 있다)
    for la, lb in itertools.combinations(fb.LEGS, 2):
        d = min(seg_dist(p1, q1, p2, q2) for p1, q1 in segs[la] for p2, q2 in segs[lb])
        if d < worst_d:
            worst_d, worst_pair = d, (f, la, lb)
    tips = {leg: segs[leg][-1][1].z for leg in fb.LEGS}
    tip_lo = min(tip_lo, min(tips.values()))
    if f in (f0, (f0 + f1) // 2, f1):
        tip_rows.append((f, {k: round(v - rest_tip, 3) for k, v in tips.items()}))
    dg = bpy.context.evaluated_depsgraph_get()
    ev = mesh.evaluated_get(dg)
    me = ev.to_mesh()
    mesh_lo = min(mesh_lo, min((ev.matrix_world @ v.co).z for v in me.vertices))
    ev.to_mesh_clear()
print("K 쉬는 자세 발끝 z", round(rest_tip, 4), "클립 발끝 최저", round(tip_lo, 4), "메시 최저", round(mesh_lo, 4))
print("K 다른 다리 마디 사이 최소 거리(mm)", round(worst_d, 4), "프레임·다리", worst_pair)
print("K 발끝 높이(쉬는 자세 대비 mm)", tip_rows)
