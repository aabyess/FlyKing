"""stroke_bias 스캔 — 서 있는 몸(날개만 움직임)에서 날개 정점이 가슴·머리·배 속에 드는지. 날개는 강체 가중치라 Driver 원하는 행렬로 직접 옮긴다.
   안쪽 판정 = +X·+Y·+Z 세 방향 광선 교차 수 홀수가 둘 이상(겹친 껍질 잡음 줄임), 경첩(뼈 머리) 0.25mm 안은 뺀다."""
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# 기본 = body/(이 파일은 body/nmf/), 시험 복사본은 환경변수 FLYBODY로
sys.path.insert(0, str(Path(__import__("os").environ.get("FLYBODY") or Path(__file__).resolve().parent.parent)))
import fly_body as fb    # noqa: E402
import fly_limbs as fl   # noqa: E402
import fly_nmf           # noqa: E402
import fly_rig           # noqa: E402
import fly_wing          # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
biases = [float(x) for x in args[0].split(",")]
asyms = [float(x) for x in args[1].split(",")]
amps = [float(x) for x in args[2].split(",")] if len(args) > 2 else [150.0]
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o.name.startswith("초파리"))
mesh = next(o for o in arm.children if o.type == "MESH")
nmf = fly_nmf.NMF()
poser = fly_rig.Poser(arm)
drv = fly_nmf.Driver(nmf, poser, arm, "v2", None, wing_frame_fn=fl.wing_frame)
BODY = {b for b in fb.BONES if b in ("Thorax", "Head") or (b[0] == "A" and b[1:2].isdigit())}
gname = {g.index: g.name for g in mesh.vertex_groups}
vg = {v.index: next((gname[g.group] for g in v.groups), None) for v in mesh.data.vertices}
mw = mesh.matrix_world
co = [mw @ v.co for v in mesh.data.vertices]
bvh = BVHTree.FromPolygons(co, [tuple(p.vertices) for p in mesh.data.polygons if all(vg[i] in BODY for i in p.vertices)])
wing = {s: [co[i] for i, g in vg.items() if g == s + "Wing"] for s in "LR"}
DIRS = (Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)))


def inside(p):
    odd = 0
    for d in DIRS:
        n, o = 0, p.copy()
        for _ in range(64):
            loc = bvh.ray_cast(o, d)[0]
            if loc is None:
                break
            n += 1
            o = loc + d * 1e-5
        odd += n % 2
    return odd >= 2


PH = 24
for amp in amps:
    for asym in asyms:
        for bias in biases:
            p = dict(fly_wing.DEFAULTS, amplitude=amp, asym=asym, stroke_bias=bias)
            worst, per = (-1, 0.0), []
            for k in range(PH):
                ph = k / PH
                ang = {}
                for s in "LR":
                    span, lead = fly_wing.wing_frame(s, ph, fly_wing.side_params(p, s))
                    ang.update(fly_wing.solve_angles(nmf, s, span, lead, [0.0, 0.0, 0.0])[0])
                des = drv.desired(ang, {"LWing", "RWing"})
                n = 0
                for s in "LR":
                    M = des[s + "Wing"] @ poser.rest[s + "Wing"].inverted()
                    hinge = des[s + "Wing"].translation
                    for q in wing[s]:
                        w = M @ q
                        if (w - hinge).length > 0.25 and inside(w):
                            n += 1
                per.append(n)
                worst = max(worst, (n, round(ph, 3)))
            print(f"B 폭 {amp:g} 비대칭 {asym:g} bias {bias:+g}°: 최대 {worst[0]}점(위상 {worst[1]})  위상별 {per}", flush=True)
