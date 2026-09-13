"""합성 날갯짓 정밀 검사 — 뼈 자세에서 날개 판(끝·앞 가장자리)을 몸 좌표로 되돌려 fly_wing 기대값과 각도 차, 박동별 폭, 날개·발·메시 최저 z.
    blender -b 결과.blend --python wing_verify.py -- 데이터.csv slow fps"""
import csv
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

B = Path(__import__("os").environ.get("FLYBODY") or Path(__file__).resolve().parent.parent)  # 기본 = body/(이 파일은 body/nmf/), 시험 복사본은 FLYBODY로
sys.path.insert(0, str(B))
import fly_body as fb     # noqa: E402
import fly_limbs as fl    # noqa: E402
import fly_wing           # noqa: E402

data, slow, fps = sys.argv[sys.argv.index("--") + 1:][:3]
slow, fps = float(slow), float(fps)
rows = [r for r in csv.reader(open(data)) if r]
arr = np.array([[float(x) for x in r] for r in rows[1:]])
col = {h.strip(): arr[:, i] for i, h in enumerate(rows[0])}
MAP = {"wingbeat_freq": "freq", "wingbeat_amp": "amplitude", "wingbeat_asym": "asym", "wingbeat_deviation": "deviation",
       "wingbeat_rotation": "rotation", "wingbeat_stroke_plane": "stroke_plane", "wingbeat_bias": "stroke_bias"}
times = col["time"]
sc = bpy.context.scene
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o.name.startswith("초파리"))
mesh = next(o for o in arm.children if o.type == "MESH")
act = arm.animation_data.action
f0, f1 = (int(round(x)) for x in act.frame_range)
gname = {g.index: g.name for g in mesh.vertex_groups}
wing_idx = np.array([v.index for v in mesh.data.vertices if any(gname[g.group].endswith("Wing") for g in v.groups)])
rest_q = {s: arm.data.bones[s + "Wing"].matrix_local.to_quaternion() for s in "LR"}
rest_axes = {}
for s in "LR":
    _, u, v = fl.wing_frame(s, "rest")
    rest_axes[s] = (u.normalized(), v.normalized())
thor_rest = arm.data.bones["Thorax"].matrix_local.to_quaternion()
cap = fps / 4.0
phase, err = 0.0, 0.0
series = {s: {"meas": [], "exp": []} for s in "LR"}
phases, wing_low, foot_low, mesh_low = [], 1e9, 1e9, 1e9


def phi_of(vec, side, plane_deg):
    v = vec.copy()
    if side == "R":
        v.y = -v.y
    v = Matrix.Rotation(math.radians(plane_deg), 3, "Y").inverted() @ v
    return math.degrees(math.atan2(v.x, v.y))


for j in range(f0, f1 + 1):
    t = times[0] + j / (fps * slow)
    p = dict(fly_wing.DEFAULTS, **{MAP[k]: float(np.interp(t, times, col[k])) for k in MAP if k in col})
    if j:
        phase += min(p["freq"] / slow, cap) / fps
    phases.append(phase)
    sc.frame_set(j)
    body = arm.pose.bones["Thorax"].matrix.to_quaternion() @ thor_rest.inverted()
    for s in "LR":
        q = arm.pose.bones[s + "Wing"].matrix.to_quaternion() @ rest_q[s].inverted()
        span, lead = body.inverted() @ (q @ rest_axes[s][0]), body.inverted() @ (q @ rest_axes[s][1])
        es, el = fly_wing.wing_frame(s, phase, fly_wing.side_params(p, s))
        err = max(err, math.degrees(span.angle(es)), math.degrees(lead.angle(el)))
        series[s]["meas"].append(phi_of(span, s, p["stroke_plane"]))
        series[s]["exp"].append(phi_of(es, s, p["stroke_plane"]))
    foot_low = min(foot_low, min((arm.matrix_world @ arm.pose.bones[l + "Tarsus5"].tail).z for l in fb.LEGS))
    dg = bpy.context.evaluated_depsgraph_get()
    ev = mesh.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) @ np.array(ev.matrix_world.to_3x3()).T + np.array(ev.matrix_world.translation)
    wing_low = min(wing_low, float(co[wing_idx, 2].min()))
    mesh_low = min(mesh_low, float(co[:, 2].min()))
    ev.to_mesh_clear()

beats = {}
for i, ph in enumerate(phases):
    beats.setdefault(int(ph), []).append(i)
keys = sorted(beats)
pick = [keys[0], keys[len(keys) // 2], keys[-2] if len(keys) > 1 else keys[-1]]
out = []
for k in pick:
    idx = beats[k] + ([beats[k + 1][0]] if k + 1 in beats else [])
    row = [k]
    for s in "LR":
        m = [series[s]["meas"][i] for i in idx]
        e = [series[s]["exp"][i] for i in idx]
        row += [f"{s} 잰 {max(m) - min(m):.1f}° / 기대 {max(e) - min(e):.1f}°"]
    out.append(row)
print(f"V 판 방향 최대 각도 차 {err:.3f}°  박동 수 {len(keys)}  (박동, 폭) {out}")
print(f"V 최저 z(mm): 날개 메시 {wing_low:.3f}  발끝 {foot_low:.3f}  메시 전체 {mesh_low:.3f}")
