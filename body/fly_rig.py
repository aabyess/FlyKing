"""초파리 뼈대·동작 — 뼈 이름은 NeuroMechFly 관절 체계, 뼈마다 강체 가중치 1(꼭짓점 속성 「뼈」→ 정점 그룹).
클립(30fps, 첫 = 끝):
  Walk_Tripod 30프레임(1초) — 삼각 보행 한 주기(LF·RM·LH ↔ RF·LM·RH). 제자리 걸음(뿌리 이동 없음), 몸통 오르내림·옆 흔들림.
  Idle_Groom  60프레임(2초) — 몸을 살짝 들고 앞다리 둘로 겹눈·더듬이 앞을 세 번 문지른다. 가운데·뒷다리 발은 제자리.
  Flight_Wingbeat 40프레임 = 날갯짓 한 박동(실제 200Hz면 5ms — 느린 원본, 재생 배속으로 맞춘다). 다리는 접은 자세.
    날개는 fly_wing 운동학 → NMF v2 날개 관절각 → fly_nmf.Driver(걷기 데이터와 같은 길)로 돌린다.
  🔸 실제 노랑초파리 걸음은 한 주기 약 0.1초(10Hz) — 보기 좋게 1초로 늘렸으니 실제 속도는 재생 10배속으로 맞춘다.
다리는 IK로 푼다: 발끝 목표 → 발목마디 다섯(한 덩어리로 돌림) → 넓적다리·종아리 두 뼈 IK(무릎은 극 방향 쪽) →
밑마디는 발 방위를 40%만 따라 돈다. 발끝이 땅 아래로 안 가게 디딤 발은 쉬는 자세 높이 그대로 둔다."""
import json
import math

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

import fly_body as fb
import fly_limbs as fl
import fly_nmf
import fly_wing

FPS = 30
CLIPS = (("Walk_Tripod", 30), ("Idle_Groom", 60), ("Flight_Wingbeat", 40))
FLIGHT_PITCH, FLIGHT_LIFT = 40.0, 1.2   # 날 때 몸 머리 들기(°)·띄우기(mm) — 제자리 떠 있는 클립
# 날 때 접은 다리 발끝 목표(왼쪽, 오른쪽은 y 반대)·발목마디 방향·무릎 극 — 앞다리는 머리 아래로 앞을 향해, 가운데·뒷다리는 배 아래 뒤로
FLIGHT_LEGS = {"F": ((1.00, 0.16, 0.20), (1.0, -0.1, -0.6), (-0.3, 0.9, 0.3)),
               "M": ((-0.60, 0.45, 0.15), (-1.0, 0.1, -0.3), (0.0, 0.3, 1.0)),
               "H": ((-1.30, 0.28, 0.25), (-1.0, -0.05, -0.2), (0.0, 0.2, 1.0))}
Z = Vector((0.0, 0.0, 1.0))
TRIPOD_A = ("LF", "RM", "LH")
STRIDE, LIFT = 0.6, 0.16          # mm — 한 걸음 앞뒤 폭·발 드는 높이(1차 0.5·0.12는 옆에서 걷는지 안 보였다)


def _smooth(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def _yaw(v):
    return math.atan2(v.y, v.x)


def _wrap(a):
    return (a + math.pi) % math.tau - math.pi


def _fcurves(action):
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    return [fc for layer in action.layers for strip in layer.strips for bag in strip.channelbags for fc in bag.fcurves]


def _set_action(arm, action):
    ad = arm.animation_data or arm.animation_data_create()
    ad.action = action
    if hasattr(ad, "action_slot") and len(action.slots):
        ad.action_slot = action.slots[0]


def build_armature(obj, collection, name):
    joints = json.loads(obj["joints"])
    data = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, data)
    collection.objects.link(arm)
    for o in bpy.context.view_layer.objects:
        o.select_set(o == arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for bone in fb.BONES:
        head, tail, parent = joints[bone]
        eb = data.edit_bones.new(bone)
        eb.head, eb.tail = Vector(head), Vector(tail)
        eb.align_roll(Z if abs((eb.tail - eb.head).normalized().z) < 0.95 else Vector((1.0, 0.0, 0.0)))
        eb.inherit_scale = "NONE"
        if parent:
            eb.parent = data.edit_bones[parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.parent = arm
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    return arm


class Poser:
    """뼈마다 원하는 뼈대 공간 행렬을 주면 부모 순서대로 matrix_basis를 푼다(안 준 뼈는 부모를 강체로 따라간다)."""

    def __init__(self, arm):
        bones = arm.data.bones
        self.rest = {b.name: b.matrix_local.copy() for b in bones}
        self.parent = {b.name: b.parent.name if b.parent else None for b in bones}
        self.head = {b.name: b.head_local.copy() for b in bones}
        self.tail = {b.name: b.tail_local.copy() for b in bones}

    def placed(self, bone, head, q):
        m = q.to_matrix().to_4x4() @ self.rest[bone].to_3x3().to_4x4()
        m.translation = head
        return m

    def basis(self, desired):
        pose, out = {}, {}
        for name in fb.BONES:
            p, rest = self.parent[name], self.rest[name]
            local = self.rest[p].inverted() @ rest if p else rest
            d = desired.get(name)
            if d is None:
                d = pose[p] @ local if p else rest
            pose[name] = d
            out[name] = local.inverted() @ (pose[p].inverted() @ d if p else d)
        return out

    def leg(self, desired, leg, body, tip, tars_dir=None, pole=Z):
        c, f, ti = leg + "Coxa", leg + "Femur", leg + "Tibia"
        base_r, ctr_r = self.head[c], self.tail[c]
        knee_r, ankle_r = self.head[ti], self.tail[ti]
        tip_r = self.tail[leg + "Tarsus5"]
        base = body @ base_r
        qc = Quaternion(Z, 0.4 * _wrap(_yaw(tip - base) - _yaw(tip_r - base_r))) @ body.to_quaternion()
        ctr = base + qc @ (ctr_r - base_r)
        chord = tip_r - ankle_r
        if tars_dir is None:
            qt = Quaternion(Z, _wrap(_yaw(tip - ctr) - _yaw(tip_r - ctr_r)))
        else:
            qt = chord.normalized().rotation_difference(tars_dir.normalized())
        ankle = tip - qt @ chord
        lf, lt = (knee_r - ctr_r).length, (ankle_r - knee_r).length
        d = ankle - ctr
        dist = min(max(d.length, abs(lf - lt) + 1e-4), lf + lt - 1e-4)
        axis = d.normalized()
        up = (pole - axis * pole.dot(axis)).normalized()
        a = (lf * lf - lt * lt + dist * dist) / (2 * dist)
        knee = ctr + axis * a + up * math.sqrt(max(0.0, lf * lf - a * a))
        ankle = ctr + axis * dist
        desired[c] = self.placed(c, base, qc)
        desired[f] = self.placed(f, ctr, (knee_r - ctr_r).rotation_difference(knee - ctr))
        desired[ti] = self.placed(ti, knee, (ankle_r - knee_r).rotation_difference(ankle - knee))
        for k in range(1, 6):
            b = f"{leg}Tarsus{k}"
            desired[b] = self.placed(b, ankle + qt @ (self.head[b] - ankle_r), qt)
        return d.length - dist


def _about(pivot, matrix):
    return Matrix.Translation(pivot) @ matrix @ Matrix.Translation(-pivot)


def walk_pose(poser, t):
    pivot = Vector((0.25, 0.0, 0.75))
    body = Matrix.Translation((0.0, 0.0, 0.012 * math.cos(4 * math.pi * t))) @ _about(
        pivot, Matrix.Rotation(math.radians(1.5) * math.sin(math.tau * t), 4, "X"))
    desired = {"Thorax": body @ poser.rest["Thorax"]}
    for leg in fb.LEGS:
        p = (t + (0.0 if leg in TRIPOD_A else 0.5)) % 1.0
        tip = poser.tail[leg + "Tarsus5"].copy()
        if p < 0.5:                                                    # 디딤: 몸 기준 뒤로 일정 속도(발 높이 그대로)
            tip.x += STRIDE * (0.5 - p / 0.5)
        else:                                                          # 흔듦: 들어 앞으로
            q = (p - 0.5) / 0.5
            tip.x += STRIDE * (-0.5 + _smooth(0.0, 1.0, q))
            tip.z += LIFT * math.sin(math.pi * q)
        poser.leg(desired, leg, body, tip)
    return desired


def groom_pose(poser, t):
    e = _smooth(0.0, 0.2, t) * _smooth(1.0, 0.8, t)
    body = Matrix.Translation((0.0, 0.0, 0.03 * e)) @ _about(Vector((0.05, 0.0, 0.42)), Matrix.Rotation(math.radians(-7) * e, 4, "Y"))
    desired = {"Thorax": body @ poser.rest["Thorax"]}
    w = math.tau * 3 * t
    tilt = _about(poser.head["Head"], Matrix.Rotation(math.radians(10) * e, 4, "Y") @ Matrix.Rotation(math.radians(6) * e * math.sin(w), 4, "X"))
    desired["Head"] = body @ tilt @ poser.rest["Head"]
    for side in "LR":
        s = 1.0 if side == "L" else -1.0
        for key in "MH":
            poser.leg(desired, side + key, body, poser.tail[side + key + "Tarsus5"])
        leg = side + "F"
        rest_tip, ankle_r = poser.tail[leg + "Tarsus5"], poser.head[leg + "Tarsus1"]
        ph = w + (0.0 if side == "L" else math.pi)                     # 좌우가 반 박자 어긋나 서로 비빈다
        goal = body @ Vector((1.22 + 0.03 * math.cos(ph), 0.15 * s + 0.05 * s * math.sin(ph), 0.78 + 0.09 * math.sin(ph)))
        tdir = (rest_tip - ankle_r).normalized().lerp(Vector((0.15, -0.35 * s, 1.0)).normalized(), e).normalized()
        pole = Z.lerp(Vector((-0.2, 0.8 * s, 0.6)).normalized(), e).normalized()
        poser.leg(desired, leg, body, rest_tip.lerp(goal, e), tdir, pole)
    return desired


def flight_pose(poser, driver, wing_angles):
    desired = {"Thorax": poser.rest["Thorax"].copy()}
    for leg in fb.LEGS:
        s = 1.0 if leg[0] == "L" else -1.0
        tip, tdir, pole = FLIGHT_LEGS[leg[1]]
        mirror = lambda v: Vector((v[0], v[1] * s, v[2]))
        poser.leg(desired, leg, Matrix.Identity(4), mirror(tip), mirror(tdir).normalized(), mirror(pole).normalized())
    wings = driver.desired(wing_angles, {"LWing", "RWing"})
    desired["LWing"], desired["RWing"] = wings["LWing"], wings["RWing"]
    # 🔴 1차(몸 수평·바닥에 앉은 채)는 몸 기준으로 앞이 30° 내려간 스트로크 면 때문에 날개 끝이 z −0.24로 바닥을 뚫었다.
    #    정지비행 초파리는 몸을 들어 스트로크 면을 수평에 가깝게 둔다 — 몸 기준 자세를 다 만든 뒤 한 몸 변환(머리 들기·띄우기)을 곱한다.
    body = Matrix.Translation((0.0, 0.0, FLIGHT_LIFT)) @ _about(Vector((0.25, 0.0, 0.75)), Matrix.Rotation(-math.radians(FLIGHT_PITCH), 4, "Y"))
    return {name: body @ m for name, m in desired.items()}


def nmf_metadata(nmf, drivers):
    """뼈대에 남기는 NMF 규격 요약 — 뼈마다 v1·v2 관절 이름, 쉬는 자세 θ_ref(도), 맞춤 잔차. 축·순서 원문은 nmf/nmf_spec.json."""
    v1_names = {}
    for j in nmf.spec["v1"]["joints"]:
        bone = fly_nmf.V1_TO_BONE.get(j["body"])
        if bone:
            v1_names.setdefault(bone, []).append(j["name"])
    v2_names = {}
    for bone in fb.BONES:
        p = fly_nmf.PARENT[bone]
        if not p:
            continue
        axes = [a for seg, a in fly_nmf.LEG_DOFS if bone[2:] == seg] if bone[:2] in fb.LEGS else list(fly_nmf.AXES)
        v2_names[bone] = [f"{fly_nmf.BONE_MAP[p][1]}-{fly_nmf.BONE_MAP[bone][1]}-{a}" for a in axes]
    return {"flygym_commit": nmf.spec["flygym_commit"], "spec": "nmf/nmf_spec.json",
            "bone_map": {b: {"v1": v1, "v2": v2} for b, (v1, v2) in fly_nmf.BONE_MAP.items()},
            "dofs_v1": v1_names, "dofs_v2": v2_names,
            "theta_ref_deg": {v: {f"{b}:{a}": round(math.degrees(x), 3) for (b, a), x in d.theta_ref.items()} for v, d in drivers.items()},
            "fit_residual_deg": {v: d.fit_deg for v, d in drivers.items()},
            "ranges": nmf.spec["ranges"]["nmf"], "wingbeat_defaults": fly_wing.DEFAULTS}


def _key(arm, basis, frame, prev):
    for name, m in basis.items():
        pb = arm.pose.bones[name]
        loc, rot, _ = m.decompose()
        if name in prev:
            rot.make_compatible(prev[name])
        prev[name] = rot
        pb.location, pb.rotation_quaternion = loc, rot
        pb.keyframe_insert("location", frame=frame)
        pb.keyframe_insert("rotation_quaternion", frame=frame)


def _pose_at(arm, frame):
    bpy.context.scene.frame_set(frame)
    return [pb.matrix.copy() for pb in arm.pose.bones]


def _lowest(obj, frames):
    low = 1e9
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = ev.to_mesh()
        co = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", co)
        low = min(low, float(co[2::3].min()))
        ev.to_mesh_clear()
    return low


def rig_and_animate(obj, collection, name):
    arm = build_armature(obj, collection, name)
    poser = Poser(arm)
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 0, max(n for _, n in CLIPS)
    info = {"bones": len(arm.data.bones), "loop": {}, "lowest": {}, "_actions": []}
    nmf = fly_nmf.NMF()
    drivers = {v: fly_nmf.Driver(nmf, poser, arm, v, wing_frame_fn=fl.wing_frame) for v in ("v1", "v2")}
    beats, wing_err = fly_wing.beat_angles(nmf, dict(CLIPS)["Flight_Wingbeat"])
    meta = nmf_metadata(nmf, drivers)
    arm["nmf"] = json.dumps(meta, ensure_ascii=False)
    for pb in arm.pose.bones:
        pb["nmf_dofs_v1"] = ",".join(meta["dofs_v1"].get(pb.name, []))
        pb["nmf_dofs_v2"] = ",".join(meta["dofs_v2"].get(pb.name, []))
    info["nmf"] = {"fit_max_deg": max(max(d.fit_deg.values()) for d in drivers.values()), "wing_solve_max": wing_err}
    for clip, frames in CLIPS:
        action = bpy.data.actions.new(clip)
        action["gen_fly"] = True          # 🔴 창에서 옛 액션을 지울 땐 이 표시·반환값으로(이름으로 찾지 않는다)
        _set_action(arm, action)
        fn = {"Walk_Tripod": walk_pose, "Idle_Groom": groom_pose}.get(clip)
        prev = {}
        for frame in range(frames + 1):
            desired = fn(poser, frame / frames) if fn else flight_pose(poser, drivers["v2"], beats[frame])
            _key(arm, poser.basis(desired), frame, prev)
        for fc in _fcurves(action):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
        info["_actions"].append((clip, action, frames))
    for clip, action, frames in info["_actions"]:
        _set_action(arm, action)
        first, last = _pose_at(arm, 0), _pose_at(arm, frames)
        info["loop"][clip] = max(abs(x - y) for a, b in zip(first, last) for ra, rb in zip(a, b) for x, y in zip(ra, rb))
        info["lowest"][clip] = round(_lowest(obj, range(0, frames + 1, 2)), 4)
    _set_action(arm, info["_actions"][0][1])
    scene.frame_set(0)
    return arm, info
