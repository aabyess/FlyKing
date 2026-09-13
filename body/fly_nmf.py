"""초파리 뼈대 ↔ NeuroMechFly(flygym) 관절각.
규격은 nmf/nmf_spec.json — nmf/extract_nmf_spec.py가 flygym 원본에서 줄 번호째 뽑은 것(여기 숫자를 손으로 적지 않는다).
  · v1 이름 joint_LFCoxa_yaw / joint_LFCoxa(=pitch) / joint_LFCoxa_roll … : 축·순서·몸 쿼터니언 = legacy/flygym1_seqikpy_yawpitchroll.xml
    (flygym v1.2.1 neuromechfly_seqik_kinorder_ypr.xml 사본). 한 몸의 joint는 적힌 순서(yaw→pitch→roll)로 곱한다.
  · v2 이름 c_thorax-lf_coxa-yaw … : 축 글자→벡터 anatomy.py(yaw=x, pitch=y, roll=z), 오른쪽 yaw·roll 부호 반전 base_fly.py,
    몸 쿼터니언 rigging.yaml, DoF 순서 = 데이터의 axis_order(기본 yaw_pitch_roll).
순운동학(MuJoCo): 몸 세계 회전 = 부모 세계 회전 · 몸 쿼터니언 · R(축1, θ1) · R(축2, θ2) · …
우리 뼈로 옮기기: 우리 뼈 회전 = N(θ) · N(θ_ref)⁻¹ · 우리 쉬는 뼈 회전.
  θ_ref = 우리 쉬는 자세에 해당하는 관절각 — 다리는 마디 방향, 날개는 끝·앞 가장자리 방향을 맞춰 LM으로 푼다(버전·축 순서마다).
  데이터에 없는 이음(모든 DoF가 빠진 뼈)은 부모를 강체로 따라간다. 있는 이음의 빠진 축은 0(NMF 모델에 그 DoF가 없다는 뜻).
NMF에는 관절 가동범위가 없다(v1 MJCF range 0개, v2 add_joints도 안 줌) — 검사용 범위는 spec['ranges'] 참고(걷기 관측값)."""
import json
import math
import re
from pathlib import Path

import numpy as np
from mathutils import Matrix, Quaternion, Vector

import fly_body as fb

SPEC_PATH = Path(__file__).with_name("nmf") / "nmf_spec.json"
AXES = ("yaw", "pitch", "roll")
ORDER_DEFAULT = ("yaw", "pitch", "roll")


def _leg_map():
    out = {}
    for leg in fb.LEGS:
        pairs = [("Coxa", "coxa"), ("Femur", "trochanterfemur"), ("Tibia", "tibia")] + [(f"Tarsus{k}", f"tarsus{k}") for k in range(1, 6)]
        for ours, new in pairs:
            out[leg + ours] = (leg + ours, f"{leg.lower()}_{new}")
    return out


# 우리 뼈 → (v1 몸 이름, v2 조각 이름). 도래마디는 NMF에서도 넓적다리와 한 몸(trochanterfemur).
BONE_MAP = {"Thorax": ("Thorax", "c_thorax"), "Head": ("Head", "c_head"),
            "LAntenna": ("LPedicel", "l_pedicel"), "RAntenna": ("RPedicel", "r_pedicel"),
            "LWing": ("LWing", "l_wing"), "RWing": ("RWing", "r_wing"),
            "LHaltere": ("LHaltere", "l_haltere"), "RHaltere": ("RHaltere", "r_haltere"),
            "A1A2": ("A1A2", "c_abdomen12"), "A3": ("A3", "c_abdomen3"), "A4": ("A4", "c_abdomen4"),
            "A5": ("A5", "c_abdomen5"), "A6": ("A6", "c_abdomen6"), **_leg_map()}
V1_TO_BONE = {v1: b for b, (v1, _) in BONE_MAP.items()}
V2_TO_BONE = {v2: b for b, (_, v2) in BONE_MAP.items()}
LEG_DOFS = [("Coxa", "yaw"), ("Coxa", "pitch"), ("Coxa", "roll"), ("Femur", "pitch"), ("Femur", "roll"), ("Tibia", "pitch")] + \
           [(f"Tarsus{k}", "pitch") for k in range(1, 6)]          # all_biological 다리 DoF(anatomy.py _get_all_biological_joints)
NEXT_SEG = {"Coxa": "trochanterfemur", "Femur": "tibia", "Tibia": "tarsus1", "Tarsus1": "tarsus2", "Tarsus2": "tarsus3",
            "Tarsus3": "tarsus4", "Tarsus4": "tarsus5", "Tarsus5": "tarsus5"}


def _parent_map():
    parent = {"Thorax": None, "Head": "Thorax", "A1A2": "Thorax", "A3": "A1A2", "A4": "A3", "A5": "A4", "A6": "A5"}
    for s in "LR":
        parent.update({s + "Antenna": "Head", s + "Wing": "Thorax", s + "Haltere": "Thorax"})
    for leg in fb.LEGS:
        chain = ["Coxa", "Femur", "Tibia"] + [f"Tarsus{k}" for k in range(1, 6)]
        prev = "Thorax"
        for seg in chain:
            parent[leg + seg] = prev
            prev = leg + seg
    return parent


PARENT = _parent_map()


def _lm(fun, x0, iters=80):
    x = np.array(x0, dtype=float)
    r = fun(x)
    cost, lam = float(r @ r), 1e-3
    for _ in range(iters):
        jac = np.empty((len(r), len(x)))
        for i in range(len(x)):
            dx = np.zeros_like(x)
            dx[i] = 1e-6
            jac[:, i] = (fun(x + dx) - r) / 1e-6
        a, g = jac.T @ jac, jac.T @ r
        while True:
            step = np.linalg.solve(a + lam * np.diag(np.diag(a) + 1e-9), -g)
            rn = fun(x + step)
            if float(rn @ rn) < cost:
                x, r, cost, lam = x + step, rn, float(rn @ rn), max(lam * 0.3, 1e-10)
                break
            lam *= 10.0
            if lam > 1e9:
                return x, cost
        if np.linalg.norm(step) < 1e-10:
            break
    return x, cost


class NMF:
    def __init__(self, spec_path=SPEC_PATH):
        self.spec = json.loads(Path(spec_path).read_text())
        v1 = self.spec["v1"]
        self.v1_axis = {(j["body"], j["axis_name"]): Vector(j["axis"]) for j in v1["joints"]}
        self.v1_order = {b: tuple(o) for b, o in v1["order_by_body"].items()}
        self.v2_rig = self.spec["v2"]["rigging"]
        self.v2_vec = {k: Vector(v["vector"]) for k, v in self.spec["v2"]["conventions"]["axis_vectors"].items()}
        for bone, parent in PARENT.items():                           # 우리 뼈대 부모 = NMF v1 몸 부모인지 확인
            if parent:
                assert v1["bodies"][BONE_MAP[bone][0]]["parent"] == BONE_MAP[parent][0], bone

    # ── 이름
    def parse(self, names):
        """열 이름들 → (버전, {열 이름: (뼈, 축)}, [못 옮긴 이름]). v1·v2가 섞이면 오류."""
        found, skipped, versions = {}, [], set()
        for name in names:
            m1 = re.match(r"^joint_([A-Za-z0-9]+?)(?:_(yaw|pitch|roll))?$", name)
            m2 = re.match(r"^([a-z0-9_]+)-([a-z0-9_]+)-(yaw|pitch|roll)$", name)
            if m1:
                versions.add("v1")
                bone = V1_TO_BONE.get(m1.group(1))
                axis = m1.group(2) or "pitch"
            elif m2:
                versions.add("v2")
                bone = V2_TO_BONE.get(m2.group(2))
                axis = m2.group(3)
                if bone and PARENT[bone] and BONE_MAP[PARENT[bone]][1] != m2.group(1):
                    bone = None
            else:
                continue
            if bone is None or bone == "Thorax":
                skipped.append(name)
            else:
                found[name] = (bone, axis)
        if len(versions) > 1:
            raise ValueError("v1 이름(joint_…)과 v2 이름(부모-자식-축)이 섞였다")
        return (versions.pop() if versions else None), found, skipped

    # ── 순운동학
    def bone_version(self, version, bone):
        """v1 MJCF에 관절이 하나도 없는 뼈(날개·평균곤·배)는 v2 축·몸 쿼터니언으로 다룬다(날갯짓 합성용)."""
        if version == "v1" and not any((BONE_MAP[bone][0], a) in self.v1_axis for a in AXES):
            return "v2"
        return version

    def axis_vec(self, version, bone, axis):
        version = self.bone_version(version, bone)
        if version == "v1":
            return self.v1_axis.get((BONE_MAP[bone][0], axis))
        vec = self.v2_vec[axis].copy()
        if BONE_MAP[bone][1][0] == "r" and axis != "pitch":
            vec.negate()
        return vec

    def body_quat(self, version, bone):
        version = self.bone_version(version, bone)
        if version == "v1":
            return Quaternion(self.spec["v1"]["bodies"][BONE_MAP[bone][0]]["quat"])
        return Quaternion(self.v2_rig.get(BONE_MAP[bone][1], {}).get("quat", [1, 0, 0, 0]))

    def order(self, version, bone, axis_order=None):
        if axis_order:
            return tuple(axis_order)
        if self.bone_version(version, bone) == "v1":
            return self.v1_order.get(BONE_MAP[bone][0], ORDER_DEFAULT)
        return ORDER_DEFAULT

    def local(self, version, bone, angles, axis_order=None):
        q = self.body_quat(version, bone)
        for ax in self.order(version, bone, axis_order):
            th = angles.get((bone, ax), 0.0)
            vec = self.axis_vec(version, bone, ax)
            if th and vec is not None:
                q = q @ Quaternion(vec, th)
        return q

    def world(self, version, angles, axis_order=None, driven=None, ref=None):
        """뼈마다 NMF 세계 회전. driven이 주어지면 그 밖의 뼈는 ref(θ_ref 자세)에서 부모를 강체로 따라간다."""
        out = {}
        for bone in fb.BONES:
            p = PARENT[bone]
            pw = out[p] if p else Quaternion()
            if driven is None or bone in driven or ref is None:
                out[bone] = pw @ self.local(version, bone, angles, axis_order)
            else:
                out[bone] = pw @ (ref[p].inverted() @ ref[bone] if p else ref[bone])
        return out

    # ── 쉬는 자세 맞춤
    def seg_dir(self, bone):
        leg, seg = bone[:2], bone[2:]
        return Vector(self.v2_rig[f"{leg.lower()}_{NEXT_SEG[seg]}"]["pos"]).normalized()

    def wing_axes(self, side):
        m = self.spec["v2"]["mesh_axes"]["l_wing"]
        span, lead = Vector(m["span"]), Vector(m["leading"])
        if side == "R":                                               # 왼쪽 메시 y 배율 −1(mirror_rule)
            span.y, lead.y = -span.y, -lead.y
        return span.normalized(), lead.normalized()

    def neutral_guess(self, version, bone, axis):
        seg = BONE_MAP[bone][1]
        key = f"{BONE_MAP[PARENT[bone]][1]}-{seg}-{axis}"
        deg = self.spec["neutral_pose_v2_deg"].get(key, {}).get("deg", -5.0 if "tarsus" in seg and axis == "pitch" else 0.0)
        if self.bone_version(version, bone) == "v1" and seg[0] == "r" and axis != "pitch":
            deg = -deg
        return math.radians(deg)

    def to_version(self, theta_v2, version):
        """v2 각 → 같은 몸자세의 다른 버전 각(v1은 오른쪽 yaw·roll 축이 반대가 아니므로 부호만 바꾼다)."""
        return {(b, a): (-v if self.bone_version(version, b) == "v1" and BONE_MAP[b][1][0] == "r" and a != "pitch" else v)
                for (b, a), v in theta_v2.items()}

    def calibrate(self, version, rest_dirs, wing_frames, axis_order=None, guess=None):
        """θ_ref. rest_dirs: 뼈 → 우리 쉬는 마디 방향(뼈대 공간), wing_frames: 'L'/'R' → (끝 방향, 앞 가장자리 방향)."""
        theta, report = {}, {}
        for leg in fb.LEGS:
            keys = [(leg + seg, ax) for seg, ax in LEG_DOFS if self.axis_vec(version, leg + seg, ax) is not None]
            x0 = [guess[k] if guess and k in guess else self.neutral_guess(version, *k) for k in keys]
            bones = [leg + s for s in ("Coxa", "Femur", "Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5")]

            def residual(x, keys=keys, bones=bones, x0=np.array(x0)):
                angles = dict(zip(keys, x))
                w, q = {}, Quaternion()
                res = []
                for b in bones:
                    q = q @ self.local(version, b, angles, axis_order)
                    d = q @ self.seg_dir(b)
                    res.extend(d - rest_dirs[b])
                res.extend(1e-3 * (x - x0))
                return np.array(res)

            x, _ = _lm(residual, x0)
            theta.update(dict(zip(keys, x)))
            q = Quaternion()
            for b in bones:
                q = q @ self.local(version, b, theta, axis_order)
                report[b] = round(math.degrees((q @ self.seg_dir(b)).angle(rest_dirs[b])), 2)
        if True:                                                       # 날개는 버전과 무관하게 v2 축(bone_version)
            for side in "LR":
                bone = side + "Wing"
                span_l, lead_l = self.wing_axes(side)
                span_t, lead_t = wing_frames[side]
                keys = [(bone, a) for a in AXES]

                def residual(x, bone=bone, keys=keys):
                    q = self.local(version, bone, dict(zip(keys, x)), axis_order)
                    return np.array([*(q @ span_l - span_t), *(q @ lead_l - lead_t)])

                x, _ = _lm(residual, [guess.get(k, 0.0) if guess else 0.0 for k in keys])
                theta.update(dict(zip(keys, x)))
                q = self.local(version, bone, theta, axis_order)
                report[bone] = round(max(math.degrees((q @ span_l).angle(span_t)), math.degrees((q @ lead_l).angle(lead_t))), 2)
        return theta, report


def rest_targets(arm, wing_frame_fn):
    bones = arm.data.bones
    dirs = {b.name: (b.tail_local - b.head_local).normalized() for b in bones}
    frames = {}
    for side in "LR":
        _, u, v = wing_frame_fn(side, "rest")
        frames[side] = (u.normalized(), v.normalized())
    return dirs, frames


class Driver:
    """버전·축 순서마다 θ_ref를 한 번 풀어 두고, 관절각 한 벌 → Poser.basis에 넣을 desired 행렬."""

    def __init__(self, nmf, poser, arm, version, axis_order=None, wing_frame_fn=None):
        self.nmf, self.poser, self.version, self.order = nmf, poser, version, axis_order
        dirs, frames = rest_targets(arm, wing_frame_fn)
        guess = None
        if version != "v2" or axis_order:
            # 🔴 1차: v1을 중립 자세에서 바로 풀었더니 오른다리가 뒤집힌 가지(밑마디 roll −212°·넓적다리 roll 185°)로 갔다 —
            #    방향은 맞아도 넓적다리 비틀림이 180° 틀린다. v2(기본 순서)로 먼저 풀고 부호만 바꿔 출발점으로 쓴다.
            base, _ = nmf.calibrate("v2", dirs, frames, None)
            guess = nmf.to_version(base, version)
        self.theta_ref, self.fit_deg = nmf.calibrate(version, dirs, frames, axis_order, guess)
        self.ref = nmf.world(version, self.theta_ref, axis_order)
        self.rest_q = {b: poser.rest[b].to_quaternion() for b in fb.BONES}

    def desired(self, angles, driven, root_offset=None):
        full = dict(self.theta_ref)
        for key in list(full):
            if key[0] in driven:
                full.pop(key)                                         # 움직이는 뼈는 데이터 각만(빠진 축 0)
        full.update(angles)
        world = self.nmf.world(self.version, full, self.order, driven=driven, ref=self.ref)
        out, pose = {}, {}
        for bone in fb.BONES:
            q = world[bone] @ self.ref[bone].inverted() @ self.rest_q[bone]
            p = PARENT[bone]
            rest = self.poser.rest[bone]
            pos = (pose[p] @ self.poser.rest[p].inverted() @ rest).translation if p else rest.translation + (root_offset or Vector())
            m = q.to_matrix().to_4x4()
            m.translation = pos
            pose[bone] = out[bone] = m
        return out
