"""flygym(NeuroMechFly) 관절각 시계열 → 초파리 뼈대 키프레임.
    blender -b 초파리.blend --python apply_joint_angles.py -- 데이터 [옵션]
데이터
  .csv  첫 줄 = 열 이름, 한 줄 = 한 시각
  .npz  열 이름마다 1차원 배열, 또는 joint_names(N) + angles(T×N) (+ time 등 1차원 배열)
  .pkl  dict(열 이름 → 1차원 배열) — flygym_demo 걷기 조각 꼴
열 이름
  v1(flygym 1.x): joint_LFCoxa_yaw · joint_LFCoxa(=pitch, joint_LFCoxa_pitch도 받음) · joint_LFCoxa_roll · joint_Head_yaw · joint_LPedicel …
  v2(flygym 2.x): c_thorax-lf_coxa-yaw · lf_coxa-lf_trochanterfemur-pitch · c_thorax-l_wing-roll · c_thorax-c_abdomen12-pitch …
  time(또는 t): 초. 없으면 --timestep, 그것도 없으면 한 줄 = 한 프레임
  날갯짓 발생기(선택, 이 열이 있으면 날개를 합성): wingbeat_freq(Hz) · wingbeat_amp(°, 앞뒤 폭) · wingbeat_asym(−1~1, +면 왼쪽 큼)
    · wingbeat_deviation(°) · wingbeat_rotation(°) · wingbeat_stroke_plane(°) · wingbeat_bias(°) — 빠진 건 fly_wing.DEFAULTS
  못 옮기는 열(더듬이 funiculus·arista, 주둥이, 눈 등 우리 뼈대에 없는 조각)은 이름을 찍고 건너뛴다.
옵션
  --fps 30  --slow K(시간 K배 늘림)  --axis-order yaw_pitch_roll(v2는 데이터와 반드시 맞출 것)  --degrees
  --ground clip|frame|none  발끝 접지 보정. NMF 다리는 우리보다 몸높이 대비 짧아 각을 그대로 옮기면 발이 땅 아래로 간다(실측 0.1~0.4mm).
            clip = 클립 전체 최저 발끝을 쉬는 자세 발끝 높이로 한 번 올림(기본), frame = 프레임마다, none = 안 함
  --wing-display-hz H  합성 날갯짓이 화면에서 초당 H박동을 넘으면 위상만 늦춘다(기본 fps/4 — 앨리어싱 방지). 진폭·비대칭은 데이터 그대로
  --pose ground|flight  ground = 서 있는 몸(기본). flight = fly_rig.flight_pose — 다리 접고 몸 머리 40° 들고 1.2mm 띄운 정지비행 몸(날갯짓만 있는 데이터용,
            다리 관절각 열과는 같이 못 씀). 🔴 서 있는 몸에 큰 폭(150°+) 날갯짓을 주면 앞이 30° 내려간 스트로크 면 때문에 날개 끝이 바닥을 뚫는다
            (wingbeat_cpg 실측 z −0.34mm) — Flight_Wingbeat 클립과 같은 까닭
  --name 액션이름  --save 결과.blend  --fbx 결과.fbx
원리는 fly_nmf.py 머리말(순운동학 → θ_ref로 우리 쉬는 자세에 맞춤)."""
import argparse
import csv
import math
import pickle
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fly_body as fb      # noqa: E402
import fly_limbs as fl     # noqa: E402
import fly_nmf             # noqa: E402
import fly_rig             # noqa: E402
import fly_wing            # noqa: E402

WING_COLS = {"wingbeat_freq": "freq", "wingbeat_amp": "amplitude", "wingbeat_asym": "asym", "wingbeat_deviation": "deviation",
             "wingbeat_rotation": "rotation", "wingbeat_stroke_plane": "stroke_plane", "wingbeat_bias": "stroke_bias"}


def load(path):
    p = Path(path)
    if p.suffix == ".csv":
        with open(p, newline="") as f:
            rows = [r for r in csv.reader(f) if r]
        head = [h.strip() for h in rows[0]]
        arr = np.array([[float(x) for x in r] for r in rows[1:]], dtype=float)
        return {h: arr[:, i] for i, h in enumerate(head)}
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=False)
        cols = {}
        if "joint_names" in z.files and "angles" in z.files:
            cols = {str(n): z["angles"][:, i].astype(float) for i, n in enumerate(z["joint_names"])}
        cols.update({k: z[k].astype(float) for k in z.files if k not in ("joint_names", "angles") and z[k].ndim == 1})
        return cols
    if p.suffix == ".pkl":
        d = pickle.load(open(p, "rb"))
        return {k: np.asarray(v, dtype=float) for k, v in d.items() if np.ndim(v) == 1}
    raise SystemExit(f"모르는 형식: {p.suffix}")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--timestep", type=float)
    ap.add_argument("--slow", type=float, default=1.0)
    ap.add_argument("--axis-order")
    ap.add_argument("--degrees", action="store_true")
    ap.add_argument("--ground", choices=("clip", "frame", "none"), default="clip")
    ap.add_argument("--wing-display-hz", type=float)
    ap.add_argument("--pose", choices=("ground", "flight"), default="ground")
    ap.add_argument("--name")
    ap.add_argument("--save")
    ap.add_argument("--fbx")
    args = ap.parse_args(argv)

    cols = load(args.data)
    lengths = {len(v) for v in cols.values()}
    assert len(lengths) == 1, f"열 길이가 다르다: {lengths}"
    rows = lengths.pop()
    nmf = fly_nmf.NMF()
    version, found, skipped = nmf.parse(list(cols))
    wing_cols = {k: v for k, v in WING_COLS.items() if k in cols}
    if version is None and not wing_cols:
        raise SystemExit("관절 열도 날갯짓 열도 없다")
    version = version or "v2"
    order = tuple(args.axis_order.split("_")) if args.axis_order else None
    if version == "v2" and not order and found:                    # 날갯짓 발생기 열만 있으면 축 순서는 쓰이지 않는다
        print("⚠️ v2 데이터인데 --axis-order가 없다 — yaw_pitch_roll로 가정(데이터를 만든 skeleton.axis_order와 맞출 것)")

    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o.name.startswith("초파리"))
    poser = fly_rig.Poser(arm)
    driver = fly_nmf.Driver(nmf, poser, arm, version, order, wing_frame_fn=fl.wing_frame)
    scale = math.pi / 180.0 if args.degrees else 1.0

    times = cols.get("time", cols.get("t"))
    if times is None:
        times = np.arange(rows) * (args.timestep if args.timestep else 1.0 / args.fps)
    out_len = float(times[-1] - times[0]) * args.slow
    frames = max(1, int(round(out_len * args.fps)))
    t_out = times[0] + np.arange(frames + 1) / (args.fps * args.slow)
    interp = {k: np.interp(t_out, times, v) for k, v in cols.items()}

    driven = {bone for bone, _ in found.values()}
    if wing_cols:
        if any(bone.endswith("Wing") for bone in driven):
            print("⚠️ 날개 관절각 열과 날갯짓 발생기 열이 함께 있다 — 관절각 열을 쓰고 발생기는 무시")
            wing_cols = {}
        else:
            driven |= {"LWing", "RWing"}
    if args.pose == "flight" and any(b[:2] in fb.LEGS for b in driven):
        raise SystemExit("--pose flight는 다리를 접은 자세로 덮어쓴다 — 다리 관절각 열과 같이 못 쓴다")
    cap = args.wing_display_hz or args.fps / 4.0
    phase, warm, slowed, desired_all = 0.0, {"L": [0.0, 0.0, 0.0], "R": [0.0, 0.0, 0.0]}, False, []
    side_amp_max = 0.0
    for j in range(frames + 1):
        angles = {found[n]: float(interp[n][j]) * scale for n in found}
        if wing_cols:
            p = dict(fly_wing.DEFAULTS, **{WING_COLS[k]: float(interp[k][j]) for k in wing_cols})
            side_amp_max = max(side_amp_max, p["amplitude"] * (1.0 + 0.5 * abs(p.get("asym", 0.0))))
            if j:
                hz = p["freq"] / args.slow
                if hz > cap:
                    hz, slowed = cap, True
                phase += hz / args.fps
            for side in "LR":
                span, lead = fly_wing.wing_frame(side, phase, fly_wing.side_params(p, side))
                a, warm[side], _ = fly_wing.solve_angles(nmf, side, span, lead, warm[side], driver.order)
                angles.update(a)
        desired_all.append(fly_rig.flight_pose(poser, driver, angles) if args.pose == "flight" else driver.desired(angles, driven))

    rest_tip = min(arm.data.bones[leg + "Tarsus5"].tail_local.z for leg in fb.LEGS)

    def tip_min(des):
        return min((des[leg + "Tarsus5"] @ poser.rest[leg + "Tarsus5"].inverted() @ arm.data.bones[leg + "Tarsus5"].tail_local.to_4d()).z
                   for leg in fb.LEGS)

    legs_driven = any(b[:2] in fb.LEGS for b in driven)
    offsets = [0.0] * len(desired_all)
    if legs_driven and args.ground != "none":
        mins = [tip_min(d) for d in desired_all]
        offsets = [rest_tip - min(mins)] * len(mins) if args.ground == "clip" else [rest_tip - m for m in mins]
    name = args.name or Path(args.data).stem
    action = bpy.data.actions.new(name)
    action["apply_joint_angles"] = True
    fly_rig._set_action(arm, action)
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    prev = {}
    for j, des in enumerate(desired_all):
        if offsets[j]:
            for m in des.values():
                m.translation.z += offsets[j]
        fly_rig._key(arm, poser.basis(des), j, prev)
    for fc in fly_rig._fcurves(action):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    scene = bpy.context.scene
    scene.render.fps = int(round(args.fps))
    scene.frame_start, scene.frame_end = 0, frames
    print(f"적용  {name}  버전 {version}  축 순서 {driver.order or 'MJCF/기본'}  열 {len(found)}개 → 뼈 {len(driven)}개  "
          f"자세 {args.pose}  프레임 0~{frames} ({args.fps:g}fps, 시간 ×{args.slow:g})  θ_ref 잔차 최대 {max(driver.fit_deg.values()):.2f}°  "
          f"접지 보정 {args.ground} {min(offsets):+.3f}~{max(offsets):+.3f}mm")
    if skipped:
        print("  못 옮긴 열:", skipped)
    if wing_cols:
        if side_amp_max > fly_wing.SAFE_AMPLITUDE:
            print(f"  ⚠️ 한쪽 날개 폭 최대 {side_amp_max:.1f}° > 안전 상한 {fly_wing.SAFE_AMPLITUDE:g}° — 날개 뿌리가 가슴 벽을 조금 스칠 수 있다(fly_wing.SAFE_AMPLITUDE 주석)")
        print(f"  날갯짓 합성 열 {sorted(wing_cols)}" + (f"  ⚠️ 화면 박동수를 {cap:g}Hz로 늦춤(앨리어싱 방지)" if slowed else ""))
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.save).resolve()), relative_remap=True)
        print("  저장", args.save)
    if args.fbx:
        obj = next(o for o in arm.children if o.type == "MESH")
        for o in bpy.context.view_layer.objects:
            o.select_set(o in (obj, arm))
        bpy.context.view_layer.objects.active = arm
        bpy.ops.export_scene.fbx(filepath=str(Path(args.fbx).resolve()), use_selection=True, object_types={"ARMATURE", "MESH"},
                                 global_scale=1.0, apply_unit_scale=False, path_mode="RELATIVE", add_leaf_bones=False,
                                 armature_nodetype="NULL", bake_anim=True, bake_anim_use_all_actions=True,
                                 bake_anim_use_nla_strips=False, bake_anim_force_startend_keying=True, bake_anim_simplify_factor=0.0)
        print("  FBX", args.fbx)
    return action


if __name__ == "__main__":
    main()
