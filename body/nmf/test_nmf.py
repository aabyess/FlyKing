"""fly_nmf 점검 — blender -b ../초파리.blend --python test_nmf.py"""
import math
import pickle
import sys
from pathlib import Path

import bpy

HERE = Path(bpy.data.filepath).parent
sys.path.insert(0, str(HERE))
import fly_body as fb      # noqa: E402
import fly_limbs as fl     # noqa: E402
import fly_nmf             # noqa: E402
import fly_rig             # noqa: E402
import fly_wing            # noqa: E402

arm = bpy.data.objects["초파리_뼈대"]
poser = fly_rig.Poser(arm)
nmf = fly_nmf.NMF()
for version in ("v1", "v2"):
    drv = fly_nmf.Driver(nmf, poser, arm, version, wing_frame_fn=fl.wing_frame)
    worst = sorted(drv.fit_deg.items(), key=lambda kv: -kv[1])[:6]
    print(version, "맞춤 잔차 최대(°)", worst)
    print(version, "θ_ref LF(°)", {f"{b}-{a}": round(math.degrees(v), 1) for (b, a), v in drv.theta_ref.items() if b.startswith("LF")})
    print(version, "θ_ref RF(°)", {f"{b}-{a}": round(math.degrees(v), 1) for (b, a), v in drv.theta_ref.items() if b.startswith("RF")})
    if version == "v2":
        print("v2 θ_ref 날개(°)", {f"{b}-{a}": round(math.degrees(v), 1) for (b, a), v in drv.theta_ref.items() if "Wing" in b})
    # 쉬는 자세 되풀이: θ_ref를 그대로 넣으면 뼈가 쉬는 행렬과 같아야 한다
    driven = {b for b, _ in drv.theta_ref}
    des = drv.desired(dict(drv.theta_ref), driven)
    err = max((des[b].to_quaternion().rotation_difference(poser.rest[b].to_quaternion()).angle for b in fb.BONES))
    perr = max(((des[b].translation - poser.rest[b].translation).length for b in fb.BONES))
    print(version, "θ_ref → 쉬는 자세 되풀이 오차 회전 %.2e rad · 위치 %.2e mm" % (err, perr))

data = pickle.load(open(Path.home() / "flybrain/flygym/src/flygym_demo/complex_terrain/assets/single_steps_untethered.pkl", "rb"))
names = [k for k in data if k.startswith("joint_")]
version, found, skipped = nmf.parse(names)
print("걷기 데이터", version, len(found), "개 옮김, 못 옮김", skipped)
drv = fly_nmf.Driver(nmf, poser, arm, version, wing_frame_fn=fl.wing_frame)
for i in (0, 11, 22, 33):
    angles = {found[n]: float(data[n][i]) for n in found}
    driven = {b for b, _ in angles.values()} if False else {found[n][0] for n in found}
    des = drv.desired(angles, driven)
    basis = poser.basis(des)
    for name, m in basis.items():
        pb = arm.pose.bones[name]
        pb.rotation_mode = "QUATERNION"
        loc, rot, _ = m.decompose()
        pb.location, pb.rotation_quaternion = loc, rot
    if arm.animation_data:
        arm.animation_data.action = None
    bpy.context.view_layer.update()
    tips = {leg: (arm.matrix_world @ arm.pose.bones[leg + "Tarsus5"].tail) for leg in fb.LEGS}
    # NMF 원형과 방향 비교: 우리 종아리 방향 vs NMF 종아리 방향(같은 각)
    world = nmf.world(version, {**angles}, None)
    ang = max(math.degrees((arm.pose.bones[leg + "Tibia"].tail - arm.pose.bones[leg + "Tibia"].head).normalized()
                           .angle(world[leg + "Tibia"] @ nmf.seg_dir(leg + "Tibia"))) for leg in fb.LEGS)
    print(f"  표본 {i}: 발끝 z", {k: round(v.z, 3) for k, v in tips.items()}, " 종아리 방향 최대 차 %.1f°" % ang)

beats, worst = fly_wing.beat_angles(nmf, 24)
print("날갯짓 24칸 각 풀이 최대 잔차", round(worst, 5))
first, last = beats[0], beats[-1]
print("  첫·끝 각(°)", {f"{b}-{a}": (round(math.degrees(first[(b, a)]), 1), round(math.degrees(last[(b, a)]), 1)) for (b, a) in first if b == "LWing"})
print("  L 날개 각 범위(°)", {a: (round(min(math.degrees(f[("LWing", a)]) for f in beats), 1), round(max(math.degrees(f[("LWing", a)]) for f in beats), 1)) for a in fly_nmf.AXES})
