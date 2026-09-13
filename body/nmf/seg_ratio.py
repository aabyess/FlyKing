"""다리 마디 길이 우리 대 NeuroMechFly v2 rigging 비율(왼쪽 3다리). blender -b 초파리.blend --python nmf/seg_ratio.py"""
import sys
from pathlib import Path
import bpy
from mathutils import Vector
# 기본 = body/(이 파일은 body/nmf/), 시험 복사본은 환경변수 FLYBODY로
sys.path.insert(0, str(Path(__import__("os").environ.get("FLYBODY") or Path(__file__).resolve().parent.parent)))
import fly_nmf  # noqa: E402
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o.name.startswith("초파리"))
rig = fly_nmf.NMF().v2_rig
NEXT = [("Coxa", "trochanterfemur"), ("Femur", "tibia"), ("Tibia", "tarsus1"), ("Tarsus1", "tarsus2"), ("Tarsus2", "tarsus3"),
        ("Tarsus3", "tarsus4"), ("Tarsus4", "tarsus5")]
for leg in ("LF", "LM", "LH"):
    row, tot_o, tot_n = [], 0.0, 0.0
    for seg, nxt in NEXT:
        o = arm.data.bones[leg + seg].length
        n = Vector(rig[f"{leg.lower()}_{nxt}"]["pos"]).length
        tot_o, tot_n = tot_o + o, tot_n + n
        row.append(f"{seg} {o:.3f}/{n:.3f}={o / n:.2f}")
    coxa = arm.data.bones[leg + "Coxa"].head_local
    print("R", leg, " · ".join(row), f"| 밑마디~발목마디4 합 {tot_o:.3f}/{tot_n:.3f}={tot_o / tot_n:.2f}", "| 밑마디 머리 z", round(coxa.z, 3))
