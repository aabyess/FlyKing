"""다리 뼈 길이 표(6다리 × 9뼈) 우리 → NMF. blender -b 초파리.blend --python nmf/leg_table.py"""
import sys
from pathlib import Path
import bpy
from mathutils import Vector
# 기본 = body/(이 파일은 body/nmf/), 시험 복사본은 환경변수 FLYBODY로
sys.path.insert(0, str(Path(__import__("os").environ.get("FLYBODY") or Path(__file__).resolve().parent.parent)))
import fly_nmf  # noqa: E402
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o.name.startswith("초파리"))
rig = fly_nmf.NMF().v2_rig
NEXT = {"Coxa": "trochanterfemur", "Femur": "tibia", "Tibia": "tarsus1", "Tarsus1": "tarsus2", "Tarsus2": "tarsus3",
        "Tarsus3": "tarsus4", "Tarsus4": "tarsus5", "Tarsus5": None}
for leg in ("LF", "RF", "LM", "RM", "LH", "RH"):
    cells = []
    for seg, nxt in NEXT.items():
        ours = arm.data.bones[leg + seg].length
        nm = Vector(rig[f"{leg.lower()}_{nxt}"]["pos"]).length if nxt else None
        cells.append(f"{seg} {ours:.3f}→{nm:.3f}" if nm else f"{seg} {ours:.3f}→(NMF 없음)")
    print("T", leg, " | ".join(cells))
