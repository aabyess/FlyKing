"""장면(초파리 + 폰 세트)을 웹 뷰어용 glTF 바이너리로 내보낸다.

  /Applications/Blender.app/Contents/MacOS/Blender -b scene/초파리_릴스.blend --python scene/export_glb.py

산출: live/web/assets/scene.glb
- 책상·카메라·조명은 빼고(웹에서 새로 만든다) 초파리 뼈대+메시, 폰_세트(폰·거치대)만 담는다.
- glTF는 Y가 위다. Blender (x, y, z) → three.js (x, z, -y). 초파리는 그대로 +X를 본다.
"""
import os

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "live", "web", "assets", "scene.glb")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    arm = bpy.data.objects["초파리_뼈대"]
    fly = bpy.data.objects["초파리"]
    mods = [m.type for m in fly.modifiers]
    assert "ARMATURE" in mods, f"초파리 메시에 아마추어 모디파이어가 없다: {mods}"
    rig = bpy.data.objects["폰_세트"]
    keep = {arm, fly, rig} | set(rig.children_recursive)
    for o in bpy.context.view_layer.objects:
        o.select_set(o in keep)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", use_selection=True, export_yup=True,
                              export_apply=False, export_skins=True, export_animations=False,
                              export_cameras=False, export_lights=False, export_materials="EXPORT")
    print("GLB_DONE", OUT, os.path.getsize(OUT), "objects", sorted(o.name for o in keep), "bones", len(arm.data.bones))


if __name__ == "__main__":
    main()
