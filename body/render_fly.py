"""창 검수 렌더(초파리 전용) — 판_초파리만 남기고 임시 카메라·태양·보조광·밝은 배경으로 찍은 뒤 전부 되돌린다."""
import math

import bpy
from mathutils import Vector

import os  # noqa: E402
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders") + os.sep   # exec할 땐 ns에 __file__을 넣어 준다
# 이름: (카메라 방향(물체에서 카메라 쪽), 거리 배율, 겨냥점 오프셋(mm), 렌즈)
SHOTS = {
    "dorsal": (Vector((-0.05, 0.0, 1.0)), 1.15, Vector((0, 0, 0)), 85),
    "lateral": (Vector((0.0, 1.0, 0.12)), 1.15, Vector((0, 0, 0)), 85),
    "front": (Vector((1.0, 0.25, 0.3)), 0.6, Vector((0.9, 0, 0.6)), 85),
    "three_q": (Vector((0.9, 0.8, 0.6)), 1.1, Vector((0, 0, 0)), 85),
    "eye": (Vector((0.45, 1.0, 0.25)), 0.22, Vector((0.95, 0.35, 0.8)), 85),
    "wing": (Vector((-0.02, 0.0, 1.0)), 0.75, Vector((-0.9, 0.45, 0.0)), 85),
}


def render(names, prefix, shots=("dorsal", "lateral", "front"), collection="판_초파리", res=(1600, 1200)):
    scene = bpy.context.scene
    col = bpy.data.collections[collection]
    bpy.context.view_layer.update()
    targets = [bpy.data.objects[n] for n in names]
    pts = [o.matrix_world @ Vector(c) for o in targets for c in o.bound_box]
    lo = Vector((min(q.x for q in pts), min(q.y for q in pts), min(q.z for q in pts)))
    hi = Vector((max(q.x for q in pts), max(q.y for q in pts), max(q.z for q in pts)))
    center = (lo + hi) / 2
    origin = targets[0].matrix_world.translation
    size = max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
    saved = dict(engine=scene.render.engine, rx=scene.render.resolution_x, ry=scene.render.resolution_y,
                 pct=scene.render.resolution_percentage, path=scene.render.filepath, cam=scene.camera,
                 vt=scene.view_settings.view_transform, look=scene.view_settings.look)
    hidden = {c.name: c.hide_render for c in scene.collection.children_recursive}
    for c in scene.collection.children_recursive:
        c.hide_render = c.name != collection
    root_objs = {o.name: o.hide_render for o in scene.collection.objects}
    for o in scene.collection.objects:
        o.hide_render = True
    others = {o.name: o.hide_render for o in col.objects if o.type == "MESH" and o not in targets}
    for n in others:
        bpy.data.objects[n].hide_render = True
    bg = scene.world.node_tree.nodes.get("Background") if scene.world and scene.world.use_nodes else None
    saved_bg = (tuple(bg.inputs[0].default_value), bg.inputs[1].default_value) if bg else None
    if bg:
        bg.inputs[0].default_value = (0.55, 0.58, 0.62, 1.0)
        bg.inputs[1].default_value = 0.9
    temp = []
    cam_data = bpy.data.cameras.new("_검수카메라")
    cam_data.clip_start, cam_data.clip_end = 0.01, 200.0
    cam = bpy.data.objects.new("_검수카메라", cam_data)
    col.objects.link(cam)
    temp.append((cam, cam_data, bpy.data.cameras))
    for label, energy, rot in (("_검수태양", 3.2, (40, 0, -35)), ("_검수보조", 1.1, (120, 0, 150))):
        ld = bpy.data.lights.new(label, "SUN")
        ld.energy = energy
        ld.angle = math.radians(8)
        lo_ = bpy.data.objects.new(label, ld)
        lo_.rotation_euler = tuple(math.radians(a) for a in rot)
        col.objects.link(lo_)
        temp.append((lo_, ld, bpy.data.lights))
    for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            scene.render.engine = eng
            break
        except TypeError:
            pass
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = res[0], res[1], 100
    scene.view_settings.view_transform = "Standard"
    scene.camera = cam
    written = []
    try:
        for shot in shots:
            direction, k, offset, lens = SHOTS[shot]
            cam_data.lens = lens
            target = (origin + offset) if offset.length else center
            cam.location = target + direction.normalized() * size * 3.2 * k
            cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = OUT + f"{prefix}_{shot}.png"
            bpy.ops.render.render(write_still=True)
            written.append(scene.render.filepath)
    finally:
        for n, v in others.items():
            bpy.data.objects[n].hide_render = v
        scene.render.engine = saved["engine"]
        scene.render.resolution_x, scene.render.resolution_y = saved["rx"], saved["ry"]
        scene.render.resolution_percentage = saved["pct"]
        scene.render.filepath = saved["path"]
        scene.camera = saved["cam"]
        scene.view_settings.view_transform = saved["vt"]
        scene.view_settings.look = saved["look"]
        for c in scene.collection.children_recursive:
            if c.name in hidden:
                c.hide_render = hidden[c.name]
        for o in scene.collection.objects:
            if o.name in root_objs:
                o.hide_render = root_objs[o.name]
        if bg and saved_bg:
            bg.inputs[0].default_value, bg.inputs[1].default_value = saved_bg
        for obj, data, store in temp:
            bpy.data.objects.remove(obj, do_unlink=True)
            store.remove(data)
    return written
