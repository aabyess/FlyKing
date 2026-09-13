"""헤드리스 정본 확인 — 초파리.blend(구운 텍스처)를 열어 3/4 한 장 렌더.
    blender -b 초파리.blend --python check_baked.py"""
import math
import os

import bpy
from mathutils import Vector

HERE = os.path.dirname(bpy.data.filepath)
scene = bpy.context.scene
obj = bpy.data.objects["초파리"]
for slot in obj.material_slots:
    nodes = slot.material.node_tree.nodes
    kinds = sorted(n.type for n in nodes)
    img = next((n.image for n in nodes if n.type == "TEX_IMAGE"), None)
    print("재질", slot.material.name, kinds, img.filepath if img else None, img.size[:] if img else None)
world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.58, 0.62, 1)
scene.world = world
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.lens, cam.data.clip_start = 85, 0.01
scene.collection.objects.link(cam)
target = Vector((0.0, 0.0, 0.6))
cam.location = target + Vector((0.9, 0.8, 0.6)).normalized() * 13.0
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.2
sun.rotation_euler = (math.radians(40), 0, math.radians(-35))
scene.collection.objects.link(sun)
scene.camera = cam
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = 1200, 900
scene.view_settings.view_transform = "Standard"
scene.render.filepath = os.path.join(HERE, "renders", "fly_baked_check.png")
bpy.ops.render.render(write_still=True)
print("렌더", scene.render.filepath)
