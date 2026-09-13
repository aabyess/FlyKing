"""공장 습격 괴물 — 좀비 개미(작고 떼로) · 거미 괴물(크고 드묾). 유니티가 다리를 흔들어 걷게 한다.

실행: Blender -b --factory-startup --python factory/blender/gen_monsters.py -- <Models 폴더> <미리보기.png>
단위·축은 gen_room.py와 같다(1 = 초파리 ×10 장면 단위, 초파리 가로 약 42.6). 머리 쪽 +X, 위 +Z.
  monster_zombie_ant  몸길이 약 60 — 초파리보다 조금 크다. 초파리 다가가기 뉴런이 켜지는 「작게 움직이는 것」 역할
  monster_spider      다리 폭 약 170 — 초파리 네 배. 다가오면 시야를 덮는 「루밍」 역할(거대섬유 도주)
오브젝트: 몸통(뿌리) · 머리 · 다리_L1…(원점이 엉덩이 관절 — 유니티가 이 점을 축으로 흔든다).
재질 이름은 「몬스터_…」 — 유니티 FactoryGame.DressMaterials가 이름으로 색·발광을 입힌다.
"""
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
MODELS, PREVIEW = argv
bpy.ops.wm.read_factory_settings(use_empty=True)
os.makedirs(MODELS, exist_ok=True)


def material(name, color, rough=0.7, metal=0.0, emit=None):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except AttributeError:
        pass
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    m.diffuse_color = (*color, 1.0)
    if emit is not None:
        bsdf.inputs["Emission Color"].default_value = (*emit, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 6.0
    return m


MAT = {
    "좀비피부": material("몬스터_좀비피부", (0.22, 0.28, 0.19), 0.85),
    "썩은살": material("몬스터_썩은살", (0.36, 0.15, 0.16), 0.6),
    "좀비다리": material("몬스터_좀비다리", (0.12, 0.14, 0.10), 0.8),
    "초록눈": material("몬스터_초록눈", (0.45, 1.0, 0.25), 0.3, emit=(0.4, 1.0, 0.2)),
    "거미털": material("몬스터_거미털", (0.06, 0.055, 0.055), 0.95),
    "거미무늬": material("몬스터_거미무늬", (0.55, 0.03, 0.02), 0.5),
    "빨간눈": material("몬스터_빨간눈", (1.0, 0.1, 0.05), 0.2, emit=(1.0, 0.08, 0.03)),
    "송곳니": material("몬스터_송곳니", (0.75, 0.72, 0.62), 0.35),
}


def ellipsoid(bm, c, r, segs=20, rings=12):
    res = bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=rings, radius=1.0)
    for v in res["verts"]:
        v.co = Vector((v.co.x * r[0], v.co.y * r[1], v.co.z * r[2])) + Vector(c)
    return res["verts"]


def limb(bm, a, b, r0, r1, sides=8):
    """a→b 가늘어지는 원기둥(마디 하나)."""
    a, b = Vector(a), Vector(b)
    axis = b - a
    rot = axis.to_track_quat("Z", "Y").to_matrix().to_4x4()
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=sides, radius1=r0, radius2=r1, depth=axis.length)
    bmesh.ops.transform(bm, matrix=Matrix.Translation((a + b) / 2) @ rot, verts=res["verts"])
    return res["verts"]


def paint(bm, verts, idx):
    for f in {f for v in verts for f in v.link_faces}:
        f.material_index = idx
        f.smooth = True


def make_object(name, parts, origin=(0, 0, 0), parent=None):
    """parts: [(생성함수, 재질키)] — 꼭짓점은 장면 좌표로 만들고 원점을 origin으로 옮긴다."""
    bm = bmesh.new()
    keys = []
    for build, key in parts:
        if key not in keys:
            keys.append(key)
        paint(bm, build(bm), keys.index(key))
    bmesh.ops.translate(bm, vec=-Vector(origin), verts=bm.verts[:])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for k in keys:
        me.materials.append(MAT[k])
    ob = bpy.data.objects.new(name, me)
    ob.location = origin
    bpy.context.scene.collection.objects.link(ob)
    if parent is not None:
        ob.parent = parent
        ob.matrix_parent_inverse = parent.matrix_world.inverted()
    return ob


def zombie_ant(prefix):
    z = 9.0
    body = make_object(f"{prefix}", [
        (lambda bm: ellipsoid(bm, (6, 0, z), (9, 5.5, 5.5)), "좀비피부"),               # 가슴
        (lambda bm: ellipsoid(bm, (-3, 0, z - 0.5), (2.5, 2, 2)), "좀비다리"),           # 허리 마디
        (lambda bm: ellipsoid(bm, (-16, 0, z + 2), (13, 9.5, 8.5)), "좀비피부"),         # 배
        (lambda bm: ellipsoid(bm, (-19, 4.5, z + 7), (4, 3, 2.2)), "썩은살"),            # 썩어 드러난 살
        (lambda bm: ellipsoid(bm, (-10, -6, z + 5), (3, 2.5, 2)), "썩은살"),
    ])
    make_object(f"{prefix}_머리", [
        (lambda bm: ellipsoid(bm, (20, 0, z + 2), (6.5, 6, 5.5)), "좀비피부"),
        (lambda bm: ellipsoid(bm, (23, 4.2, z + 4), (1.8, 1.6, 1.6)), "초록눈"),
        (lambda bm: ellipsoid(bm, (23, -4.2, z + 4), (1.8, 1.6, 1.6)), "초록눈"),
        (lambda bm: limb(bm, (25, 2.5, z), (31, 1, z - 2), 1.1, 0.3), "좀비다리"),       # 큰턱
        (lambda bm: limb(bm, (25, -2.5, z), (31, -1, z - 2), 1.1, 0.3), "좀비다리"),
        (lambda bm: limb(bm, (22, 3, z + 7), (30, 9, z + 16), 0.6, 0.3), "좀비다리"),    # 더듬이(한쪽은 부러짐)
        (lambda bm: limb(bm, (22, -3, z + 7), (25, -6, z + 11), 0.6, 0.4), "좀비다리"),
    ], origin=(14, 0, z), parent=body)
    for i, x in enumerate((11, 6, 1)):
        for side, s in (("L", 1), ("R", -1)):
            hip = (x, s * 5, z - 1)
            knee = (x + (4 - i * 4), s * 17, z + 6)
            foot = (x + (6 - i * 7), s * 25, 0.3)
            make_object(f"{prefix}_다리_{side}{i + 1}", [
                (lambda bm, h=hip, k=knee: limb(bm, h, k, 1.3, 0.9), "좀비다리"),
                (lambda bm, k=knee, f=foot: limb(bm, k, f, 0.9, 0.35), "좀비다리"),
            ], origin=hip, parent=body)
    return body


def spider(prefix):
    z = 22.0
    body = make_object(f"{prefix}", [
        (lambda bm: ellipsoid(bm, (6, 0, z), (15, 13, 9)), "거미털"),                    # 머리가슴
        (lambda bm: ellipsoid(bm, (-26, 0, z + 8), (24, 21, 18)), "거미털"),             # 배
        (lambda bm: ellipsoid(bm, (-26, 0, z + 25.5), (5, 3, 1.2)), "거미무늬"),         # 붉은 모래시계 무늬
        (lambda bm: ellipsoid(bm, (-33, 0, z + 23), (4, 3.5, 1.5)), "거미무늬"),
    ])
    eyes = [(19, 3.5, z + 6, 2.2), (19, -3.5, z + 6, 2.2), (17, 7, z + 7, 1.5), (17, -7, z + 7, 1.5),
            (15, 2.5, z + 8.5, 1.3), (15, -2.5, z + 8.5, 1.3), (13, 6, z + 8.5, 1.1), (13, -6, z + 8.5, 1.1)]
    make_object(f"{prefix}_머리", [
        (lambda bm: ellipsoid(bm, (18, 0, z + 1), (6, 8, 6)), "거미털"),
        *[(lambda bm, e=e: ellipsoid(bm, e[:3], (e[3], e[3], e[3])), "빨간눈") for e in eyes],
        (lambda bm: limb(bm, (23, 3, z - 2), (27, 2.5, z - 11), 1.8, 0.2), "송곳니"),
        (lambda bm: limb(bm, (23, -3, z - 2), (27, -2.5, z - 11), 1.8, 0.2), "송곳니"),
    ], origin=(18, 0, z), parent=body)
    for i, (x, fwd) in enumerate(((14, 1.0), (8, 0.35), (1, -0.35), (-5, -1.0))):
        for side, s in (("L", 1), ("R", -1)):
            hip = (x, s * 10, z - 1)
            knee = (x + fwd * 22, s * 42, z + 30)
            ankle = (x + fwd * 38, s * 70, z + 8)
            foot = (x + fwd * 44, s * 84, 0.5)
            make_object(f"{prefix}_다리_{side}{i + 1}", [
                (lambda bm, h=hip, k=knee: limb(bm, h, k, 3.2, 2.4), "거미털"),
                (lambda bm, k=knee, a=ankle: limb(bm, k, a, 2.4, 1.5), "거미털"),
                (lambda bm, a=ankle, f=foot: limb(bm, a, f, 1.5, 0.4), "거미털"),
            ], origin=hip, parent=body)
    return body


def export(root, path):
    bpy.ops.object.select_all(action="DESELECT")
    root.select_set(True)
    for c in root.children_recursive:
        c.select_set(True)
    bpy.context.view_layer.objects.active = root
    bpy.ops.export_scene.fbx(filepath=path, use_selection=True, object_types={"MESH"}, axis_forward="-Z", axis_up="Y",
                             bake_space_transform=True, apply_unit_scale=False, global_scale=1.0, mesh_smooth_type="FACE",
                             path_mode="STRIP", bake_anim=False, add_leaf_bones=False)
    dims = [o.dimensions for o in [root, *root.children_recursive]]
    print("EXPORTED", os.path.basename(path), "parts", 1 + len(root.children_recursive))


ant = zombie_ant("좀비개미")
spi = spider("거미괴물")
export(ant, os.path.join(MODELS, "monster_zombie_ant.fbx"))
export(spi, os.path.join(MODELS, "monster_spider.fbx"))

# 미리보기: 초파리 크기 비교용 막대(가로 42.6) · 개미 · 거미
ant.location = (-40, -70, 0)
spi.location = (40, 60, 0)
bm = bmesh.new()
bmesh.ops.create_cube(bm, size=1.0)
bmesh.ops.scale(bm, vec=(42.6, 2, 2), verts=bm.verts[:])
bmesh.ops.translate(bm, vec=(-40, -130, 1), verts=bm.verts[:])
me = bpy.data.meshes.new("초파리_크기")
bm.to_mesh(me)
bm.free()
bpy.context.scene.collection.objects.link(bpy.data.objects.new("초파리_크기", me))
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "MATERIAL"
scene.render.resolution_x, scene.render.resolution_y = 1000, 700
cam = bpy.data.objects.new("미리보기", bpy.data.cameras.new("미리보기"))
scene.collection.objects.link(cam)
cam.location = (230, -260, 190)
cam.rotation_euler = (Vector((0, -10, 20)) - cam.location).to_track_quat("-Z", "Y").to_euler()
cam.data.lens = 40
scene.camera = cam
scene.render.filepath = PREVIEW
bpy.ops.render.render(write_still=True)
print("PREVIEW", PREVIEW)
