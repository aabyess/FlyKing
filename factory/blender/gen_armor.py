"""초파리 장비(갑옷) 3단계 — body/초파리.blend 위에 갑옷을 뼈에 붙여 게임용 FBX로 내보낸다.

실행(원본 .blend는 읽기만, 저장하지 않음):
  Blender -b body/초파리.blend --python factory/blender/gen_armor.py -- <출력.fbx> <미리보기_접두어>

단위: Blender 1 = 1mm(body/gen_fly.py와 같음). 머리 쪽 +X, 위 +Z, 왼쪽 +Y.
갑옷 치수는 몸 메시에서 뼈 가중치로 가슴(Thorax)·머리(Head) 상자를 재서 정한다 — 숫자를 손으로 맞추지 않는다.
  1단계 가죽 조끼: 가슴 윗면 껍데기
  2단계 쇠 판금: 가슴 판 + 테두리 + 리벳 + 투구
  3단계 황금 갑옷: 가슴 판 + 테두리 + 어깨 보호대 + 투구 + 붉은 볏
게임(FactoryGame.cs)은 오브젝트 이름 앞머리 "갑옷{단계}_"로 켜고 끈다. 게임 효과는 몸 동작 속도뿐, 뇌 판단은 바꾸지 않는다.
"""
import math
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
OUT_FBX, OUT_PNG = argv[0], argv[1]

arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
body = next(o for o in bpy.data.objects if o.type == "MESH" and "Thorax" in o.vertex_groups)
arm.data.pose_position = "REST"
bpy.context.view_layer.update()


def part_box(name):
    gi = body.vertex_groups[name].index
    pts = [body.matrix_world @ v.co for v in body.data.vertices if v.groups and max(v.groups, key=lambda g: g.weight).group == gi]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return (lo + hi) / 2, (hi - lo) / 2


TH_C, TH_H = part_box("Thorax")
HD_C, HD_H = part_box("Head")
print("MEASURE thorax", tuple(round(v, 3) for v in TH_C), tuple(round(v, 3) for v in TH_H),
      "head", tuple(round(v, 3) for v in HD_C), tuple(round(v, 3) for v in HD_H))


def material(name, color, metal, rough):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except AttributeError:
        pass
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Roughness"].default_value = rough
    m.diffuse_color = (*color, 1.0)
    m.metallic, m.roughness = metal, rough
    return m


MAT = {
    "가죽": material("갑옷_가죽", (0.30, 0.15, 0.06), 0.0, 0.8),
    "가죽_테": material("갑옷_가죽_테", (0.12, 0.06, 0.03), 0.0, 0.9),
    "쇠": material("갑옷_쇠", (0.56, 0.57, 0.60), 1.0, 0.35),
    "쇠_테": material("갑옷_쇠_테", (0.20, 0.20, 0.22), 1.0, 0.45),
    "금": material("갑옷_금", (0.85, 0.62, 0.16), 1.0, 0.25),
    "금_테": material("갑옷_금_테", (0.55, 0.33, 0.06), 1.0, 0.3),
    "볏": material("갑옷_볏", (0.62, 0.03, 0.03), 0.0, 0.6),
}


def finish(name, bm, mat, bone):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(mat)
    ob.parent = arm
    ob.parent_type = "BONE"
    ob.parent_bone = bone
    bpy.context.view_layer.update()
    ob.matrix_world = Matrix.Identity(4)   # 뼈에 붙여도 꼭짓점이 뼈대(원점) 공간 좌표 그대로 놓이게
    return ob


def shell(name, center, radii, cut_z, mat, bone, thickness=0.035):
    """타원체 윗부분(z ≥ cut_z) 껍데기. 안쪽 면이 radii에 오고 두께는 바깥으로."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=40, v_segments=24, radius=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * radii[0], v.co.y * radii[1], v.co.z * radii[2])) + Vector(center)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < cut_z], context="VERTS")
    bm.normal_update()
    bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=-thickness)
    return finish(name, bm, mat, bone)


def ring(name, center, radii, z, tube, mat, bone):
    """껍데기 가장자리(z 높이 타원) 둘레 테."""
    k = (z - center[2]) / radii[2]
    s = math.sqrt(max(1.0 - k * k, 0.0))
    rx, ry = radii[0] * s + tube, radii[1] * s + tube
    bm = bmesh.new()
    segs, sides = 48, 8
    grid = []
    for i in range(segs):
        a = math.tau * i / segs
        c = Vector((center[0] + rx * math.cos(a), center[1] + ry * math.sin(a), z))
        out = Vector((math.cos(a), math.sin(a), 0.0))
        grid.append([bm.verts.new(c + out * (tube * math.cos(math.tau * j / sides)) + Vector((0, 0, tube * math.sin(math.tau * j / sides))))
                     for j in range(sides)])
    for i in range(segs):
        for j in range(sides):
            bm.faces.new((grid[i][j], grid[(i + 1) % segs][j], grid[(i + 1) % segs][(j + 1) % sides], grid[i][(j + 1) % sides]))
    return finish(name, bm, mat, bone)


def ball(name, center, radii, mat, bone):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * radii[0], v.co.y * radii[1], v.co.z * radii[2])) + Vector(center)
    return finish(name, bm, mat, bone)


def box(name, lo, hi, mat, bone):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector(((v.co.x + 0.5) * (hi[0] - lo[0]) + lo[0], (v.co.y + 0.5) * (hi[1] - lo[1]) + lo[1], (v.co.z + 0.5) * (hi[2] - lo[2]) + lo[2]))
    return finish(name, bm, mat, bone)


def chest(tier, grow, cut_frac, mat, rim_mat, rivets=False):
    c = (TH_C.x - 0.06, 0.0, TH_C.z)
    r = (TH_H.x * 0.95 * grow, TH_H.y * grow, TH_H.z * grow)
    cut = TH_C.z + TH_H.z * cut_frac
    objs = [shell(f"갑옷{tier}_가슴", c, r, cut, mat, "Thorax")]
    objs.append(ring(f"갑옷{tier}_가슴테", c, r, cut + 0.01, 0.028 if tier > 1 else 0.02, rim_mat, "Thorax"))
    if rivets:
        k = (cut + 0.06 - c[2]) / r[2]
        s = math.sqrt(1 - k * k)
        for i in range(10):
            a = math.tau * (i + 0.5) / 10
            p = (c[0] + (r[0] * s + 0.02) * math.cos(a), c[1] + (r[1] * s + 0.02) * math.sin(a), cut + 0.06)
            objs.append(ball(f"갑옷{tier}_리벳{i}", p, (0.025, 0.025, 0.025), rim_mat, "Thorax"))
    return c, r, objs


def helmet(tier, mat, rim_mat, crest=False):
    c = (HD_C.x - 0.02, 0.0, HD_C.z)
    r = (HD_H.x * 1.12, HD_H.y * 0.98, HD_H.z * 1.1)
    cut = HD_C.z + HD_H.z * 0.45          # 윗머리만 덮어 겹눈(옆면)은 보이게
    shell(f"갑옷{tier}_투구", c, r, cut, mat, "Head", thickness=0.03)
    ring(f"갑옷{tier}_투구테", c, r, cut + 0.005, 0.022, rim_mat, "Head")
    top = c[2] + r[2]
    box(f"갑옷{tier}_코막이", (c[0] + r[0] * 0.78, -0.035, cut - 0.18), (c[0] + r[0] * 0.95, 0.035, top - 0.06), rim_mat, "Head")
    if crest:
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=1.0)
        for v in bm.verts:
            v.co = Vector((v.co.x * r[0] * 0.85, v.co.y * 0.035, v.co.z * 0.26)) + Vector((c[0] - 0.05, 0.0, top - 0.02))
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < top - 0.02], context="VERTS")
        finish(f"갑옷{tier}_볏", bm, MAT["볏"], "Head")


# 1단계 가죽 조끼
chest(1, 1.05, 0.12, MAT["가죽"], MAT["가죽_테"])
# 2단계 쇠 판금 + 투구
chest(2, 1.09, 0.0, MAT["쇠"], MAT["쇠_테"], rivets=True)
helmet(2, MAT["쇠"], MAT["쇠_테"])
# 3단계 황금 갑옷 + 어깨 + 볏 투구
c3, r3, _ = chest(3, 1.12, -0.08, MAT["금"], MAT["금_테"], rivets=True)
for side, sgn in (("L", 1), ("R", -1)):
    pc = (TH_C.x + TH_H.x * 0.45, sgn * TH_H.y * 0.98, TH_C.z + TH_H.z * 0.15)
    shell(f"갑옷3_어깨_{side}", pc, (TH_H.x * 0.33, TH_H.y * 0.3, TH_H.z * 0.38), pc[2] - TH_H.z * 0.05, MAT["금"], "Thorax", thickness=0.03)
helmet(3, MAT["금"], MAT["금_테"], crest=True)

# 병정 초파리 장비: 오른 어깨에 멘 소총(총구가 머리 앞) + 철모. 유니티가 이름 앞머리 「병정_」으로 켠다(갑옷 2단계부터는 철모 대신 갑옷 투구).
MAT["총몸"] = material("병정_총몸", (0.12, 0.12, 0.13), 0.8, 0.4)
MAT["총열"] = material("병정_총열", (0.25, 0.25, 0.27), 1.0, 0.3)
MAT["개머리"] = material("병정_개머리", (0.32, 0.20, 0.10), 0.0, 0.7)
MAT["조준경"] = material("병정_조준경", (0.04, 0.04, 0.045), 0.6, 0.3)
MAT["철모"] = material("병정_철모", (0.24, 0.28, 0.16), 0.0, 0.8)
G_TOP = TH_C.z + TH_H.z * 1.12                                  # 가슴 윗면 위(날개 뿌리 z≈1.01보다 높게)
G_Y = -TH_H.y * 0.62                                            # 오른 어깨(−Y)
G_X0 = TH_C.x - TH_H.x * 0.55                                   # 개머리 끝
G_FRONT = HD_C.x + HD_H.x                                       # 머리 앞 끝


def gun_cyl(name, x0, x1, y, z, r, key):
    bm = bmesh.new()
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=r, radius2=r, depth=x1 - x0)
    rot = Vector((1.0, 0.0, 0.0)).to_track_quat("Z", "Y").to_matrix().to_4x4()
    bmesh.ops.transform(bm, matrix=Matrix.Translation(((x0 + x1) / 2, y, z)) @ rot, verts=res["verts"])
    finish(name, bm, MAT[key], "Thorax")


box("병정_개머리", (G_X0, G_Y - 0.07, G_TOP - 0.05), (G_X0 + 0.45, G_Y + 0.07, G_TOP + 0.13), MAT["개머리"], "Thorax")
box("병정_총몸", (G_X0 + 0.45, G_Y - 0.09, G_TOP - 0.02), (G_X0 + 1.25, G_Y + 0.09, G_TOP + 0.2), MAT["총몸"], "Thorax")
box("병정_탄창", (G_X0 + 0.85, G_Y - 0.06, G_TOP - 0.3), (G_X0 + 1.0, G_Y + 0.06, G_TOP - 0.02), MAT["총몸"], "Thorax")
gun_cyl("병정_총열", G_X0 + 1.25, G_FRONT + 0.55, G_Y, G_TOP + 0.1, 0.04, "총열")
gun_cyl("병정_소염기", G_FRONT + 0.55, G_FRONT + 0.7, G_Y, G_TOP + 0.1, 0.065, "총몸")
gun_cyl("병정_조준경", G_X0 + 0.6, G_X0 + 1.05, G_Y, G_TOP + 0.3, 0.055, "조준경")
bm = bmesh.new()
bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=0.03)
bmesh.ops.translate(bm, vec=Vector((G_FRONT + 0.72, G_Y, G_TOP + 0.1)), verts=bm.verts[:])
finish("병정_총구", bm, MAT["총몸"], "Thorax")                 # 총알이 나가는 자리(유니티 FireBullet)
H_C = (HD_C.x - 0.03, 0.0, HD_C.z)
H_R = (HD_H.x * 1.18, HD_H.y * 1.04, HD_H.z * 1.14)
shell("병정_철모", H_C, H_R, HD_C.z + HD_H.z * 0.4, MAT["철모"], "Head", thickness=0.035)
ring("병정_철모테", H_C, H_R, HD_C.z + HD_H.z * 0.4 + 0.005, 0.03, MAT["철모"], "Head")

arm.data.pose_position = "POSE"
bpy.context.view_layer.update()
armor = [o for o in bpy.data.objects if o.name.startswith("갑옷")]
print("ARMOR", len(armor), sorted({o.name.split("_")[0] for o in armor}))

# 미리보기(Workbench) — 단계마다 한 장
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "MATERIAL"
scene.display.shading.show_specular_highlight = True
scene.render.resolution_x, scene.render.resolution_y = 900, 600
cam_data = bpy.data.cameras.new("미리보기_카메라")
cam_data.lens = 60
cam = bpy.data.objects.new("미리보기_카메라", cam_data)
scene.collection.objects.link(cam)
cam.location = (3.6, -4.2, 3.0)
target = Vector((0.0, 0.0, 0.7))
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = cam
for tier in (1, 2, 3):
    for o in armor:
        o.hide_render = not o.name.startswith(f"갑옷{tier}_")
    scene.render.filepath = f"{OUT_PNG}_{tier}.png"
    bpy.ops.render.render(write_still=True)
cam.location = (5.0, -1.2, 2.6)                                # 오른쪽 앞에서 — 창·방패가 함께 보이게
cam.rotation_euler = (Vector((0.4, 0.0, 0.6)) - cam.location).to_track_quat("-Z", "Y").to_euler()
cam_data.lens = 38
for o in armor:
    o.hide_render = True                                         # 병정 미리보기: 갑옷 없이 철모·소총만
scene.render.filepath = f"{OUT_PNG}_soldier.png"
bpy.ops.render.render(write_still=True)
for o in armor:
    o.hide_render = False
bpy.data.objects.remove(cam)

# 게임용 FBX(body/gen_fly.py와 같은 설정)
bpy.ops.object.select_all(action="DESELECT")
for o in [arm, *[o for o in bpy.data.objects if o.type == "MESH" and (o.parent == arm or o.name.startswith("갑옷"))]]:
    o.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.export_scene.fbx(filepath=OUT_FBX, use_selection=True, object_types={"ARMATURE", "MESH"}, global_scale=1.0,
                         apply_unit_scale=False, path_mode="RELATIVE", mesh_smooth_type="FACE", add_leaf_bones=False,
                         armature_nodetype="NULL", bake_anim=True, bake_anim_use_all_actions=True,
                         bake_anim_use_nla_strips=False, bake_anim_force_startend_keying=True, bake_anim_simplify_factor=0.0)
print("ACTIONS", [a.name for a in bpy.data.actions])
print("EXPORTED", OUT_FBX)
