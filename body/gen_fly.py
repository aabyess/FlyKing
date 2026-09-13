"""초파리(노랑초파리 암컷 성체) 몸 — 헤드리스 정본. 게임 에셋 아님(FlyWire 커넥톰·뇌 시뮬레이션에 붙일 몸).
    blender -b --factory-startup --python gen_fly.py -- [--pose=rest|spread] [--no-bake]
산출: 초파리.fbx(메시 + 뼈대 「초파리_뼈대」 + 클립 Walk_Tripod·Idle_Groom·Flight_Wingbeat) · 초파리.blend · Textures/*.png (이 폴더).
NeuroMechFly 관절각 재생: fly_nmf.py(규격 nmf/nmf_spec.json) · apply_joint_angles.py · 날갯짓 fly_wing.py.
클립은 30fps. Walk_Tripod는 실제 한 주기 약 0.1초를 1초로 늘린 것 — 실제 속도는 재생 10배속(fly_rig.py 머리말).
단위: Blender 1 = 1mm(.blend 장면 단위 mm). FBX는 숫자 그대로 mm 값을 쓴다(apply_unit_scale 끔 — 다시 읽으면 몸길이 2.5).
축: +X 앞(머리) · +Y 왼쪽 · Z 위 — NeuroMechFly와 같다. 원점 = 가슴 아래 바닥(발끝이 z≈0)."""
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fly_body as fb      # noqa: E402
import fly_limbs as fl     # noqa: E402
import fly_rig as fr       # noqa: E402
import fly_tex as ft       # noqa: E402
from fly_mesh import FlyMesh  # noqa: E402

NAME = "초파리"
TEX = os.path.join(HERE, "Textures")
TRI_LIMIT = 200000


def build_mesh(pose):
    mesh = FlyMesh(fb.BONES)
    facets = fb.build_head(mesh)
    fb.build_thorax(mesh)
    fb.build_abdomen(mesh)
    for side in "LR":
        fb.build_haltere(mesh, side)
        fl.build_wing(mesh, side, pose)
        for key in "FMH":
            fl.build_leg(mesh, side, key)
    return mesh, facets


def _select_only(obj):
    for o in bpy.context.view_layer.objects:
        o.select_set(o == obj)
    bpy.context.view_layer.objects.active = obj


def assemble(collection, pose="rest", folder=TEX, name=NAME, reuse_images=False):
    mesh, facets = build_mesh(pose)
    bm = mesh.bm
    bm.normal_update()
    tris = sum(len(f.verts) - 2 for f in bm.faces)
    assert tris <= TRI_LIMIT, f"삼각형 {tris} > {TRI_LIMIT}"
    zs = [v.co.z for v in bm.verts]
    body_len = (fb.HEAD[-1][0] + 0.015) - (fb.abd_axis(1.0)[0] - 0.02)
    assert 2.4 <= body_len <= 2.6, f"몸길이 {body_len:.3f}mm (약 2.5)"
    assert 2.1 <= fl.WING_LENGTH <= 2.3
    assert set(mesh.joints) == set(fb.BONES), f"관절 빠짐: {set(fb.BONES) ^ set(mesh.joints)}"
    data = bpy.data.meshes.new(name)
    bm.to_mesh(data)
    bm.free()
    data.shade_smooth()
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    bone_of = [0] * len(data.vertices)
    data.attributes["뼈"].data.foreach_get("value", bone_of)
    for index, bone in enumerate(fb.BONES):
        group = obj.vertex_groups.new(name=bone)
        members = [i for i, b in enumerate(bone_of) if b == index]
        if members:
            group.add(members, 1.0, "REPLACE")
    for mat_name in mesh.mats:
        if mat_name in ft.DRAWN:
            img = bpy.data.images.get(mat_name) if reuse_images else None
            mat = ft.image_material(mat_name, img or ft.save_png(mat_name, ft.DRAWN[mat_name](), folder))
        elif reuse_images and bpy.data.images.get(mat_name):
            mat = ft.image_material(mat_name, bpy.data.images[mat_name])
        else:
            mat = ft.bake_material(mat_name)
        data.materials.append(mat)
    obj["joints"] = json.dumps({k: [list(v[0]), list(v[1]), v[2]] for k, v in mesh.joints.items()})
    lo = [min(v.co[k] for v in data.vertices) for k in range(3)]
    hi = [max(v.co[k] for v in data.vertices) for k in range(3)]
    info = dict(tris=tris, body_length=round(body_len, 3), wing_length=fl.WING_LENGTH, facets=facets,
                size=[round(h - l, 3) for l, h in zip(lo, hi)], lowest=round(min(zs), 4), mats=list(mesh.mats))
    return obj, info


def unwrap(obj):
    _select_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    baked = {i for i, slot in enumerate(obj.material_slots) if slot.material.name in ft.BAKED}
    for f in bm.faces:
        f.select = f.material_index in baked
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.002)
    bpy.ops.object.mode_set(mode="OBJECT")


def bake(obj, folder=TEX):
    os.makedirs(folder, exist_ok=True)
    scene = bpy.context.scene
    saved = (scene.render.engine, scene.cycles.samples)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    dummy = bpy.data.images.new("_버림", 8, 8)
    temp, images = [], {}
    try:
        _select_only(obj)
        for slot in obj.material_slots:
            m = slot.material
            nt = m.node_tree
            node = nt.nodes.new("ShaderNodeTexImage")
            if m.name in ft.BAKED:
                old = bpy.data.images.get(m.name)
                if old is not None:
                    bpy.data.images.remove(old)
                size = ft.BAKED[m.name]
                node.image = images[m.name] = bpy.data.images.new(m.name, size, size)
            else:
                node.image = dummy
                temp.append((nt, node))
            for n in nt.nodes:
                n.select = False
            node.select = True
            nt.nodes.active = node
        bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, use_clear=True, margin=6)
        for nt, node in temp:
            nt.nodes.remove(node)
        for name, img in images.items():
            img.filepath_raw = os.path.join(folder, name + ".png")
            img.file_format = "PNG"
            img.save()
            ft.image_material(name, img)
    finally:
        bpy.data.images.remove(dummy)
        scene.render.engine, scene.cycles.samples = saved


def build_in_window(collection, pose="rest", folder=TEX, name=NAME, do_bake=True, rig=False):
    obj, info = assemble(collection, pose, folder, name, reuse_images=not do_bake)
    unwrap(obj)
    if do_bake:
        bake(obj, folder)
    arm = None
    if rig:
        arm, rig_info = fr.rig_and_animate(obj, collection, name + "_뼈대")
        info.update(rig_info)
    return obj, arm, info


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    pose = next((a.split("=", 1)[1] for a in args if a.startswith("--pose=")), "rest")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit = "MILLIMETERS"
    col = bpy.data.collections.new("판_초파리")
    scene.collection.children.link(col)
    obj, info = assemble(col, pose)
    unwrap(obj)
    if "--no-bake" not in args:
        bake(obj)
    arm, rig_info = fr.rig_and_animate(obj, col, NAME + "_뼈대")
    info.update({k: v for k, v in rig_info.items() if not k.startswith("_")})
    assert all(gap < 1e-4 for gap in rig_info["loop"].values()), f"첫 ≠ 끝: {rig_info['loop']}"
    assert all(low > -0.03 for low in rig_info["lowest"].values()), f"발끝이 땅 아래: {rig_info['lowest']}"
    assert rig_info["nmf"]["fit_max_deg"] < 1.0 and rig_info["nmf"]["wing_solve_max"] < 1e-3, f"NMF 맞춤 잔차: {rig_info['nmf']}"
    actions = sorted(a.name for a in bpy.data.actions)
    assert actions == ["Flight_Wingbeat", "Idle_Groom", "Walk_Tripod"], f"액션이 셋이 아니다: {actions}"
    for act in bpy.data.actions:
        act.use_fake_user = True    # 🔴 사용자 0인 액션은 .blend 저장 때 빠진다 — 11:14판 .blend엔 활성 Walk_Tripod 하나만 남았다(FBX엔 셋)
    for o in bpy.context.view_layer.objects:
        o.select_set(o in (obj, arm))
    bpy.context.view_layer.objects.active = arm
    fbx = os.path.join(HERE, NAME + ".fbx")
    # 🔴 use_all_actions=True여야 테이크 이름이 「초파리_뼈대|Walk_Tripod」 꼴(False면 장면 이름 테이크 하나)
    bpy.ops.export_scene.fbx(filepath=fbx, use_selection=True, object_types={"ARMATURE", "MESH"}, global_scale=1.0,
                             apply_unit_scale=False, path_mode="RELATIVE", mesh_smooth_type="FACE", add_leaf_bones=False,
                             armature_nodetype="NULL", bake_anim=True, bake_anim_use_all_actions=True,
                             bake_anim_use_nla_strips=False, bake_anim_force_startend_keying=True, bake_anim_simplify_factor=0.0)
    blend = os.path.join(HERE, NAME + ".blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend, relative_remap=True)
    print(f"만듦  {NAME}  자세 {pose}  {info}  → {fbx} · {blend}")


if __name__ == "__main__":
    main()
