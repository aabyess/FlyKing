"""「초파리가 스마트폰으로 릴스 보는」 장면 — 헤드리스 정본.

  /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python scene/gen_scene.py -- [--no-render] [--samples=48] [--shots=wide,shoulder,face]

산출(scene/):
  초파리_릴스.blend       장면 전체(초파리 + 스마트폰 + 거치대 + 책상 + 카메라 3대)
  스마트폰.fbx            폰만, 원점·로컬 축 그대로(유니티용). 화면 재질 이름 「폰_화면」
  Textures/릴스_자리.png  실제 릴스 붙이기 전 화면 자리 표시
  renders/*.png          미리보기(git 제외)

초파리는 body/초파리.blend의 「판_초파리」 컬렉션을 복사해 온다. body/는 수정하지 않는다.
단위 1 = 1mm, 축 +X 앞 · +Y 왼쪽 · Z 위(body/와 같다). 초파리는 +X(폰 쪽)를 본다.
"""
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BODY = os.path.join(ROOT, "body")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import phone as ph  # noqa: E402

FLY_POS = Vector((-100.0, 0.0, 0.0))   # 폰 앞 10cm. 더 가까우면 어깨 너머 샷에 화면이 다 안 들어온다
TILT_DEG = 15.0            # 폰이 수직에서 뒤로 기운 각
BASE_TOP = 4.0             # 거치대 바닥판 윗면 높이
_SHOULDER = Vector((-112.0, -3.0, 0.9))
_PITCH = math.radians(22.0)                                           # 화면 아래 끝~위 끝 가운데를 올려다봄
_AIM = Vector((122.0, 3.0, 0.0)).normalized()
SHOTS = {
    # 이름: (카메라 위치, 겨냥점, 렌즈 mm, 초점 거리 기준점 또는 None, 조리개)
    "wide": ((-260.0, -170.0, 70.0), (-40.0, 0.0, 70.0), 30, None, None),
    "shoulder": (tuple(_SHOULDER), tuple(_SHOULDER + Vector((_AIM.x * math.cos(_PITCH), _AIM.y * math.cos(_PITCH), math.sin(_PITCH))) * 50),
                 22, (-100.0, 0.0, 0.7), 2.8),
    "face": ((-88.0, 4.5, 2.0), (-98.8, 0.0, 0.8), 60, (-98.8, 0.0, 0.8), 2.8),
}


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opt = dict(render="--no-render" not in a, samples=48, shots=list(SHOTS))
    for s in a:
        if s.startswith("--samples="):
            opt["samples"] = int(s.split("=", 1)[1])
        if s.startswith("--shots="):
            opt["shots"] = s.split("=", 1)[1].split(",")
    return opt


def append_fly(scene):
    path = os.path.join(BODY, "초파리.blend")
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.collections = ["판_초파리"]
    col = dst.collections[0]
    scene.collection.children.link(col)
    for img in bpy.data.images:              # //Textures/... 는 body 기준이라 절대 경로로 고친다
        if img.filepath.startswith("//"):
            img.filepath = os.path.join(BODY, img.filepath[2:])
            img.reload()
    arm = next(o for o in col.objects if o.type == "ARMATURE")
    arm.location = FLY_POS
    return col, arm


def phone_matrix():
    """로컬(+X 폭, +Y 위, +Z 화면) → 세계. 화면이 초파리(-X 쪽)를 보고 뒤로 TILT_DEG 기운다."""
    a = math.radians(TILT_DEG)
    ex, ey, ez = Vector((0, -1, 0)), Vector((math.sin(a), 0, math.cos(a))), Vector((-math.cos(a), 0, math.sin(a)))
    rot = Matrix([[ex.x, ey.x, ez.x], [ex.y, ey.y, ez.y], [ex.z, ey.z, ez.z]]).to_4x4()
    # 뒷면 아래 모서리(로컬 0, -H/2, 0)가 바닥판 위 (0, 0, BASE_TOP)에 닿게
    loc = Vector((0, 0, BASE_TOP)) + ey * (ph.H / 2)
    return Matrix.Translation(loc) @ rot


def mesh_object(name, bm, mat, col):
    me = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    return ob


def build_stand_and_table(col):
    plastic = ph.material("거치대", (0.78, 0.79, 0.80), roughness=0.55)
    wood = bpy.data.materials.new("책상")
    wood.use_nodes = True
    nt = wood.node_tree
    p = nt.nodes["Principled BSDF"]
    p.inputs["Roughness"].default_value = 0.6
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.02
    noise.inputs["Detail"].default_value = 6.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.20, 0.13, 0.08, 1)
    ramp.color_ramp.elements[1].color = (0.36, 0.25, 0.16, 1)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], p.inputs["Base Color"])

    bm = bmesh.new()
    ph._box(bm, (19.0, 0.0, BASE_TOP / 2), (70.0, 80.0, BASE_TOP), bevel=0.8)       # 바닥판
    ph._box(bm, (-10.4, 0.0, BASE_TOP + 2.6), (4.2, 76.0, 5.2), bevel=0.6)          # 앞 턱(폰 미끄럼 막이)
    a = math.radians(TILT_DEG)
    top = Vector((80 * math.sin(a), 0, BASE_TOP + 80 * math.cos(a)))                 # 폰 뒷면을 받치는 쐐기
    tri = [(0.4, BASE_TOP), (top.x + 0.4, top.z), (top.x + 0.4, BASE_TOP)]
    left = [bm.verts.new((x, 30.0, z)) for x, z in tri]
    right = [bm.verts.new((x, -30.0, z)) for x, z in tri]
    bm.faces.new(left)
    bm.faces.new(list(reversed(right)))
    for i in range(3):
        j = (i + 1) % 3
        bm.faces.new((left[j], left[i], right[i], right[j]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    stand = mesh_object("거치대", bm, plastic, col)

    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=600.0)
    table = mesh_object("책상", bm, wood, col)
    table.location.z = -0.01
    return stand, table


def camera(col, name, loc, target, lens, focus_pt, fstop):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.clip_start, cd.clip_end = 0.05, 5000.0
    cam = bpy.data.objects.new(name, cd)
    col.objects.link(cam)
    cam.location = Vector(loc)
    cam.rotation_euler = (Vector(target) - cam.location).to_track_quat("-Z", "Y").to_euler()
    if focus_pt:
        cd.dof.use_dof = True
        cd.dof.focus_distance = (Vector(focus_pt) - cam.location).length
        cd.dof.aperture_fstop = fstop
    return cam


def use_gpu(scene):
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        scene.cycles.device = "GPU"
        return "METAL"
    except Exception as e:  # noqa: BLE001
        scene.cycles.device = "CPU"
        return f"CPU ({e})"


def export_phone_fbx(root, path):
    saved = root.matrix_world.copy()
    root.matrix_world = Matrix.Identity(4)
    bpy.context.view_layer.update()
    objs = [root] + list(root.children)
    for o in bpy.context.view_layer.objects:
        o.select_set(o in objs)
    bpy.context.view_layer.objects.active = root
    bpy.ops.export_scene.fbx(filepath=path, use_selection=True, object_types={"EMPTY", "MESH"}, global_scale=1.0,
                             apply_unit_scale=False, path_mode="AUTO", mesh_smooth_type="FACE", use_mesh_modifiers=True,
                             bake_anim=False)
    root.matrix_world = saved
    bpy.context.view_layer.update()


def main():
    opt = args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit = "MILLIMETERS"
    scene.frame_current = 1

    fly_col, arm = append_fly(scene)
    col = bpy.data.collections.new("판_폰")
    scene.collection.children.link(col)

    tex_dir = os.path.join(HERE, "Textures")
    os.makedirs(tex_dir, exist_ok=True)
    img = ph.placeholder_image(os.path.join(tex_dir, "릴스_자리.png"))
    root, info = ph.build_phone(col, img)

    fbx = os.path.join(HERE, "스마트폰.fbx")
    export_phone_fbx(root, fbx)
    root.matrix_world = phone_matrix()
    bpy.context.view_layer.update()

    screen = bpy.data.objects["폰_화면"]
    normal = (screen.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
    assert normal.x < -0.95, f"화면이 초파리를 안 본다: {normal}"
    lowest = min((o.matrix_world @ Vector(c)).z for o in root.children for c in o.bound_box)
    assert lowest > BASE_TOP - 0.05, f"폰이 거치대를 뚫는다: {lowest}"

    build_stand_and_table(col)

    world = bpy.data.worlds.new("어두운_방")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.020, 0.022, 0.028, 1)
    bg.inputs[1].default_value = 1.0
    scene.world = world
    sun_d = bpy.data.lights.new("창빛", "SUN")
    sun_d.energy = 1.2
    sun_d.color = (1.0, 0.92, 0.82)
    sun_d.angle = math.radians(12)
    sun = bpy.data.objects.new("창빛", sun_d)
    sun.rotation_euler = (math.radians(55), 0, math.radians(-150))
    col.objects.link(sun)

    cams = {n: camera(col, "카메라_" + n, *SHOTS[n]) for n in SHOTS}
    scene.camera = cams["shoulder"]

    scene.render.engine = "CYCLES"
    device = use_gpu(scene)
    scene.cycles.samples = opt["samples"]
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 1600, 1000

    blend = os.path.join(HERE, "초파리_릴스.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend, relative_remap=True)

    written = []
    if opt["render"]:
        out = os.path.join(HERE, "renders")
        os.makedirs(out, exist_ok=True)
        for n in opt["shots"]:
            scene.camera = cams[n]
            scene.render.filepath = os.path.join(out, f"{n}.png")
            bpy.ops.render.render(write_still=True)
            written.append(scene.render.filepath)
        scene.camera = cams["shoulder"]

    info.update(screen_normal=[round(v, 3) for v in normal], phone_lowest=round(lowest, 3), device=device,
                fly=arm.name, blend=blend, fbx=fbx, renders=written)
    print("SCENE_DONE " + json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
