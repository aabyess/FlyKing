"""「초파리가 스마트폰으로 릴스 보는」 장면 — 헤드리스 정본.

  /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python scene/gen_scene.py -- [--no-render] [--samples=48] [--shots=wide,shoulder,face]

산출(scene/):
  초파리_릴스.blend       장면 전체(초파리 + 스마트폰 + 거치대 + 책상 + 카메라 3대)
  스마트폰.fbx            폰만, 실제 치수(mm)·원점·로컬 축 그대로(유니티용). 화면 재질 이름 「폰_화면」
  Textures/릴스_자리.png  실제 릴스 붙이기 전 화면 자리 표시
  renders/*.png          미리보기(git 제외)

크기(2026-09-13 사장님: 「스마트폰 크기 초파리랑 1:1, 초파리 크기 키워줘」):
  - 초파리는 FLY_SCALE배(몸길이 2.5mm → 25mm). 균일 확대라 관절각·뼈대 비율은 그대로다.
  - 폰은 높이 = 확대된 초파리 몸길이 × PHONE_TO_FLY. 거치대도 같이 줄어든다.
  - 물리·뇌 쪽 실제 치수는 body/(1 = 1mm)가 정본이다. 이 장면의 크기는 보기용이다.

초파리는 body/초파리.blend의 「판_초파리」 컬렉션을 복사해 온다. body/는 수정하지 않는다.
축 +X 앞 · +Y 왼쪽 · Z 위(body/와 같다). 초파리는 +X(폰 쪽)를 본다.
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
BODY = os.environ.get("FLYKING_BODY") or os.path.join(ROOT, "body")   # 다른 체크아웃의 최신 몸을 쓰려면 FLYKING_BODY로 지정
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import phone as ph  # noqa: E402

FLY_BODY_MM = 2.5          # body/gen_fly.py가 검사하는 실제 몸길이
FLY_SCALE = 10.0
PHONE_TO_FLY = 1.0         # 폰 높이 ÷ 초파리 몸길이
TILT_DEG = 15.0            # 폰이 수직에서 뒤로 기운 각
BASE_TOP = 4.0             # 거치대 바닥판 윗면 높이(폰 실제 치수 기준, 세트 전체가 함께 줄어든다)
LIP_FRONT = -12.5          # 거치대 앞 턱 앞면 x(실제 치수 기준)
GAP_TO_PHONE = 0.5         # 초파리 머리 끝 ~ 거치대 앞 턱 거리(초파리 몸길이 배수)
RES = (1600, 1000)


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opt = dict(render="--no-render" not in a, samples=48, shots=["wide", "shoulder", "face"])
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
    return col, arm


def corners(objs):
    return [o.matrix_world @ Vector(c) for o in objs if o.type == "MESH" for c in o.bound_box]


def bbox(pts):
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def phone_matrix():
    """로컬(+X 폭, +Y 위, +Z 화면) → 세트 좌표. 화면이 -X(초파리 쪽)를 보고 뒤로 TILT_DEG 기운다."""
    a = math.radians(TILT_DEG)
    ex, ey, ez = Vector((0, -1, 0)), Vector((math.sin(a), 0, math.cos(a))), Vector((-math.cos(a), 0, math.sin(a)))
    rot = Matrix([[ex.x, ey.x, ez.x], [ex.y, ey.y, ez.y], [ex.z, ey.z, ez.z]]).to_4x4()
    loc = Vector((0, 0, BASE_TOP)) + ey * (ph.H / 2)   # 뒷면 아래 모서리가 바닥판 위 (0, 0, BASE_TOP)
    return Matrix.Translation(loc) @ rot


def mesh_object(name, bm, mat, col, parent=None):
    me = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    ob.parent = parent
    return ob


def build_stand(col, parent):
    plastic = ph.material("거치대", (0.78, 0.79, 0.80), roughness=0.55)
    bm = bmesh.new()
    ph._box(bm, (19.0, 0.0, BASE_TOP / 2), (70.0, 80.0, BASE_TOP), bevel=0.8)                 # 바닥판
    ph._box(bm, (LIP_FRONT + 2.1, 0.0, BASE_TOP + 2.6), (4.2, 76.0, 5.2), bevel=0.6)          # 앞 턱
    a = math.radians(TILT_DEG)
    top = Vector((80 * math.sin(a), 0, BASE_TOP + 80 * math.cos(a)))                           # 뒷면 받침 쐐기
    tri = [(0.4, BASE_TOP), (top.x + 0.4, top.z), (top.x + 0.4, BASE_TOP)]
    left = [bm.verts.new((x, 30.0, z)) for x, z in tri]
    right = [bm.verts.new((x, -30.0, z)) for x, z in tri]
    bm.faces.new(left)
    bm.faces.new(list(reversed(right)))
    for i in range(3):
        j = (i + 1) % 3
        bm.faces.new((left[j], left[i], right[i], right[j]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return mesh_object("거치대", bm, plastic, col, parent)


def build_table(col):
    wood = bpy.data.materials.new("책상")
    wood.use_nodes = True
    nt = wood.node_tree
    p = nt.nodes["Principled BSDF"]
    p.inputs["Roughness"].default_value = 0.6
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.05
    noise.inputs["Detail"].default_value = 6.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.20, 0.13, 0.08, 1)
    ramp.color_ramp.elements[1].color = (0.36, 0.25, 0.16, 1)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], p.inputs["Base Color"])
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=600.0)
    table = mesh_object("책상", bm, wood, col)
    table.location.z = -0.01
    return table


def new_camera(col, name, lens):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.clip_start, cd.clip_end = 0.05, 5000.0
    cam = bpy.data.objects.new(name, cd)
    col.objects.link(cam)
    return cam


def frame(cam, target, direction, points, margin=1.1):
    """target을 겨냥하고 direction 쪽에서, points가 전부 화면 안에 들어오는 가장 가까운 거리에 카메라를 둔다."""
    d = Vector(direction).normalized()
    rot = (-d).to_track_quat("-Z", "Y")
    m = rot.to_matrix()
    right, up = m.col[0], m.col[1]
    tan_h = (cam.data.sensor_width / 2) / cam.data.lens
    tan_v = tan_h * RES[1] / RES[0]
    dist = 0.0
    for p in points:
        v = Vector(p) - Vector(target)
        dist = max(dist, v.dot(d) + abs(v.dot(right)) * margin / tan_h, v.dot(d) + abs(v.dot(up)) * margin / tan_v)
    cam.location = Vector(target) + d * dist
    cam.rotation_euler = rot.to_euler()
    return dist


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
    """세트에 붙이기 전에 부른다 — 루트가 원점·실제 치수일 때 내보낸다."""
    objs = [root] + list(root.children)
    for o in bpy.context.view_layer.objects:
        o.select_set(o in objs)
    bpy.context.view_layer.objects.active = root
    bpy.ops.export_scene.fbx(filepath=path, use_selection=True, object_types={"EMPTY", "MESH"}, global_scale=1.0,
                             apply_unit_scale=False, path_mode="AUTO", mesh_smooth_type="FACE", use_mesh_modifiers=True,
                             bake_anim=False)


def main():
    opt = args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit = "MILLIMETERS"
    scene.frame_current = 1

    # 폰 세트(실제 치수로 만든 뒤 한꺼번에 줄인다)
    col = bpy.data.collections.new("판_폰")
    scene.collection.children.link(col)
    tex_dir = os.path.join(HERE, "Textures")
    os.makedirs(tex_dir, exist_ok=True)
    img = ph.placeholder_image(os.path.join(tex_dir, "릴스_자리.png"))
    root, info = ph.build_phone(col, img)
    fbx = os.path.join(HERE, "스마트폰.fbx")
    export_phone_fbx(root, fbx)

    phone_scale = FLY_BODY_MM * FLY_SCALE * PHONE_TO_FLY / ph.H
    rig = bpy.data.objects.new("폰_세트", None)
    rig.empty_display_size = 10
    col.objects.link(rig)
    root.parent = rig
    root.matrix_basis = phone_matrix()
    stand = build_stand(col, rig)
    rig.scale = (phone_scale,) * 3
    bpy.context.view_layer.update()

    screen = bpy.data.objects["폰_화면"]
    normal = (screen.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
    assert normal.x < -0.95, f"화면이 초파리를 안 본다: {normal}"
    phone_parts = list(root.children)
    lowest = min(p.z for p in corners(phone_parts))
    assert lowest > (BASE_TOP - 0.05) * phone_scale, f"폰이 거치대를 뚫는다: {lowest}"
    phone_height = bbox(corners(phone_parts))[1].z

    # 초파리: 확대 후 머리 끝이 거치대 앞 턱에서 GAP_TO_PHONE 몸길이만큼 떨어지게
    fly_col, arm = append_fly(scene)
    arm.scale = (FLY_SCALE,) * 3
    bpy.context.view_layer.update()
    fly_meshes = [o for o in fly_col.objects if o.type == "MESH"]
    lo, hi = bbox(corners(fly_meshes))
    front_target = LIP_FRONT * phone_scale - GAP_TO_PHONE * FLY_BODY_MM * FLY_SCALE
    arm.location.x += front_target - hi.x
    bpy.context.view_layer.update()
    fly_pts = corners(fly_meshes)
    flo, fhi = bbox(fly_pts)
    fly_center = (flo + fhi) / 2

    build_table(col)
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

    # 카메라 3대 — 크기를 바꿔도 자동으로 맞춰 잡는다
    set_pts = corners(phone_parts + [stand])
    everything = fly_pts + set_pts
    alo, ahi = bbox(everything)
    cams = {}
    cams["wide"] = new_camera(col, "카메라_wide", 35)
    frame(cams["wide"], (alo + ahi) / 2, (-0.8, -0.7, 0.45), everything, 1.08)

    cams["shoulder"] = new_camera(col, "카메라_shoulder", 35)
    screen_pts = corners([screen])
    s_lo, s_hi = bbox(screen_pts)
    shoulder_target = (fly_center + (s_lo + s_hi) / 2) / 2
    frame(cams["shoulder"], shoulder_target, (-1.0, -0.25, 0.30), fly_pts + screen_pts, 1.05)
    cams["shoulder"].data.dof.use_dof = True
    cams["shoulder"].data.dof.focus_distance = (fly_center - cams["shoulder"].location).length
    cams["shoulder"].data.dof.aperture_fstop = 4.0

    cams["face"] = new_camera(col, "카메라_face", 60)
    head = Vector((fhi.x - 0.2 * (fhi.x - flo.x), fly_center.y, fly_center.z))
    frame(cams["face"], head, (0.55, 1.0, 0.30), fly_pts, 1.05)

    scene.camera = cams["shoulder"]
    scene.render.engine = "CYCLES"
    device = use_gpu(scene)
    scene.cycles.samples = opt["samples"]
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = RES

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

    info.update(phone_scale=round(phone_scale, 4), phone_height_in_scene=round(phone_height, 2),
                fly_scale=FLY_SCALE, fly_size_in_scene=[round(v, 2) for v in (fhi - flo)],
                fly_front_x=round(fhi.x, 2), lip_front_x=round(LIP_FRONT * phone_scale, 2),
                screen_normal=[round(v, 3) for v in normal], device=device, blend=blend, fbx=fbx, renders=written)
    print("SCENE_DONE " + json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
