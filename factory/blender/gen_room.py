"""공장 방 — 어둡고 회색인 공장 분위기: 콘크리트 바닥 · 강철 벽(경고 띠) · H형 강철 기둥 · 매다는 공장 전등.

실행: Blender -b --factory-startup --python factory/blender/gen_room.py -- <Models 폴더> <Textures 폴더> <미리보기.png>
단위·축은 ~/flybrain/factory_assets/gen_factory.py와 같다(1 = 초파리 ×10 장면 단위, FBX axis_forward -Z / up Y).
  floor_concrete  100×100 타일, 윗면 높이 0 — 가장자리 줄눈이 이어 깔면 콘크리트 이음매가 된다
  wall_steel      폭 100 · 높이 60 강철 골판 벽, 아래 경고 띠, 위 철골 보
  pillar_h        H형 강철 기둥 높이 150, 받침판 20×20, 아래 경고 띠
  lamp_hanging    지름 32 전등갓 + 전구 + 줄(위로 200) — 원점은 전등갓 아래
텍스처는 PNG로 저장하고 유니티(FactoryGame.DressMaterials)가 재질 이름으로 입힌다.
"""
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
MODELS, TEX, PREVIEW = argv
bpy.ops.wm.read_factory_settings(use_empty=True)
os.makedirs(MODELS, exist_ok=True)
os.makedirs(TEX, exist_ok=True)


# ── 텍스처(이어 붙여도 끊기지 않는 값 잡음) ───────────────────────────────
def noise(size, rng, cells):
    g = rng.random((cells, cells))
    x = np.arange(size) * cells / size
    i0 = x.astype(int) % cells
    i1 = (i0 + 1) % cells
    f = x - np.floor(x)
    f = f * f * (3 - 2 * f)
    top = g[np.ix_(i0, i0)] * (1 - f[None, :]) + g[np.ix_(i0, i1)] * f[None, :]
    bot = g[np.ix_(i1, i0)] * (1 - f[None, :]) + g[np.ix_(i1, i1)] * f[None, :]
    return top * (1 - f[:, None]) + bot * f[:, None]


def fbm(size, rng, cells_list):
    out, total = np.zeros((size, size)), 0.0
    for k, c in enumerate(cells_list):
        w = 1.0 / (k + 1)
        out += noise(size, rng, c) * w
        total += w
    return out / total


def tex_concrete(size=1024):
    rng = np.random.default_rng(31)
    n = fbm(size, rng, (4, 16, 64, 256))
    stain = fbm(size, rng, (2, 4, 8))
    g = 0.27 + 0.07 * (n - 0.5) - 0.06 * np.clip(stain - 0.55, 0, 1) * 3
    speck = rng.random((size, size))
    g = np.where(speck > 0.997, g + 0.08, g)
    g = np.where(speck < 0.004, g - 0.07, g)
    j = 5                                                             # 줄눈(타일 가장자리)
    g[:j, :] *= 0.45
    g[-j:, :] *= 0.45
    g[:, :j] *= 0.45
    g[:, -j:] *= 0.45
    return np.dstack([g * 0.97, g * 0.99, g * 1.03])


def tex_steelwall(size=512):
    rng = np.random.default_rng(32)
    x = np.arange(size) / size
    ribs = 0.5 + 0.5 * np.cos(x * 10 * math.tau)                        # 골판 10줄
    n = fbm(size, rng, (4, 32, 128))
    y = np.linspace(1, 0, size)[:, None]                                  # 아래로 갈수록 때
    g = 0.20 + 0.05 * ribs[None, :] + 0.05 * (n - 0.5) - 0.05 * np.clip(0.35 - y, 0, 1) * 2.5
    streak = fbm(size, rng, (8, 64))
    g -= 0.03 * np.clip(streak - 0.6, 0, 1) * 4
    return np.dstack([g * 0.96, g * 0.99, g * 1.04])


def tex_hazard(w=512, h=128):
    rng = np.random.default_rng(33)
    yy, xx = np.mgrid[0:h, 0:w]
    stripe = ((xx + yy) // 48) % 2 == 0
    yellow, black = np.array([0.80, 0.60, 0.06]), np.array([0.07, 0.07, 0.07])
    out = np.where(stripe[..., None], yellow, black)
    wear = fbm(w, rng, (8, 64, 256))[:h, :w]
    out = out * (0.8 + 0.3 * wear[..., None])
    worn = wear > 0.72
    out[worn] = out[worn] * 0.5 + np.array([0.18, 0.18, 0.18]) * 0.5
    return out


def save_png(name, rgb):
    path = os.path.join(TEX, f"{name}.png")
    rgb = np.clip(rgb, 0, 1)
    h, w = rgb.shape[:2]
    img = bpy.data.images.new(name, w, h)
    img.pixels.foreach_set(np.dstack([rgb[::-1], np.ones((h, w))]).astype(np.float32).ravel())
    img.filepath_raw, img.file_format = path, "PNG"
    img.save()
    return img


IMG = {"concrete": save_png("factory_concrete", tex_concrete()),
       "steelwall": save_png("factory_steelwall", tex_steelwall()),
       "hazard": save_png("factory_hazard", tex_hazard())}


# ── 재질 ────────────────────────────────────────────────────────────────
def material(name, color=None, image=None, rough=0.7, metal=0.0, emit=None):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except AttributeError:
        pass
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if image is not None:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = image
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        m.diffuse_color = (*color, 1.0)
    if emit is not None:
        bsdf.inputs["Emission Color"].default_value = (*emit, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 8.0
    return m


MAT = {
    "콘크리트": material("공장_콘크리트", image=IMG["concrete"], rough=0.9),
    "강철벽": material("공장_강철벽", image=IMG["steelwall"], rough=0.6, metal=0.6),
    "경고띠": material("공장_경고띠", image=IMG["hazard"], rough=0.7),
    "철골": material("공장_철골", color=(0.10, 0.105, 0.11), rough=0.55, metal=0.7),
    "전등갓": material("공장_전등갓", color=(0.12, 0.15, 0.14), rough=0.4, metal=0.5),
    "전구": material("공장_전구", color=(1.0, 0.86, 0.6), rough=0.2, emit=(1.0, 0.82, 0.55)),
}


# ── 도형(UV는 면 방향으로 평면 투영, uv_size 단위마다 텍스처 한 번) ──────────
class Part:
    def __init__(self, name):
        self.name = name
        self.bm = bmesh.new()
        self.uv = self.bm.loops.layers.uv.new("UVMap")
        self.mats = []

    def _mat_index(self, mat):
        if mat not in self.mats:
            self.mats.append(mat)
        return self.mats.index(mat)

    def _uv(self, faces, uv_size, origin):
        for f in faces:
            nx, ny, nz = (abs(c) for c in f.normal)
            for loop in f.loops:
                p = loop.vert.co
                if nz >= nx and nz >= ny:
                    u, v = p.x, p.y
                elif nx >= ny:
                    u, v = p.y, p.z
                else:
                    u, v = p.x, p.z
                loop[self.uv].uv = ((u - origin[0]) / uv_size[0], (v - origin[1]) / uv_size[1])

    def box(self, lo, hi, mat, uv_size=(100.0, 100.0), origin=(0.0, 0.0)):
        res = bmesh.ops.create_cube(self.bm, size=1.0)
        verts = res["verts"]
        for v in verts:
            v.co.x = (v.co.x + 0.5) * (hi[0] - lo[0]) + lo[0]
            v.co.y = (v.co.y + 0.5) * (hi[1] - lo[1]) + lo[1]
            v.co.z = (v.co.z + 0.5) * (hi[2] - lo[2]) + lo[2]
        faces = list({f for v in verts for f in v.link_faces})
        self.bm.normal_update()
        idx = self._mat_index(mat)
        for f in faces:
            f.material_index = idx
        self._uv(faces, uv_size, origin)

    def cone(self, z0, z1, r0, r1, mat, segs=32, cap=False):
        res = bmesh.ops.create_cone(self.bm, cap_ends=cap, segments=segs, radius1=r0, radius2=r1, depth=z1 - z0)
        for v in res["verts"]:
            v.co.z += (z0 + z1) / 2
        faces = list({f for v in res["verts"] for f in v.link_faces})
        idx = self._mat_index(mat)
        for f in faces:
            f.material_index = idx
            f.smooth = True
        return faces

    def sphere(self, c, r, mat):
        res = bmesh.ops.create_uvsphere(self.bm, u_segments=16, v_segments=10, radius=r)
        for v in res["verts"]:
            v.co += Vector(c)
        idx = self._mat_index(mat)
        for f in {f for v in res["verts"] for f in v.link_faces}:
            f.material_index = idx
            f.smooth = True

    def build(self):
        me = bpy.data.meshes.new(self.name)
        self.bm.normal_update()
        self.bm.to_mesh(me)
        self.bm.free()
        for m in self.mats:
            me.materials.append(m)
        ob = bpy.data.objects.new(self.name, me)
        bpy.context.scene.collection.objects.link(ob)
        return ob


def floor_concrete():
    p = Part("floor_concrete")
    p.box((-50, -50, -2), (50, 50, 0), MAT["콘크리트"], uv_size=(100, 100), origin=(-50, -50))
    return p.build()


def wall_steel():
    p = Part("wall_steel")
    p.box((-50, -2, 7), (50, 2, 58), MAT["강철벽"], uv_size=(100, 51), origin=(-50, 7))
    p.box((-50, -2.4, 0), (50, 2.4, 7), MAT["경고띠"], uv_size=(50, 12.5), origin=(-50, 0))
    p.box((-50, -3.5, 58), (50, 3.5, 62), MAT["철골"])
    for x in (-50, 50):                                                  # 칸 이음 기둥(이웃 벽과 반씩 겹침)
        p.box((x - 1.5, -3.2, 0), (x + 1.5, 3.2, 58), MAT["철골"])
    return p.build()


def pillar_h():
    p = Part("pillar_h")
    p.box((-10, -10, 0), (10, 10, 2), MAT["철골"])
    for s in (-1, 1):
        p.box((-7, s * 5 - 1, 2), (7, s * 5 + 1, 150), MAT["철골"])      # 플랜지
    p.box((-1, -4, 2), (1, 4, 150), MAT["철골"])                          # 웹
    p.box((-7.4, -6.4, 2), (7.4, 6.4, 16), MAT["경고띠"], uv_size=(15, 3.75), origin=(-7.4, 2))
    return p.build()


def lamp_hanging():
    p = Part("lamp_hanging")
    faces = p.cone(4, 16, 16, 4, MAT["전등갓"])
    bmesh.ops.solidify(p.bm, geom=faces, thickness=0.8)
    p.box((-3.5, -3.5, 16), (3.5, 3.5, 20), MAT["철골"])
    p.box((-0.35, -0.35, 20), (0.35, 0.35, 220), MAT["철골"])
    p.sphere((0, 0, 7.5), 3.6, MAT["전구"])
    return p.build()


objs = [floor_concrete(), wall_steel(), pillar_h(), lamp_hanging()]

for ob in objs:
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.export_scene.fbx(filepath=os.path.join(MODELS, f"{ob.name}.fbx"), use_selection=True, object_types={"MESH"},
                             axis_forward="-Z", axis_up="Y", bake_space_transform=True, apply_unit_scale=False,
                             global_scale=1.0, mesh_smooth_type="FACE", path_mode="STRIP", bake_anim=False, add_leaf_bones=False)
    print("EXPORTED", ob.name, tuple(round(v, 1) for v in ob.dimensions))

# ── 미리보기: 바닥 3×3 · 뒤 벽 · 기둥 · 전등 ─────────────────────────────
scene = bpy.context.scene
for ob in objs:
    ob.hide_render = True
col = scene.collection


def place(src, loc, rot_z=0.0):
    o = src.copy()
    o.hide_render = False
    o.location = loc
    o.rotation_euler = (0, 0, rot_z)
    col.objects.link(o)


for ix in (-1, 0, 1):
    for iy in (-1, 0, 1):
        place(objs[0], (ix * 100, iy * 100, 0))
for ix in (-1, 0, 1):
    place(objs[1], (ix * 100, 150, 0))
place(objs[1], (-150, 100, 0), math.pi / 2)
place(objs[2], (-140, 140, 0))
place(objs[3], (0, 0, 110))

world = bpy.data.worlds.new("어두운 공장")
world.color = (0.02, 0.022, 0.025)
scene.world = world
try:
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.02, 0.022, 0.025, 1)
    bg.inputs["Strength"].default_value = 1.0
except (AttributeError, StopIteration):
    pass
spot = bpy.data.lights.new("전등빛", "SPOT")
spot.energy, spot.spot_size, spot.color = 2.5e6, math.radians(80), (1.0, 0.85, 0.65)
sp = bpy.data.objects.new("전등빛", spot)
sp.location = (0, 0, 112)
col.objects.link(sp)
fill = bpy.data.lights.new("채움빛", "SUN")
fill.energy, fill.color = 0.4, (0.7, 0.75, 0.85)
fo = bpy.data.objects.new("채움빛", fill)
fo.rotation_euler = (math.radians(50), 0, math.radians(30))
col.objects.link(fo)

cam_data = bpy.data.cameras.new("미리보기")
cam = bpy.data.objects.new("미리보기", cam_data)
col.objects.link(cam)
cam.location = (260, -300, 260)
cam.rotation_euler = (Vector((-20, 40, 30)) - cam.location).to_track_quat("-Z", "Y").to_euler()
cam_data.lens = 32
scene.camera = cam
scene.render.resolution_x, scene.render.resolution_y = 1000, 700
for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"):
    try:
        scene.render.engine = engine
        break
    except TypeError:
        continue
scene.render.filepath = PREVIEW
bpy.ops.render.render(write_still=True)
print("PREVIEW", scene.render.engine, PREVIEW)
