"""스마트폰(특정 제품 복제가 아닌 일반형) — 헤드리스 생성 모듈.

단위 1 = 1mm(초파리 body/와 같다).
로컬 축: +X 폭(화면을 볼 때 오른쪽) · +Y 높이(위) · +Z 화면 앞. 뒷면이 z=0.

화면은 따로 떨어진 오브젝트 「폰_화면」이고 재질은 「폰_화면」이다.
- UV 0~1이 표시 영역 전체를 덮는다(둥근 모서리 바깥은 잘림).
- 유니티: 이 재질 자리에 VideoPlayer의 RenderTexture를 넣으면 릴스가 재생된다.
- Blender: 재질의 이미지 노드 「릴스」에 이미지 시퀀스를 넣고, fit_cover()로 9:16 영상을 가운데 맞춤한다.
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

W, H = 71.6, 147.8          # 몸체 폭·높이
FRAME_T = 7.2               # 금속 프레임 두께(뒷면~앞유리 아래)
GLASS_T = 0.6               # 앞유리 두께
CORNER_R = 9.5
BEZEL = 1.6
DISPLAY_W, DISPLAY_H = W - 2 * BEZEL, H - 2 * BEZEL
DISPLAY_R = CORNER_R - BEZEL
SCREEN_Z = FRAME_T + GLASS_T + 0.01
THICKNESS = FRAME_T + GLASS_T


def rounded_rect(w, h, r, seg=12):
    """가운데 원점 둥근 사각형 윤곽(반시계)."""
    pts = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90),
                       (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for i in range(seg + 1):
            a = math.radians(a0 + 90 * i / seg)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _slab(bm, pts, z0, z1, offset=(0.0, 0.0)):
    """윤곽을 z0~z1로 세운 기둥. 새로 만든 면 목록을 돌려준다."""
    ox, oy = offset
    bot = [bm.verts.new((x + ox, y + oy, z0)) for x, y in pts]
    top = [bm.verts.new((x + ox, y + oy, z1)) for x, y in pts]
    faces = [bm.faces.new(list(reversed(bot))), bm.faces.new(top)]
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        faces.append(bm.faces.new((bot[i], bot[j], top[j], top[i])))
    return faces


def _bevel_caps(bm, faces, width, segments):
    """기둥 위아래 모서리만 둥글린다."""
    caps = [f for f in faces[:2]]
    edges = {e for f in caps for e in f.edges}
    bmesh.ops.bevel(bm, geom=list(edges), offset=width, segments=segments, profile=0.5,
                    affect="EDGES", clamp_overlap=True)


def _cylinder(bm, r, z0, z1, x, y, segments=48):
    before = set(bm.faces)
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments, radius1=r, radius2=r,
                          depth=z1 - z0, matrix=Matrix.Translation((x, y, (z0 + z1) / 2)))
    return list(set(bm.faces) - before)


def _box(bm, center, dims, bevel=0.0):
    before = set(bm.faces)
    bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.LocRotScale(center, None, dims))
    new = list(set(bm.faces) - before)
    if bevel:
        edges = {e for f in new for e in f.edges}
        bmesh.ops.bevel(bm, geom=list(edges), offset=bevel, segments=2, profile=0.5, affect="EDGES", clamp_overlap=True)
        new = list(set(bm.faces) - before)
    return new


def material(name, color, metallic=0.0, roughness=0.5, emission=None, strength=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*color, 1.0)
    p.inputs["Metallic"].default_value = metallic
    p.inputs["Roughness"].default_value = roughness
    if emission:
        p.inputs["Emission Color"].default_value = (*emission, 1.0)
        p.inputs["Emission Strength"].default_value = strength
    m.diffuse_color = (*color, 1.0)
    return m


def fit_cover(mat, img_w, img_h):
    """영상 비율이 화면과 달라도 늘리지 않고 가운데를 꽉 채운다(인스타 앱처럼 잘라서)."""
    disp, img = DISPLAY_W / DISPLAY_H, img_w / img_h
    sx, sy = (disp / img, 1.0) if img > disp else (1.0, img / disp)
    mp = mat.node_tree.nodes["화면_맞춤"]
    mp.inputs["Scale"].default_value = (sx, sy, 1.0)
    mp.inputs["Location"].default_value = (0.5 - 0.5 * sx, 0.5 - 0.5 * sy, 0.0)


def screen_material(image, strength=1.6):
    m = material("폰_화면", (0.004, 0.004, 0.005), roughness=0.06)
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.name = "화면_맞춤"
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.name = tex.label = "릴스"
    tex.image = image
    tex.extension = "CLIP"
    nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], p.inputs["Emission Color"])
    p.inputs["Emission Strength"].default_value = strength
    fit_cover(m, *image.size)
    return m


def placeholder_image(path, w=720, h=1280):
    """실제 릴스를 붙이기 전 자리 표시 화면(9:16). 노을 그라데이션 + 재생 표시 + 캡션 줄."""
    y = np.linspace(0.0, 1.0, h)[:, None, None]          # Blender 픽셀은 아래 줄부터
    x = np.linspace(0.0, 1.0, w)[None, :, None]
    top, bot = np.array([0.96, 0.58, 0.32]), np.array([0.22, 0.10, 0.42])
    rgb = np.broadcast_to(bot * (1 - y) + top * y, (h, w, 3)).copy()
    X, Y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    px, py = (X - 0.5) * w / h, Y - 0.55
    glow = np.clip(0.20 - np.sqrt(px ** 2 + py ** 2), 0, None) / 0.20
    rgb += glow[..., None] * 0.30
    tri = (px > -0.035) & (px < 0.05) & (np.abs(py) < (0.05 - px) * 0.55)
    rgb[tri] = 1.0
    for y0, y1, x1 in ((0.080, 0.095, 0.62), (0.110, 0.125, 0.40)):
        band = (Y > y0) & (Y < y1) & (X > 0.06) & (X < x1)
        rgb[band] = rgb[band] * 0.35 + 0.65
    for cy in (0.30, 0.38, 0.46):
        dot = np.sqrt(((X - 0.9) * w / h) ** 2 + (Y - cy) ** 2) < 0.018
        rgb[dot] = rgb[dot] * 0.3 + 0.7
    rgba = np.concatenate([np.clip(rgb, 0, 1), np.ones((h, w, 1))], axis=2).astype(np.float32)
    img = bpy.data.images.new("릴스_자리", w, h)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    return img


def _object(name, bm, mats, collection, parent, sharp_deg=35.0):
    me = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    me.shade_smooth()
    me.set_sharp_from_angle(angle=math.radians(sharp_deg))
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    ob.parent = parent
    return ob


def build_phone(collection, screen_image):
    """스마트폰을 만들어 루트 Empty 「스마트폰」과 치수 정보를 돌려준다."""
    frame = material("폰_프레임", (0.09, 0.095, 0.10), metallic=1.0, roughness=0.28)
    back = material("폰_뒷면", (0.10, 0.13, 0.17), roughness=0.42)
    glass = material("폰_유리", (0.003, 0.003, 0.004), roughness=0.04)
    lens = material("폰_렌즈", (0.005, 0.005, 0.008), roughness=0.02)
    flash = material("폰_플래시", (0.85, 0.80, 0.62), roughness=0.3)
    screen = screen_material(screen_image)

    root = bpy.data.objects.new("스마트폰", None)
    root.empty_display_size = 20
    collection.objects.link(root)

    # 몸체: 금속 테두리 + 뒷면(아래를 향한 면)
    bm = bmesh.new()
    faces = _slab(bm, rounded_rect(W, H, CORNER_R), 0.0, FRAME_T)
    _bevel_caps(bm, faces, 1.0, 4)
    bm.normal_update()
    for f in bm.faces:
        f.material_index = 1 if f.normal.z < -0.99 else 0
    body = _object("폰_몸체", bm, [frame, back], collection, root)

    # 앞유리
    bm = bmesh.new()
    faces = _slab(bm, rounded_rect(W - 0.5, H - 0.5, CORNER_R - 0.25), FRAME_T - 0.05, THICKNESS)
    _bevel_caps(bm, faces, 0.25, 2)
    _object("폰_앞유리", bm, [glass], collection, root)

    # 화면(UV 0~1 = 표시 영역)
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    verts = [bm.verts.new((x, y, SCREEN_Z)) for x, y in rounded_rect(DISPLAY_W, DISPLAY_H, DISPLAY_R, 16)]
    face = bm.faces.new(verts)
    for loop in face.loops:
        loop[uv].uv = ((loop.vert.co.x + DISPLAY_W / 2) / DISPLAY_W, (loop.vert.co.y + DISPLAY_H / 2) / DISPLAY_H)
    _object("폰_화면", bm, [screen], collection, root, sharp_deg=180.0)

    # 앞 카메라 구멍
    bm = bmesh.new()
    _cylinder(bm, 1.6, SCREEN_Z, SCREEN_Z + 0.03, 0.0, DISPLAY_H / 2 - 5.5, 32)
    _object("폰_앞카메라", bm, [lens], collection, root)

    # 뒷면 카메라 섬 + 렌즈 3개 + 플래시 (앞에서 볼 때 오른쪽 위 = 뒤에서 볼 때 왼쪽 위)
    cx, cy = W / 2 - 17.0, H / 2 - 17.0
    bm = bmesh.new()
    island = _slab(bm, rounded_rect(27, 27, 7), -1.3, 0.05, (cx, cy))
    _bevel_caps(bm, island, 0.5, 3)
    island_faces = set(bm.faces)
    parts = []
    for lx, ly in ((-6.5, 6.5), (-6.5, -6.5), (6.5, -6.5)):
        parts += [(f, 0) for f in _cylinder(bm, 4.8, -2.1, -1.25, cx + lx, cy + ly)]
        parts += [(f, 2) for f in _cylinder(bm, 3.8, -2.2, -2.05, cx + lx, cy + ly)]
    parts += [(f, 3) for f in _cylinder(bm, 1.6, -1.45, -1.25, cx + 6.5, cy + 6.5, 24)]
    for f in island_faces:
        f.material_index = 1
    for f, idx in parts:
        f.material_index = idx
    _object("폰_뒷카메라", bm, [frame, back, lens, flash], collection, root)

    # 옆 버튼: 오른쪽 전원, 왼쪽 음량 둘
    bm = bmesh.new()
    _box(bm, (W / 2 + 0.15, 22.0, FRAME_T / 2), (0.9, 18.0, 2.6), bevel=0.3)
    _box(bm, (-W / 2 - 0.15, 36.0, FRAME_T / 2), (0.9, 13.0, 2.6), bevel=0.3)
    _box(bm, (-W / 2 - 0.15, 52.0, FRAME_T / 2), (0.9, 13.0, 2.6), bevel=0.3)
    _object("폰_버튼", bm, [frame], collection, root)

    info = dict(size=[W, H, THICKNESS], display=[round(DISPLAY_W, 2), round(DISPLAY_H, 2)],
                display_aspect=round(DISPLAY_W / DISPLAY_H, 4), body_tris=sum(len(p.vertices) - 2 for p in body.data.polygons))
    return root, info
