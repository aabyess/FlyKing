"""초파리 텍스처 — 그린 것(numpy: 겹눈 낱눈·강모·홑눈·평균곤·날개 시맥)과 굽는 셰이더(몸·다리: 꼭짓점 색 × 잔 얼룩).
그린 PNG 값은 sRGB, 굽는 셰이더 색은 선형."""
import os

import bpy
import numpy as np

from fly_limbs import UV_BOX, VEINS, outline, smooth_line


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _hash(a, b):
    return np.modf(np.abs(np.sin(a * 12.9898 + b * 78.233) * 43758.5453))[0]


def save_png(name, rgba, folder):
    os.makedirs(folder, exist_ok=True)
    old = bpy.data.images.get(name)
    if old is not None:
        bpy.data.images.remove(old)
    h, w = rgba.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.pixels.foreach_set(rgba.astype(np.float32).ravel())
    img.filepath_raw = os.path.join(folder, name + ".png")
    img.file_format = "PNG"
    img.save()
    return img


def eye_texture(size=256):
    """낱눈 하나 — UV 가운데 (0.5,0.5) 반지름 0.5 육각. 가운데 밝은 주홍 → 가장자리 짙은 적갈, 바깥은 낱눈 사이 틈."""
    y, x = np.mgrid[0:size, 0:size]
    X, Y = (x + 0.5) / size - 0.5, (y + 0.5) / size - 0.5
    ang = np.arctan2(Y, X)
    hexr = np.hypot(X, Y) * np.cos((ang % (np.pi / 3)) - np.pi / 6) / np.cos(np.pi / 6) / 0.5
    center, edge, gap = np.array((0.76, 0.10, 0.06)), np.array((0.56, 0.06, 0.04)), np.array((0.30, 0.03, 0.02))
    col = center + (edge - center) * _smooth(0.2, 0.9, hexr)[..., None]
    col = col + (gap - col) * _smooth(0.92, 1.0, hexr)[..., None]
    return np.dstack([col, np.ones((size, size))])


def hair_texture(w=32, h=256):
    v = ((np.arange(h) + 0.5) / h)[:, None, None]
    col = np.array((0.13, 0.08, 0.05)) + (np.array((0.32, 0.23, 0.15)) - np.array((0.13, 0.08, 0.05))) * v
    return np.dstack([np.broadcast_to(col, (h, w, 3)), np.ones((h, w))])


def ocellus_texture(size=128):
    y, x = np.mgrid[0:size, 0:size]
    r = np.hypot((x + 0.5) / size - 0.5, (y + 0.5) / size - 0.5) * 2
    col = np.array((0.62, 0.26, 0.12)) + (np.array((0.22, 0.07, 0.04)) - np.array((0.62, 0.26, 0.12))) * _smooth(0.0, 1.0, r)[..., None]
    return np.dstack([col, np.ones((size, size))])


def haltere_texture(w=64, h=256):
    v = ((np.arange(h) + 0.5) / h)[:, None]
    u = ((np.arange(w) + 0.5) / w)[None, :]
    stalk, knob = np.array((0.86, 0.78, 0.60)), np.array((0.80, 0.68, 0.46))
    t = _smooth(0.5, 0.6, v) * (0.8 + 0.2 * np.sin(u * np.pi * 2) ** 2)
    col = stalk + (knob - stalk) * t[..., None]
    return np.dstack([col, np.ones((h, w))])


def _segments(points):
    p = np.array(points, dtype=np.float64)
    return p[:-1], p[1:]


def wing_texture(width=2048, height=1024):
    """날개 막(알파 약 0.22, 옅은 무지개 얼룩·미세털 점) + 시맥(알파 1, 갈색) + 앞가장자리 굵은 costa + 가장자리 잔털."""
    s0, w0, s1, w1 = UV_BOX
    S = s0 + (np.arange(width) + 0.5) / width * (s1 - s0)
    W = w0 + (np.arange(height) + 0.5) / height * (w1 - w0)
    S, W = np.meshgrid(S, W)
    poly = outline()
    a, b = _segments(poly + [poly[0]])
    inside = np.zeros(S.shape, dtype=bool)
    dist = np.full(S.shape, 1e9)
    arc = np.zeros(S.shape)
    lengths = np.hypot(*(b - a).T)
    starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    for (ax, ay), (bx, by), L, st in zip(a, b, lengths, starts):
        if (ay > W).any() or (by > W).any():
            cross = ((ay > W) != (by > W)) & (S < (bx - ax) * (W - ay) / ((by - ay) if by != ay else 1e-12) + ax)
            inside ^= cross
        t = np.clip(((S - ax) * (bx - ax) + (W - ay) * (by - ay)) / max(L * L, 1e-12), 0, 1)
        d = np.hypot(S - (ax + t * (bx - ax)), W - (ay + t * (by - ay)))
        closer = d < dist
        dist = np.where(closer, d, dist)
        arc = np.where(closer, st + t * L, arc)
    px = (s1 - s0) / width
    membrane = np.where(inside, 1.0, 0.0) * 1.0
    edge_soft = np.where(inside, 1.0, _smooth(px * 1.5, 0.0, dist))
    alpha = 0.22 * edge_soft
    col = np.dstack([np.full(S.shape, 0.78), np.full(S.shape, 0.79), np.full(S.shape, 0.80)])
    irid = 0.5 + 0.5 * np.sin(S * 7.0 + W * 11.0 + np.sin(S * 3.0) * 2.0)
    col = col + np.dstack([0.06 * irid, 0.03 * (1 - irid), 0.07 * (1 - irid)])
    speck = (_hash(np.floor(S / 0.006), np.floor(W / 0.006)) > 0.72) & inside
    alpha = np.where(speck, alpha + 0.12, alpha)
    base = _smooth(0.22, 0.05, S) * membrane                            # 뿌리 쪽 경화부
    alpha = np.maximum(alpha, 0.85 * base)
    col = col + (np.array((0.45, 0.33, 0.2)) - col) * base[..., None]
    vein_col = np.array((0.40, 0.29, 0.17))
    lines = [(pts, width_mm) for pts, width_mm in VEINS.values()]
    tip_index = min(range(len(poly)), key=lambda i: (poly[i][0] - 2.12) ** 2 + (poly[i][1] + 0.29) ** 2)
    lines.append((poly[:tip_index + 1], 0.02))                           # costa: 뿌리 → L4 끝까지 앞 가장자리
    for pts, width_mm in lines:
        pts = smooth_line(pts, 8) if len(pts) > 2 else pts
        va, vb = _segments(pts)
        vd = np.full(S.shape, 1e9)
        vt = np.zeros(S.shape)
        seg_len = np.hypot(*(vb - va).T)
        total = seg_len.sum()
        acc = 0.0
        for (ax, ay), (bx, by), L in zip(va, vb, seg_len):
            t = np.clip(((S - ax) * (bx - ax) + (W - ay) * (by - ay)) / max(L * L, 1e-12), 0, 1)
            d = np.hypot(S - (ax + t * (bx - ax)), W - (ay + t * (by - ay)))
            closer = d < vd
            vd = np.where(closer, d, vd)
            vt = np.where(closer, (acc + t * L) / total, vt)
            acc += L
        half = width_mm * (1.0 - 0.35 * vt) / 2
        cover = _smooth(half + px, half - px, vd) * edge_soft
        alpha = np.maximum(alpha, cover)
        col = col + (vein_col - col) * cover[..., None]
    hair_phase = np.abs(np.modf(arc / 0.011)[0] - 0.5)
    hair_len = 0.018 + 0.012 * _hash(np.floor(arc / 0.011), 3.0)
    hair = (~inside) & (hair_phase < 0.1) & (dist < hair_len)
    alpha = np.where(hair, np.maximum(alpha, 0.55), alpha)
    col = np.where(hair[..., None], np.array((0.35, 0.3, 0.26)), col)
    return np.dstack([np.clip(col, 0, 1), np.clip(alpha, 0, 1)])


# ──────────────────────────────────────────────────────────── 재질

def _clear(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.type not in {"BSDF_PRINCIPLED", "OUTPUT_MATERIAL"}:
            nt.nodes.remove(n)
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Metallic"].default_value = 0.0
    return nt, bsdf


ROUGH = {"초파리_몸": 0.42, "초파리_다리": 0.45, "초파리_겹눈": 0.28, "초파리_홑눈": 0.12, "초파리_강모": 0.5,
         "초파리_평균곤": 0.55, "초파리_날개_막": 0.25}
DRAWN = {"초파리_겹눈": eye_texture, "초파리_강모": hair_texture, "초파리_홑눈": ocellus_texture,
         "초파리_평균곤": haltere_texture, "초파리_날개_막": wing_texture}
BAKED = {"초파리_몸": 2048, "초파리_다리": 1024}


def image_material(name, img):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nt, bsdf = _clear(mat)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = ROUGH[name]
    # 날개 막은 반투명 알파 그대로 섞는다 — 게임 에셋이 아니라 알파 컷 규칙(`_잎카드`)을 따르지 않는다(PM 09-13)
    if name.endswith("_막"):
        nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
        mat.use_backface_culling = False
        if hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "BLENDED"
    return mat


def bake_material(name):
    """굽기 전 셰이더 — 꼭짓점 색 「색」 × 물체 좌표 잔 얼룩(1mm에 70·260 번)."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nt, bsdf = _clear(mat)
    attr = nt.nodes.new("ShaderNodeVertexColor")
    attr.layer_name = "색"
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mults = []
    for scale, amount in ((70.0, 0.25), (260.0, 0.18)):
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = scale
        noise.inputs["Detail"].default_value = 4.0
        nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = 1.0 - amount
        mr.inputs["To Max"].default_value = 1.0 + amount * 0.6
        nt.links.new(noise.outputs["Fac"], mr.inputs["Value"])
        mults.append(mr)
    m1 = nt.nodes.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    nt.links.new(mults[0].outputs["Result"], m1.inputs[0])
    nt.links.new(mults[1].outputs["Result"], m1.inputs[1])
    mix = nt.nodes.new("ShaderNodeVectorMath")
    mix.operation = "SCALE"
    nt.links.new(attr.outputs["Color"], mix.inputs[0])
    nt.links.new(m1.outputs["Value"], mix.inputs["Scale"])
    nt.links.new(mix.outputs["Vector"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = ROUGH[name]
    return mat
