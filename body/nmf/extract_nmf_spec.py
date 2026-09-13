"""NeuroMechFly 관절 규격 추출 — flygym 저장소 원본에서 몸 좌표계·관절 축·순서·부호·가동범위 근거를 읽어 nmf_spec.json으로.
추측 금지: 모든 값에 출처 파일:줄을 단다. 이 스크립트가 줄 번호를 원본에서 직접 찾는다(손으로 적은 줄 번호 없음).
    ~/flybrain/shiu-brain-model/.venv/bin/python extract_nmf_spec.py [flygym 저장소 경로]
출처(flygym NeLy-EPFL/flygym, sparse clone ~/flybrain/flygym):
  v1 = legacy/flygym1_seqikpy_yawpitchroll.xml  (= flygym v1.2.1 flygym/data/mjcf/neuromechfly_seqik_kinorder_ypr.xml 사본)
  v2 = rigging.yaml(몸 위치·쿼터니언) + anatomy.py(축 글자→벡터, 이음 목록) + compose/fly/base_fly.py(오른쪽 yaw·roll 부호 뒤집기)
  참고 = pose/neutral/yaw_pitch_roll.yaml(v2 중립 자세, 도), flygym_demo/complex_terrain/assets/single_steps_untethered.pkl(v1 이름 걷기 관측값)"""
import json
import math
import pickle
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "flybrain/flygym")
SRC = REPO / "src/flygym"
NMF = SRC / "assets/model/neuromechfly"
V1_XML = NMF / "legacy/flygym1_seqikpy_yawpitchroll.xml"
RIGGING = NMF / "rigging.yaml"
NEUTRAL = NMF / "pose/neutral/yaw_pitch_roll.yaml"
ANATOMY = SRC / "anatomy.py"
BASE_FLY = SRC / "compose/fly/base_fly.py"
WALK_PKL = REPO / "src/flygym_demo/complex_terrain/assets/single_steps_untethered.pkl"


def rel(p, line=None):
    s = str(p.relative_to(REPO))
    return f"{s}:{line}" if line else s


def find_line(path, pattern, start=1):
    for no, text in enumerate(path.read_text().split("\n"), 1):
        if no >= start and re.search(pattern, text):
            return no, text.strip()
    raise SystemExit(f"못 찾음: {pattern} in {path}")


def v1_tree():
    """worldbody의 body·joint를 줄 번호와 함께. MuJoCo: 한 body의 여러 hinge는 적힌 순서대로 부모 쪽부터 곱해진다."""
    bodies, joints, stack = {}, [], []
    lines = V1_XML.read_text().split("\n")
    inside = False
    for no, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith("<worldbody"):
            inside = True
        if not inside:
            continue
        m = re.match(r"<body (.*?)(/?)>", s)
        if m:
            a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
            name = a["name"]
            bodies[name] = {"parent": stack[-1] if stack else None, "pos": [float(x) for x in a.get("pos", "0 0 0").split()],
                            "quat": [float(x) for x in a.get("quat", "1 0 0 0").split()], "source": rel(V1_XML, no)}
            if not m.group(2):
                stack.append(name)
            continue
        if s.startswith("</body"):
            stack.pop()
            continue
        m = re.match(r"<joint (.*?)/?>", s)
        if m:
            a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
            name = a["name"]
            axis_name = name.rsplit("_", 1)[1] if name.endswith(("_yaw", "_roll")) else "pitch"
            joints.append({"name": name, "body": stack[-1], "axis": [float(x) for x in a["axis"].split()], "axis_name": axis_name,
                           "type": a.get("type"), "range": a.get("range"), "limited": a.get("limited"), "source": rel(V1_XML, no)})
        if s.startswith("</worldbody"):
            break
    order = {}
    for j in joints:
        order.setdefault(j["body"], []).append(j["axis_name"])
    return bodies, joints, order


def v2_rigging():
    out, cur = {}, None
    for no, line in enumerate(RIGGING.read_text().split("\n"), 1):
        m = re.match(r"^(\w+):\s*$", line)
        if m:
            cur = m.group(1)
            out[cur] = {"source": rel(RIGGING, no)}
            continue
        m = re.match(r"^\s+(pos|quat):\s*\[(.*)\]", line)
        if m and cur:
            out[cur][m.group(1)] = [float(x) for x in m.group(2).split(",")]
    return out


def v2_conventions():
    text = ANATOMY.read_text().split("\n")
    axis = {}
    for key in ("pitch", "roll", "yaw"):
        no, s = find_line(ANATOMY, rf'^\s*"{key}":\s*\(')
        axis[key] = {"vector": [float(x) for x in re.search(r"\((.*)\)", s).group(1).split(",")], "source": rel(ANATOMY, no)}
    flip_no, flip_s = find_line(BASE_FLY, r'jointdof\.child\.pos\[0\] == "r" and not self\._is_pitch')
    neg_no, _ = find_line(BASE_FLY, r"vec = -vec", flip_no)
    order_no, _ = find_line(BASE_FLY, r"for jointdof in skeleton\.iter_jointdofs")
    iter_no, _ = find_line(ANATOMY, r"for axis in axis_order\.value")
    bio_no, _ = find_line(ANATOMY, r"def _get_all_biological_joints")
    pairs_no, _ = find_line(ANATOMY, r"^ALL_CONNECTED_SEGMENT_PAIRS")
    return {
        "axis_vectors": axis,
        "right_side_flip": {"rule": "오른쪽(child.pos[0]=='r') 몸의 pitch 아닌 축(yaw·roll)은 축 벡터 부호를 뒤집는다",
                            "source": f"{rel(BASE_FLY, flip_no)}-{neg_no}", "code": flip_s},
        "dof_order": {"rule": "한 이음의 DoF는 skeleton.axis_order 순서로 차례로 만든다(=MuJoCo에서 부모 쪽부터 곱)",
                      "source": [rel(BASE_FLY, order_no), rel(ANATOMY, iter_no)]},
        "biological_preset": {"rule": "all_biological: 넓적다리(trochanterfemur)는 yaw 없음, 종아리·발목마디는 pitch만. 날개·평균곤·배·더듬이는 3축",
                              "source": rel(ANATOMY, bio_no)},
        "connected_pairs": rel(ANATOMY, pairs_no),
        "name_pattern": "{parent}-{child}-{axis}  예: c_thorax-lf_coxa-yaw, lf_coxa-lf_trochanterfemur-pitch, c_thorax-l_wing-roll",
    }


NMF_PY = SRC / "compose/fly/neuromechfly.py"
MESH_DIR = NMF / "meshes/simplified_max2000faces"


def mesh_axes():
    """날개·평균곤 메시(몸 좌표계, 단위 m → 모델에서 ×1000)의 주축. 날개: 뿌리(원점)에서 가장 먼 점 = 날개 끝 방향(span),
    넓게 퍼진 쪽(뒤 가장자리 쪽 돌출이 더 큼)의 반대 = 앞 가장자리(leading). 오른쪽은 왼쪽 메시를 y 배율 −1로 뒤집어 쓴다."""
    import struct
    import numpy as np
    out = {}
    for name in ("l_wing", "l_haltere"):
        path = MESH_DIR / f"{name}.stl"
        data = path.read_bytes()
        n = struct.unpack("<I", data[80:84])[0]
        rows = np.frombuffer(data, dtype=np.uint8, count=n * 50, offset=84).reshape(n, 50)
        pts = np.frombuffer(rows[:, 12:48].tobytes(), dtype="<f4").reshape(-1, 3).astype(float) * 1000.0
        tip = pts[np.argmax(np.linalg.norm(pts, axis=1))]
        span = tip / np.linalg.norm(tip)
        perp = pts - np.outer(pts @ span, span)
        _, _, vt = np.linalg.svd(perp - perp.mean(0), full_matrices=False)
        chord, normal = vt[0], vt[1]
        c = perp @ chord
        lead = chord if c.max() < -c.min() else -chord          # 앞 가장자리 = 덜 튀어나온 쪽
        dorsal = np.cross(lead, span)                           # 오른손: 앞 × 끝 = 등(우리 날개 v × u와 같은 약속)
        out[name] = {"span": [round(float(x), 3) for x in span], "leading": [round(float(x), 3) for x in lead],
                     "dorsal": [round(float(x), 3) for x in dorsal], "length_mm": round(float(np.linalg.norm(tip)), 3),
                     "extent_along_leading_mm": [round(float((perp @ lead).min()), 3), round(float((perp @ lead).max()), 3)],
                     "source": rel(path) + " (주축 계산)"}
    mno, _ = find_line(NMF_PY, r'segment_name\[0\] == "r"')
    sno, _ = find_line(NMF_PY, r"scale=\(self\.SCALE, y_sign")
    out["mirror_rule"] = {"rule": "오른쪽 조각은 왼쪽 메시를 y 배율 −1로 씀 → 오른쪽 span·dorsal의 y 성분 부호 반대, 오른손 좌표 유지 위해 dorsal = leading × span 다시 계산",
                          "source": f"{rel(NMF_PY, mno)}, {rel(NMF_PY, sno)}"}
    return out


def neutral_pose():
    out = {}
    for no, line in enumerate(NEUTRAL.read_text().split("\n"), 1):
        m = re.match(r"^\s+([\w-]+):\s*(-?[\d.]+)\s*$", line)
        if m:
            out[m.group(1)] = {"deg": float(m.group(2)), "source": rel(NEUTRAL, no)}
    return out


def walking_ranges():
    data = pickle.load(open(WALK_PKL, "rb"))
    out = {}
    for k, v in data.items():
        if k.startswith("joint_"):
            vals = [float(x) for x in v]
            out[k] = {"min": min(vals), "max": max(vals), "samples": len(vals)}
    return out


def main():
    commit = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%H %cI"], capture_output=True, text=True).stdout.strip()
    bodies, joints, order = v1_tree()
    ranged = [j["name"] for j in joints if j["range"] or j["limited"]]
    spec = {
        "flygym_commit": commit,
        "units": {"length": "mm", "angle": "rad (v1 MJCF compiler angle=radian)", "quat": "MuJoCo (w, x, y, z)"},
        "frame": "+X 앞 · +Y 왼 · Z 위 (v1 Thorax 쿼터니언 1 0 0 0, 다리 몸 쿼터니언 전부 1 0 0 0 → 모든 각 0이면 다리가 −Z로 곧게 늘어짐)",
        "v1": {"file": rel(V1_XML), "bodies": bodies, "joints": joints, "order_by_body": order,
               "joint_count": len(joints), "joints_with_range": ranged,
               "missing": "날개·평균곤·배·주둥이에 관절 없음(몸만 있음)"},
        "v2": {"rigging": v2_rigging(), "conventions": v2_conventions(), "mesh_axes": mesh_axes()},
        "ranges": {"nmf": "없음 — v1 MJCF에 range/limited 속성이 하나도 없고, v2 base_fly.add_joints도 range를 주지 않는다(stiffness·damping·springref만)",
                   "walking_observed_v1": walking_ranges(), "walking_source": rel(WALK_PKL)},
        "neutral_pose_v2_deg": neutral_pose(),
    }
    out = Path(__file__).with_name("nmf_spec.json")
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=1))
    print("wrote", out, "| v1 joints", len(joints), "| with range", len(ranged), "| v2 segments", len(spec["v2"]["rigging"]),
          "| walking dofs", len(spec["ranges"]["walking_observed_v1"]), "| commit", commit)
    for k, v in spec["v2"]["conventions"]["axis_vectors"].items():
        print(" ", k, v)
    print(" ", spec["v2"]["conventions"]["right_side_flip"])
    print("  mesh", {k: v for k, v in spec["v2"]["mesh_axes"].items()})
    print("  v1 order", {b: o for b, o in order.items() if b in ("LFCoxa", "LFFemur", "Head", "LPedicel")})


main()
