# FlyKing — 초파리 뇌 신경지도 × Blender 몸

사장님 개인 프로젝트. 초파리 커넥톰 기반 뇌 시뮬레이션을 Blender로 만든 초파리 몸에 연결해 **관절을 세세하게 나눠 걷고·몸단장하고·날갯짓하게** 만든다.
게임 프로젝트(`~/GitHub/GuilRandomDefense`)와는 무관하다 — 거기 규칙·파일을 섞지 말 것.

## 목표(2026-09-13 사장님)
- 「초파리 뇌 신경 연결할건데 진짜 세세하게 관절 나눠서 구현 가능해? 날개짓이랑」 → 가능하다고 답하고 시작함.
- 연결 경로(계획):
  1. **뇌** — Shiu et al. 2024(Nature) 전뇌 LIF 모델(FlyWire v783, 뉴런 약 13.9만). 자극(예: 당 감각뉴런) → 하강뉴런(DN, 약 1,100개) 발화율.
  2. **해독** — DN 발화율 → 행동 명령(걷기 좌우 구동·몸단장·도약·비행). 참고 구현: `~/flybrain/embodied-fly-brain/brain_body_bridge.py`(원본은 CUDA용, 맥에 맞게 옮겨야 함).
  3. **몸 물리** — NeuroMechFly v2 / flygym(MuJoCo, **CPU로 돎**): 87관절(다리 7축×6 = 42 + 머리·더듬이·배·날개). 출력 = 관절 각도 시계열.
  4. **Blender** — `body/apply_joint_angles.py`로 관절각을 뼈대에 키프레임으로 굽고 렌더.
  - 날갯짓: 뉴런이 박동(약 200Hz)을 하나하나 치지 않는다(비동기 비행근). **박동 발생기(날개당 yaw·pitch·roll)** + 조향 신호가 진폭·주파수·좌우 차이를 조절하는 방식으로 구현.
  - 나중: MaleCNS v1.0에 몸통 신경절(VNC) 운동뉴런이 있으니 DN 해독 대신 운동뉴런 → 관절 직접 매핑도 가능.

## 이 저장소
| 경로 | 내용 |
|---|---|
| `body/` | Blender 초파리 몸 **헤드리스 정본 스크립트**(blender 세션 제작). `gen_fly.py`(메인) · `fly_mesh/fly_body/fly_limbs/fly_wing/fly_tex.py` · `fly_rig.py` · `fly_nmf.py`(flygym MJCF 관절 축·한계) · `apply_joint_angles.py` · `render_fly.py` · `check_baked.py` · `nmf/`(규격 추출·예제 생성·애니 검사) · `examples/`(flygym 관절각 csv/npz, 날갯짓 CPG csv) · `초파리.blend/.fbx` · `Textures/` |
| `brain/shiu/` | Shiu 모델 실행 스크립트: `smoke_test.py`(0.1초 동작 확인) · `sugar_test.py`(당 뉴런 23개 150Hz 1초 → 상위 반응) · `analyze_sugar.py`(자극 뉴런 뺀 하류 반응) |
| `scene/` | 「초파리가 폰으로 릴스 보는」 장면(이 세션 제작, body/는 읽기만). `phone.py`(일반형 스마트폰 — 화면은 별도 오브젝트·재질 「폰_화면」, UV 0~1 = 표시 영역, `fit_cover`로 9:16 가운데 맞춤) · `gen_scene.py`(초파리 어펜드 + 폰·거치대·책상·카메라 3대 → `초파리_릴스.blend`, 유니티용 `스마트폰.fbx`, 미리보기 `renders/`) |
| `insta/` | 초파리 전용 인스타 계정 브라우저 자동화(Playwright). `reels.py`(릴스 읽기·재생속도·넘기기·좋아요, 화면 좌표로 버튼 찾음) · `common.py`(프로필·실행 설정) · `login.py`(창 띄우고 사장님이 **직접** 로그인할 때까지 대기) · `probe_reels.py`(릴스 화면 구조 탐색, 좋아요 안 누름) |

- 뼈대: 뼈 61개, 이름이 NeuroMechFly v2 body 이름과 같다 — `Thorax, Head, LAntenna, RAntenna, LWing, RWing, LHaltere, RHaltere, A1A2, A3~A6` + 다리 `LF/LM/LH/RF/RM/RH` × `Coxa, Femur, Tibia, Tarsus1~5`. 축 +X 앞·+Y 왼·Z 위, 1 단위 = 1mm.
- 클립: `Walk_Tripod`(30f) · `Idle_Groom`(60f). 날개 3축·관절 한계·`Flight_Wingbeat`는 blender 세션이 확장 중(아래).

## git 밖 작업공간 `~/flybrain/` (대용량·외부 저장소)
| 폴더 | 출처 | 비고 |
|---|---|---|
| `shiu-brain-model/` | github.com/philshiu/Drosophila_brain_model | 데이터 포함(Connectivity_783.parquet 100MB, Completeness_783.csv). 파이썬 환경 `.venv` |
| `embodied-fly-brain/` | github.com/erojasoficial-byte/fly-brain | 뇌+NeuroMechFly 체화판. **NVIDIA GPU·CUDA 필요 → 맥에선 원본 그대로 못 돎.** `data/flywire_annotations.tsv`(세포 유형) 여기 있음 |
| `eon-fly-brain/` | github.com/eonsystemspbc/fly-brain | 벤치마크. 빠른 백엔드는 GPU 전용 |
| `flygym/` | github.com/NeLy-EPFL/flygym | NeuroMechFly v2 몸(MuJoCo) |
| `male-cns-v1.0/` | male-cns.janelia.org/download (CC-BY) | 수컷 전체 CNS 커넥톰 평면 파일: 주석 21만 행·연결 가중치 1.5억 행(1.1GB) |
| `flybody-blender` | → `~/GitHub/FlyKing/body` 링크 | 옛 경로 호환용 |
| `insta/` | 직접 만듦 | `.venv`(uv, Python 3.11, playwright 1.62) · `profile/`(로그인 쿠키 — **비밀번호·세션은 git·메모리에 절대 넣지 않는다**) · `results/` |

## 환경·실행
- 뇌 모델 환경: `~/flybrain/shiu-brain-model/.venv`(uv, Python 3.10, brian2 2.5.1, numpy 1.24, **setuptools<70**(없으면 pkg_resources 오류), **cython<3**(3이면 brian2가 numpy 방식으로 떨어져 4배 느림)).
  실측: 뇌 전체 1초 시뮬 ≈ 5초(Cython 첫 컴파일 46초).
- 실행(스크립트가 모델 폴더로 스스로 이동한다):
  `~/flybrain/shiu-brain-model/.venv/bin/python brain/shiu/sugar_test.py`
- Blender: `/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --python body/gen_fly.py`
- flygym 환경은 아직 안 만듦(다음 단계) — MuJoCo CPU.

## 함께 일하는 세션
- **blender** 세션(ListAgents로 확인, SendMessage로 지시)이 몸 모델·뼈대·렌더를 맡는다. 사장님 Blender 창에서 실시간으로 보여 주며 만들고, 정본은 헤드리스 스크립트.
- 2026-09-13 blender에 지시해 둔 것: flygym MJCF 기준 관절 축·한계, `Flight_Wingbeat` 클립, `apply_joint_angles.py`, 관절 한계 검사 렌더.
  (blender는 게임 쪽 유닛 FBX 정리를 먼저 끝내고 초파리로 돌아온다.)

## 규칙
- 대용량(커넥톰·결과·렌더)은 git에 넣지 않는다 — GitHub 파일 한도 100MB. `.gitignore` 참고.
- 커밋은 `git add <파일들>` 후 `git commit -m "…" -- <파일들>`(여러 세션이 같은 저장소를 만질 수 있다).
- 추측으로 해부학·관절 축을 정하지 않는다 — flygym MJCF·논문 출처를 주석에 남긴다.
