# 초파리 뇌로 릴스 판정하기 — 구조와 한계

릴스 첫 1초 → 겹눈 → 시각 특징 → Shiu 2024 전뇌 LIF 모델(FlyWire v783) → 뉴런 발화율 → 좋아요·넘기기 판정.
허브(`live/hub.py --decider=brain`)가 릴스마다 이 과정을 돌리고 근거 수치를 `~/flybrain/insta/results/brain/`에 남긴다.

## 한 줄 결론
초파리는 영상 내용(사람·자막·웃김)을 모른다. 이 판정은 **밝기·움직임·다가옴에 대한 타고난 시각 반응이 실제 배선을 거쳐 어떤 명령 뉴런을 켰는가**이다. 연결 세기가 고정이라 학습·취향은 없고, 포아송 입력 때문에 같은 영상도 조금씩 흔들린다. 그래서 화면 문구는 「초파리 뇌가 강하게 반응한 영상에 좋아요」다.

## 파이프라인
| 단계 | 파일 | 내용 |
|---|---|---|
| 겹눈 | `eye.py` | 눈당 낱눈 721(flygym `vision.yaml`), 시야 157°, 두 광수용체 채널(pale·yellow, RGB 근사). 폰 화면이 가로 70°·세로 110°를 차지 |
| 시각 특징 | `features.py` | 운동(하센슈타인–라이하르트), 작은 물체, 다가옴(한 덩어리가 사방으로 커짐), 밝기·밝기 변화 |
| 입력 뉴런 | `brain_eval.py` `VPN_INPUT` | 특징 → 시각 투사 뉴런 무리(LC4·LC6·LC16·LPLC2·LC11·LC10·HS·VS·버섯체 VPN 20유형)에 최대 150Hz 포아송 |
| 뇌 | `brain_eval.py` | Shiu `create_model` 그대로, 네트워크 한 번 만들고 재사용. 뇌 1초 ≈ 실제 1.6초 |
| 판정 | `live/hub.py` `judge()` | 아래 규칙 |
| 문턱 | `calibrate.py` → `calibration.json` | 회색 대조군 + 기준 릴스 분포 |

## 판정 규칙(앞이 우선)
1. **도주 → 바로 넘김**: 거대섬유 DNp01 50ms 최고 ≥ 60Hz. 루밍 도약 도주(von Reyn 2014). 문턱은 embodied `brain_body_bridge.py`의 0.3×200Hz.
2. **처벌 → 바로 넘김**: PPL1 도파민 뉴런 평균 ≥ 기준 릴스 90백분위(Aso 2010·2012).
3. **보상 → 좋아요**: PAM 도파민 뉴런 평균 ≥ 회색 대조군 평균+3SD(Liu 2012; Burke 2012).
4. **다가가기 → 좋아요**: oDN1·P9 전진 명령 평균 ≥ 기준 릴스 67백분위, PPL1 ≤ 67백분위(Bidaye 2020).
5. 그 밖 → 조금 보고 넘김.

4번은 절대 기준이 아니라 「기준 릴스들 중 상대적으로 강하게 켜짐」이다.

## 검증에서 알게 된 것(2026-09-13)
| 시험 | 파일 | 결과 |
|---|---|---|
| 광수용체 직접 자극 | `sweep_gain.py` | R1-6을 283Hz까지 올려도 신호가 라미나·수질 입구에서 멈춤. 루밍에 LC4·LPLC2 반응 0. **앞단 시각 회로는 이 LIF 모델로 재현 안 됨**(실제로는 연속 전위·억제 해제로 신호 전달, 광수용체 부호도 모델에선 반대) |
| VPN 직접 자극 | `vpn_probe.py` | LC4 150Hz → 거대섬유 112Hz, LPLC2 → 162Hz. **루밍 도주 경로 재현**. LC11 → oDN1 28Hz |
| 시각→도파민 경로 조회 | `vpn_paths.py` | 버섯체로 가는 VPN: aMe12(케니언 시냅스 2,888) 등. PAM까지 두 단계 경로 최강 aMe12·aMe26, PPL1은 LC9·LC6 |
| 당 자극 | `dan_probe.py` | 당 감각뉴런 23개 150Hz → MN9 70Hz(섭식 재현), **PAM 0Hz**. 즉 PAM이 안 켜지는 건 시각 탓이 아니라 이 모델의 한계 |
| 특징 → 뇌 | `brain_eval.py` | 루밍 → 거대섬유 190Hz, 가로 줄무늬 40Hz(문턱 아래), 회색 0. 실제 릴스 6개: 거대섬유 0~140Hz, oDN1 43~58Hz, PPL1 0.56~0.94Hz, PAM ≈ 0 |

그래서 도파민(PAM·PPL1)은 매 릴스 기록하지만, 좋아요는 대부분 4번(다가가기 명령)에서 나온다.

## 근사와 한계
- 겹눈 색: RGB에 자외선이 없어 pale·yellow 채널은 근사.
- 광수용체 → 낱눈 배정(`retina_map.py`)은 위치 좌표 근사. VPN 모드에서는 쓰지 않음.
- 시각 특징은 뉴런 시뮬레이션이 아니라 문헌 모델 근사. VPN 무리 안의 수용장(어느 뉴런이 시야 어디를 보는지)은 좌우 눈만 구분.
- 1초·10fps만 본다. 초파리 시각은 훨씬 빠르다(깜빡임 융합 약 200Hz).
- 도파민은 화학물질 확산·학습이 없는 「발화 여부」만.

## 실행
```
PY=~/flybrain/shiu-brain-model/.venv/bin/python
$PY brain/visual/brain_eval.py --synthetic gray,loom,bars,vbars,dot,flicker     # 인공 자극 검증
~/flybrain/insta/.venv/bin/python brain/visual/capture_clips.py 24 1.2 10       # 기준 릴스 모으기(좋아요 안 누름)
$PY brain/visual/calibrate.py                                                    # 문턱 → calibration.json
~/flybrain/insta/.venv/bin/python live/hub.py                                    # 뇌 판정 + 가짜 좋아요
```
