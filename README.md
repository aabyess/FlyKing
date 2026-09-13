# FlyKing

초파리(Drosophila melanogaster) 커넥톰 뇌 시뮬레이션을 Blender로 만든 초파리 몸에 연결하는 개인 프로젝트.

- `body/` — Blender 초파리 몸(뼈 61개, NeuroMechFly v2 이름 체계) 생성·리깅·관절각 적용 스크립트
- `brain/shiu/` — Shiu et al. 2024 전뇌 LIF 모델(FlyWire v783) 실행 스크립트

대용량 데이터와 외부 저장소는 `~/flybrain/`에 두고 git에는 넣지 않는다. 구성·실행 방법은 [CLAUDE.md](CLAUDE.md).

## 출처
- Shiu, P. K. et al. (2024). A Drosophila computational brain model reveals sensorimotor processing. *Nature*. — github.com/philshiu/Drosophila_brain_model
- FlyWire connectome v783 — flywire.ai
- NeuroMechFly v2 / flygym — github.com/NeLy-EPFL/flygym
- Male CNS connectome v1.0 (Janelia·Google, CC-BY) — male-cns.janelia.org
