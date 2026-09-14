#!/bin/zsh
# 뇌 서버 + 공장 게임을 함께 켜서 화면을 찍고, 몸짓·방어 기록과 오류를 요약한다(검증용).
#   zsh factory/tools/play_check.sh <빈 캡처 폴더> <기다릴 캡처 장수> [게임 인자...]
# 예:
#   작업대·몸짓·갑옷  zsh factory/tools/play_check.sh ~/flybrain/factory/shots_work 8 -shotPlan "45:all,52:s0,58:s3,64:s3,70:s2,76:s1" -testGear 1
#   병정 방어        zsh factory/tools/play_check.sh ~/flybrain/factory/shots_defense 8 -shotPlan "18:all,26:fight" -testSoldiers 1 -testGear 1
#   게임 오버 화면    zsh factory/tools/play_check.sh ~/flybrain/factory/shots_gameover 3 -shotPlan "8:all" -testSoldiers 1 -testFactoryHp 10
# 게임 인자: -shotPlan "초:보기,…"(보기 = all · sN 작업대 · fight 싸움 · gateN 문), 계획이 끝나면 몸짓 사건(event_*.png)을 찍고 스스로 꺼진다.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SHOTS=$1
WANT=$2
shift 2
if [ -n "$(ls -A "$SHOTS" 2>/dev/null)" ]; then
  echo "캡처 폴더가 비어 있지 않아요: $SHOTS (빈 폴더를 주세요)"
  exit 1
fi
mkdir -p "$SHOTS"
PY=${PY:-$HOME/flybrain/shiu-brain-model/.venv/bin/python}
PLAYER="$HOME/Library/Logs/DefaultCompany/초파리 공장/Player.log"
LOG="$SHOTS/brain_server.log"
"$PY" "$ROOT/factory/server/brain_server.py" --workers 2 --port 8790 > "$LOG" 2>&1 &
SRV=$!
for i in {1..120}; do [ $(grep -c WORKER_READY "$LOG") -ge 2 ] && break; sleep 1; done
echo "WORKERS_READY $(grep -c WORKER_READY "$LOG")"
open -n "$ROOT/factory/unity/FlyFactory/Build/FlyFactory.app" --args -shotDir "$SHOTS" "$@"
for i in {1..240}; do [ $(ls "$SHOTS"/*.png 2>/dev/null | wc -l) -ge $WANT ] && break; sleep 1; done
sleep 5
echo "SHOTS $(ls "$SHOTS" | grep png | tr '\n' ' ')"
grep -E "^(BODY|DEFENSE)" "$PLAYER" | head -40
grep -E "Exception|error CS" "$PLAYER" | head -10
echo "SERVER errors $(grep -c JOB_ERROR "$LOG")"
pkill -f FlyFactory.app/Contents/MacOS 2>/dev/null
kill $SRV 2>/dev/null
sleep 1
pkill -f "factory/server/brain_server.py" 2>/dev/null
echo "PLAY_CHECK_DONE"
