#!/bin/zsh
# 초파리 공장 유니티 맥 빌드(배치 모드) → factory/unity/FlyFactory/Build/FlyFactory.app
#   zsh factory/tools/build_mac.sh
#   UNITY=<유니티 실행 파일> LOG=<빌드 기록> 로 바꿀 수 있다.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
UNITY=${UNITY:-/Applications/Unity/Hub/Editor/6000.0.82f1/Unity.app/Contents/MacOS/Unity}
LOG=${LOG:-$ROOT/factory/unity/FlyFactory/Logs/build_mac.log}
mkdir -p "$(dirname "$LOG")"
"$UNITY" -batchmode -projectPath "$ROOT/factory/unity/FlyFactory" -executeMethod FactoryBuild.BuildMac -quit -logFile "$LOG"
echo "UNITY_BUILD_EXIT $?"
grep -E "BUILD_RESULT|error CS" "$LOG" | head -20
