# source 해서 쓰는 환경 설정. wrapper(scripts/factory)와 같은 분석 환경을 현재 셸에 만든다.
#   source scripts/env.sh
# 이후 python · jupyter 를 직접 실행할 수 있다. Isaac Sim 은 $ISAACSIM/python.sh 로 실행한다.
_WS="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
if [[ ! -f "$_WS/.venv/bin/activate" ]]; then
  echo "분석 환경(.venv)이 없습니다. 먼저 실행: ./scripts/factory setup" >&2; return 1 2>/dev/null || exit 1
fi
[[ "${VIRTUAL_ENV:-}" == "$_WS/.venv" ]] || source "$_WS/.venv/bin/activate"
export AI_MANUFAC_HOME="$_WS" PYTHONNOUSERSITE=1 ISAACSIM="${ISAACSIM_PATH:-$HOME/isaacsim}"
echo "factory env: $_WS, python $(python -c 'import sys;print(sys.version.split()[0])'), Isaac Sim $([[ -x $ISAACSIM/python.sh ]] && echo $ISAACSIM || echo 없음)"
