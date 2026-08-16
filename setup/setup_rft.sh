#!/usr/bin/env bash
# =============================================================================
# setup/setup_rft.sh
# Phase 4 (RFT) 환경 구축 — 다중 L40S 인스턴스
#
# ★ 아키텍처: 프로세스 분리 (env/rft.requirements.txt 헤더의 결정 참조)
#
#   이 스크립트는 **두 개의 venv** 를 만든다:
#     1) ~/env_isaaclab  : Isaac Sim 5.1.0 + torch 2.7.0  — 롤아웃 워커용
#                          (setup_isaaclab.sh 를 그대로 재사용)
#     2) ~/env_rft       : veRL + vLLM                    — 학습/정책용
#
#   둘은 같은 venv 에 합치지 않는다. SimpleVLA-RL 의 rob_rollout.py 가 이미
#   시뮬레이터를 별도 프로세스 + Queue 로 격리해 돌리고 있어, 그 자리에
#   Isaac Sim 워커를 끼우는 것이 자연스럽고 torch 충돌이 구조적으로 사라진다.
#
# 사용법:
#   chmod +x setup/setup_rft.sh && ./setup/setup_rft.sh
#   SKIP_ISAACLAB=1 ./setup/setup_rft.sh    # isaaclab venv 가 이미 있을 때
#   FULL_SMOKE=1 ./setup/setup_rft.sh       # 워커 왕복까지 검증
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RFT_VENV="${RFT_VENV:-${HOME}/env_rft}"
ISAACLAB_VENV="${ISAACLAB_VENV:-${HOME}/env_isaaclab}"
SIMPLEVLA_DIR="${SIMPLEVLA_DIR:-${HOME}/SimpleVLA-RL}"
CONSTRAINTS="${REPO_ROOT}/env/constraints.rft.txt"
SKIP_ISAACLAB="${SKIP_ISAACLAB:-0}"
FULL_SMOKE="${FULL_SMOKE:-0}"

TOTAL=6
log()  { echo -e "\n\033[1;32m[SETUP ${1}/${TOTAL}]\033[0m ${2}"; }
info() { echo -e "\033[1;36m[INFO]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
die()  { echo -e "\033[1;31m[FAIL]\033[0m $*" >&2; exit 1; }

trap 'echo -e "\n\033[1;31m[FAIL]\033[0m ${STAGE:-알 수 없는 단계} 에서 중단됨 (line $LINENO)" >&2' ERR

# =============================================================================
STAGE="1. 사전 확인"
log 1 "사전 환경 확인"
# =============================================================================
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi 없음."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
case "${GPU_NAME}" in
    *A100*|*H100*|*H20*)
        die "RT 코어 없는 GPU (${GPU_NAME}) — RFT 는 오프스크린 렌더가 필수라 불가.
     라이브스트림은 불필요하지만 렌더는 필요하다. 둘은 별개다 (계획서 §Phase4b)." ;;
esac
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
info "GPU ${NUM_GPUS}개 (${GPU_NAME})"
(( NUM_GPUS < 2 )) && warn "GPU 1개 — 시뮬 렌더와 학습이 한 카드를 공유한다. num_envs 를 줄일 것."

[[ -f "${CONSTRAINTS}" ]] || die "constraints 파일 없음: ${CONSTRAINTS}"

# =============================================================================
STAGE="2. isaaclab venv (롤아웃 워커용)"
log 2 "isaaclab venv — 롤아웃 워커가 이 인터프리터로 실행된다"
# =============================================================================
if [[ "${SKIP_ISAACLAB}" == "1" ]]; then
    info "SKIP_ISAACLAB=1 — 건너뜀"
    [[ -x "${ISAACLAB_VENV}/bin/python" ]] || die "그런데 ${ISAACLAB_VENV} 가 없다."
elif [[ -x "${ISAACLAB_VENV}/bin/python" ]] && \
     "${ISAACLAB_VENV}/bin/python" -c "import isaacsim" 2>/dev/null; then
    info "isaaclab venv 이미 준비됨 — 건너뜀"
else
    info "setup_isaaclab.sh 를 호출한다 (수십 분 소요)"
    bash "${REPO_ROOT}/setup/setup_isaaclab.sh"
fi

# =============================================================================
STAGE="3. rft venv"
log 3 "rft venv: ${RFT_VENV}"
# =============================================================================
# SimpleVLA-RL 은 CUDA 12.4 / Python 3.10 기준으로 검증되어 있다.
if ! command -v python3.10 >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update -y
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
fi
sudo apt-get install -y tmux rsync ninja-build cmake build-essential || true

[[ -d "${RFT_VENV}" ]] || python3.10 -m venv "${RFT_VENV}"
# shellcheck disable=SC1091
source "${RFT_VENV}/bin/activate"
pip install --upgrade pip setuptools wheel packaging

# =============================================================================
STAGE="4. SimpleVLA-RL"
log 4 "SimpleVLA-RL clone + 설치"
# =============================================================================
if [[ ! -d "${SIMPLEVLA_DIR}" ]]; then
    git clone https://github.com/PRIME-RL/SimpleVLA-RL.git "${SIMPLEVLA_DIR}"
else
    info "이미 clone 됨 — 건너뜀"
fi

# ⚠ 여기가 Day 2 의 실측 지점이다.
#   constraints.rft.txt 의 torch/vllm 핀은 잠정값이고, 실제 기준은 저장소의
#   SETUP.md 다. 아래에서 그 파일을 띄워 주니 반드시 눈으로 확인하고,
#   어긋나면 constraints.rft.txt 를 고친 뒤 이 스크립트를 다시 돌릴 것.
if [[ -f "${SIMPLEVLA_DIR}/SETUP.md" ]]; then
    echo
    warn "===== SimpleVLA-RL/SETUP.md 의 버전 요구사항을 확인할 것 ====="
    grep -nE "torch|vllm|cuda|python|flash" "${SIMPLEVLA_DIR}/SETUP.md" | head -30 || true
    warn "위와 env/constraints.rft.txt 가 어긋나면 constraints 를 먼저 고칠 것 (§4-7)."
    echo
fi

pip install -c "${CONSTRAINTS}" -r "${REPO_ROOT}/env/rft.requirements.txt"
pip install -c "${CONSTRAINTS}" -e "${SIMPLEVLA_DIR}"

# flash-attn 은 torch 이후, 빌드 격리 없이.
python -c "import flash_attn" 2>/dev/null || \
    pip install flash-attn --no-build-isolation

# =============================================================================
STAGE="5. 어댑터 배선"
log 5 "Isaac Lab 롤아웃 어댑터 배선 확인"
# =============================================================================
# 워커는 isaaclab venv 의 python 으로 실행되지만, 코드는 이 저장소에 있다.
# 두 venv 가 같은 저장소를 보게 하고, 워커 인터프리터 경로를 파일로 남긴다.
WORKER_PY="${ISAACLAB_VENV}/bin/python"
[[ -x "${WORKER_PY}" ]] || die "워커 인터프리터가 없다: ${WORKER_PY}"

mkdir -p "${REPO_ROOT}/rft/runtime"
cat > "${REPO_ROOT}/rft/runtime/paths.env" <<EOF
# setup_rft.sh 가 생성. 어댑터가 워커를 띄울 때 읽는다.
ISAACLAB_PYTHON=${WORKER_PY}
RFT_PYTHON=${RFT_VENV}/bin/python
REPO_ROOT=${REPO_ROOT}
SIMPLEVLA_DIR=${SIMPLEVLA_DIR}
EOF
info "경로 기록: rft/runtime/paths.env"

# 워커 쪽 venv 에도 태스크 패키지가 설치되어 있어야 한다.
"${WORKER_PY}" -m pip install -c "${REPO_ROOT}/env/constraints.txt" \
    -e "${REPO_ROOT}/source" >/dev/null
info "태스크 패키지를 isaaclab venv 에 편집 가능 설치 완료"

# =============================================================================
STAGE="6. 스모크 테스트"
log 6 "스모크 테스트"
# =============================================================================
echo
info "pip check 결과 (참고용 — 종료 코드 미반영)"
pip check || true

cd "${REPO_ROOT}"
SMOKE_ARGS=(--isaaclab-python "${WORKER_PY}")
[[ "${FULL_SMOKE}" == "1" ]] && SMOKE_ARGS+=(--full)

if python env/smoke/check_rft.py "${SMOKE_ARGS[@]}"; then
    mkdir -p env/locks
    pip freeze > env/locks/rft.lock.txt
    info "lock 박제: env/locks/rft.lock.txt"
else
    die "스모크 테스트 실패 — lock 을 갱신하지 않았다."
fi

trap - ERR
cat <<DONE

=============================================================
 rft 환경 준비 완료 (venv 2개).

   학습/정책 : ${RFT_VENV}
   롤아웃워커: ${ISAACLAB_VENV}   ← Isaac Sim 은 여기에만 있다

 다음 단계 — 워커 왕복부터 반드시 확인 (이게 최대 리스크의 조기 검증 지점):
   source ${RFT_VENV}/bin/activate
   python env/smoke/check_rft.py --full

 통과하면 관측 스펙 대조 → RFT 본 학습:
   python scripts/dump_obs_reference.py --compare datasets/obs_reference
   bash rft/run_rft_rigid.sh

 모니터링은 포트 개방 없이 SSH 터널로:
   ssh -L 6006:localhost:6006 <서버>
=============================================================
DONE
