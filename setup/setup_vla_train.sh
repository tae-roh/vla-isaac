#!/usr/bin/env bash
# =============================================================================
# setup/setup_vla_train.sh
# Phase 3 (SFT) 환경 구축 — H100/A100 인스턴스
#
# ★ 이 인스턴스에는 Isaac Sim 을 설치하지 않는다.
#   계획서 §3-1 표: "VLA SFT — Isaac Sim 구동 ✗, 렌더 ✗".
#   그래서 RT 코어 제약이 적용되지 않고, 전 구간에서 유일하게 H100/A100 을
#   쓸 수 있는 구간이다. torch 도 Isaac Sim 핀(2.7.0)이 아니라 openvla-oft 가
#   검증한 2.2.0 을 쓴다 → constraints.vla-train.txt (constraints.txt 아님)
#
# 사용법:
#   chmod +x setup/setup_vla_train.sh && ./setup/setup_vla_train.sh
#   FULL_SMOKE=1 ./setup/setup_vla_train.sh    # 모델 로드까지 검증 (7B 다운로드 발생)
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${HOME}/env_vla_train}"
OFT_DIR="${OFT_DIR:-${HOME}/openvla-oft}"
SIMPLEVLA_DIR="${SIMPLEVLA_DIR:-${HOME}/SimpleVLA-RL}"
CONSTRAINTS="${REPO_ROOT}/env/constraints.vla-train.txt"
FULL_SMOKE="${FULL_SMOKE:-0}"

TOTAL=7
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
info "이 인스턴스는 Isaac Sim 을 쓰지 않으므로 RT 코어 없는 카드(H100/A100)도 정상이다."

[[ -f "${CONSTRAINTS}" ]] || die "constraints 파일 없음: ${CONSTRAINTS}"

FREE_GB=$(df -BG --output=avail "${HOME}" | tail -1 | tr -dc '0-9')
(( FREE_GB < 200 )) && warn "홈 여유 공간 ${FREE_GB}GB — 7B 체크포인트가 누적되므로 200GB 권장."

# =============================================================================
STAGE="2. 시스템 패키지"
log 2 "시스템 패키지"
# =============================================================================
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y software-properties-common cmake build-essential git curl \
    tmux rsync ninja-build
if ! command -v python3.10 >/dev/null 2>&1; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
fi
info "$(python3.10 --version) — openvla-oft 는 3.10 기준"

# =============================================================================
STAGE="3. 가상환경"
log 3 "가상환경: ${VENV_DIR}"
# =============================================================================
[[ -d "${VENV_DIR}" ]] || python3.10 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel packaging

# =============================================================================
STAGE="4. PyTorch"
log 4 "PyTorch 2.2.0 (cu121)"
# =============================================================================
# flash-attn 2.5.5 의 프리빌트 휠이 이 torch 버전에 맞춰져 있다. 올리면 소스 빌드가
# 발생해 30분 이상 소모된다 — 3~4일 일정에서 그럴 여유가 없다.
pip install -c "${CONSTRAINTS}" torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# =============================================================================
STAGE="5. openvla-oft + 의존성"
log 5 "openvla-oft clone + 의존성"
# =============================================================================
if [[ ! -d "${OFT_DIR}" ]]; then
    git clone https://github.com/moojink/openvla-oft.git "${OFT_DIR}"
else
    info "openvla-oft 이미 clone 됨 — 건너뜀"
fi

pip install -c "${CONSTRAINTS}" -r "${REPO_ROOT}/env/vla-train.requirements.txt"
pip install -c "${CONSTRAINTS}" -e "${OFT_DIR}"

# flash-attn 은 torch 설치 이후, --no-build-isolation 으로 설치해야 한다.
# (빌드 격리를 켜면 torch 를 못 찾아 실패한다)
if python -c "import flash_attn" 2>/dev/null; then
    info "flash-attn 이미 설치됨 — 건너뜀"
else
    pip install "flash-attn==2.5.5" --no-build-isolation
fi

# SimpleVLA-RL 은 SFT 단계에서 직접 쓰지 않지만, 벤더링된 openvla_oft 모델 코드와
# 우리 체크포인트의 호환을 Day 3 에 대조해야 하므로 미리 받아 둔다.
if [[ ! -d "${SIMPLEVLA_DIR}" ]]; then
    git clone https://github.com/PRIME-RL/SimpleVLA-RL.git "${SIMPLEVLA_DIR}"
else
    info "SimpleVLA-RL 이미 clone 됨 — 건너뜀"
fi

# =============================================================================
STAGE="6. 상수셋 확인"
log 6 "openvla-oft 상수셋 확인 (ACTION_DIM=7 / PROPRIO_DIM=8 / CHUNK=8)"
# =============================================================================
# prismatic/vla/constants.py 의 detect_robot_platform() 은 실행 커맨드에 "libero" 가
# 있어야 LIBERO 상수셋을 고른다. 없으면 ALOHA(14/14/25)로 잡히면서 SFT 가 조용히
# 잘못된 차원으로 학습된다 — Phase 4 에서 원인불명 붕괴로 나타나는 종류의 버그다.
# 우리 학습 진입 스크립트 이름에 libero 를 넣어 두었지만, 확인은 스모크가 한다.
CONST_FILE="${OFT_DIR}/prismatic/vla/constants.py"
if [[ -f "${CONST_FILE}" ]]; then
    info "상수 파일: ${CONST_FILE}"
    grep -nE "NUM_ACTIONS_CHUNK|ACTION_DIM|PROPRIO_DIM" "${CONST_FILE}" | head -20 || true
    warn "위 값이 LIBERO 셋(8/7/8)으로 잡히지 않으면 스모크 테스트가 잡아낸다.
       그때는 이 파일에서 LIBERO 상수를 무조건 쓰도록 직접 편집할 것."
fi

pip install -c "${CONSTRAINTS}" -e "${REPO_ROOT}/source" 2>/dev/null || \
    info "태스크 패키지는 이 환경에 불필요 (Isaac Lab 의존) — 건너뜀"

# =============================================================================
STAGE="7. 스모크 테스트"
log 7 "스모크 테스트"
# =============================================================================
echo
info "pip check 결과 (참고용 — 종료 코드 미반영)"
pip check || true

cd "${REPO_ROOT}"
SMOKE_ARGS=()
[[ "${FULL_SMOKE}" == "1" ]] && SMOKE_ARGS+=(--full)

if python env/smoke/check_vla_train.py "${SMOKE_ARGS[@]}"; then
    mkdir -p env/locks
    pip freeze > env/locks/vla-train.lock.txt
    info "lock 박제: env/locks/vla-train.lock.txt"
else
    die "스모크 테스트 실패 — lock 을 갱신하지 않았다."
fi

trap - ERR
cat <<DONE

=============================================================
 vla-train 환경 준비 완료.

   source ${VENV_DIR}/bin/activate
   cd ${REPO_ROOT}

 다음 단계 — 데이터셋을 받고 SFT 시작 (tmux 안에서 실행할 것):
   tmux new -s sft
   bash scripts/sft/run_sft_libero_spec.sh

 openvla-oft : ${OFT_DIR}
 SimpleVLA-RL: ${SIMPLEVLA_DIR}
=============================================================
DONE
