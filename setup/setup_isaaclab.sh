#!/usr/bin/env bash
# =============================================================================
# setup/setup_isaaclab.sh
# Phase 1 · 2 · 4b 용 환경 구축 (L40S, Ubuntu 22.04)
#
# 기존 D:\vla-proj\setup_isaaclab.sh 에서 바뀐 점 (계획서 §4-4 반영):
#   1) ./isaaclab.sh --install  →  --install none
#      학습 프레임워크(sb3/rl_games/skrl)를 아예 끌어오지 않아 충돌을 원천 회피한다.
#      초기 세팅에서 겪은 연쇄 충돌의 상당수가 여기서 발생했다.
#   2) 모든 pip install 에 -c env/constraints.txt 를 건다 (§4-7).
#   3) 우리 태스크 패키지(source/vla_isaac_tasks)를 편집 가능 설치한다.
#   4) 마지막에 스모크 테스트를 자동 실행하고, 통과 시 lock 을 박제한다.
#
# 사용법:
#   chmod +x setup/setup_isaaclab.sh && ./setup/setup_isaaclab.sh
#   FULL_SMOKE=1 ./setup/setup_isaaclab.sh     # Isaac Sim 기동까지 검증 (수 분 추가)
#
# 멱등성: 재실행하면 이미 끝난 단계는 건너뛴다.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${HOME}/env_isaaclab}"
ISAACLAB_DIR="${ISAACLAB_DIR:-${HOME}/IsaacLab}"
ISAACSIM_VERSION="5.1.0"
TORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
CONSTRAINTS="${REPO_ROOT}/env/constraints.txt"
FULL_SMOKE="${FULL_SMOKE:-0}"

# SkillGen(cuRobo) 설치 여부. 기본 켜짐.
#   USE_SKILLGEN=0 ./setup/setup_isaaclab.sh   로 건너뛸 수 있다.
# 설치에 실패해도 전체 세팅은 계속 진행된다 — MimicGen 경로가 그대로 살아 있으므로
# 여기서 스크립트를 죽이면 얻는 것 없이 Day 1 오전을 잃는다.
USE_SKILLGEN="${USE_SKILLGEN:-1}"
# Isaac Lab 문서가 지정한 검증 커밋. USD 를 충돌체로 파싱할 때 필요한
# quad-face 삼각분할 지원이 들어 있어, 최신으로 바꾸면 호환성 문제가 난다.
CUROBO_COMMIT="ebb71702f3f70e767f40fd8e050674af0288abe8"

TOTAL=9
log()  { echo -e "\n\033[1;32m[SETUP ${1}/${TOTAL}]\033[0m ${2}"; }
info() { echo -e "\033[1;36m[INFO]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
die()  { echo -e "\033[1;31m[FAIL]\033[0m $*" >&2; exit 1; }

trap 'echo -e "\n\033[1;31m[FAIL]\033[0m ${STAGE:-알 수 없는 단계} 에서 중단됨 (line $LINENO)" >&2' ERR

# =============================================================================
STAGE="1. 사전 확인"
log 1 "사전 환경 확인"
# =============================================================================
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi 없음. GPU 인스턴스가 맞는지 확인할 것."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
case "${GPU_NAME}" in
    *A100*|*H100*|*H20*)
        die "RT 코어가 없는 GPU (${GPU_NAME}) — Isaac Sim 렌더 불가.
     headless 여도 오프스크린 렌더는 RTX 렌더러를 쓰므로 우회 불가하다.
     L40S / L40 / RTX 6000 Ada / RTX 4090 등을 사용할 것 (계획서 §3-1)." ;;
    *) info "GPU: ${GPU_NAME} — RT 코어 보유 카드로 판단" ;;
esac

GLIBC_VER=$(ldd --version | head -n1 | grep -oP '\d+\.\d+$')
GLIBC_MAJOR=${GLIBC_VER%%.*}; GLIBC_MINOR=${GLIBC_VER##*.}
if (( GLIBC_MAJOR < 2 || (GLIBC_MAJOR == 2 && GLIBC_MINOR < 35) )); then
    die "GLIBC ${GLIBC_VER} < 2.35. Ubuntu 22.04+ 가 필요하다."
fi
info "GLIBC ${GLIBC_VER} OK"

FREE_GB=$(df -BG --output=avail "${HOME}" | tail -1 | tr -dc '0-9')
(( FREE_GB < 150 )) && warn "홈 여유 공간 ${FREE_GB}GB — 데이터셋까지 감안하면 150GB 이상 권장."

[[ -f "${CONSTRAINTS}" ]] || die "constraints 파일 없음: ${CONSTRAINTS}"

# =============================================================================
STAGE="2. 시스템 패키지"
log 2 "시스템 패키지 (Python 3.11, build tools, libglu1-mesa)"
# =============================================================================
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
# libglu1-mesa 누락 시 libGLU.so.1 not found → Iray 렌더러 로드 실패 (계획서 §4-5)
sudo apt-get install -y software-properties-common cmake build-essential git curl \
    libglu1-mesa tmux rsync

if ! command -v python3.11 >/dev/null 2>&1; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi
info "$(python3.11 --version) — Isaac Sim 5.X 는 3.11 필수 (3.10/3.12 불가)"

# =============================================================================
STAGE="3. 가상환경"
log 3 "가상환경: ${VENV_DIR}"
# =============================================================================
if [[ ! -d "${VENV_DIR}" ]]; then
    python3.11 -m venv "${VENV_DIR}"
else
    info "이미 존재 — 건너뜀"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip

# =============================================================================
STAGE="4. Isaac Sim + PyTorch"
log 4 "Isaac Sim ${ISAACSIM_VERSION} + PyTorch ${TORCH_VERSION} (cu128) — 수십 GB, 오래 걸림"
# =============================================================================
# 버전까지 확인한다. `import isaacsim` 성공만으로 건너뛰면, 직접 설치해 둔
# 다른 버전이 그대로 남아 조용히 진행된다 — constraints 는 5.1.0 기준이므로
# 버전이 다르면 나중에 원인 불명 오류로 나타난다.
INSTALLED_ISAACSIM="$(pip show isaacsim 2>/dev/null | awk '/^Version:/{print $2}')"
if [[ "${INSTALLED_ISAACSIM}" == "${ISAACSIM_VERSION}" ]]; then
    info "isaacsim ${ISAACSIM_VERSION} 이미 설치됨 — 건너뜀"
else
    [[ -n "${INSTALLED_ISAACSIM}" ]] && \
        warn "isaacsim ${INSTALLED_ISAACSIM} 가 설치되어 있다 (기대: ${ISAACSIM_VERSION}) — 교체한다."
    pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
        --extra-index-url https://pypi.nvidia.com
fi

# torch 는 반드시 cu128 인덱스에서. 기본 PyPI 에서 받으면 다른 CUDA 빌드가 덮어쓴다.
# torchaudio 를 처음부터 함께 설치 (누락 시 isaacsim-core 의존성 경고).
pip install -U "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"

export OMNI_KIT_ACCEPT_EULA=YES
grep -q "OMNI_KIT_ACCEPT_EULA" "${HOME}/.bashrc" || \
    echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> "${HOME}/.bashrc"

# =============================================================================
STAGE="5. Isaac Lab"
log 5 "Isaac Lab clone + install none"
# =============================================================================
if [[ ! -d "${ISAACLAB_DIR}" ]]; then
    git clone https://github.com/isaac-sim/IsaacLab.git --branch main "${ISAACLAB_DIR}"
else
    info "이미 clone 됨 — 건너뜀: ${ISAACLAB_DIR}"
fi

# 디렉토리는 있는데 Isaac Lab 이 아닌 경우를 여기서 잡는다.
# ISAACLAB_DIR 을 잘못 지정하면 아래 ./isaaclab.sh 가 "No such file" 로 죽는데,
# 그 메시지만으로는 경로 문제인지 설치 문제인지 구분이 안 된다.
[[ -x "${ISAACLAB_DIR}/isaaclab.sh" ]] || die \
    "${ISAACLAB_DIR} 는 Isaac Lab 체크아웃이 아니다 (isaaclab.sh 없음).
     경로가 다르면 ISAACLAB_DIR 로 지정할 것:
       ISAACLAB_DIR=~/workspace/IsaacLab ./setup/setup_isaaclab.sh"

# 이후 페이즈의 명령들이 \$ISAACLAB_DIR 을 쓴다 (docs/RUNBOOK.md).
# 경로가 홈 바로 아래가 아닐 수 있으므로 셸에 남겨 둔다.
if grep -q "^export ISAACLAB_DIR=" "${HOME}/.bashrc" 2>/dev/null; then
    sed -i "s|^export ISAACLAB_DIR=.*|export ISAACLAB_DIR=${ISAACLAB_DIR}|" "${HOME}/.bashrc"
else
    echo "export ISAACLAB_DIR=${ISAACLAB_DIR}" >> "${HOME}/.bashrc"
fi
info "ISAACLAB_DIR=${ISAACLAB_DIR} 를 ~/.bashrc 에 기록"

pushd "${ISAACLAB_DIR}" >/dev/null
# ★ --install none 이 핵심이다. all 로 설치하면 sb3/rl_games/skrl 이 isaacsim 의
#   고정 핀을 덮어쓰며 torch/starlette/click/psutil 연쇄 충돌을 일으킨다 (계획서 §4-4).
#   본 프로젝트는 Isaac Lab 을 환경·Mimic 용도로만 쓰므로 학습 프레임워크가 불필요하다.
./isaaclab.sh --install none
popd >/dev/null

# =============================================================================
STAGE="6. 프로젝트 의존성"
log 6 "프로젝트 의존성 + 태스크 패키지 설치 (constraints 경유)"
# =============================================================================
pip install -c "${CONSTRAINTS}" -r "${REPO_ROOT}/env/isaaclab.requirements.txt"

# 우리 태스크 패키지 — 편집 가능 설치라 코드를 고쳐도 재설치가 필요 없다.
pip install -c "${CONSTRAINTS}" -e "${REPO_ROOT}/source"

# ★ gym 자동 등록.
#   Isaac Lab 의 도구 스크립트들(record_demos / replay_demos / annotate_demos /
#   generate_dataset)은 `import isaaclab_tasks` 와 `import isaaclab_mimic.envs` 만
#   한다. 우리 패키지는 아무도 import 하지 않으므로 gym.make("VlaPlace-v0") 이
#   NameNotFound 로 죽는다.
#   .pth 파일에 적힌 `import` 줄은 인터프리터 기동 시 site 모듈이 실행해 준다
#   (stdlib 표준 동작). 이 한 줄이면 상류 스크립트를 건드리지 않고 등록이 끝난다.
#   패키지 __init__ 은 gymnasium 과 spec 만 import 하고 entry_point 는 문자열이라
#   Isaac Sim 을 끌어오지 않는다 → 기동 비용도 거의 없다.
#   경로는 sysconfig 로 잡는다. site.getsitepackages()[0] 은 venv 안에서도
#   베이스 파이썬의 site-packages 를 돌려줄 수 있는데, 거기에 쓰면 venv 실행 시
#   로드되지 않아 조용히 아무 일도 일어나지 않는다.
SITE_DIR="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "import vla_isaac_tasks" > "${SITE_DIR}/vla_isaac_tasks.pth"
info "gym 자동 등록: ${SITE_DIR}/vla_isaac_tasks.pth"

# 실제로 먹었는지 확인. 여기서 걸러야 Day 1 오후에 record_demos.py 가
# NameNotFound 로 죽는 것을 막는다.
REGISTERED="$(python -c "import gymnasium as gym; print(len([k for k in gym.registry if k.startswith('VlaPlace')]))")"
# 기본 3종 + 클리어런스 사다리 4단계 × 3종 = 15종.
if [[ "${REGISTERED}" -ge 3 ]]; then
    info "gym 등록 확인: VlaPlace 태스크 ${REGISTERED}종"
else
    warn "gym 자동 등록 실패 (VlaPlace ${REGISTERED}종). 다음으로 원인을 확인할 것:
       python -c 'import vla_isaac_tasks'          # 임포트 자체가 되는지
       python -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])'"
fi

# =============================================================================
STAGE="7. 의존성 충돌 정리"
log 7 "의존성 충돌 정리 (계획서 §4-3 확정 핀 재확인)"
# =============================================================================
# --install none 으로 대부분 회피되지만, isaacsim 이 끌어온 것들 사이의 잔여 모순을 정리한다.
pip install -c "${CONSTRAINTS}" "click>=8.2,<9" "typing_extensions>=4.15" \
    "starlette==0.49.1" "psutil>=7"
pip install -c "${CONSTRAINTS}" -U fastapi

echo
info "pip check 결과 (참고용 — 종료 코드에 반영하지 않는다, §4-6 규약 6)"
pip check || true
warn "isaacsim-kernel 의 click/psutil/typing_extensions 경고는 정상이다.
       이 스택에서 pip check 는 완전히 깨끗해질 수 없다 (§4-1 원칙 4).
       환경 검증 기준은 아래 스모크 테스트다."

# =============================================================================
STAGE="8. cuRobo (SkillGen)"
log 8 "cuRobo 설치 — SkillGen 데이터 생성용"
# =============================================================================
# SkillGen 은 서브태스크 사이의 자유공간을 cuRobo 로 계획해 충돌 없는 궤적을 만든다.
# MimicGen 의 선형 보간보다 생성 성공률이 높고, 그게 Phase 2 최대 리스크
# (계획서 §6 "Mimic 생성 성공률 저조 <10%") 에 대한 가장 강한 대응이다.
#
# ★ 이 단계는 실패해도 스크립트를 중단시키지 않는다.
#   MimicGen 경로가 그대로 살아 있어서 --use_skillgen 플래그만 빼면 진행 가능하고,
#   여기서 죽이면 Day 1 오전을 통째로 잃는다. 실패 시 무엇을 하면 되는지만 알린다.
SKILLGEN_OK=0
if [[ "${USE_SKILLGEN}" != "1" ]]; then
    info "USE_SKILLGEN=0 — 건너뜀. MimicGen 방식으로 데이터를 생성하게 된다."
elif python -c "import curobo" 2>/dev/null; then
    info "cuRobo 이미 설치됨 — 건너뜀"
    SKILLGEN_OK=1
else
    # 문서의 설치 명령은 conda 전제(`$CONDA_PREFIX`)지만 우리는 venv 다.
    # CUDA 툴킷을 apt 로 깔고 CUDA_HOME 을 직접 잡는다.
    if ! command -v nvcc >/dev/null 2>&1; then
        info "CUDA 툴킷 설치 (nvcc 없음)"
        sudo apt-get install -y cuda-toolkit-12-8 || \
            sudo apt-get install -y nvidia-cuda-toolkit || \
            warn "CUDA 툴킷 설치 실패 — cuRobo 빌드가 실패할 수 있다."
    fi

    CUDA_HOME_GUESS="${CUDA_HOME:-/usr/local/cuda-12.8}"
    [[ -d "${CUDA_HOME_GUESS}" ]] || CUDA_HOME_GUESS="/usr/local/cuda"
    [[ -d "${CUDA_HOME_GUESS}" ]] || CUDA_HOME_GUESS="$(dirname "$(dirname "$(command -v nvcc 2>/dev/null || echo /usr/bin/nvcc)")")"

    # ★ 아키텍처는 GPU 에 맞춰야 한다.
    #   Isaac Lab 문서는 8.0+PTX (Ampere/A100) 를 예시로 주는데, 우리는 L40S =
    #   Ada Lovelace = sm_89 다. 문서를 그대로 복붙하면 PTX JIT 로 돌긴 하지만
    #   최적이 아니고, 첫 실행 때마다 JIT 지연이 붙는다.
    ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
    [[ -n "${ARCH}" ]] || ARCH="89"
    TORCH_ARCH="${ARCH:0:1}.${ARCH:1}+PTX"
    info "감지된 compute capability: ${TORCH_ARCH} (L40S 는 8.9 이어야 정상)"

    info "cuRobo 소스 빌드 시작 — 프리빌트 휠이 없어 20분 이상 걸린다."
    set +e
    CUDA_HOME="${CUDA_HOME_GUESS}" \
    PATH="${CUDA_HOME_GUESS}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME_GUESS}/lib64:${LD_LIBRARY_PATH:-}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}" \
    pip install -c "${CONSTRAINTS}" \
        -e "git+https://github.com/NVlabs/curobo.git@${CUROBO_COMMIT}#egg=nvidia-curobo" \
        --no-build-isolation
    CUROBO_RC=$?
    set -e

    if [[ ${CUROBO_RC} -eq 0 ]] && python -c "import curobo" 2>/dev/null; then
        info "cuRobo 설치 성공 — SkillGen 사용 가능"
        SKILLGEN_OK=1
    else
        warn "cuRobo 설치 실패 (exit ${CUROBO_RC}). 세팅은 계속 진행한다.
       SkillGen 없이 MimicGen 방식으로 데이터를 생성하면 된다:
         generate_dataset.py 에서 --use_skillgen 플래그를 빼고,
         annotate_demos.py 에서 --annotate_subtask_start_signals 를 뺀다.
       환경 코드는 양쪽을 모두 지원하므로 고칠 것이 없다.
       (docs/RUNBOOK.md §1-5 의 'MimicGen 후퇴' 절 참조)"
    fi
fi

# =============================================================================
STAGE="9. 스모크 테스트"
log 9 "스모크 테스트"
# =============================================================================
cd "${REPO_ROOT}"
SMOKE_ARGS=()
[[ "${FULL_SMOKE}" == "1" ]] && SMOKE_ARGS+=(--full)

if python env/smoke/check_isaaclab.py "${SMOKE_ARGS[@]}"; then
    mkdir -p env/locks
    pip freeze > env/locks/isaaclab.lock.txt
    info "lock 박제: env/locks/isaaclab.lock.txt"
else
    die "스모크 테스트 실패 — lock 을 갱신하지 않았다. 위 실패 항목을 해결할 것."
fi

trap - ERR

if [[ ${SKILLGEN_OK} -eq 1 ]]; then
    GEN_MODE="SkillGen (cuRobo 모션 플래닝)"
    GEN_FLAGS="--use_skillgen"
else
    GEN_MODE="MimicGen (선형 보간) — cuRobo 없음"
    GEN_FLAGS=""
fi

cat <<DONE

=============================================================
 isaaclab 환경 준비 완료.

   source ${VENV_DIR}/bin/activate
   cd ${REPO_ROOT}

 데이터 생성 방식: ${GEN_MODE}

 다음 단계 (Day 1) — 씬을 눈으로 확인:
   export PUBLIC_IP=\$(curl -s ifconfig.me)
   python scripts/dump_obs_reference.py --task VlaPlace-v0 --save --livestream 2

 FULL_SMOKE=1 로 돌리지 않았다면, 태스크 코드 작업 전에 한 번은 실행할 것:
   python env/smoke/check_isaaclab.py --full
=============================================================
DONE
