#!/usr/bin/env bash
# =============================================================================
# scripts/sft/run_sft_libero_spec.sh
# Phase 3 — VLA SFT (H100/A100 인스턴스, env_vla_train)
#
# ★ 상수셋(ACTION_DIM=7 / PROPRIO_DIM=8 / NUM_ACTIONS_CHUNK=8)은
#   register_dataset.py 가 constants.py 에 못 박아 확정한다.
#
#   openvla-oft 의 detect_robot_platform() 은 sys.argv 를 전부 이어붙인 문자열에
#   "libero"/"aloha"/"bridge" 가 있는지로 상수셋을 고른다 (없으면 LIBERO 로 폴백).
#   즉 **실행 커맨드에 우연히 섞인 문자열이 학습 차원을 바꾼다** — 저장소를
#   ~/bridge/ 아래 두기만 해도 BRIDGE 상수셋이 잡힌다.
#   기본 폴백이 LIBERO 라 대부분 우연히 맞지만, 그건 안전이 아니라 운이다.
#
#   참고: 이 셸 스크립트의 파일명은 감지에 아무 영향이 없다. bash 프로세스의
#   이름일 뿐 python 의 sys.argv 가 아니기 때문이다. argv 에 들어가는 것은
#   아래 --run_id_note 뿐이며, 그것도 보조 수단이다. 확정은 constants.py 고정이다.
#
# ★ 핵심 플래그: --use_l1_regression False
#   기본값이 True 인데, True 면 연속 L1 회귀 헤드가 붙는다. 그건 결정론적이라
#   log π 가 정의되지 않아 **RL 을 붙일 수 없다** (계획서 §2-1).
#   이산 액션 토큰 + cross-entropy 여야 GRPO 의 ratio 를 계산할 수 있다.
#   이 한 줄이 이 프로젝트에서 SFT 를 하는 이유의 절반이다.
#
# 사용:
#   tmux new -s sft
#   bash scripts/sft/run_sft_libero_spec.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OFT_DIR="${OFT_DIR:-${HOME}/openvla-oft}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets/rlds}"
DATASET_NAME="${DATASET_NAME:-vla_pick}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs}"
VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"

# --- 일정에 맞춘 축소 설정 ---------------------------------------------------
# 계획서는 1× L40S 기준 2~5일을 예상했다. H100 으로 옮기고 스텝 수를 줄여
# 하룻밤(~8-14h)에 끝나게 잡는다. 증강 1,000~1,500개면 이 스텝 수로 충분히 수렴한다.
MAX_STEPS="${MAX_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
LORA_RANK="${LORA_RANK:-32}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
NUM_GPUS="${NUM_GPUS:-1}"

log()  { echo -e "\n\033[1;32m[SFT]\033[0m $*"; }
die()  { echo -e "\033[1;31m[FAIL]\033[0m $*" >&2; exit 1; }

# --- 사전 확인 ---------------------------------------------------------------
log "사전 확인"
[[ -d "${OFT_DIR}" ]] || die "openvla-oft 없음: ${OFT_DIR} (setup_vla_train.sh 를 먼저 실행)"
[[ -d "${DATA_ROOT}" ]] || die "데이터셋 없음: ${DATA_ROOT} (convert_hdf5_to_rlds.py 를 먼저 실행)"

NORM_STATS="${DATA_ROOT}/norm_stats.json"
[[ -f "${NORM_STATS}" ]] || die "정규화 통계 없음: ${NORM_STATS}
     이 파일이 없으면 RFT 에서 액션 디토크나이즈 구간이 어긋난다.
     scripts/convert_hdf5_to_rlds.py 를 다시 실행할 것."

python "${REPO_ROOT}/configs/vla_spec.py"

# 데이터셋 등록 + 상수셋 고정 (둘 다 멱등)
log "openvla-oft 설정 (데이터셋 등록 + 상수셋 고정)"
python "${REPO_ROOT}/scripts/sft/register_dataset.py" --oft-dir "${OFT_DIR}"

# 고정이 실제로 먹었는지 확인
log "상수셋 확인 (7 / 8 / 8 이어야 정상)"
python - <<'PYEOF' || exit 1
from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK
print(f"  ACTION_DIM={ACTION_DIM}, PROPRIO_DIM={PROPRIO_DIM}, "
      f"NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}")
assert (ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK) == (7, 8, 8), (
    "상수셋이 LIBERO(7/8/8)가 아니다. register_dataset.py 의 고정이 먹지 않았다 — "
    "openvla-oft 구조가 바뀌었을 수 있으니 prismatic/vla/constants.py 를 직접 확인할 것."
)
PYEOF

python "${REPO_ROOT}/env/smoke/check_vla_train.py" || \
    die "스모크 테스트 실패."

# --- 설정 기록 ---------------------------------------------------------------
# RFT 가 반드시 같은 설정을 써야 하므로 파일로 남긴다 (계획서 §6 스펙 불일치 리스크).
mkdir -p "${RUN_ROOT}"
CONFIG_RECORD="${RUN_ROOT}/sft_config.json"
cat > "${CONFIG_RECORD}" <<EOF
{
  "vla_path": "${VLA_PATH}",
  "dataset_name": "${DATASET_NAME}",
  "data_root_dir": "${DATA_ROOT}",
  "use_l1_regression": false,
  "use_diffusion": false,
  "num_images_in_input": 1,
  "use_proprio": false,
  "lora_rank": ${LORA_RANK},
  "batch_size": ${BATCH_SIZE},
  "grad_accumulation_steps": ${GRAD_ACCUM},
  "learning_rate": ${LEARNING_RATE},
  "max_steps": ${MAX_STEPS},
  "note": "액션 헤드는 이산 토큰 + CE (RL 호환). 연속 L1 회귀를 쓰면 log-prob 이 정의되지 않아 GRPO 불가."
}
EOF
log "설정 기록: ${CONFIG_RECORD}  ★ 체크포인트와 함께 Phase 4 로 가져갈 것"

# --- 학습 -------------------------------------------------------------------
log "SFT 시작 (GPU ${NUM_GPUS}개, 최대 ${MAX_STEPS} 스텝)"
echo "  중단 복구를 위해 ${SAVE_FREQ} 스텝마다 체크포인트를 저장한다."
echo "  tmux 밖에서 실행 중이라면 지금 중단하고 tmux 안에서 다시 시작할 것."
echo

cd "${OFT_DIR}"

# torchrun 커맨드에 'libero' 문자열이 포함되도록 --dataset_name 이 vla_pick 이어도
# 아래 --run_id_note 에 명시적으로 넣어 둔다 (detect_robot_platform 이 argv 전체를 본다).
torchrun --standalone --nnodes 1 --nproc-per-node "${NUM_GPUS}" \
    vla-scripts/finetune.py \
    --vla_path "${VLA_PATH}" \
    --data_root_dir "${DATA_ROOT}" \
    --dataset_name "${DATASET_NAME}" \
    --run_root_dir "${RUN_ROOT}" \
    --use_l1_regression False \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 1 \
    --use_proprio False \
    --batch_size "${BATCH_SIZE}" \
    --grad_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LEARNING_RATE}" \
    --use_lora True \
    --lora_rank "${LORA_RANK}" \
    --image_aug True \
    --max_steps "${MAX_STEPS}" \
    --save_freq "${SAVE_FREQ}" \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training True \
    --run_id_note "libero_spec_vla_pick"

log "SFT 완료. 다음 단계:"
cat <<NEXT

  1) 베이스라인 성공률 측정 (강체 — in-distribution)
     python scripts/eval_rollout.py --checkpoint ${RUN_ROOT}/<run>/  \\
         --task VlaPick-v0 --num-episodes 32 --out logs/baseline_rigid.json

  2) 변형체 성공률 (out-of-distribution — RFT 개선폭의 출발점)
     python scripts/eval_rollout.py --checkpoint ${RUN_ROOT}/<run>/  \\
         --task VlaPick-Deformable-v0 --num-episodes 16 \\
         --out logs/baseline_deformable.json

  3) 체크포인트 + 설정 업로드
     python scripts/upload_hub.py --path ${RUN_ROOT}/<run> \\
         --repo <user>/vla-pick-sft --repo-type model
NEXT
