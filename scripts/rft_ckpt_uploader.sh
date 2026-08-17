#!/usr/bin/env bash
# =============================================================================
# scripts/rft_ckpt_uploader.sh
# RFT 체크포인트 감시 → HF 업로드 → 로컬 정리.
#
# 학습 프로세스와 **독립**으로 돈다 (nohup). 학습이 20스텝마다 남기는
# logs/grpo_15h_lora/checkpoint-N 을 발견하는 대로 HF 에 올린다.
#
# ★ 토큰은 이 파일에 넣지 않는다. `hf auth login` 이 ~/.cache/huggingface/token
#   에 저장해 둔 것을 쓴다 (저장소 밖이라 커밋될 일이 없다).
#
# 사용:
#   nohup bash scripts/rft_ckpt_uploader.sh > logs/uploader.log 2>&1 &
# =============================================================================
set -uo pipefail

REPO_ID="${REPO_ID:-tae-roh/vla-pick-rft-ckpts}"
CKPT_DIR="${CKPT_DIR:-/home/shadeform/vla-isaac/logs/grpo_batched}"
HF="${HF:-$HOME/.local/bin/hf}"
POLL_SEC="${POLL_SEC:-120}"
# 로컬에 남겨 둘 최근 체크포인트 수. 업로드가 끝난 것만 지운다.
KEEP_LOCAL="${KEEP_LOCAL:-3}"
# 이 여유 아래로 떨어지면 업로드 끝난 오래된 것부터 지운다 [GB].
MIN_FREE_GB="${MIN_FREE_GB:-40}"
STATE="${CKPT_DIR}/.uploaded"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

mkdir -p "${CKPT_DIR}"
touch "${STATE}"

log "감시 시작: ${CKPT_DIR} → ${REPO_ID} (${POLL_SEC}s 주기, 로컬 ${KEEP_LOCAL}개 유지)"

free_gb() { df -BG --output=avail /home/shadeform | tail -1 | tr -dc '0-9'; }

# 업로드가 끝난 것 중 오래된 순으로 지워 여유를 확보한다.
cleanup() {
    local keep="$1" reason="$2"
    # step 번호 오름차순 = 오래된 순
    local done_list
    mapfile -t done_list < <(
        find "${CKPT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null |
        sed 's/checkpoint-//' | sort -n | sed 's/^/checkpoint-/'
    )
    local n=${#done_list[@]}
    local i=0
    for d in "${done_list[@]}"; do
        (( n - i <= keep )) && break
        if grep -qxF "${d}" "${STATE}"; then
            log "  정리(${reason}): ${d} 삭제 (업로드 완료됨)"
            rm -rf "${CKPT_DIR:?}/${d}"
        fi
        i=$((i+1))
    done
}

# ★ 학습이 잠깐 안 보인다고 바로 끝내지 않는다. 크래시 후 재시작하는 몇 초
#   사이에 감시가 죽어 버리면, 이후 체크포인트가 통째로 업로드되지 않는다
#   (2026-08-16 실제로 그렇게 유실될 뻔했다 — 학습 재시작 1분 전에 종료됨).
#   MISS_LIMIT 번 연속으로 안 보일 때만 종료한다.
MISS_LIMIT="${MISS_LIMIT:-5}"
miss=0

while true; do
    if ! pgrep -f "grpo_fallback.p[y]" >/dev/null 2>&1; then
        miss=$((miss+1))
        log "학습 프로세스 미검출 (${miss}/${MISS_LIMIT})"
        if (( miss >= MISS_LIMIT )); then
            log "학습 프로세스 없음 — 남은 것 업로드하고 감시 종료"
            # 마지막으로 한 바퀴 더 돌아 남은 체크포인트를 올린다
            for d in $(find "${CKPT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null |
                       sed 's/checkpoint-//' | sort -n | sed 's/^/checkpoint-/'); do
                grep -qxF "${d}" "${STATE}" && continue
                "${HF}" upload "${REPO_ID}" "${CKPT_DIR}/${d}" "${d}" \
                    --commit-message "RFT ${d} (final)" >/dev/null 2>&1 && \
                    { echo "${d}" >> "${STATE}"; log "  ✓ 최종 업로드: ${d}"; }
            done
            break
        fi
    else
        miss=0
    fi

    for d in $(find "${CKPT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null |
               sed 's/checkpoint-//' | sort -n | sed 's/^/checkpoint-/'); do
        grep -qxF "${d}" "${STATE}" && continue
        # 저장이 끝났는지 확인 — adapter 파일이 있고 60초간 크기가 안 변해야 한다
        f="${CKPT_DIR}/${d}/adapter_model.safetensors"
        [[ -f "${f}" ]] || continue
        s1=$(stat -c%s "${f}" 2>/dev/null || echo 0)
        sleep 20
        s2=$(stat -c%s "${f}" 2>/dev/null || echo 0)
        [[ "${s1}" == "${s2}" && "${s1}" != "0" ]] || { log "  ${d} 아직 기록 중 — 다음 주기에"; continue; }

        log "업로드 시작: ${d} ($(du -sh "${CKPT_DIR}/${d}" | cut -f1))"
        if "${HF}" upload "${REPO_ID}" "${CKPT_DIR}/${d}" "${d}" \
               --commit-message "RFT ${d}" >/dev/null 2>&1; then
            echo "${d}" >> "${STATE}"
            log "  ✓ 업로드 완료: ${d}"
        else
            log "  ✗ 업로드 실패: ${d} — 다음 주기에 재시도"
        fi
    done

    # history.json 은 매 스텝 갱신되므로 매번 덮어쓴다 (작다)
    if [[ -f "${CKPT_DIR}/history.json" ]]; then
        "${HF}" upload "${REPO_ID}" "${CKPT_DIR}/history.json" "history.json" \
            --commit-message "history $(date -u +%H:%M)" >/dev/null 2>&1 || true
    fi

    # 용량 관리
    cleanup "${KEEP_LOCAL}" "보관 상한"
    fg=$(free_gb)
    if (( fg < MIN_FREE_GB )); then
        log "여유 ${fg}GB < ${MIN_FREE_GB}GB — 추가 정리"
        cleanup 1 "여유 부족"
    fi

    sleep "${POLL_SEC}"
done

log "종료"
