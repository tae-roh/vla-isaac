# rft/ — RFT 어댑터

## 아키텍처: 프로세스 분리

```
┌──────────────────────────────┐         ┌───────────────────────────────┐
│  학습 프로세스 (env_rft)      │  stdin  │  롤아웃 워커 (env_isaaclab)     │
│                              │ ──────▶ │                               │
│  veRL / vLLM / GRPO          │         │  Isaac Sim 5.1.0 + torch 2.7  │
│  정책 forward + 업데이트       │ ◀────── │  num_envs 배치 벡터화           │
└──────────────────────────────┘  stdout └───────────────────────────────┘
            ipc_bridge.py                  isaaclab_rollout_worker.py
```

**왜 한 venv 로 합치지 않는가.** Isaac Sim 5.1.0 은 torch 2.7.0 에 고정되어 있고
veRL/vLLM 은 자체 요구사항이 있다. 계획서 §Phase4b-1 은 단일 venv 통합을 먼저
시도하라고 했지만, SimpleVLA-RL 의 `verl/workers/rollout/rob_rollout.py` 를 실제로
읽어 보면 `_generate_minibatch_libero()` 가 **이미 `multiprocessing.Process` + Queue 로
시뮬레이터를 격리해서 돌리고 있다.** 프로세스 분리는 우리가 발명하는 우회로가 아니라
상류가 이미 채택한 구조다. 그래서 1순위로 뒤집었다.

얻는 것:
- torch 버전 충돌이 구조적으로 소멸 (계획서가 최대 난점으로 꼽은 항목)
- 상류 수정면이 "워커 프로세스 하나 교체" 로 좁아짐
- 시뮬 GPU 와 학습 GPU 분리가 자연스럽게 맞물림

## 파일

| 파일 | 실행 venv | 역할 |
|---|---|---|
| `ipc_bridge.py` | 양쪽 | 프로토콜. 순수 stdlib — numpy 버전이 달라도 안전하게 배열을 주고받는다 |
| `isaaclab_rollout_worker.py` | `env_isaaclab` | 배치 환경을 굴린다. 직접 실행하지 않고 브리지가 띄운다 |
| `grpo_fallback.py` | `env_rft` | 자작 GRPO 루프 (fallback 경로) |
| `configs/*.yaml` | — | 축소 규모 GRPO 설정 |

## 두 갈래 경로

### 1순위 — SimpleVLA-RL(veRL) 어댑터
`RobHFRollout._generate_minibatch_libero()` 자리에 `RolloutClient` 를 끼운다.
LIBERO 는 프로세스마다 env 1개지만 Isaac Lab 은 한 프로세스가 배치를 담당하므로,
"env 마다 하나씩 Queue" 를 "배치 하나에 브리지 하나" 로 바꾸는 것이 작업의 실체다.

### Fallback — 자작 GRPO (`grpo_fallback.py`)
**전환 트리거: Day 3 정오까지 어댑터로 롤아웃 1회가 안 돌면.**

환경·보상·관측·브리지·평가 코드가 전부 재사용되므로 전환 비용은 RL 루프 부분뿐이다.
포기하는 것: vLLM 가속 생성, FSDP 멀티 GPU 샤딩. 즉 느리다 — 하지만 `num_envs` 가
작아 감당되고, 계획서 §6 도 "단일 노드에선 자작이 오히려 단순할 수 있음" 이라 적었다.

## 순서 (이 순서를 지킬 것)

```bash
# 1. 브리지 왕복 — 정책 없이. 최대 리스크의 가장 이른 검증 지점
python env/smoke/check_rft.py --full

# 2. 관측 스펙 대조 — 어댑터 완성 직후 최우선 (계획서 §Phase4b-3)
#    라이브 뷰포트로는 검출할 수 없다. 뷰포트 화면 ≠ 모델 입력 텐서
python scripts/dump_obs_reference.py --compare datasets/obs_reference \
    --candidate rft/debug_frames

# 3. 랜덤 정책 롤아웃 — 액션 규약(특히 그리퍼 부호)이 맞는지
python scripts/eval_rollout.py --random-policy --num-episodes 4

# 4. SFT 정책 베이스라인
python scripts/eval_rollout.py --checkpoint <ckpt> --num-episodes 32 \
    --out logs/sft_c5mm.json

# 5. RFT 본 학습
python rft/grpo_fallback.py --config rft/configs/grpo_rigid.yaml \
    --checkpoint <ckpt>
```

## 모니터링 — 포트 개방 불필요

RFT 는 라이브스트림을 쓰지 않는다. **렌더링과 라이브스트림은 별개다** (계획서 §Phase4b):
오프스크린 렌더(`--enable_cameras`)는 필수지만, WebRTC 인코딩은 GPU 사이클만 먹고
인스턴스당 클라이언트 1개 제약 때문에 멀티 롤아웃 구조와 맞지도 않는다.

```bash
ssh -L 6006:localhost:6006 -L 8000:localhost:8000 <서버>
# 서버: tensorboard --logdir logs --port 6006
# 서버: cd rft/debug_frames && python -m http.server 8000
```
wandb 는 아웃바운드라 포트와 무관하다.

## 흔한 실패와 해석

| 증상 | 먼저 의심할 것 |
|---|---|
| 워커가 응답 없이 죽음 | `--enable_cameras` 누락, GPU 메모리 부족. stderr 의 Isaac Sim 로그를 볼 것 |
| 그리퍼가 전혀 안 움직임 | 부호를 두 번 뒤집어 상쇄됨. 변환은 워커의 `GRIPPER_INVERT_FOR_VLA` 한 곳에서만 |
| 성공률이 계속 0 | 관측 스펙 불일치(2번을 건너뛰었는지), 또는 SFT prior 부족 |
| 무신호그룹 비율 ~100% | temperature 를 올릴 것. 그룹이 전멸/전승으로 쏠려 advantage 가 0 |
| loss 는 도는데 커브가 평평 | `grasp_lift_diag`(weight=0 진단항) 로그를 볼 것 — 파지까지는 되는지 구분된다 |
