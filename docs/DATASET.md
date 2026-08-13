# 데이터셋 — 무엇을 어떻게 만들었고, 무엇이 걸리는가

`datasets/generated.hdf5` (2026-08-13, 1,500 에피소드) 를 만들기까지의 설정과
점검 결과를 남긴다. **RLDS 변환 이후 단계에서 판단이 필요한 항목이 하나 있다**
(아래 "미해결: 궤적 길이").

---

## 산출물

| 파일 | 내용 | 크기 |
|---|---|---|
| `datasets/src_v2_01.hdf5` | 사람 데모 10개 (텔레옵) | 0.12 GB |
| `datasets/src_v2_02.hdf5` | 사람 데모 10개 (텔레옵) | 0.14 GB |
| `datasets/annotated_all.hdf5` | 위 둘의 어노테이션 결과 병합 **12개** | 0.15 GB |
| `datasets/generated.hdf5` | Mimic 증강 **1,500개** | 21.2 GB |

박스 제거(`cf43a55`) 이후에 수집한 것들이다. 그 이전 데이터(`source_*.hdf5`,
`annotated.hdf5`, `generated_small_*.hdf5`)는 씬이 달라 더 이상 유효하지 않다.

## 파이프라인 설정 (재현용)

```bash
# 1) 텔레옵 수집 — 감도를 낮춰 정밀도를 얻었다
VLA_ANNOUNCE_TARGET=1 VLA_TELEOP_POS_SENS=0.02 \
python $ISAACLAB_DIR/scripts/tools/record_demos.py \
    --task VlaPlace-v0 --teleop_device keyboard --enable_cameras \
    --dataset_file ./datasets/src_v2_01.hdf5 --num_demos 10 \
    --step_hz 24 --num_success_steps 1 --livestream 2

# 2) 어노테이션 (SkillGen 시작 경계 포함)
python $ISAACLAB_DIR/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task VlaPlace-Visuomotor-Mimic-v0 --auto --enable_cameras --headless \
    --annotate_subtask_start_signals \
    --input_file ./datasets/src_v2_01.hdf5 --output_file ./datasets/annotated_v2.hdf5

# 3) 병합
python $ISAACLAB_DIR/scripts/tools/merge_hdf5_datasets.py \
    --input_files datasets/annotated_v2.hdf5 datasets/annotated_v2_02.hdf5 \
    --output_file datasets/annotated_all.hdf5

# 4) 증강 (MimicGen) — 성공 1,500개를 채울 때까지
python $ISAACLAB_DIR/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task VlaPlace-Visuomotor-Mimic-v0 --enable_cameras --headless \
    --num_envs 60 --generation_num_trials 1500 \
    --input_file ./datasets/annotated_all.hdf5 --output_file ./datasets/generated.hdf5
```

핵심 파라미터는 `SPEC` 과 환경변수에 있다:

| 값 | 설정 | 왜 |
|---|---|---|
| 텔레옵 감도 0.02 | `VLA_TELEOP_POS_SENS` | 기본 0.05 는 1스텝 25mm 라 미세 조작이 안 됐다 |
| `action_noise` 0.003 | `VLA_ACTION_NOISE` | 0.03 은 사람 데모 액션 크기(0.0037)를 덮었다 — 아래 참조 |
| `num_envs` 60 | CLI | VRAM 24.6/46 GB, 0.43초/시도 |

## 실측 수율

| 단계 | 결과 |
|---|---|
| 어노테이션 (src_v2_01) | 5/10 |
| 어노테이션 (src_v2_02) | 7/10 |
| 증강 (MimicGen) | **43.1%** (1,500 / 3,481 시도), 2시간 3분 |
| 증강 (SkillGen) | **0%** — 미해결, 아래 참조 |

수율은 20분 구간마다 43.4 / 41.3 / 41.5 / 42.6 / 42.8 / 43.1% 로 끝까지
안정적이었고 VRAM 도 24.6 GB 에서 변동이 없었다.

소스를 5개 → 12개로 늘려도 수율은 44% → 43% 로 그대로였다. 소스 확대의 효과는
수율이 아니라 **생성 데이터의 다양성** 쪽이다.

### action_noise 가 수율을 지배했다

`SubTaskConfig.action_noise` 를 0.03(Isaac Lab Franka stack 레퍼런스 값)에서
0.003 으로 낮추자 수율이 **7.1% → 45%** 로 뛰었다. 우리 데모는 텔레옵 감도를
낮춰 액션이 곱기 때문에(평균 크기 0.0037) 0.03 노이즈가 신호를 덮고 있었다.

| | 스텝간 변화 | 움직이는 구간의 방향 반전 |
|---|---|---|
| 사람 데모 | 0.1 mm | 4.8% |
| 생성 (noise 0.03) | 17.7 mm | 76.9% |
| 생성 (noise 0.003) | 2.4 mm | 47.9% |

상류도 태스크마다 값이 다르다 — Franka stack 0.03 / AgiBot 0.01 / G1 0.003.
소스 데모가 늘면 다양성을 위해 0.005~0.01 로 올려 재평가할 것.

## 점검 결과 (`--inspect`)

```
데모 수    1,500
총 스텝    897,712
지시문 3종  green 516 / red 511 / blue 473
액션 7차원  전부 유효 (한 번도 움직이지 않은 차원 없음), 그리퍼 -1/+1 이진
```

표본 200개 검사에서 **타깃 블록이 실제로 트레이 안에 있는 비율 200/200**,
이미지 `224×224×3 uint8`, 에피소드 내 타깃 일정 — 모두 정상이다.

소스 어노테이션의 타깃 분포는 `red 2 / blue 4 / green 6` 으로 치우쳐 있었지만
생성 결과는 고르게 나왔다. 생성 시 타깃은 초기 상태 뱅크에서 균등하게 뽑히고
`object_ref="target"` 이 소스를 어느 색으로든 변환하기 때문이다.

용량의 98.7% 가 `obs/table_cam` 이다 (224×224×3 uint8). 나머지 전부를 합쳐도
1.3% — 이 데이터셋은 사실상 이미지 파일이다.

---

## ★ 미해결 1: 궤적 길이 — 판단이 필요하다

`--inspect` 가 경고 2건을 냈고, 둘 다 같은 원인이다.

```
[WARN] 최대 길이가 스펙 상한(300)을 넘는다.
[WARN] 궤적이 전반적으로 길다.

궤적 길이 : 평균 598, 중앙 621, 범위 484~706   (SPEC.MAX_EPISODE_STEPS = 300)
```

**모든 데모가 예산의 1.6~2.4배다.** 텔레옵 감도를 0.02 로 낮춰 정밀도를 얻은
대가로 사람 데모가 길어졌고(중앙 518 → 625스텝), 증강이 그 길이를 물려받았다.

영향:

| 단계 | 영향 |
|---|---|
| RLDS 변환 | 없음 |
| SFT | 897,712 프레임인데 20,000스텝 × 배치16 = 320,000 샘플 → **0.36 에폭**. 데이터를 한 번도 다 못 본다 |
| 평가 / RFT | **300스텝에서 잘린다.** 621스텝짜리 행동을 배운 정책은 예산 안에 못 끝낸다 |

대응 후보:

1. **평가·RFT 예산을 600~700스텝으로 올린다** — `eval_rollout.py --max-steps 700`,
   `rft/configs/grpo_rigid.yaml` 의 `max_steps_per_episode`. 대가는 롤아웃 시간 2배.
   평가 기준은 SFT 이후에도 조정 가능한 값이므로 되돌리기 쉽다.
2. **`MAX_STEPS` 를 늘려 SFT 에폭을 맞춘다** — 20,000 → 56,000 이면 1에폭.
   하룻밤(8~14h)을 넘긴다.
3. **더 짧은 데모로 재수집** — 근본적이지만 수집·어노테이션·증강을 전부 다시 돌린다.

현재 판단: **1번**. 데이터 자체는 건강하고, 0.36 에폭은 과적합 측면에서 오히려
안전하다. 다만 이건 확정이 아니라 기록이다 — SFT 베이스라인을 재 보고
재검토할 것.

## ★ 미해결 2: SkillGen 0%

MimicGen 은 43% 인데 SkillGen 은 **0/296, 0/20 어떤 조건에서도 0%** 다.

진행 과정에서 상류 버그 두 개를 고쳐 SkillGen 을 실제로 구동시키는 데까지는
갔다 (`patches/isaaclab-annotate-start-signals.patch`, 마지막 서브태스크 이름
규약). 그 뒤 막힌 지점은 **두 번째 플래닝(운반 구간) 이 20회 시도 20회 실패**
하는 것이다. 첫 번째(접근)는 30/30 성공한다.

기각된 가설 — 셋 다 결과를 전혀 바꾸지 못했다 (30성공/20실패 고정):

- 부착 물체(들고 있는 블록)를 충돌체에서 제외
- 로봇·블록을 정적 월드 충돌체에서 제외
- 박스 제거 (빈손 플래닝 성공률은 41% → 100% 로 올랐다)

남은 후보 (미검증):

- **목표 자세가 IK 로 도달 불가** — 블록 yaw 가 ±π 무작위라, 변환된 파지 자세가
  관절 한계를 넘거나 다른 IK 분기를 요구할 수 있다. 가장 유력하다.
- `max_planning_attempts=1` — 재시도가 없다
- `approach_distance=0.0 / retreat_distance=0.0 / collision_activation_distance=0.0`
  — 상류의 `franka_stack_cube_bin_config` 는 0.05 / 0.07 / 0.02 를 쓴다. 우리가
  떨어지는 `franka_config()` 는 전부 0 이다

Isaac Lab 에 `get_subtask_start_signals()` 를 구현한 환경이 하나도 없다 —
이 경로는 상류에서 검증된 적이 없다. RUNBOOK §1-5 의 지침("오후까지 안 되면
후퇴하고 밤 배치를 지킬 것")대로 **MimicGen 으로 진행 중**이다.
