# RUNBOOK — Day 1~4 실행 순서

인스턴스에서 그대로 복붙해 쓰는 문서. 각 블록은 순서대로 실행한다.
계획 근거는 `../vla-sft-rft-project-plan.md`, 스펙은 `configs/vla_spec.py`.

**전체 구조**

| Day | 인스턴스 | 산출물 |
|---|---|---|
| 1 | 1× L40S (포트 개방) | 환경 3종 동작 + 데모 12~15개 + 야간 증강 배치 |
| 2 | L40S 유지 + H100 신규 | RLDS 데이터셋 / SFT 착수 ∥ RFT 인프라 |
| 3 | RFT 인스턴스 (2~4× L40S) | SFT 베이스라인 + 강체 RFT 착수 |
| 4 | 동일 | **강체 RFT 개선 커브 (핵심 결과)** + 변형체 스트레치 |

**Day 2 부터는 두 인스턴스가 동시에 돈다.** SFT 가 도는 동안 L40S 에서 RFT
인프라를 만드는 것이 3~4일이 성립하는 이유다 — SFT 를 기다리지 않는다.

---

## Day 0 — 로컬 (인스턴스 없이)

```bash
git init && git add -A && git commit -m "vla-isaac scaffolding"
gh repo create <user>/vla-isaac --private --source=. --push
```

---

## Day 1 — 환경 구축 + 데이터 수집

### 1-1. 인스턴스 준비 (~1시간, 대기 시간)

Brev 에서 **L40S** 인스턴스 생성, 디스크 200GB+, 포트 **47998 / 49100** 개방.

```bash
git clone https://github.com/<user>/vla-isaac.git && cd vla-isaac
chmod +x setup/*.sh
FULL_SMOKE=1 ./setup/setup_isaaclab.sh
```

`--install none` 으로 설치하므로 sb3/rl_games 충돌이 원천 회피된다.
`pip check` 경고는 정상 — 통과 기준은 스모크 테스트뿐이다.

이 스크립트는 마지막에 **cuRobo(SkillGen 용)** 를 소스 빌드한다. 20분 이상 걸리고,
**실패해도 세팅은 계속 진행된다** — MimicGen 경로가 살아 있기 때문이다.
끝날 때 어느 방식을 쓰게 되는지 출력해 주니 그걸 보고 §1-5 로 간다.
cuRobo 를 아예 건너뛰려면 `USE_SKILLGEN=0 ./setup/setup_isaaclab.sh`.

```bash
source ~/env_isaaclab/bin/activate
cd ~/vla-isaac
```

### 1-2. 씬 확인 + 카메라 확정 (~1시간) ★ 되돌릴 수 없는 결정

```bash
export PUBLIC_IP=$(curl -s ifconfig.me)
python scripts/dump_obs_reference.py --task VlaPick-v0 --save \
    --out datasets/obs_reference --num-frames 8 --livestream 2
```

저장된 PNG 를 **반드시 눈으로 볼 것**:
- [ ] 자재와 목표 영역이 모두 화면 안에 있는가
- [ ] 파지 순간이 로봇 팔에 가려지지 않는가
- [ ] 이미지가 뒤집혀 있지 않은가 (뒤집혔으면 `configs/vla_spec.py` 의 `ROTATE_IMAGE_180`)

맞지 않으면 `configs/vla_spec.py` 의 `CAMERA_POS` / `CAMERA_ROT` 을 고치고 다시.
**여기서 확정한 카메라 설정이 곧 RFT 설정이다. Phase 2 이후에는 바꿀 수 없다.**

### 1-3. 변형체 예비 테스트 (15~20분, 타임박스 엄수)

```bash
python scripts/test_deformable_grasp.py --headless --youngs 5e5 1e5 5e4
```

권장 영률만 받아 적고 `source/vla_isaac_tasks/deformable_env_cfg.py` 에 반영한 뒤 **넘어간다.**
여기서 물리 튜닝에 빠지면 일정이 무너진다. 어떤 값에서도 파지가 안 되면
변형체는 스트레치에서 제외하고 강체에 집중한다.

### 1-4. 텔레옵 데모 수집 (~2시간)

먼저 우리 태스크가 gym 에 보이는지 확인한다. Isaac Lab 의 도구 스크립트들은
`isaaclab_tasks` 만 import 하므로, `setup_isaaclab.sh` 가 깔아 둔
`.pth` 자동 등록이 없으면 `NameNotFound` 로 죽는다.

```bash
python -c "import gymnasium as gym; print([k for k in gym.registry if k.startswith('VlaPick')])"
# 4개가 나와야 한다. 비어 있으면:
#   pip install -e source
#   echo "import vla_isaac_tasks" > "$(python -c 'import site;print(site.getsitepackages()[0])')/vla_isaac_tasks.pth"
```

```bash
mkdir -p datasets
python ~/IsaacLab/scripts/tools/record_demos.py \
    --task VlaPick-v0 --teleop_device keyboard --enable_cameras \
    --dataset_file ./datasets/source.hdf5 --num_demos 15 \
    --step_hz 24 --livestream 2
```

`--enable_cameras` 는 필수다 — 씬에 `TiledCamera` 가 있어 없으면 센서 초기화가 실패한다.
`--step_hz 24` 는 우리 제어 주기(decimation 5 × dt 1/120 = 24Hz)에 맞춘 값이다.
`--num_success_steps` 는 기본 10 이고 `SPEC.SUCCESS_HOLD_STEPS` 와 같으므로 건드리지 않는다.

**품질 체크리스트 — 생성 성공률에 직결된다:**
- [ ] 궤적이 짧은가 (불필요한 이동 최소화)
- [ ] 직선 경로인가 (축 단위로 나눠 움직이지 말 것)
- [ ] **일시정지가 없는가** ← 키보드 조작의 최대 함정. 멈춤은 정책이 학습하기 어렵다
- [ ] 녹화 끝에 여유 버퍼가 있는가

실수하면 `R` 로 폐기 후 리셋. 최소 10개 성공 데모.

**게임패드가 있으면 반드시 쓸 것** — 아날로그 스틱이라 SE(3) 연속 조작이 되어
데모 품질이 확연히 다르다. 키보드는 누를 때마다 팔이 계단식으로 튀는데,
그 계단이 그대로 Mimic 생성 성공률을 깎는다.

```bash
python ~/IsaacLab/scripts/tools/record_demos.py \
    --task VlaPick-v0 --teleop_device gamepad --enable_cameras \
    --dataset_file ./datasets/source.hdf5 --num_demos 15 \
    --step_hz 24 --livestream 2
```

두 장치 모두 `pickplace_env_cfg.py` 의 `teleop_devices` 에 등록되어 있다
(키보드 감도 0.05 / 게임패드 1.0·1.6 + dead_zone 0.01).
감도가 안 맞으면 그 값을 조정한다.

```bash
# 재생 검증 — 물리 비결정성으로 일부는 실패한다. 넉넉히 수집했으면 정상.
python ~/IsaacLab/scripts/tools/replay_demos.py \
    --task VlaPick-v0 --enable_cameras --dataset_file ./datasets/source.hdf5
```

### 1-5. 어노테이션 + 시험 생성 (~1시간) ★ 조기 경보 지점

**기본 경로는 SkillGen 이다.** 서브태스크 사이의 자유공간을 cuRobo 가 충돌 없이
계획하므로, MimicGen 의 선형 보간보다 생성 성공률이 높다 — 그게 Phase 2 최대
리스크(생성 성공률 저조)에 대한 가장 강한 대응이다.

먼저 SkillGen 이 실제로 쓸 수 있는 상태인지 확인:

```bash
python -c "import curobo; print('SkillGen 사용 가능')"
```

실패하면 아래 **"MimicGen 후퇴"** 로 간다. 성공하면:

```bash
# 어노테이션 — SkillGen 은 시작 경계가 필수다.
# --annotate_subtask_start_signals 를 빼면 생성이 실패한다.
python ~/IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task VlaPick-Visuomotor-Mimic-v0 --auto --enable_cameras \
    --annotate_subtask_start_signals \
    --input_file ./datasets/source.hdf5 --output_file ./datasets/annotated.hdf5

# 소량 시험 생성 — 성공률을 여기서 본다
python ~/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task VlaPick-Visuomotor-Mimic-v0 --enable_cameras --use_skillgen \
    --num_envs 10 --generation_num_trials 20 \
    --input_file ./datasets/annotated.hdf5 --output_file ./datasets/generated_small.hdf5
```

#### MimicGen 후퇴 (cuRobo 가 막혔거나 SkillGen 이 에러를 낼 때)

**환경 코드는 고칠 것이 없다.** 두 플래그만 빼면 된다:

```bash
python ~/IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task VlaPick-Visuomotor-Mimic-v0 --auto --enable_cameras \
    --input_file ./datasets/source.hdf5 --output_file ./datasets/annotated.hdf5
#   ↑ --annotate_subtask_start_signals 없음

python ~/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task VlaPick-Visuomotor-Mimic-v0 --enable_cameras \
    --num_envs 10 --generation_num_trials 20 \
    --input_file ./datasets/annotated.hdf5 --output_file ./datasets/generated_small.hdf5
#   ↑ --use_skillgen 없음
```

후퇴를 판단할 시점: **Day 1 오후까지 SkillGen 으로 시험 생성이 안 돌면 후퇴한다.**
가장 가능성 있는 원인은 cuRobo 자체가 아니라 Isaac Lab `main` 의 SkillGen 코드가
Isaac Sim 6.0 API 를 기대하는 것이다(문서가 6.0.0 기준, 우리는 5.1.0 고정).
그건 여기서 고칠 수 있는 종류의 문제가 아니므로 붙잡지 말 것.

#### 성공률 판정 (두 방식 공통)

**성공률이 10% 미만이면 야간 배치를 돌리지 말 것.** 밤을 통째로 날린다. 대응 순서:

1. **SkillGen 이면** `configs/vla_spec.py` 의 `APPROACH_START_DISTANCE` (0.15) 를
   키워 접촉 구간을 넓힌다 — 파지 직전 정렬까지 사람 데모를 쓰게 된다
2. `pickplace_mimic_env_cfg.py` 의 `num_interpolation_steps` 를 5 → 10 으로
3. `subtask_term_offset_range` 를 (10,20) → (20,35) 로 (경계를 더 뒤로)
4. 원본 데모 재수집 (더 짧고 매끄럽게)
5. **MimicGen 을 쓰고 있었다면 SkillGen 으로 올라가는 것도 대응책이다** —
   스티칭 충돌이 원인일 때 가장 효과가 크다

### 1-6. 야간 배치 생성

```bash
tmux new -s gen
python ~/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task VlaPick-Visuomotor-Mimic-v0 --enable_cameras --headless --use_skillgen \
    --num_envs 30 --generation_num_trials 3000 \
    --input_file ./datasets/annotated.hdf5 --output_file ./datasets/generated.hdf5
```

MimicGen 으로 후퇴했다면 `--use_skillgen` 을 뺀다. 그 외에는 동일하다.

`generation_num_trials` 는 **시도 횟수**지 성공 개수가 아니다. 시험 생성 성공률로
역산해 목표 1,000~1,500개가 나오게 잡는다. `--num_envs` 는 생성 속도이자
**형상 다양성**이기도 하다 (자재는 env 별로 배정되므로).

> SkillGen 은 cuRobo 플래닝이 붙어 시도당 시간이 MimicGen 보다 길 수 있다.
> 시험 생성 20회의 실제 소요를 재서 `generation_num_trials` 를 역산할 것 —
> 성공률만 보고 개수를 잡으면 밤 시간이 모자랄 수 있다.

---

## Day 2 — SFT 착수 ∥ RFT 인프라 (두 인스턴스 동시)

### 2-1. [L40S] 데이터 점검 → RLDS 변환

```bash
python scripts/convert_hdf5_to_rlds.py --inspect datasets/generated.hdf5
```

경고를 흘리지 말 것 — 특히 "한 번도 움직이지 않은 액션 차원" 과
"궤적이 전반적으로 길다".

```bash
python scripts/convert_hdf5_to_rlds.py \
    --hdf5 datasets/generated.hdf5 --out datasets/rlds

python scripts/upload_hub.py --path datasets/rlds \
    --repo <user>/vla-pick-rlds --repo-type dataset
```

### 2-2. [H100 신규] SFT 시작 — 오전에 반드시 띄울 것

H100/A100 인스턴스 생성 (RT 코어 불필요 — Isaac Sim 을 쓰지 않는 유일한 구간).

```bash
git clone https://github.com/<user>/vla-isaac.git && cd vla-isaac
chmod +x setup/*.sh && ./setup/setup_vla_train.sh
source ~/env_vla_train/bin/activate

huggingface-cli download <user>/vla-pick-rlds --repo-type dataset \
    --local-dir datasets/rlds
python scripts/upload_hub.py --verify datasets/rlds     # sha256 확인

tmux new -s sft
bash scripts/sft/run_sft_libero_spec.sh
```

스크립트가 학습 전에 상수셋을 **고정**하고 검증한다 (`7 / 8 / 8`).
어긋나면 학습을 시작하지 않고 중단하므로, 그 출력을 확인하고 넘어가면 된다.

> 왜 고정이 필요한가: openvla-oft 는 `sys.argv` 를 이어붙인 문자열에
> `libero`/`aloha`/`bridge` 가 있는지로 액션 차원을 고른다(없으면 LIBERO 폴백).
> 즉 **저장소를 `~/bridge/` 아래 두기만 해도 BRIDGE 상수셋이 잡힌다.**
> 폴백이 LIBERO라 대개 우연히 맞지만 그건 안전이 아니라 운이라,
> `register_dataset.py` 가 `constants.py` 끝에 재대입 블록을 붙여 확정한다.

### 2-3. [L40S] RFT 인프라 (SFT 를 기다리지 않는다)

```bash
SKIP_ISAACLAB=1 ./setup/setup_rft.sh
source ~/env_rft/bin/activate

# ★ 최대 리스크의 가장 이른 검증 지점
python env/smoke/check_rft.py --full
```

이게 통과하면 남은 RFT 작업은 GRPO 루프 배선뿐이다. 통과하지 못하면
`rft/README.md` 의 "흔한 실패와 해석" 표부터 볼 것.

### 2-4. [L40S, 선택] 변형체 데모 수집

강체 RFT 가 우선이다. 시간이 남을 때만:

```bash
python ~/IsaacLab/scripts/tools/record_demos.py \
    --task VlaPick-Deformable-v0 --teleop_device keyboard --enable_cameras \
    --dataset_file ./datasets/deformable_source.hdf5 --num_demos 20 \
    --step_hz 24 --livestream 2
```

---

## Day 3 — 베이스라인 + RFT 착수

### 3-1. SFT 체크포인트 회수 + 베이스라인

```bash
# [H100] 업로드
python scripts/upload_hub.py --path runs/<run> --repo <user>/vla-pick-sft --repo-type model

# [RFT 인스턴스] 다운로드 + 검증
huggingface-cli download <user>/vla-pick-sft --local-dir ckpt/sft
python scripts/upload_hub.py --verify ckpt/sft
```

### 3-2. ★ 관측 스펙 대조 — 무엇보다 먼저

```bash
python scripts/eval_rollout.py --random-policy --num-episodes 4     # 브리지+액션 규약
python scripts/dump_obs_reference.py --compare datasets/obs_reference \
    --candidate rft/debug_frames                                     # 픽셀 대조
```

라이브 뷰포트로는 검출할 수 없는 종류의 불일치다. 여기서 잡지 않으면
"RFT 커브가 오르지 않는다" 는 형태로만 드러나고 원인 추적에 며칠이 든다.

### 3-3. 베이스라인 성공률

```bash
python scripts/eval_rollout.py --checkpoint ckpt/sft \
    --task VlaPick-v0 --num-episodes 32 --out logs/baseline_rigid.json

python scripts/eval_rollout.py --checkpoint ckpt/sft \
    --task VlaPick-Deformable-v0 --num-episodes 16 --out logs/baseline_deformable.json
```

해석: 강체 60% 이상이면 계획대로. 60% 미만이어도 **0 이 아니면 진행한다** —
SimpleVLA-RL 은 데모 1개 SFT(17.3%)에서 RL 로 91.7% 까지 올렸다.
0 이면 RFT 에 학습 신호가 없으므로 데이터/스펙부터 다시 본다.

### 3-4. ⚠ Fallback 트리거 — 정오 판단

**정오까지 어댑터로 롤아웃 1회가 안 돌면 자작 GRPO 로 전환한다.**
환경·보상·브리지·평가가 전부 재사용되므로 전환 비용은 RL 루프뿐이다.

### 3-5. 강체 RFT 착수

```bash
tmux new -s rft
python rft/grpo_fallback.py --config rft/configs/grpo_rigid.yaml \
    --checkpoint ckpt/sft
```

첫 20스텝의 실제 소요를 보고 `total_steps` 를 조정한다. **끝까지 돌릴 필요 없다 —
커브가 오르기 시작하는 것만 확인되면 결과로 성립한다.**

모니터링 (포트 개방 불필요):
```bash
ssh -L 6006:localhost:6006 <서버>
```

---

## Day 4 — 결과 확보

### 4-1. 커브 확인 + 개선폭 측정

```bash
python scripts/eval_rollout.py --checkpoint logs/grpo_rigid/checkpoint-<N> \
    --task VlaPick-v0 --num-episodes 32 --out logs/rft_rigid.json
```

`logs/baseline_rigid.json` 과 비교한 것이 **이 프로젝트의 핵심 결과**다.

### 4-2. 커브가 오르지 않을 때

로그의 `무신호그룹` 비율부터 본다:
- ~100% → `temperature` 를 1.4 → 1.6 으로 (그룹이 전멸/전승으로 쏠림)
- 낮은데도 평평 → `grasp_lift_diag` 진단항으로 파지까지는 되는지 확인
- 발산 → `learning_rate` 를 1e-6 → 3e-7, 또는 `kl_coef` 를 0.01 로

### 4-3. 변형체 스트레치 (강체 커브 확보 + 변형체 베이스라인 > 0 일 때만)

```bash
python rft/grpo_fallback.py --config rft/configs/grpo_deformable.yaml \
    --checkpoint logs/grpo_rigid/checkpoint-<N>
```

### 4-4. 마무리

```bash
python scripts/upload_hub.py --path logs --repo <user>/vla-pick-rft --repo-type model
pip freeze > env/locks/rft.lock.txt
git add -A && git commit -m "Day 4 결과" && git push
```

인스턴스 stop.

---

## 자주 쓰는 것

```bash
# 스펙 확인 (어느 venv 에서든)
python configs/vla_spec.py

# SkillGen 사용 가능 여부
python -c "import curobo; print('SkillGen 사용 가능')"

# 스모크
python env/smoke/check_isaaclab.py --full
python env/smoke/check_vla_train.py --full
python env/smoke/check_rft.py --full

# 데이터 점검
python scripts/convert_hdf5_to_rlds.py --inspect datasets/generated.hdf5

# 무결성
python scripts/upload_hub.py --verify <경로>
```

## 절대 하지 말 것

- `./isaaclab.sh --install` (all) — sb3/rl_games 충돌. 반드시 `--install none`
- `pip install` 을 constraints 없이 — 항상 `-c env/constraints.<환경>.txt`
- torch 버전 상향 — Isaac Sim 5.1.0 이 2.7.0 에 고정
- `pip check` 통과를 검증 기준으로 — 이 스택에서는 불가능. 기준은 스모크 테스트
- Phase 2 이후 `configs/vla_spec.py` 의 카메라/액션/청크 변경 — SFT 데이터를 다시 만들어야 한다
- 시험 생성 성공률 <10% 인데 야간 배치 강행 — 밤을 통째로 날린다
- SkillGen 을 쓰면서 `--annotate_subtask_start_signals` 누락 — 시작 경계가 없으면
  생성이 실패한다. 어노테이션과 생성의 플래그는 **한 쌍으로** 맞춰야 한다
- cuRobo 가 안 된다고 Day 1 을 통째로 쓰기 — MimicGen 후퇴는 플래그 2개다.
  오후까지 안 되면 후퇴하고 밤 배치를 지킬 것
