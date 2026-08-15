# RUNBOOK — Day 1~4 실행 순서

인스턴스에서 그대로 복붙해 쓰는 문서. 각 블록은 순서대로 실행한다.
계획 근거는 `../vla-sft-rft-project-plan.md`, 스펙은 `configs/vla_spec.py`.

**전체 구조**

| Day | 인스턴스 | 산출물 |
|---|---|---|
| 1 | 1× L40S (포트 개방) | 환경 3종 동작 + 데모 12~15개 + 야간 증강 배치 |
| 2 | L40S 유지 + H100 신규 | RLDS 데이터셋 / SFT 착수 ∥ RFT 인프라 |
| 3 | RFT 인스턴스 (2~4× L40S) | SFT 베이스라인 + RFT 착수 |
| 4 | 동일 | **SFT 대비 RFT 성공률 개선폭 (핵심 결과)** |

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

**Isaac Lab 이 이미 다른 경로에 있으면** `ISAACLAB_DIR` 로 알려 준다.
스크립트가 이 값을 `~/.bashrc` 에 남기고, 이후 모든 명령이 `$ISAACLAB_DIR` 을 쓴다:

```bash
ISAACLAB_DIR=~/workspace/IsaacLab FULL_SMOKE=1 ./setup/setup_isaaclab.sh
```

`--install none` 으로 설치하므로 sb3/rl_games 충돌이 원천 회피된다.
`pip check` 경고는 정상 — 통과 기준은 스모크 테스트뿐이다.

이 스크립트는 마지막에 **cuRobo(SkillGen 용)** 를 소스 빌드한다. 20분 이상 걸리고,
**실패해도 세팅은 계속 진행된다** — MimicGen 경로가 살아 있기 때문이다.
끝날 때 어느 방식을 쓰게 되는지 출력해 주니 그걸 보고 §1-5 로 간다.
cuRobo 를 아예 건너뛰려면 `USE_SKILLGEN=0 ./setup/setup_isaaclab.sh`.

```bash
source ~/env_isaaclab/bin/activate
cd "$(git -C ~/workspace/vla-isaac rev-parse --show-toplevel 2>/dev/null || echo ~/vla-isaac)"
```

이후 명령은 모두 저장소 루트 기준이다. setup 스크립트들은 자기 위치에서
`REPO_ROOT` 를 계산하므로 저장소를 어디에 두든 상관없다.

### 1-2. 씬 확인 + 카메라 확정 (~1시간) ★ 되돌릴 수 없는 결정

먼저 **초기 상태 뱅크**를 만든다 (개정 §3). Isaac Sim 없이 몇 초면 끝나지만,
나중에 붙이면 그 전 실험 결과가 전부 재현 불가가 되므로 여기서 한다.

```bash
python scripts/make_init_states.py --show train          # 이미 커밋돼 있다 — 확인만
python scripts/make_init_states.py --show eval_base      # 평가 홀드아웃 64개
```

없거나 스펙이 바뀌어 차원이 어긋나면 그때만 만든다 (기존 파일은 `--force` 필요):

```bash
python scripts/make_init_states.py --name train --force
python scripts/make_init_states.py --name eval_base --size 64 --seed 12345 --force
git add datasets/init_states && git commit -m "초기 상태 뱅크"
```

```bash
export PUBLIC_IP=$(curl -s ifconfig.me)
python scripts/dump_obs_reference.py --task VlaPlace-v0 --save \
    --out datasets/obs_reference --num-frames 8 --livestream 2
```

> `datasets/obs_reference/*.png` 는 2026-08-11 트레이 개편 후 씬으로 재생성해
> 두었다 (블록 3개 + 정사각 트레이 1개). 씬 지오메트리를 다시 건드리면 위
> 명령으로 덮어쓸 것 — Phase 4 스펙 대조는 항상 최신 씬 기준이어야 한다.

저장된 PNG 를 **반드시 눈으로 볼 것**:
- [ ] 박스(블록 3개)와 타깃 트레이가 **둘 다** 화면 안에 있는가
- [ ] 블록 3색이 서로 구분되는가 (구분이 안 되면 언어 채널이 죽는다)
- [ ] 트레이가 블록보다 확실히 넉넉해 보이는가 (한 변 72mm vs 블록 60×30mm)
- [ ] 파지 순간이 로봇 팔에 가려지지 않는가
- [ ] 이미지가 뒤집혀 있지 않은가 (뒤집혔으면 `configs/vla_spec.py` 의 `ROTATE_IMAGE_180`)

맞지 않으면 `configs/vla_spec.py` 의 `CAMERA_POS` / `CAMERA_ROT` 을 고치고 다시.
**여기서 확정한 카메라 설정이 곧 RFT 설정이다. Phase 2 이후에는 바꿀 수 없다.**

> **카메라 확정 완료** (2026-08-10): `CAMERA_POS=(1.2, 0.0, 0.8)`,
> `CAMERA_ROT=(0.32818, -0.66487, -0.6017, 0.297)`. 초기값은 시선이 작업공간이 아니라
> 로봇 베이스를 향해 목표 영역 중심이 프레임 밖(u=225.0)이었다. focal·해상도·
> center crop 등 OpenVLA 규약 항목은 건드리지 않았고 위치와 조준만 바꿨다.
>
> `configs/vla_spec.py` 의 `assert_workspace_visible()` 이 스펙 자체 검사로
> "작업공간이 center crop 후에도 화면 안" 을 확인한다. Isaac Sim 없이 도는 순수
> 계산이라 `python configs/vla_spec.py` 만으로 즉시 검증된다 — 카메라를 다시 만질
> 일이 생기면 렌더 전에 이걸 먼저 볼 것.
>
> **개편 후 검사 대상**: 소스 박스 내부 + 타깃 트레이 + 리프트 높이.
> 카메라 값은 확정한 것을 그대로 두고 ROI 만 지오메트리에 맞췄으며, 같은 검사를
> 통과한다.

### 1-3. 변형체 — 지금은 하지 않는다

개정 §8 로 변형체는 후속 확장으로 밀렸고, 시작점도 체적 FEM 이 아니라 **박막(2D)**
으로 바뀌었다. `deformable_env_cfg.py` 는 등록조차 하지 않았다 (파일 머리말 참조).
이 시간은 §1-4 데모 수집에 쓴다.

### 1-4. 텔레옵 데모 수집 (~2시간)

먼저 우리 태스크가 gym 에 보이는지 확인한다. Isaac Lab 의 도구 스크립트들은
`isaaclab_tasks` 만 import 하므로, `setup_isaaclab.sh` 가 깔아 둔
`.pth` 자동 등록이 없으면 `NameNotFound` 로 죽는다.

```bash
python -c "import gymnasium as gym; print([k for k in gym.registry if k.startswith('VlaPlace')])"
# 3개가 나와야 한다. 비어 있으면 먼저 어느 쪽 문제인지 가른다:
#   python -c "import vla_isaac_tasks; import gymnasium as gym; print([k for k in gym.registry if k.startswith('VlaPlace')])"
#     → 이쪽에서 나오면 .pth 만 안 걸린 것 / 에러 나면 패키지 설치 문제
#
# .pth 재설치 (경로는 반드시 sysconfig 로 — site.getsitepackages()[0] 은
# venv 안에서도 베이스 파이썬 경로를 돌려줄 수 있어 조용히 무시된다):
#   pip install -e source
#   echo "import vla_isaac_tasks" > "$(python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/vla_isaac_tasks.pth"
```

**★ `VLA_ANNOUNCE_TARGET=1` 을 반드시 붙일 것.** 블록 3개는 색만 다르고 형상이
같아서, 지시문을 모르면 어느 것을 집어야 하는지 알 방법이 없다. 이 환경변수를
켜면 **리셋할 때마다** 터미널에 타깃이 찍힌다:

```
============================================================
  [env 0 / 뱅크 #7]  ▶  PUT THE GREEN BLOCK INTO THE TRAY
============================================================
```

타깃을 틀리면 성공 판정이 안 뜨고, `record_demos.py` 는 성공한 것만 저장하므로
그 에피소드는 통째로 버려진다 — 에러는 안 나고 "왜 저장이 안 되지" 로만 보인다.
`R` 로 폐기해도 리셋이므로 안내가 다시 찍힌다. 세면서 따라갈 필요가 없다.

```bash
mkdir -p datasets
VLA_ANNOUNCE_TARGET=1 python $ISAACLAB_DIR/scripts/tools/record_demos.py \
    --task VlaPlace-v0 --teleop_device keyboard --enable_cameras \
    --dataset_file ./datasets/source.hdf5 --num_demos 15 \
    --step_hz 24 --num_success_steps 1 --livestream 2
```

`--enable_cameras` 는 필수다 — 씬에 `TiledCamera` 가 있어 없으면 센서 초기화가 실패한다.
`--step_hz 24` 는 우리 제어 주기(decimation 5 × dt 1/120 = 24Hz)에 맞춘 값이다.

**`--num_success_steps 1` 을 반드시 넘길 것.** 유지 시간은 환경이 이미 센다
(`SPEC.SUCCESS_HOLD_STEPS` = 48스텝 = 2초). `record_demos.py` 는 그 위에서 또
`--num_success_steps` 만큼 연속 성공을 세므로, 기본값 10 을 그대로 두면 총
57스텝(2.4초)을 요구하게 된다. 1 로 두면 정확히 2초다.

> **에피소드는 시간으로 끝나지 않는다.** `record_demos.py` 가
> `env_cfg.terminations.time_out = None` 으로 타임아웃을 꺼 버린다("목표에
> 도달할 때까지 무한히 돌린다"). 그래서 12.5초가 지나도 리셋되지 않는 것이
> **정상**이다. 녹화 중 종료 조건은 성공 판정과 `object_dropped` 둘뿐이다.
> 다만 12.5초(300스텝)는 평가·RFT 롤아웃에서는 실제로 잘리는 예산이므로,
> 데모가 그보다 길면 정책이 따라 할 시간이 없다는 뜻이다 — 짧게 만들 것.

> **녹화는 리스폰 즉시 시작된다.** 움직이기 시작할 때가 아니다
> (`running_recording_instance` 가 처음부터 True). 리셋 직후 멍하니 있는
> 시간도 그대로 데이터에 들어가므로, 지시문을 확인했으면 바로 움직일 것.

**품질 체크리스트 — 생성 성공률에 직결된다:**
- [ ] **안내에 찍힌 색의 블록을 집었는가** ← 틀리면 그 에피소드는 저장되지 않는다
- [ ] 트레이에 넣은 뒤 **2초는 그대로 둘 것** — 유지 조건을 못 채우면 성공이 안 뜬다
- [ ] 궤적이 짧은가 (불필요한 이동 최소화)
- [ ] 직선 경로인가 (축 단위로 나눠 움직이지 말 것)
- [ ] **일시정지가 없는가** ← 키보드 조작의 최대 함정. 멈춤은 정책이 학습하기 어렵다
- [ ] 녹화 끝에 여유 버퍼가 있는가

실수하면 `R` 로 폐기 후 리셋. 최소 10개 성공 데모.

**게임패드가 있으면 반드시 쓸 것** — 아날로그 스틱이라 SE(3) 연속 조작이 되어
데모 품질이 확연히 다르다. 키보드는 누를 때마다 팔이 계단식으로 튀는데,
그 계단이 그대로 Mimic 생성 성공률을 깎는다.

```bash
VLA_ANNOUNCE_TARGET=1 python $ISAACLAB_DIR/scripts/tools/record_demos.py \
    --task VlaPlace-v0 --teleop_device gamepad --enable_cameras \
    --dataset_file ./datasets/source.hdf5 --num_demos 15 \
    --step_hz 24 --num_success_steps 1 --livestream 2
```

두 장치 모두 `pickplace_env_cfg.py` 의 `teleop_devices` 에 등록되어 있다
(키보드 감도 0.05 / 게임패드 1.0·1.6 + dead_zone 0.01).
감도가 안 맞으면 그 값을 조정한다.

### 재생 검증 + 품질 선별

데모는 **한 파일에** `demo_0, demo_1, ...` 로 쌓인다. 성공했다고 품질이 좋은 것은
아니므로, 나쁜 것을 골라내고 간다 — 궤적이 길거나 중간에 멈춘 데모는 Mimic 생성
성공률을 직접 깎는다.

```bash
# 1) 무엇이 들어 있나 (길이·정지구간 통계, 의심 후보에 ←? 표시)
python scripts/filter_demos.py --inspect datasets/source.hdf5

# 2) 전체 재생 — 물리 비결정성으로 일부는 실패한다. 넉넉히 수집했으면 정상.
python $ISAACLAB_DIR/scripts/tools/replay_demos.py \
    --task VlaPlace-v0 --enable_cameras --dataset_file ./datasets/source.hdf5

# 3) 의심스러운 것만 골라 다시 재생 (--select_episodes 는 Isaac Lab 기본 기능)
python $ISAACLAB_DIR/scripts/tools/replay_demos.py \
    --task VlaPlace-v0 --enable_cameras \
    --dataset_file ./datasets/source.hdf5 --select_episodes 3 7 11

# 4) 좋은 것만 남긴다 (원본은 그대로 둔다)
python scripts/filter_demos.py --input datasets/source.hdf5 \
    --output datasets/source_clean.hdf5 --drop 3 7 11
```

이후 §1-5 어노테이션의 `--input_file` 을 `source_clean.hdf5` 로 바꿔 쓴다.
재생 성공률만 자동으로 보고 싶으면 `--validate_success_rate` 를 붙이면 된다.

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
python $ISAACLAB_DIR/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task VlaPlace-Visuomotor-Mimic-v0 --auto --enable_cameras \
    --annotate_subtask_start_signals \
    --input_file ./datasets/source.hdf5 --output_file ./datasets/annotated.hdf5

# 소량 시험 생성 — 성공률을 여기서 본다
python $ISAACLAB_DIR/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task VlaPlace-Visuomotor-Mimic-v0 --enable_cameras --use_skillgen \
    --num_envs 10 --generation_num_trials 20 \
    --input_file ./datasets/annotated.hdf5 --output_file ./datasets/generated_small.hdf5
```

#### MimicGen 후퇴 (cuRobo 가 막혔거나 SkillGen 이 에러를 낼 때)

**환경 코드는 고칠 것이 없다.** 두 플래그만 빼면 된다:

```bash
python $ISAACLAB_DIR/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task VlaPlace-Visuomotor-Mimic-v0 --auto --enable_cameras \
    --input_file ./datasets/source.hdf5 --output_file ./datasets/annotated.hdf5
#   ↑ --annotate_subtask_start_signals 없음

python $ISAACLAB_DIR/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task VlaPlace-Visuomotor-Mimic-v0 --enable_cameras \
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
python $ISAACLAB_DIR/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task VlaPlace-Visuomotor-Mimic-v0 --enable_cameras --headless --use_skillgen \
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

같은 이유로 **정책 경로도 여기서 대조한다.** RFT 의 샘플링은 OFT 의 parallel
decoding(placeholder 56토큰 + stop, 1회 forward, 액션 구간 양방향 어텐션)을
그대로 재현해야 한다 — `model.generate()` 같은 autoregressive 생성은 모델이
학습된 적 없는 방식이라 정책 분포가 조용히 달라진다.

```bash
python rft/grpo_fallback.py --verify-checkpoint --checkpoint ckpt/sft
# ✓ 정책 경로가 predict_action 과 일치한다   ← 이 줄이 나와야 진행
```

어긋나면 로짓 슬라이스 위치(`num_patches + num_prompt_tokens`)나 디토크나이즈
(`model.vocab_size` / `-1` / `bin_centers`)가 상류와 다른 것이다.

### 3-3. 베이스라인 성공률

```bash
python scripts/eval_rollout.py --checkpoint ckpt/sft \
    --task VlaPlace-v0 --out logs/sft_base.json
```

평가는 `eval_base` 홀드아웃 64개를 **인덱스 순서대로** 돈다. 시드가 아니라
파일이라, 나중에 RFT 체크포인트를 재면 정확히 같은 씬에서 비교된다.

해석:
- **≥30%** → 그대로 RL 을 돌린다
- 5~30% → 개선폭이 나오기 어렵지만 0 이 아니면 진행한다 (SimpleVLA-RL 은
  17.3% 에서 91.7% 까지 올린 사례가 있다)
- <5% → 학습 신호가 없다. 데이터/스펙부터 본다

**0 이면** 정책 문제가 아니라 파이프라인 문제다. 워커가 보내는
진단값 `lifted` (박스에서 꺼내는 데까지는 되는가) 와 `yaw_err` (각도) 를 먼저
볼 것. 둘 다 정상인데 성공률만 0 이면 성공 판정 배선을 의심한다 (§4-2).

추가로 Language split 도 여기서 한 번 재 둔다 (씬은 그대로, 문장만 바꾼다):

```bash
python scripts/eval_rollout.py --checkpoint ckpt/sft --rephrase 0 \
    --out logs/sft_lang0.json
```

### 3-4. ⚠ Fallback 트리거 — 정오 판단

**정오까지 어댑터로 롤아웃 1회가 안 돌면 자작 GRPO 로 전환한다.**
환경·보상·브리지·평가가 전부 재사용되므로 전환 비용은 RL 루프뿐이다.

### 3-5. RFT 착수

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

### 4-1. 개선폭 측정 — 이 프로젝트의 핵심 결과물

같은 홀드아웃으로 다시 잰다:

```bash
python scripts/eval_rollout.py --checkpoint logs/grpo_rigid/checkpoint-<N> \
    --task VlaPlace-v0 --out logs/rft_base.json
```

`logs/sft_base.json` 과 나란히 놓은 것이 핵심 결과다 — 같은 64개 초기 상태에서
SFT 대비 RFT 가 얼마나 올랐는가. 커브가 오르기 시작하는 것만 확인돼도 성립한다.

### 4-2. 커브가 오르지 않을 때

로그의 `그룹 used/attempts` 부터 본다. 무신호 그룹은 이제 재샘플링으로 흡수되므로
(dynamic sampling), **배치를 채웠는지**가 첫 지표다:

- `used < groups_per_step` 이 계속됨 → 시도 상한까지 가도 신호 있는 그룹이 안
  나온다. `temperature` 를 올리거나(1.6 → 1.8), SFT 베이스라인이
  30% 게이트를 넘는지 다시 볼 것 — 대개 후자다
- 배치는 채우는데 커브가 평평 → 워커 `diag` 로 어디서 막혔는지 가른다:
  `lifted` 가 낮으면 박스에서 꺼내지를 못하는 것, 높은데 성공률이 낮으면
  배치 정밀도 — `yaw_err` 로 각도 문제인지 본다
- 성공률이 **정확히 0 으로 고정** → 학습이 아니라 배선을 의심할 것. env 보상은
  `func × weight × dt` 라 성공해도 ≈0.042 다. 워커의 성공 임계가 절대값
  (0.5 같은 것)으로 돌아가 있으면 성공이 영원히 안 잡힌다
- 발산 → `learning_rate` 를 1e-6 → 3e-7, 또는 `kl_coef` 를 0.01 로

### 4-3. 후속 확장 (지금은 하지 않는다)

embodiment 축(팔만 교체), 변형체 박막, 그리고 **난이도 축 복원** — 트레이를
다시 좁혀 클리어런스 곡선을 그리는 것. 셋 다 기본 성공률을 확보한 뒤의
이야기다. 난이도를 되돌릴 때는 `SPEC.TRAY_INNER_SIZE` 를 줄이고
`SPEC.SUCCESS_REQUIRE_YAW` 를 True 로 올리면 되지만, 그 순간 텔레옵 데모를
다시 만들 수 있는지가 병목이 된다는 점을 기억할 것.

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

# 초기 상태 뱅크 확인
python scripts/make_init_states.py --show eval_base

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
- 초기 상태 뱅크를 다시 만들고 이전 결과와 비교 — 씬이 달라져 비교가 성립하지 않는다
- SFT 성공률 30% 미만인 split 에서 RL 돌리기 — 개선폭이 안 나온다 (개정 5절)
- SkillGen 을 쓰면서 `--annotate_subtask_start_signals` 누락 — 시작 경계가 없으면
  생성이 실패한다. 어노테이션과 생성의 플래그는 **한 쌍으로** 맞춰야 한다
- cuRobo 가 안 된다고 Day 1 을 통째로 쓰기 — MimicGen 후퇴는 플래그 2개다.
  오후까지 안 되면 후퇴하고 밤 배치를 지킬 것
