# vla-isaac

Isaac Lab 환경에서 VLA(OpenVLA-OFT)를 **SFT → RFT** 하는 프로젝트.
태스크: **얕은 박스 안의 블록 3개 중 지시문이 지정한 하나를 꺼내, 지시문이
지정한 맞춤 포켓에 안착** (Franka Panda). 난이도 손잡이는 두 개 — 블록의
**대칭 차수**(이산)와 포켓 **클리어런스**(연속 5mm→0.5mm)다.
핵심 결과물은 클리어런스 곡선 `SR_SFT(c)` vs `SR_SFT+RL(c)`.

전체 실행 순서는 **[docs/RUNBOOK.md](docs/RUNBOOK.md)** 를 볼 것.
설계 근거는 상위 디렉토리의 `vla-sft-rft-project-plan.md`.

## 이 저장소가 푸는 문제

계획서의 원안은 2~5일 SFT + LIBERO 재현 + veRL 어댑터로 총 1~2주 규모였다.
**3~4일 안에 유의미한 결과를 내기 위해** 세 가지를 바꿨다:

1. **로컬에서 전체 코드를 먼저 쓴다.** 인스턴스 시간은 실행·디버깅에만 쓴다.
2. **Day 2 부터 두 인스턴스가 병렬로 돈다.** SFT(H100)를 기다리며 L40S 를 놀리지 않는다.
3. **핵심 결과를 강체 RFT 커브로 잡는다.** 변형체는 스트레치 — FEM 튜닝이
   일정을 잡아먹는 가장 큰 변수라, 결과 보장을 물리 디버깅에 걸지 않는다.

## 설계 결정

### 0. Embodiment: Franka Emika Panda

세 가지가 동시에 맞아떨어지는 유일한 선택이라서다.

| 근거 | 내용 |
|---|---|
| **LIBERO 가 Franka 다** | 액션 차원(7)만 맞는 게 아니라 **의미**까지 맞는다 — 같은 팔, 같은 평행 2지 그리퍼, 같은 델타 EEF 제어 |
| **OpenVLA 사전학습 분포** | `openvla/openvla-7b` 의 Open X-Embodiment 학습 데이터에 Franka 계열 비중이 크다 → prior 활용도가 높다 |
| **Isaac Lab Mimic 레퍼런스가 전부 Franka** | `FrankaCubeStackIKRelMimicEnv` 복사 개조가 헬퍼 구현 리스크 대응이었다. 로봇을 바꾸면 이 레퍼런스를 못 쓴다 |

제어는 `DifferentialIKController` 상대 포즈 모드(dls), 델타는 **로봇 베이스 프레임**,
그리퍼는 이진 개폐(최대 8cm). 프리셋은 `FRANKA_PANDA_HIGH_PD_CFG` — 기본 게인
(80/4)으로는 태스크공간 IK 추종이 안 되고 HIGH_PD(400/80)여야 한다.

> **제어 규약이 LIBERO 와 똑같아야 하는 건 아니다.** 우리는 우리 Isaac Lab 데이터로
> 처음부터 SFT 하므로 정책이 우리 규약을 학습한다. 반드시 맞아야 하는 것은
> "SFT 데이터 ↔ RFT 롤아웃" 인데 둘 다 같은 환경이라 구조적으로 보장된다.
> LIBERO 일치는 정확성 요건이 아니라 **사전학습 prior 활용도**의 문제다.

관련 식별자(관절/바디/링크 이름, 오프셋, 그리퍼 기하)는 전부
[`configs/vla_spec.py`](configs/vla_spec.py) 의 embodiment 섹션에 모여 있고,
환경 cfg 는 문자열을 직접 쓰지 않고 거기서 가져온다.

### 1. LIBERO 스펙에 정확히 맞춘다 → 모델 코드 수정 0

openvla-oft 의 `prismatic/vla/constants.py` 는 실행 커맨드에 `libero` 가 있으면
`ACTION_DIM=7, PROPRIO_DIM=8, NUM_ACTIONS_CHUNK=8` 을 고른다. 우리 Isaac Lab
태스크를 이 숫자에 맞추면 **openvla-oft 와 SimpleVLA-RL 의 모델 코드를 한 줄도
고치지 않아도 된다.** 3~4일이 성립하는 가장 큰 전제다.

- 액션 7차원 = IK 상대 델타 포즈(6) + 이진 그리퍼(1)
- proprio 8차원 = eef 위치(3) + axis-angle(3) + 그리퍼 관절(2)
- 3인칭 단일 뷰 224×224, 청크 8

모든 수치는 [`configs/vla_spec.py`](configs/vla_spec.py) 한 곳에 있고,
Isaac Lab 환경·SFT·RFT 가 전부 이 파일만 읽는다.

### 2. 이산 액션 토큰 헤드 (연속 L1 회귀 아님)

GRPO 는 `log π(a|s)` 와 정책 ratio 가 필요하다. 연속 L1 회귀 헤드는
결정론적이라 확률분포가 없어 **RL 을 붙일 수 없다.** 그래서 SFT 단계부터
`--use_l1_regression False` 로 이산 토큰 + cross-entropy 를 쓴다.
이 한 줄이 "SFT 를 왜 이렇게 하는가" 의 절반이다.

### 3. 초기 상태를 시드가 아니라 **뱅크 인덱스**로 지정한다

GRPO 는 같은 s₀ 에서 G개 궤적을 뽑는 것을 전제한다. 리셋마다 배치가 달라지면

```
Â_i = (R_i − mean(R)) / std(R)
```

가 "정책이 잘했는가" 가 아니라 "이번 리스폰이 쉬웠는가" 를 재게 되어 보상이
노이즈가 된다. 시드만 맞추는 것으로는 부족하다 — 배치 안의 env 들이 여전히
서로 다른 배치를 받기 때문이다.

그래서 초기 배치·타깃 블록·타깃 슬롯을 미리 만들어 `datasets/init_states/*.npz`
로 굳혀 두고, 리셋 때 **인덱스로 꺼낸다.** 정수 하나를 주면 전 env 가 같은
s₀ 에서 출발한다 (`client.reset(init_index=…)`).

- 학습/생성: `train` 뱅크를 순회
- 평가: `eval_base` 홀드아웃 64개를 인덱스 순서대로 (시드 12345, 학습과 분리)
- **RL 코드보다 먼저 만들었다.** 나중에 붙이면 그 전 실험이 전부 재현 불가가 된다.

### 4. 언어 채널이 죽지 않게 만든다

블록 3개는 **색만 다르고 형상·크기가 같다.** "하나만 튀는" 구성이면 모델이
지시문을 무시하고 "튀는 색으로 가라" 만 배워도 만점이 나온다. 포켓 3개도 동일
규격이라, 블록–슬롯 대응은 **오직 지시문으로만** 결정된다.

같은 이유로 Mimic 의 `object_ref` 는 씬 엔티티 이름이 아니라 역할 이름
(`target` / `pocket`)이다. 고정 엔티티로 두면 증강 궤적이 전부 같은 블록·같은
포켓으로 가고, 데이터에서 언어 조건이 사라진다.

지시문 문자열의 출처는 관측의 `target_ids` 하나다 — 롤아웃 워커와 RLDS 변환이
같은 값에서 문장을 만든다. 두 곳에서 따로 만들면 SFT 와 RFT 의 지시문이
어긋나고, 증상은 "RFT 를 켜니 성능이 무너진다" 로만 나타난다.

### 5. 데이터 증강은 SkillGen 기본, MimicGen 은 한 플래그로 후퇴

두 방식의 차이는 **서브태스크 사이의 자유공간을 어떻게 잇는가** 하나다.

| | 자유공간 처리 | 결과 |
|---|---|---|
| MimicGen | 선형 보간 | 스티칭 구간에서 충돌 → 생성 실패 |
| SkillGen | cuRobo 모션 플래닝 | 충돌 회피 → 생성 성공률↑ |

계획서 §6 이 꼽은 Phase 2 최대 리스크가 "생성 성공률 저조(<10%)" 인데,
그 주 원인이 바로 선형 보간 스티칭 충돌이다. 그래서 SkillGen 을 기본으로 둔다.

다만 cuRobo 는 프리빌트 휠이 없어 소스 빌드(~20분)가 필요하고, Isaac Lab 문서가
**Isaac Sim 6.0.0 기준**인데 우리는 5.1.0 에 고정되어 있다. Day 1 에 막힐 수 있는
지점이라, 환경 코드가 **양쪽을 동시에 지원**하도록 만들었다:

- `SubTaskConfig` 에 `subtask_start_signal` 을 넣어 둔다 (SkillGen 전용, MimicGen 은 무시)
- 관측 그룹에 시작 시그널 2종을 등록해 둔다 (`grasp_start`, `place_start`)
- `use_skillgen` 은 cfg 에 박지 않고 **CLI 플래그로만** 지배한다

→ 후퇴할 때 고칠 코드가 없다. `--use_skillgen` 과
`--annotate_subtask_start_signals` 두 플래그만 빼면 된다.

### 6. RFT 는 프로세스 분리 (계획서의 순서를 뒤집음)

계획서는 "단일 venv 통합을 먼저 시도, 실패하면 프로세스 분리" 였다. 그런데
SimpleVLA-RL 의 `rob_rollout.py` 를 실제로 읽어 보면 LIBERO 를
`multiprocessing.Process` + Queue 로 **이미 격리해서 돌리고 있다.**
프로세스 분리는 우회로가 아니라 상류가 이미 쓰는 구조다 → 1순위로 채택했다.

결과: Isaac Sim(torch 2.7.0)과 veRL/vLLM 의 torch 충돌이 **구조적으로 소멸**한다.
계획서가 "이 페이즈 최대의 의존성 난점" 으로 꼽은 항목이 사라진 것이다.

## 구조

```
configs/vla_spec.py          ★ 관측·액션 스펙의 단일 진실 공급원 (SSOT)
env/                         의존성 정의 + 스모크 테스트 (코드보다 먼저 만든 것)
  constraints*.txt             환경별 버전 핀 — 모든 pip install 에 -c 로 건다
  smoke/check_*.py             ★ 유일한 환경 검증 기준 (pip check 아님)
setup/setup_*.sh             환경 구축 (멱등, 스모크 자동 실행, lock 박제)
source/vla_isaac_tasks/      Isaac Lab 태스크 — gym 등록 15종
  pickplace_env_cfg.py         VlaPlace-v0             텔레옵·재생·평가
  pickplace_mimic_env*.py      VlaPlace-*-Mimic-v0     Mimic 증강 (헬퍼 6종)
                               VlaPlace-c{5,2,1,0p5}mm-*  클리어런스 split
  scene_assets.py              블록 3개 + 소스 박스 + 포켓 트레이 (전부 프리미티브)
  init_states.py               ★ 초기 상태 뱅크 — s0 를 인덱스로 지정한다
  deformable_env_cfg.py        보류 (등록 안 됨. 후속 확장에서 박막부터 재개)
scripts/                     데이터 파이프라인 + 평가
  make_init_states.py          ★ 초기 상태 뱅크 생성 — RL 코드보다 먼저 돌린다
  dump_obs_reference.py        ★ 관측 PNG 덤프/대조 — 스펙 불일치의 유일한 검출 수단
  convert_hdf5_to_rlds.py      HDF5 → RLDS + 정규화 통계
  eval_rollout.py              성공률 측정
  sft/run_sft_libero_spec.sh   SFT 실행
rft/                         RFT 어댑터
  ipc_bridge.py                프로세스 간 프로토콜 (순수 stdlib)
  isaaclab_rollout_worker.py   배치 롤아웃 워커 (isaaclab venv 에서 실행)
  grpo_fallback.py             자작 GRPO (Day 3 정오 fallback)
docs/RUNBOOK.md              ★ Day 1~4 실행 순서
```

## 세 개의 venv, 세 개의 constraints

같은 저장소를 세 환경이 공유하지만 **의존성은 절대 섞지 않는다.**

| venv | 용도 | torch | constraints |
|---|---|---|---|
| `env_isaaclab` | Isaac Sim, 데이터 생성, 롤아웃 워커 | 2.7.0 (고정) | `constraints.txt` |
| `env_vla_train` | SFT (H100 — Isaac Sim 없음) | 2.2.0 (openvla-oft 기준) | `constraints.vla-train.txt` |
| `env_rft` | veRL / vLLM / GRPO | veRL 기준 | `constraints.rft.txt` |

`torch==2.7.0` 핀은 **Isaac Sim 5.1.0 이 검증한 버전이라는 이유 하나로** 존재한다.
SFT 인스턴스에는 Isaac Sim 이 없으므로 그 핀을 적용할 근거가 없고, 적용하면
오히려 openvla-oft 가 검증한 스택을 깨뜨린다. 그래서 파일을 나눴다.

## 시작하기

```bash
# L40S 인스턴스에서
chmod +x setup/*.sh && FULL_SMOKE=1 ./setup/setup_isaaclab.sh
source ~/env_isaaclab/bin/activate
python configs/vla_spec.py          # 스펙 확인
```

이후는 [docs/RUNBOOK.md](docs/RUNBOOK.md).

## 절대 하지 말 것

- `./isaaclab.sh --install` (all) → 반드시 `--install none`. sb3/rl_games 가
  isaacsim 의 핀을 덮어쓰며 연쇄 충돌을 일으킨다
- constraints 없는 `pip install`
- torch 버전 상향
- `pip check` 통과를 검증 기준으로 삼기 — 이 스택에서는 원리적으로 불가능
- Phase 2 이후 `configs/vla_spec.py` 의 카메라/액션/청크 변경 — SFT 데이터를
  전부 다시 만들어야 한다
