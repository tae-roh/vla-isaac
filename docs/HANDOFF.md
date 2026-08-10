# 인수인계 — 원격(Brev) 인스턴스, 스모크 통과 후 카메라 확정 대기

원격 Claude Code용 브리핑. 로컬에서 코드를 다 짜 두었고, 실제 인스턴스에서
돌려 보며 깨지는 곳을 고치는 단계다.

**현재 위치: 스모크 전체 통과 → RUNBOOK §1-2 카메라 확정에서 사람의 판단 대기 중.**
데모 수집(§1-4)은 카메라가 확정되기 전에는 시작하지 않는다.

## 지금 할 일

스모크는 **전체 통과했다** (2026-08-10, isaaclab 0.54.4 / isaacsim 5.1.0.0 / L40S).
lock도 `env/locks/isaaclab.lock.txt`에 박제·커밋됐다. 다시 확인하려면:

```bash
cd ~/workspace/vla-isaac && python env/smoke/check_isaaclab.py --full
```

스모크가 이 환경의 **유일한 검증 기준**이다 (`pip check`는 이 스택에서 원리적으로
깨끗해질 수 없으므로 기준이 아니다).

**RUNBOOK §1-2 카메라도 확정됐다.** 초기 설정은 목표 영역 중심이 프레임 밖
(u=225.0 / 224px)이었다 — 시선이 작업공간이 아니라 **로봇 베이스**(테이블 교점
(0.097, 0.0))를 향해 있었기 때문이다. 정책이 "어디에 놓아야 하는지" 를 볼 수 없는
상태였다. 확정값:

| | 이전 | 확정 |
|---|---|---|
| `CAMERA_POS` | `(1.05, 0.0, 0.55)` | `(1.2, 0.0, 0.8)` |
| `CAMERA_ROT` | `(0.35355, -0.61237, -0.61237, 0.35355)` | `(0.32818, -0.66487, -0.6017, 0.297)` |
| 시선-테이블 교점 | `(0.097, 0.0)` = 로봇 베이스 | `(0.36, 0.09)` = 작업공간 |
| 목표 영역 중심 | u=225.0 (**프레임 밖**) | u=175.4 |
| 작업공간 화면 점유 | 19% | 38% |

**focal length·aperture·클리핑·해상도는 원래 값 그대로다 — 위치와 조준만 바꿨다.**
OpenVLA 규약(224×224 / 단일 3인칭 뷰 / uint8 / center crop 0.9 / 7·8·8)은 어느 것도
건드리지 않았다. 카메라 외부 파라미터는 그 규약에 들어가지 않는다. 결과 기하는
거리 1.08m·하각 48도로 LIBERO agentview 에 오히려 더 가깝다.

이제 `configs/vla_spec.py` 의 `assert_workspace_visible()` 이 **스펙 자체 검사로**
작업공간이 화각 안에 있는지 확인한다 (`assert_consistent()` 이 호출 → 세 venv 의
스모크가 모두 검사). Isaac Sim 없이 순수 stdlib 로 도는 투영이라, 카메라·목표 영역·
스폰 범위 중 하나만 바뀌어도 즉시 잡힌다. 이전 설정을 넣으면 12개 지점에서 실패한다.
해석적 투영과 실제 렌더는 1px 안에서 일치하는 것을 확인했다.

## 프로젝트 한 줄 요약

Isaac Lab에서 형상 랜덤화된 자재를 집어 목표 영역으로 옮기는 태스크를 만들고,
OpenVLA-OFT를 SFT한 뒤 GRPO로 RFT까지 3~4일 안에 완주한다. Franka Emika Panda.
전체 실행 순서는 `docs/RUNBOOK.md`.

## 환경

| | |
|---|---|
| 저장소 | `~/workspace/vla-isaac` |
| venv | `~/env_isaaclab` (`source ~/env_isaaclab/bin/activate`) |
| Isaac Lab | `~/workspace/IsaacLab` (isaaclab 0.54.4 / isaacsim 5.1.0.0) |
| GPU | L40S 1장 |

⚠ **`ISAACLAB_DIR`이 `~/.bashrc`에 없다.** `setup_isaaclab.sh`가 기록하도록 되어
있는데(스크립트 152~159줄) 실제로는 비어 있다. `OMNI_KIT_ACCEPT_EULA`도 없다
(Isaac Sim은 지금 잘 뜨므로 이쪽은 급하지 않다). RUNBOOK §1-4 이후의 명령들이
`python $ISAACLAB_DIR/scripts/tools/record_demos.py` 형태라 그대로 두면
`No such file`로 죽는다. **데모 수집 전에** 한 번 넣어 둘 것:

```bash
echo 'export ISAACLAB_DIR=~/workspace/IsaacLab' >> ~/.bashrc && source ~/.bashrc
```

## 이미 확인된 상태

- Isaac Sim, torch 2.7.0, gymnasium: 설치됨
- `isaaclab` / `isaaclab_tasks` / `isaaclab_mimic` / `isaaclab_assets`: 설치됨
- `vla_isaac_tasks`: editable 설치 + `.pth` 자동 등록 완료, gym에 4종 등록됨
- 씬 생성, 액션 매니저(7차원 = arm 6 + gripper 1)까지 정상 동작 확인

- `Visuomotor-Mimic 환경 생성`까지 통과, 카메라 관측 `(2, 224, 224, 3) uint8` 확인
- 텔레옵 장치 keyboard/gamepad 등록 확인

**해결된 막힘**: `gripper_joint_names` 수정은 맞았고, 바로 다음 것에서 죽었다 —
`AttributeError: 'PickPlaceVisuomotorMimicEnvCfg' object has no attribute 'gripper_open_val'`.
이 리비전의 `stack.mdp.observations.object_grasped`는 `gripper_joint_names`만
`hasattr`로 확인하고 `gripper_open_val`·`gripper_threshold`는 그냥 참조한다.
**셋이 한 세트다.** 지금은 세 개 모두 설정되어 있다.

## ★ 가장 중요한 교훈 — 이것 때문에 세 번 깨졌다

**이 저장소는 Isaac Lab `main` 문서/소스를 보고 작성됐는데, 이 인스턴스의 체크아웃은
리비전이 다를 수 있다.** API가 어긋나면 원격 문서가 아니라 **로컬 소스를 직접 읽어라.**

```bash
grep -rn "찾는_심볼" $ISAACLAB_DIR/source/ | head -20
```

실제로 이렇게 깨졌다:

| 증상 | 원인 |
|---|---|
| `SubTaskConfig got unexpected keyword 'subtask_start_signal'` | 문서 요약엔 있었지만 실제 dataclass엔 없는 필드. 시작 신호는 `get_subtask_start_signals()` 메서드로 받는다 |
| `Cannot find gripper_joint_names` | 위치를 main 소스에서 못 찾음. 로컬 grep이 확정 |
| `No module named 'pxr'` | `AppLauncher`보다 먼저 `isaaclab`을 import함 |
| `no attribute 'gripper_open_val'` | `gripper_joint_names`만 넣어서. 셋이 한 세트다 |
| 성공 판정이 영원히 False | `gripper_pos`가 `[f1, -f2]`를 돌려주는데 `.sum()`을 씀 (아래) |

**Isaac Lab 함수의 반환 규약도 소스로 확인할 것.** 시그니처가 맞아도 의미가 다를 수 있다.
`gripper_pos`는 두 번째 손가락 부호를 뒤집어 `[f1, -f2]`를 돌려준다(LIBERO/robosuite
qpos 규약). Franka 두 손가락은 둘 다 `0~+0.04`로 대칭 이동하므로 `.sum()`은 항상 ≈0 이다.
`placed_signal`이 그 합으로 그리퍼 개방을 판정하고 있어서 **성공이 절대 뜨지 않았다.**
에러가 없어서 데모 수집·RFT 보상 0 으로만 드러나는 종류의 버그다. `.abs()`로 고쳤다.
실측으로 확인한 값 (`VlaPick-v0`, 임계값 `GRIPPER_OPEN_QPOS_SUM=0.06`):

| 그리퍼 명령 | `gripper_pos` | 수정 전 `sum()` | 수정 후 `abs().sum()` |
|---|---|---|---|
| OPEN | `[+0.0400, -0.0400]` | `-0.0000` → False | `+0.0800` → **True** |
| CLOSE | `[+0.0001, -0.0001]` | `-0.0000` → False | `+0.0002` → False |

곁가지로 하나 더: `reset_robot` 이벤트의 `reset_joints_by_offset` 이 `joint_names` 없이
로봇 전체에 걸려 있어서 **손가락 관절까지 ±0.05 랜덤화된다** (리셋 직후 실측
`[0.0400, 0.0029]` — 한쪽이 거의 닫힌 채로 시작). 첫 OPEN 명령이면 풀리므로
급하지는 않은데, 팔 관절만 흔들 의도였다면 `joint_names=[SPEC.ARM_JOINT_REGEX]`를
넣어야 한다.

## ★ 운영 함정 — 실패한 Isaac Sim은 죽지 않는다

`gym.make`가 예외를 던지면 그 뒤 `simulation_app.close()`가 **돌아오지 않는다.**
프로세스가 GPU 4.3GB를 물고 CPU를 계속 태우며 남는다. 이게 두 번 시간을 잡아먹었다:

- 남은 프로세스 2개 때문에 L40S 하나를 셋이 나눠 쓰게 되어, 환경 생성이 **10분 넘게
  멈춘 것처럼** 보였다. 정리하니 **12초** 만에 진짜 에러가 났다.
- 즉 "느리다/멈췄다"를 디버깅하기 전에 **먼저 유령 프로세스를 확인할 것.**

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
ps -eo pid,etime,time,args | grep -E "[c]heck_isaaclab|[t]est_deformable|[d]ump_obs"
pkill -9 -f check_isaaclab    # 필요하면
```

스크립트를 하나 돌리기 전에 GPU가 비어 있는지 보는 습관이 제일 싸게 먹힌다.

## §1-3 변형체 예비 테스트 — 결과를 믿지 말 것

두 가지 문제를 확인했다. **고치지 않았다** (변형체는 Phase 4b 스트레치이고
RUNBOOK이 타임박스를 지시하므로). 다만 결과를 그대로 해석하면 안 된다.

**(1) `--youngs` 를 여러 개 주면 두 번째에서 멈춘다.** 한 `SimulationApp` 안에서
`gym.make` 를 두 번 하는 구조인데, Isaac Lab은 같은 앱에서 `InteractiveScene` 을
허물고 다시 세우는 것을 지원하지 않는다. 두 번째 환경 생성에서 로그 한 줄 없이
9분 넘게 CPU만 태웠다. **값 하나당 프로세스 하나로 돌릴 것:**

```bash
for Y in 5e5 1e5 5e4; do python scripts/test_deformable_grasp.py --headless --youngs $Y; done
```

**(2) 그렇게 돌린 결과는 영률과 무관하다** — 10배 범위에서 값이 전부 똑같다:

| 영률 | 상승 | 퍼짐배율 | FPS | 파지 |
|---|---|---|---|---|
| 5.0e+05 | 3.9cm | 1.00x | 13.3 | ✗ |
| 1.0e+05 | 3.9cm | 1.00x | 13.4 | ✗ |
| 5.0e+04 | 3.9cm | 1.00x | 13.5 | ✗ |

퍼짐배율이 정확히 1.00x = **노드가 한 번도 변형되지 않았다** = 그리퍼가 물체에
닿은 적이 없다. 원인은 FEM이 아니라 스크립트다: `phase_action()` 이 물체 위치를
보지 않고 `dz=-0.35` 로 곧장 내려가는데, 물체는 리셋마다 `y∈[-0.22,-0.02]` 로
랜덤 배치된다(팔은 `y≈0` 에서 내려간다). 최대 22cm 빗나가므로 어떤 영률에서도
잡힐 수 없다.

**따라서 "어떤 설정에서도 파지 실패 → 변형체 제외" 라는 결론을 여기서 내리면 안 된다.**
그건 물리 결과가 아니라 스크립트가 물체를 조준하지 않는다는 사실일 뿐이다. FEM 자체는
안정적이고(발산 없음) 13 FPS 로 돈다 — 그 두 가지는 유효한 정보다. 판단이 필요하면
`phase_action()` 이 `deformable_center()` 로 물체 xy 를 먼저 잡아 가로로 정렬한 뒤
하강하게 고쳐야 한다.

## 절대 바꾸지 말 것

이것들은 서로 맞물려 있어서, 하나 바꾸면 SFT 데이터를 다시 만들거나 모델 코드를
고쳐야 한다. 3~4일 일정이 여기 걸려 있다.

1. **`configs/vla_spec.py`의 `ACTION_DIM=7` / `PROPRIO_DIM=8` / `NUM_ACTIONS_CHUNK=8`**
   openvla-oft의 LIBERO 상수셋과 일치시킨 값이다. 이 일치 덕분에 openvla-oft와
   SimpleVLA-RL의 모델 코드를 한 줄도 안 고친다. 바꾸는 순간 그 전제가 깨진다.
2. **카메라 설정** (`CAMERA_POS` / `CAMERA_ROT` / 224×224)
   Day 1에 사람이 눈으로 보고 확정할 값이다. 스모크를 통과시키려고 임의로 만지지 말 것.
3. **torch 버전** — Isaac Sim 5.1.0이 2.7.0에 고정. 상향 금지.
4. **`./isaaclab.sh --install none`** — `--install`(all)로 바꾸면 sb3/rl_games가
   isaacsim의 핀을 덮어쓰며 torch·starlette·click 연쇄 충돌이 난다.
5. **`AppLauncher`보다 먼저 isaaclab 계열 import 금지** — `pxr`(OpenUSD)은
   SimulationApp이 뜬 뒤에야 경로에 잡힌다.
6. **constraints 없는 `pip install` 금지** — 항상 `-c env/constraints.txt`.

## 설계 불변식 (고칠 때 유지해야 하는 것)

- `configs/vla_spec.py`가 관측·액션 스펙의 **단일 진실 공급원**이다. 환경 cfg, SFT,
  RFT가 모두 이 파일만 읽는다. 값을 두 곳에 적지 말 것.
- 성공 판정은 `mdp/observations.py`의 `placed_signal()` 하나에서 나오고,
  termination·reward·Mimic 시그널이 전부 그것을 재사용한다. 따로 구현하지 말 것.
- Mimic 관측 그룹은 `concatenate_terms=False`여야 한다. Mimic이 `obs_buf["policy"]["eef_pos"]`
  처럼 **키로** 접근하므로 이어 붙이면 KeyError로 죽는다.
- 씬은 `replicate_physics=False`여야 한다. `MultiAssetSpawnerCfg`가 env마다 다른
  지오메트리를 만들기 때문. True로 두면 **에러 없이** 모든 env가 같은 형상이 된다.
- SkillGen과 MimicGen 두 경로를 분기 없이 지원하도록 만들어 뒀다. SkillGen이 막히면
  `--use_skillgen`과 `--annotate_subtask_start_signals` 두 플래그만 빼면 되고,
  환경 코드는 고치지 않는다.

## 고칠 때

- 수정은 **최소 diff**로. 스모크를 통과시키려고 구조를 재설계하지 말 것.
- 스펙 값을 바꿔서 통과시키는 것과 코드를 고쳐서 통과시키는 것은 다르다. 후자를 할 것.
- 무엇을 왜 고쳤는지 짧은 주석을 남길 것. 특히 Isaac Lab 버전 차이 때문이라면
  "이 리비전에서는 X가 Y다"를 적어 두면 다음 페이즈에서 같은 일을 겪지 않는다.
- 커밋 메시지는 한 줄로 짧게. 통과할 때마다 커밋하고 푸시할 것.

```bash
git add -A && git commit -m "<한 줄>" && git push origin main
```

## 스모크가 통과하면

`docs/RUNBOOK.md` §1-2(카메라 확정)로 넘어간다. 그 단계는 **사람이 눈으로 봐야 하므로**
자동으로 진행하지 말고 사용자에게 알릴 것.

## 참고 파일

| 파일 | 무엇 |
|---|---|
| `configs/vla_spec.py` | 스펙 SSOT + embodiment 정의 |
| `source/vla_isaac_tasks/` | 태스크 환경 (gym 4종 등록) |
| `env/smoke/check_isaaclab.py` | 지금 돌리는 스모크 |
| `docs/RUNBOOK.md` | Day 1~4 전체 실행 순서 |
| `rft/README.md` | RFT 아키텍처 + 흔한 실패 표 |
