# 인수인계 — 원격(Brev) 인스턴스에서 스모크 테스트 통과시키기

원격 Claude Code용 브리핑. 로컬에서 코드를 다 짜 두었고, 이제 실제 인스턴스에서
돌려 보며 깨지는 곳을 고치는 단계다.

## 지금 할 일

```bash
cd ~/workspace/vla-isaac && python env/smoke/check_isaaclab.py --full
```

**통과할 때까지 반복한다.** 실패 항목을 고치고, 다시 돌리고, 통과하면 lock을 박제한다.
스모크가 이 환경의 **유일한 검증 기준**이다 (`pip check`는 이 스택에서 원리적으로
깨끗해질 수 없으므로 기준이 아니다).

통과 후:

```bash
pip freeze > env/locks/isaaclab.lock.txt
```

## 프로젝트 한 줄 요약

Isaac Lab에서 형상 랜덤화된 자재를 집어 목표 영역으로 옮기는 태스크를 만들고,
OpenVLA-OFT를 SFT한 뒤 GRPO로 RFT까지 3~4일 안에 완주한다. Franka Emika Panda.
전체 실행 순서는 `docs/RUNBOOK.md`.

## 환경

| | |
|---|---|
| 저장소 | `~/workspace/vla-isaac` |
| venv | `~/env_isaaclab` (`source ~/env_isaaclab/bin/activate`) |
| Isaac Lab | `$ISAACLAB_DIR` = `~/workspace/IsaacLab` (`~/.bashrc`에 기록됨) |
| GPU | L40S |

## 이미 확인된 상태

- Isaac Sim, torch 2.7.0, gymnasium: 설치됨
- `isaaclab` / `isaaclab_tasks` / `isaaclab_mimic` / `isaaclab_assets`: 설치됨
- `vla_isaac_tasks`: editable 설치 + `.pth` 자동 등록 완료, gym에 4종 등록됨
- 씬 생성, 액션 매니저(7차원 = arm 6 + gripper 1)까지 정상 동작 확인

**마지막으로 막힌 지점**: `Visuomotor-Mimic 환경 생성`에서
`NotImplementedError: Cannot find gripper_joint_names in the environment config`

이에 대한 수정(`PickPlaceEnvCfg.__post_init__`에서 `self.gripper_joint_names` 설정)을
커밋했지만 **아직 실행 검증되지 않았다.** 이것부터 확인할 것.

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
