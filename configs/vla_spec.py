# =============================================================================
# configs/vla_spec.py
#
# 관측 / 액션 스펙의 단일 진실 공급원 (SSOT).
#
# 계획서 §6 리스크 표의 "SFT-RFT 스펙 불일치 (뷰/proprio/정규화) → 원인불명 성능
# 붕괴" 를 구조적으로 막기 위한 파일이다. Isaac Lab 환경 코드, SFT 실행 스크립트,
# RFT 어댑터가 **모두 이 파일 하나만 읽는다.** 값을 바꾸려면 여기서 바꾸고,
# 바꾼 뒤에는 SFT 데이터를 다시 만들어야 한다 (Phase 2 이후 변경 금지).
#
# ★ 설계 원칙: LIBERO 스펙과 정확히 일치시킨다
#
#   openvla-oft 의 prismatic/vla/constants.py 는 실행 커맨드에 "libero" 가
#   포함되면 LIBERO 상수셋을 자동 선택한다:
#       NUM_ACTIONS_CHUNK = 8, ACTION_DIM = 7, PROPRIO_DIM = 8,
#       ACTION_PROPRIO_NORMALIZATION_TYPE = BOUNDS_Q99
#
#   우리 Isaac Lab 태스크를 이 숫자에 정확히 맞추면 openvla-oft 와
#   SimpleVLA-RL 의 모델 코드를 **한 줄도 고치지 않아도 된다.**
#   이것이 3~4일 일정이 성립하는 핵심 전제이므로, 아래 값들은 임의로 바꾸지 말 것.
#
# 이 파일은 순수 stdlib 로만 작성한다 — isaaclab / vla-train / rft 세 venv 어디에서든
# 의존성 없이 import 되어야 하기 때문이다.
# =============================================================================

from __future__ import annotations

import math

# =============================================================================
# Embodiment (로봇 플랫폼)
# =============================================================================
# ★ 왜 Franka Emika Panda 인가 — 세 가지가 동시에 맞아떨어지는 유일한 선택이다
#
#   1) LIBERO 가 Franka Panda 다.
#      우리는 액션/proprio 차원을 LIBERO 스펙(7/8/8)에 맞춰 openvla-oft 와
#      SimpleVLA-RL 의 모델 코드를 그대로 쓰기로 했다. 그런데 그 일치는 숫자만의
#      문제가 아니다 — LIBERO 는 robosuite 위의 Franka Panda 를 델타 EEF 로
#      제어한다. 같은 팔·같은 그리퍼·같은 제어 방식이라 7차원의 **의미**까지
#      일치한다. 다른 로봇을 골랐다면 차원만 맞고 의미는 어긋났을 것이다.
#
#   2) OpenVLA 의 사전학습 분포에 Franka 가 두텁게 들어 있다.
#      openvla/openvla-7b 는 Open X-Embodiment 로 학습되었고 Franka 계열
#      데이터셋의 비중이 크다. 사전학습 prior 를 가장 잘 활용할 수 있는 몸체다.
#
#   3) Isaac Lab 의 Mimic 레퍼런스가 전부 Franka 다.
#      FrankaCubeStackIKRelMimicEnv 를 복사 개조하는 것이 계획서 §6 의
#      "Mimic 헬퍼 구현 난이도" 리스크 대응이었다. 로봇을 바꾸면 그 레퍼런스를
#      못 쓰고, 3~4일 예산에서 그럴 여유가 없다.
#
# ★ 제어 방식이 LIBERO 와 정확히 같아야 하는가 → 아니다 (오해하기 쉬운 지점)
#   우리는 **우리 Isaac Lab 데이터로 처음부터 SFT** 한다. 따라서 델타 스케일이나
#   좌표계 규약이 LIBERO 와 달라도 정책이 우리 데이터의 규약을 학습한다.
#   반드시 맞아야 하는 것은 "SFT 데이터와 RFT 롤아웃이 같은 규약" 인데, 둘 다
#   같은 Isaac Lab 환경을 쓰므로 구조적으로 보장된다.
#   LIBERO 와의 일치는 정확성 요건이 아니라 **사전학습 prior 활용도**의 문제다.

ROBOT_NAME = "Franka Emika Panda"
# Isaac Lab 프리셋. HIGH_PD 여야 한다 — 기본 게인(stiffness 80/damping 4)으로는
# 태스크공간 IK 제어가 목표 포즈를 제대로 추종하지 못한다. HIGH_PD 는 400/80 이고
# 중력보상이 켜져 있어 Isaac Lab 문서가 differential IK 용으로 명시 권장한다.
ROBOT_ASSET_CFG = "isaaclab_assets.robots.franka:FRANKA_PANDA_HIGH_PD_CFG"

ARM_DOF = 7                          # 팔 관절 수 (그리퍼 제외)
ARM_JOINT_REGEX = "panda_joint.*"    # panda_joint1..7

# 엔드이펙터
BASE_LINK_NAME = "panda_link0"       # FrameTransformer 의 기준 프레임
EEF_BODY_NAME = "panda_hand"         # IK 가 제어하는 바디
# IK 제어점 오프셋 — panda_hand 원점에서 손가락 사이 파지점까지.
EEF_BODY_OFFSET = (0.0, 0.0, 0.107)
# 관측용 eef 프레임 오프셋. IK 제어점과 미세하게 다른 것은 Isaac Lab 레퍼런스
# 그대로다 (제어점 0.107 vs 관측점 0.1034). 굳이 통일하지 않는다 —
# 레퍼런스에서 검증된 값이고, Mimic 헬퍼가 관측값을 그대로 쓰기 때문이다.
EEF_FRAME_OFFSET = (0.0, 0.0, 0.1034)

# 그리퍼 (평행 2지)
GRIPPER_JOINT_REGEX = "panda_finger.*"       # 액션이 제어할 관절 선택용
GRIPPER_COMMAND_REGEX = "panda_finger_.*"    # 열림/닫힘 명령 표현식의 키
# 정규식이 아닌 실제 관절 이름. Isaac Lab Mimic 이 환경 cfg 에서
# `gripper_joint_names` 를 찾는다 (없으면 NotImplementedError).
# 정규식으로는 안 되고 이름 목록이어야 한다.
GRIPPER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
# Isaac Lab 의 그리퍼 상태 판정 헬퍼(stack.mdp 의 object_grasped 등)가 요구하는
# 임계값. "열림 기준값에서 이만큼 벗어나면 닫힌 것으로 본다" 는 뜻이다 [m].
# gripper_joint_names / gripper_open_val / gripper_threshold 는 **세 개가 한 세트**로,
# 하나만 설정하면 나머지에서 AttributeError 가 난다 (이 리비전에서 확인).
# Isaac Lab Franka stack 레퍼런스와 같은 값을 쓴다.
GRIPPER_STATUS_THRESHOLD = 0.005
FINGER_LINK_NAMES = ("panda_leftfinger", "panda_rightfinger")
FINGER_FRAME_OFFSET = (0.0, 0.0, 0.046)
GRIPPER_OPEN_QPOS = 0.04             # 손가락 1개의 최대 개방 [m]
GRIPPER_CLOSE_QPOS = 0.0
# 최대 파지 폭 [m]. 자재 에셋 크기의 상한을 정하는 값이다 —
# 이보다 두꺼운 물체는 아예 잡을 수 없어 태스크가 성립하지 않는다.
GRIPPER_MAX_WIDTH = 2 * GRIPPER_OPEN_QPOS    # 0.08

# 제어 방식
CONTROL_MODE = "ik_rel_pose_dls"     # DifferentialIKController, 상대 포즈, damped least squares
# 델타가 표현되는 좌표계. Isaac Lab 의 IK 액션 term 이 로봇 root 기준 포즈를
# 컨트롤러에 넘기므로 델타도 **로봇 베이스 프레임**이다.
# (컨트롤러 자체는 프레임을 가정하지 않는다 — "It is up to the user to ensure
#  that the command is given in the correct frame")
# Day 1 텔레옵에서 육안 확인할 것: +x 를 밀면 팔이 로봇 정면으로 나가야 한다.
CONTROL_FRAME = "robot_base"


# -----------------------------------------------------------------------------
# 액션 스펙
# -----------------------------------------------------------------------------
# [dx, dy, dz, drx, dry, drz, gripper] — LIBERO/OpenVLA 와 동일한 7차원.
#
# Isaac Lab 쪽 대응:
#   dx..drz : DifferentialInverseKinematicsActionCfg(
#                 controller=DifferentialIKControllerCfg(
#                     command_type="pose", use_relative_mode=True, ik_method="dls"),
#                 scale=0.5)
#   gripper : BinaryJointPositionActionCfg  (열림 +1 / 닫힘 -1)
ACTION_DIM = 7
ACTION_LAYOUT = ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")

# 액션 청크 크기. openvla-oft LIBERO 상수셋과 동일해야 한다.
NUM_ACTIONS_CHUNK = 8

# IK 델타 액션 스케일. Isaac Lab 의 Franka stack IK-Rel 레퍼런스와 동일값.
IK_ACTION_SCALE = 0.5

# 그리퍼 부호 규약.
#   LIBERO 는 [-1, +1] 에서 -1 = 열림, +1 = 닫힘 이고,
#   rob_rollout 이 normalize_gripper_action() → invert_gripper_action() 순으로 변환한다.
#   Isaac Lab 의 BinaryJointPositionActionCfg 는 양수 = 열림 이므로 부호가 반대다.
#   → 어댑터에서 이 상수로 한 번만 뒤집는다. 두 군데서 뒤집으면 서로 상쇄되어
#     "그리퍼가 아예 안 움직이는" 증상으로 나타난다. 반드시 여기만 참조할 것.
GRIPPER_OPEN_ISAACLAB = 1.0
GRIPPER_CLOSE_ISAACLAB = -1.0
GRIPPER_INVERT_FOR_VLA = True

# -----------------------------------------------------------------------------
# Proprioception 스펙
# -----------------------------------------------------------------------------
# LIBERO PROPRIO_DIM = 8 과 정확히 일치시킨다:
#   eef 위치 3 + eef 방향(axis-angle) 3 + 그리퍼 관절 2 = 8
# rob_rollout 의 _obs_to_input() 이 LIBERO 에서 만드는 벡터와 같은 레이아웃이다.
PROPRIO_DIM = 8
PROPRIO_LAYOUT = (
    "eef_pos_x", "eef_pos_y", "eef_pos_z",
    "eef_axisangle_x", "eef_axisangle_y", "eef_axisangle_z",
    "gripper_qpos_0", "gripper_qpos_1",
)

# SFT 와 RFT 가 동일해야 하는 스위치. openvla-oft finetune.py 의 --use_proprio 와,
# RFT 어댑터가 관측을 만들 때 proprio 를 넣을지 여부를 함께 지배한다.
#
# 기본값 False 로 시작한다: proprio 를 켜면 SFT/RFT 양쪽에서 정규화 통계까지
# 일치시켜야 하고, 어긋나면 원인 추적이 가장 어려운 종류의 버그가 된다.
# 3~4일 일정에서는 켜지 않는 쪽이 안전하며, SimpleVLA-RL 의 LIBERO 기본 설정도 이쪽이다.
USE_PROPRIO = False

# -----------------------------------------------------------------------------
# 이미지 관측 스펙
# -----------------------------------------------------------------------------
# 3인칭 단일 뷰. 손목 카메라는 넣지 않는다 (계획서 §2-1: SimpleVLA-RL 판은 단일 뷰).
NUM_IMAGES_IN_INPUT = 1
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

# Isaac Lab 씬 안에서 이 카메라 센서의 이름. 관측 dict 의 키로도 쓰인다.
CAMERA_SENSOR_NAME = "table_cam"

# TiledCamera 배치. Phase 1 에서 확정했고 이후 변경 금지.
#
# ★ 왜 초기값을 고쳤는가 (Day 1 §1-2 에서 확정)
#   출발점은 Isaac Lab Franka stack visuomotor 의 table_cam = (1.05, 0, 0.55) +
#   쿼터니언 (0.35355, -0.61237, -0.61237, 0.35355) 이었다. 그런데 그 조합의
#   시선은 테이블 평면과 (0.097, 0.0) 에서 만난다 — **로봇 베이스**다. 우리
#   작업공간(자재 스폰 x 0.38~0.60 / y -0.22~0.05, 목표 영역 중심 (0.45, 0.35))이
#   아니라 로봇을 조준하고 있었다. 그 결과:
#     - 목표 영역 중심이 u=225.0 에 투영 → 224px 프레임 **밖**
#     - 작업공간이 화면 아래 1/4 에 몰려 픽셀을 낭비
#   즉 정책이 "자재를 어디에 놓아야 하는지" 를 볼 수 없는 상태였다.
#
#   아래 값은 시선을 테이블의 (0.36, 0.09) 로 맞춘 것이다. 자재 스폰 박스 +
#   목표 영역 원주 + 운반/접근 높이(z ≤ 0.30)를 42점으로 샘플해, 전부 center crop
#   이후에도 여유 12px 이상으로 들어오도록 풀었다. 화면 점유율 19% → 38%.
#
#   ★ OpenVLA 규약은 건드리지 않았다. 고정해야 하는 것은 224x224 / 단일 3인칭 뷰 /
#     uint8 / center crop 이고, 카메라 외부 파라미터는 그중 어디에도 없다.
#     focal length·aperture·클리핑도 원래 값 그대로다 — **위치와 조준만** 바꿨다.
#     결과 기하는 오히려 LIBERO agentview 에 더 가깝다 (거리 1.08m, 하각 48도).
CAMERA_POS = (1.2, 0.0, 0.8)
CAMERA_ROT = (0.32818, -0.66487, -0.6017, 0.297)      # ROS 규약 쿼터니언 (w, x, y, z)
CAMERA_CONVENTION = "ros"
CAMERA_FOCAL_LENGTH = 24.0
CAMERA_FOCUS_DISTANCE = 400.0
CAMERA_HORIZONTAL_APERTURE = 20.955
CAMERA_CLIPPING_RANGE = (0.1, 3.0)

# 추론 시 center crop.
#
# ★ 학습에 --image_aug True 를 쓰면 추론에서 반드시 켜야 한다.
#   image_aug 는 random resized crop(면적 기준 0.9)을 걸어 학습한다. 추론에서
#   crop 을 안 하면 모델이 학습 때보다 "줌아웃"된 그림을 보게 되어 성능이
#   조용히 깎인다. openvla-oft 의 run_libero_eval.py 는 center_crop 기본값이
#   True 이고 아예 assert 로 막아 둔다:
#     "Expecting center_crop==True because model was trained with image augmentations!"
#
#   scripts/sft/run_sft_libero_spec.sh 가 --image_aug True 로 학습하므로 True 다.
#   학습 플래그를 끄면 여기도 함께 꺼야 한다 — 둘은 한 쌍이다.
IMAGE_CENTER_CROP = True
# 면적 비율. openvla 는 sqrt 를 취해 각 변에 적용한다 (면적이 0.9 가 되도록).
IMAGE_CENTER_CROP_SCALE = 0.9

# 이미지를 180도 회전할지 여부.
#   LIBERO 는 MuJoCo 렌더 출력이 상하 반전이라 get_libero_image() 가 180도 돌린다.
#   Isaac Lab 의 TiledCamera 는 이미 바로 선 이미지를 준다 → 돌리면 안 된다.
#   Day 1 에 dump_obs_reference.py 로 실제 PNG 를 눈으로 보고 확정할 것.
ROTATE_IMAGE_180 = False

# -----------------------------------------------------------------------------
# 언어 instruction
# -----------------------------------------------------------------------------
# SimpleVLA-RL 의 process_input() 이 만드는 프롬프트 형식과 정확히 일치시킨다:
#     "In: What action should the robot take to {task}?\nOut:"
# 아래는 그 {task} 자리에 들어갈 문자열이다.
TASK_INSTRUCTION = "pick up the material and place it in the target zone"
PROMPT_TEMPLATE = "In: What action should the robot take to {task}?\nOut:"


def build_prompt(instruction: str = TASK_INSTRUCTION) -> str:
    """VLA 에 넣을 최종 프롬프트 문자열을 만든다."""
    return PROMPT_TEMPLATE.format(task=instruction.lower().strip())


def prepare_image_for_vla(image):
    """관측 이미지를 VLA 입력용 PIL 이미지로 만든다 (평가·RFT 공용).

    추론 경로가 두 곳(scripts/eval_rollout.py, rft/grpo_fallback.py)이라 한 곳에
    모아 둔다. 두 곳에 각각 구현하면 반드시 한쪽만 바뀐다.

    Args:
        image: (H, W, 3) uint8 numpy 배열.

    PIL 은 함수 안에서 import 한다 — 이 모듈은 PIL 이 없는 환경에서도
    import 될 수 있어야 하기 때문이다 (스모크 테스트가 세 venv 에서 돈다).
    """
    from PIL import Image

    pil = image if hasattr(image, "crop") else Image.fromarray(image)

    if not IMAGE_CENTER_CROP:
        return pil

    # openvla 와 동일: 면적이 SCALE 이 되도록 각 변에 sqrt(SCALE) 를 적용하고,
    # 잘라낸 뒤 원래 크기로 되돌린다.
    w, h = pil.size
    ratio = IMAGE_CENTER_CROP_SCALE**0.5
    cw, ch = int(w * ratio), int(h * ratio)
    left, top = (w - cw) // 2, (h - ch) // 2
    return pil.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)


# -----------------------------------------------------------------------------
# 정규화
# -----------------------------------------------------------------------------
# openvla-oft LIBERO 상수셋과 동일. 데이터셋에서 산출한 실제 통계는
# datasets/norm_stats.json 에 저장되고 SFT/RFT 양쪽이 그 파일을 읽는다.
ACTION_PROPRIO_NORMALIZATION_TYPE = "bounds_q99"
NORM_STATS_FILENAME = "norm_stats.json"

# -----------------------------------------------------------------------------
# 에피소드 / 태스크
# -----------------------------------------------------------------------------
# 한 에피소드 최대 스텝. RFT 롤아웃 비용에 직결되므로 짧게 잡는다.
# 텔레옵 데모가 이보다 길면 Mimic 증강도 길어지고 RFT 가 급격히 느려진다
# → Day 1 데모 수집 시 "궤적을 짧게" 체크리스트가 여기에 연결된다.
MAX_EPISODE_STEPS = 300

# 제어 주기. Isaac Lab decimation 과 맞물린다.
SIM_DT = 1.0 / 120.0
DECIMATION = 5                       # → 정책 주기 24Hz
EPISODE_LENGTH_S = MAX_EPISODE_STEPS * SIM_DT * DECIMATION

# -----------------------------------------------------------------------------
# 성공 판정 (Mimic 서브태스크 시그널 + RFT 0/1 보상이 공유)
# -----------------------------------------------------------------------------
# 자재 스폰 범위 (리셋마다 이 안에서 무작위 배치). mdp/events.py 의
# randomize_material_pose 가 이 값을 읽는다 — 두 곳에 적지 않는다.
# ★ 카메라 화각이 이 범위와 목표 영역을 모두 덮어야 한다.
#   assert_workspace_visible() 이 그것을 검사한다.
MATERIAL_SPAWN_X_RANGE = (0.38, 0.60)
MATERIAL_SPAWN_Y_RANGE = (-0.22, 0.05)

# 목표 영역: 테이블 위 원기둥 형태 영역 (중심 xy + 반경).
GOAL_REGION_CENTER = (0.45, 0.35)    # 로봇 베이스 기준 xy [m]
GOAL_REGION_RADIUS = 0.09            # [m]
GOAL_HEIGHT_TOLERANCE = 0.06         # 목표 z 로부터 허용 편차 [m]
TABLE_HEIGHT = 0.0                   # 테이블 상면 z (env 로컬 기준) [m]

# "정지" 판정 속도 임계값 [m/s]. 물체가 굴러가는 중에 성공이 뜨는 것을 막는다.
SUCCESS_LIN_VEL_THRESHOLD = 0.03
# 그리퍼가 "열렸다"고 볼 관절 위치 합 [m]. Franka 는 손가락 2개 각 0~0.04.
GRIPPER_OPEN_QPOS_SUM = 0.06
# 파지 판정: eef 와 물체 중심 거리 [m].
GRASP_DISTANCE_THRESHOLD = 0.06
# 들어올림 판정: 테이블면으로부터의 높이 [m]. 서브태스크 1 의 종료 신호에 쓰인다.
LIFT_HEIGHT_THRESHOLD = 0.10

# -----------------------------------------------------------------------------
# 서브태스크 "시작" 경계 (SkillGen 전용)
# -----------------------------------------------------------------------------
# SkillGen 은 MimicGen 과 달리 각 서브태스크의 **시작 경계도 필수**로 요구한다.
# 시작~종료 사이는 사람 데모의 접촉 구간(skill)을 그대로 재생하고, 서브태스크
# 사이의 자유공간은 cuRobo 가 충돌 회피 경로를 새로 깐다.
#   MimicGen : 서브태스크 사이를 선형 보간 → 스티칭 구간에서 충돌 가능
#   SkillGen : 그 구간을 모션 플래닝 → 충돌 없음, 생성 성공률이 높다
#
# 경계를 어디에 두는가 = "어디부터가 접촉이 중요한 구간인가" 를 정하는 것이다.
#   접근(자유공간) → [파지 시작] 파지·들어올림 → 운반(자유공간) → [배치 시작] 배치·놓기
#
# 이 값들은 SkillGen 을 쓰지 않을 때는 아무 영향이 없다 (관측 항목만 하나 늘 뿐).
# 그래서 두 방식을 분기 코드 없이 같은 환경으로 지원할 수 있다.

# 파지 시작: eef 가 자재에 이만큼 가까워지면 접촉 구간의 시작으로 본다 [m].
# GRASP_DISTANCE_THRESHOLD(0.06) 보다 넉넉해야 한다 — 실제로 닿기 전에
# 정렬·하강하는 구간까지 사람 데모를 써야 파지가 재현된다.
APPROACH_START_DISTANCE = 0.15

# 배치 시작: 자재를 든 채로 목표 영역 중심에서 이 반경 안에 들어오면 배치 구간 시작 [m].
# 목표 반경(0.09)보다 커야 한다 — 영역 위에 정확히 도달한 뒤가 아니라,
# 그 위로 접근해 내려놓기 시작하는 지점부터가 사람 데모가 필요한 구간이다.
PLACE_START_RADIUS = 0.20

# 성공 후 녹화를 바로 끊지 않고 유지할 스텝 수.
# 계획서 §Phase1-3 경고: 성공 시점에 녹화가 즉시 끊기면 replay 에서 성공 조건이
# 재트리거되지 않아 데모가 통째로 버려진다.
SUCCESS_HOLD_STEPS = 10


# -----------------------------------------------------------------------------
# 자기 검증
# -----------------------------------------------------------------------------
def assert_consistent() -> None:
    """스펙 내부 정합성 확인. 세 환경의 스모크 테스트가 모두 이 함수를 호출한다."""
    assert ACTION_DIM == len(ACTION_LAYOUT), (
        f"ACTION_DIM({ACTION_DIM}) != len(ACTION_LAYOUT)({len(ACTION_LAYOUT)})"
    )
    assert PROPRIO_DIM == len(PROPRIO_LAYOUT), (
        f"PROPRIO_DIM({PROPRIO_DIM}) != len(PROPRIO_LAYOUT)({len(PROPRIO_LAYOUT)})"
    )
    # openvla-oft LIBERO 상수셋과의 일치 — 어긋나면 모델 코드를 고쳐야 한다는 뜻이므로
    # 조용히 넘어가면 안 된다.
    assert (ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK) == (7, 8, 8), (
        "openvla-oft 의 LIBERO 상수셋(ACTION_DIM=7, PROPRIO_DIM=8, "
        "NUM_ACTIONS_CHUNK=8)과 어긋났다. 이 값을 바꾸려면 prismatic/vla/constants.py "
        "도 함께 고쳐야 하며, 그 순간 3~4일 일정 전제가 깨진다."
    )
    assert NUM_IMAGES_IN_INPUT == 1, "SimpleVLA-RL 판 OpenVLA-OFT 는 3인칭 단일 뷰다."
    assert IMAGE_HEIGHT == IMAGE_WIDTH == 224, "OpenVLA 입력 해상도는 224x224 고정."
    assert GOAL_REGION_RADIUS > 0 and GOAL_HEIGHT_TOLERANCE > 0

    # --- embodiment ↔ 액션 스펙 교차 검증 ---
    # 7차원 = 델타 포즈 6 (위치3 + 회전3) + 그리퍼 1.
    # 팔의 관절 수(7)와 우연히 같은 숫자라 헷갈리기 쉬운데, 액션은 태스크공간이라
    # 관절 수와 무관하다. 팔이 6축이든 7축이든 이 값은 7 이다.
    assert ACTION_DIM == 6 + 1, (
        "액션은 태스크공간 델타 포즈(6) + 그리퍼(1) 이다. 관절 수와 무관하다."
    )
    assert ARM_DOF >= 6, (
        f"팔 자유도 {ARM_DOF} < 6 이면 임의의 6D 포즈를 낼 수 없어 "
        "델타 포즈 제어가 성립하지 않는다."
    )
    # 그리퍼 "열림" 판정 임계값이 물리적으로 도달 가능한 범위 안에 있어야 한다.
    # 이걸 넘겨 잡으면 성공 판정이 영원히 False 가 되어 보상이 0 으로 고정된다.
    assert 0 < GRIPPER_OPEN_QPOS_SUM < GRIPPER_MAX_WIDTH, (
        f"GRIPPER_OPEN_QPOS_SUM({GRIPPER_OPEN_QPOS_SUM}) 은 "
        f"0 과 최대 개방폭({GRIPPER_MAX_WIDTH}) 사이여야 한다."
    )
    # 파지 판정 거리는 그리퍼 크기 규모와 맞아야 한다.
    assert GRASP_DISTANCE_THRESHOLD > GRIPPER_MAX_WIDTH / 2, (
        "파지 판정 거리가 그리퍼 반폭보다 작으면 실제로 잡고 있어도 "
        "파지로 인정되지 않는다."
    )
    assert CONTROL_FRAME in ("robot_base", "world", "eef")

    # 자재 스폰 범위가 목표 영역과 겹치면 아무것도 안 해도 성공으로 잡힌다
    # (mdp/events.py 의 randomize_material_pose 주석 참조).
    gx, gy = GOAL_REGION_CENTER
    assert not (
        MATERIAL_SPAWN_X_RANGE[0] - GOAL_REGION_RADIUS <= gx <= MATERIAL_SPAWN_X_RANGE[1] + GOAL_REGION_RADIUS
        and MATERIAL_SPAWN_Y_RANGE[0] - GOAL_REGION_RADIUS <= gy <= MATERIAL_SPAWN_Y_RANGE[1] + GOAL_REGION_RADIUS
    ), (
        f"자재 스폰 범위 x{MATERIAL_SPAWN_X_RANGE} y{MATERIAL_SPAWN_Y_RANGE} 가 "
        f"목표 영역(중심 {GOAL_REGION_CENTER}, 반경 {GOAL_REGION_RADIUS})과 겹친다. "
        "리셋 직후 성공이 뜨는 에피소드가 생겨 RFT 보상이 오염된다."
    )

    # 카메라가 작업공간 전체를 담고 있는지 — 에러 없이 성능만 깎는 불일치를 막는다.
    assert_workspace_visible()


def _camera_axes():
    """CAMERA_ROT(ROS 규약)에서 카메라 축 3개를 뽑는다 → (right, down, forward).

    ROS 규약: +X 오른쪽, +Y 아래, +Z 광축(전방). 회전행렬의 열이 그 순서다.
    """
    w, x, y, z = CAMERA_ROT
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    right = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    down = (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))
    forward = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    return right, down, forward


def project_to_image(point):
    """env 로컬 좌표의 점을 카메라 픽셀 (u, v) 로 투영한다. 카메라 뒤면 None.

    Isaac Lab 을 띄우지 않고도 "이게 화면에 들어오나" 를 답할 수 있어야 한다 —
    한 번 렌더해 보려면 Isaac Sim 기동에 수 분이 들고, 그 비용 때문에 아무도
    확인하지 않게 된다. 실제로 그래서 목표 영역이 화면 밖인 채로 굴러갔다.
    """
    right, down, forward = _camera_axes()
    v = (
        point[0] - CAMERA_POS[0],
        point[1] - CAMERA_POS[1],
        point[2] - CAMERA_POS[2],
    )

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    z_cam = dot(v, forward)
    if z_cam <= 1e-6:
        return None
    fx = CAMERA_FOCAL_LENGTH / CAMERA_HORIZONTAL_APERTURE * IMAGE_WIDTH
    return (
        IMAGE_WIDTH / 2.0 + fx * dot(v, right) / z_cam,
        IMAGE_HEIGHT / 2.0 + fx * dot(v, down) / z_cam,
    )


def center_crop_bounds() -> tuple[float, float]:
    """center crop 이후에도 살아남는 픽셀 구간 [lo, hi). crop 을 끄면 프레임 전체."""
    if not IMAGE_CENTER_CROP:
        return 0.0, float(IMAGE_WIDTH)
    ratio = IMAGE_CENTER_CROP_SCALE**0.5
    cw = int(IMAGE_WIDTH * ratio)
    lo = (IMAGE_WIDTH - cw) // 2
    return float(lo), float(lo + cw)


def workspace_roi_points() -> list:
    """정책이 반드시 봐야 하는 점들 — 자재가 놓일 수 있는 곳 + 목표 영역 + 운반 높이."""
    x0, x1 = MATERIAL_SPAWN_X_RANGE
    y0, y1 = MATERIAL_SPAWN_Y_RANGE
    gx, gy = GOAL_REGION_CENTER
    r = GOAL_REGION_RADIUS
    pts = []
    # 자재 정지 위치 (테이블면 ~ 자재 높이)
    for x in (x0, x1):
        for y in (y0, y1):
            for z in (TABLE_HEIGHT, TABLE_HEIGHT + 0.08):
                pts.append((x, y, z))
    # 목표 영역 원주
    for i in range(16):
        a = 2.0 * math.pi * i / 16.0
        pts.append((gx + r * math.cos(a), gy + r * math.sin(a), TABLE_HEIGHT))
    # 운반 중 자재 (스폰 위 / 목표 위)
    for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1), (gx, gy), (gx, gy + r)):
        pts.append((x, y, TABLE_HEIGHT + 0.20))
    return pts


def assert_workspace_visible(margin_px: float = 6.0) -> None:
    """작업공간 전체가 center crop 이후에도 화면 안에 있는지 확인한다.

    ★ 이 검사가 왜 있는가 (Day 1 §1-2 에서 실제로 당했다)
      초기 카메라 설정은 시선이 로봇 베이스를 향해 있어서 **목표 영역 중심이
      프레임 밖**(u=225.0 / 224px)이었다. 에러는 전혀 나지 않는다 — 씬도 만들어지고
      관측 shape 도 맞고 스모크도 통과한다. 정책이 "어디에 놓아야 하는지" 를 못 보는
      것뿐이고, 그건 SFT 를 다 돌린 뒤 "성능이 안 나온다" 로만 드러난다.
      카메라·목표·스폰 범위는 서로 다른 파일에 있어서 한쪽만 바뀌기 쉽다.
    """
    lo, hi = center_crop_bounds()
    lo, hi = lo + margin_px, hi - margin_px
    bad = []
    for p in workspace_roi_points():
        uv = project_to_image(p)
        if uv is None:
            bad.append((p, "카메라 뒤"))
            continue
        u, v = uv
        if not (lo <= u <= hi and lo <= v <= hi):
            bad.append((p, f"u={u:.1f} v={v:.1f}"))
    if bad:
        lines = "\n".join(f"    {p} → {why}" for p, why in bad[:8])
        raise AssertionError(
            f"작업공간이 카메라 화각을 벗어난다 (crop 후 허용 {lo:.1f}~{hi:.1f}px, "
            f"{len(bad)}개 지점 실패):\n{lines}\n"
            "  CAMERA_POS / CAMERA_ROT / CAMERA_FOCAL_LENGTH 와 "
            "GOAL_REGION_CENTER / MATERIAL_SPAWN_*_RANGE 중 하나가 어긋났다.\n"
            "  이건 에러 없이 조용히 성능만 깎는 종류의 불일치다 — 반드시 맞출 것."
        )


def summary() -> str:
    """사람이 읽을 스펙 요약. 스모크 테스트와 학습 로그 머리말에 찍는다."""
    return (
        "=== VLA 태스크 스펙 (configs/vla_spec.py) ===\n"
        f"  로봇      : {ROBOT_NAME} ({ARM_DOF}축 + 평행 2지 그리퍼)\n"
        f"              제어 {CONTROL_MODE}, 델타 좌표계 {CONTROL_FRAME}, "
        f"최대 파지폭 {GRIPPER_MAX_WIDTH * 100:.0f}cm\n"
        f"  액션      : {ACTION_DIM}차원 {ACTION_LAYOUT}, 청크 {NUM_ACTIONS_CHUNK}\n"
        f"  proprio   : {PROPRIO_DIM}차원, 사용={USE_PROPRIO}\n"
        f"  이미지    : {NUM_IMAGES_IN_INPUT}뷰 {IMAGE_HEIGHT}x{IMAGE_WIDTH} "
        f"(센서 '{CAMERA_SENSOR_NAME}', 180도회전={ROTATE_IMAGE_180})\n"
        f"  정규화    : {ACTION_PROPRIO_NORMALIZATION_TYPE}\n"
        f"  에피소드  : 최대 {MAX_EPISODE_STEPS}스텝 ({EPISODE_LENGTH_S:.1f}s), "
        f"제어 {1.0 / (SIM_DT * DECIMATION):.0f}Hz\n"
        f"  instruction: {TASK_INSTRUCTION!r}\n"
        f"  목표영역  : 중심 {GOAL_REGION_CENTER}, 반경 {GOAL_REGION_RADIUS}m\n"
    )


def _self_check_center_crop() -> None:
    """center crop 이 크기를 유지하고 실제로 중앙을 확대하는지 확인."""
    try:
        import numpy as np
        from PIL import Image  # noqa: F401
    except ImportError:
        print("(numpy/PIL 없음 — center crop 자체 검사 건너뜀)")
        return

    # 중앙에만 흰 사각형이 있는 이미지. crop 하면 흰 비율이 커져야 한다.
    arr = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    q = IMAGE_HEIGHT // 4
    arr[q : IMAGE_HEIGHT - q, q : IMAGE_WIDTH - q] = 255

    out = np.asarray(prepare_image_for_vla(arr))
    assert out.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3), (
        f"crop 후 크기가 바뀌었다: {out.shape}"
    )
    before = (arr > 127).mean()
    after = (out > 127).mean()
    if IMAGE_CENTER_CROP:
        assert after > before, (
            f"center crop 이 확대 효과를 내지 못했다 (흰 비율 {before:.3f} → {after:.3f})"
        )
        print(f"center crop OK: 중앙 영역 비율 {before:.3f} → {after:.3f}")
    else:
        assert abs(after - before) < 1e-6, "crop 을 껐는데 이미지가 바뀌었다"
        print("center crop 비활성 — 원본 그대로 통과")


if __name__ == "__main__":
    assert_consistent()
    print(summary())
    _self_check_center_crop()
    print("스펙 정합성 확인 통과.")
