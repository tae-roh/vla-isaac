# =============================================================================
# vla_isaac_tasks/pickplace_mimic_env.py
#
# Isaac Lab Mimic 환경 래퍼 — 헬퍼 6종 구현.
#
# 레퍼런스: isaaclab_mimic 의 FrankaCubeStackIKRelMimicEnv.
# 액션 레이아웃(IK 상대 델타 6 + 그리퍼 1)이 그 환경과 동일하므로 포즈↔액션
# 변환 로직은 검증된 구현을 그대로 따르고, 오브젝트가 여러 개가 아니라 하나라는
# 점만 반영했다. 계획서 §6 의 "Mimic 환경 헬퍼 구현 난이도" 리스크에 대한
# 대응이 "복사 개조로 시작" 이었고, 그대로 했다.
# =============================================================================

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

# 이 문자열은 MimicEnvCfg.subtask_configs 의 키이자 헬퍼들이 주고받는 eef 이름이다.
# cfg 쪽과 반드시 같아야 한다 → 여기서 한 번만 정의하고 cfg 가 import 해 쓴다.
EEF_NAME = "franka"


class PickPlaceMimicEnv(ManagerBasedRLMimicEnv):
    """형상 랜덤 자재 픽앤플레이스용 Mimic 환경."""

    # -------------------------------------------------------------------------
    def get_robot_eef_pose(
        self, eef_name: str, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """현재 엔드이펙터 포즈를 4x4 행렬로. Shape (len(env_ids), 4, 4).

        로봇 엔드이펙터 컨트롤러가 쓰는 프레임과 같아야 한다 — 여기서는
        관측의 eef_pos / eef_quat 이 그 프레임이다 (씬의 ee_frame 이
        panda_hand + 0.1034 오프셋이고, IK 액션의 body_offset 이 0.107 로
        거의 같은 지점을 가리킨다).
        """
        if env_ids is None:
            env_ids = slice(None)

        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    # -------------------------------------------------------------------------
    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """목표 eef 포즈 → env.step() 에 넣을 액션 (상대 델타 포즈 + 그리퍼).

        Returns:
            Shape (7,) 텐서. [dx, dy, dz, drx, dry, drz, gripper]
        """
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        curr_pose = self.get_robot_eef_pose(EEF_NAME, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        delta_position = target_pos - curr_pos

        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        (gripper_action,) = gripper_action_dict.values()

        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None:
            # 액션 노이즈는 정책 강건성에 기여한다 (계획서 §Phase2 품질 체크리스트).
            noise = action_noise_dict[EEF_NAME] * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)

    # -------------------------------------------------------------------------
    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """액션 → 목표 eef 포즈. target_eef_pose_to_action 의 역변환.

        데모의 기록된 액션 시퀀스로부터 컨트롤러 목표 포즈열을 복원할 때 쓴다.

        Args:
            action: Shape (num_envs, action_dim)
        """
        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        curr_pose = self.get_robot_eef_pose(EEF_NAME, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        target_pos = curr_pos + delta_position

        # 회전 델타를 축-각으로 분해한다. 각이 0 에 가까우면 축이 정의되지 않아
        # 0/0 이 되므로 그 경우를 명시적으로 0 축으로 눌러 준다.
        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle

        is_close_to_zero_angle = torch.isclose(
            delta_rotation_angle, torch.zeros_like(delta_rotation_angle)
        ).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(
            delta_rotation_axis
        )[is_close_to_zero_angle]

        delta_quat = PoseUtils.quat_from_angle_axis(
            delta_rotation_angle.squeeze(1), delta_rotation_axis
        ).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()
        return {EEF_NAME: target_poses}

    # -------------------------------------------------------------------------
    def actions_to_gripper_actions(
        self, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """액션 시퀀스에서 그리퍼 성분만 뽑는다 (마지막 1개 차원)."""
        return {EEF_NAME: actions[:, -1:]}

    # -------------------------------------------------------------------------
    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """씬 오브젝트들의 포즈를 4x4 행렬 dict 로.

        Mimic 은 이 포즈를 기준으로 원본 데모의 오브젝트 중심 구간을 새 씬으로
        변환 재배치한다 — 증강의 핵심 입력이다.

        오브젝트가 하나뿐이라 dict 항목도 하나다. 키 "object" 는
        SubTaskConfig.object_ref 와 일치해야 한다.
        """
        if env_ids is None:
            env_ids = slice(None)

        pose7 = self.obs_buf["policy"]["object_pose"][env_ids]   # (N, 7) pos+quat
        pos = pose7[..., :3]
        rot = PoseUtils.matrix_from_quat(pose7[..., 3:7])
        return {"object": PoseUtils.make_pose(pos, rot)}

    # -------------------------------------------------------------------------
    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """서브태스크 경계 시그널. --auto 어노테이션이 이 값을 읽는다.

        종료 경계 1개 + 시작 경계 2개를 함께 반환한다.
          - MimicGen 은 종료 경계만 쓴다 (시작 경계는 무시된다)
          - SkillGen 은 **시작 경계가 필수**다. 없으면 데이터 생성이 실패한다.
        둘을 모두 내보내면 분기 없이 양쪽을 지원할 수 있다.

        서브태스크가 2개이고 규약상 마지막 것은 종료 시그널이 없으므로
        (SubTaskConfig.subtask_term_signal=None) 종료 항목은 1개다.
        """
        if env_ids is None:
            env_ids = slice(None)

        subtask_terms = self.obs_buf["subtask_terms"]
        return {
            # 종료 경계
            "grasp_lift": subtask_terms["grasp_lift"][env_ids],
            # 시작 경계 (SkillGen)
            "grasp_start": subtask_terms["grasp_start"][env_ids],
            "place_start": subtask_terms["place_start"][env_ids],
        }

    # -------------------------------------------------------------------------
    def get_expected_attached_object(
        self, eef_name: str, subtask_index: int, env_cfg
    ) -> str | None:
        """(SkillGen) 해당 서브태스크 시작 시점에 그리퍼에 붙어 있어야 할 오브젝트.

        cuRobo 가 자유공간 경로를 계획할 때, 로봇이 물체를 들고 있으면 그 물체까지
        충돌체에 포함해야 한다. 그러지 않으면 플래너가 "로봇만" 통과하는 경로를
        만들어 **들고 있는 자재가 테이블이나 목표 마커를 뚫고 지나가는** 궤적이 나온다.

        우리 서브태스크 구성:
          index 0 (파지) → None      아직 아무것도 안 들었다
          index 1 (배치) → "object"  앞 서브태스크에서 파지했으므로 들고 있다

        레퍼런스(FrankaCubeStackIKRelMimicEnv)는 서브태스크 이름을 문자열로 검사해
        추론하지만, 우리는 서브태스크가 2개로 고정이라 인덱스로 직접 판정하는 편이
        명확하고 오작동 여지가 없다.
        """
        if eef_name not in env_cfg.subtask_configs:
            return None

        subtask_configs = env_cfg.subtask_configs[eef_name]
        if not (0 <= subtask_index < len(subtask_configs)):
            return None

        # 첫 서브태스크(파지) 시작 시점에는 빈손이다.
        if subtask_index == 0:
            return None

        # 그 이후는 파지를 마친 뒤이므로 자재를 들고 있다.
        return subtask_configs[subtask_index].object_ref
