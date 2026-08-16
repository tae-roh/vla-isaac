# vendor/ — 추론 전용 prismatic shim

체크포인트의 원격 코드(`modeling_prismatic.py`)가 `prismatic` 에서 딱 두 모듈만
쓴다:

    prismatic.vla.constants      (ACTION_DIM 등 — stdlib 만 씀)
    prismatic.training.train_utils (마스크 헬퍼 — torch + 위 모듈)

그런데 openvla-oft 의 `prismatic/__init__.py` 는 패키지 전체를 끌어오면서
`draccus` / `dlimp`(PyPI 에 없는 GitHub 패키지) 같은 **학습 데이터 파이프라인**
의존성을 요구한다. 그걸 이 인스턴스(env_isaaclab)에 설치하면 openvla-oft 가
검증한 torch 2.2.0 스택이 딸려 들어와 **Isaac Sim 이 고정한 torch 2.7.0 을
깨뜨린다.**

그래서 필요한 두 모듈만 원본 그대로 복사한 shim 을 둔다. 평가·RFT 롤아웃에서
    PYTHONPATH=vendor
로 얹으면 `import prismatic` 이 이것으로 해결된다.

★ 학습(SFT)에는 쓰지 말 것. 그쪽은 env_vla_train 에 openvla-oft 를 정식으로
  설치해 쓴다 (setup/setup_vla_train.sh).

출처: https://github.com/moojink/openvla-oft  (~/openvla-oft 체크아웃에서 복사)
