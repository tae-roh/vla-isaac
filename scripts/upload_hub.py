"""산출물을 HF Hub private repo 로 올리고 sha256 을 기록한다.

계획서 §3-3: 인스턴스는 언제든 사라질 수 있으므로 오브젝트 스토리지를 단일
진실 공급원(SSOT)으로 삼는다. 각 마일스톤 종료 시 이 스크립트를 돌린다.

sha256 을 남기는 이유가 형식적인 게 아니다 — 조용한 파일 손상은 SFT 중반에
원인 불명 에러로 나타나고, 그때는 데이터가 문제인지 코드가 문제인지 구분하는
데만 반나절이 든다. 업로드 직후 확인해 두면 그 분기를 없앨 수 있다.

사용 예:
    python scripts/upload_hub.py --path datasets/rlds \
        --repo myuser/vla-pick-rlds --repo-type dataset

    # 다운로드 후 무결성 확인
    python scripts/upload_hub.py --verify datasets/rlds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MANIFEST_NAME = "SHA256SUMS.json"
# 큰 체크포인트를 한 번에 읽지 않도록 청크 단위로 해싱한다.
_CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path) -> dict[str, dict]:
    manifest: dict[str, dict] = {}
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST_NAME)
    for i, path in enumerate(files, 1):
        rel = path.relative_to(root).as_posix()
        manifest[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        print(f"  [{i}/{len(files)}] {rel}  ({manifest[rel]['bytes'] / 1e6:.1f} MB)")
    return manifest


def cmd_verify(root: Path) -> int:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        print(f"매니페스트가 없다: {manifest_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad, missing = [], []

    for rel, meta in manifest.items():
        path = root / rel
        if not path.exists():
            missing.append(rel)
            continue
        if sha256_file(path) != meta["sha256"]:
            bad.append(rel)

    if missing:
        print(f"[FAIL] 누락 {len(missing)}개:")
        for r in missing[:20]:
            print(f"  - {r}")
    if bad:
        print(f"[FAIL] 해시 불일치 {len(bad)}개 (전송 중 손상):")
        for r in bad[:20]:
            print(f"  - {r}")
    if missing or bad:
        return 1

    print(f"[ OK ] {len(manifest)}개 파일 무결성 확인 완료")
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    root: Path = args.path
    if not root.is_dir():
        print(f"디렉토리가 아니다: {root}", file=sys.stderr)
        return 2

    print(f"sha256 계산 중: {root}")
    manifest = build_manifest(root)
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    total_gb = sum(m["bytes"] for m in manifest.values()) / 1e9
    print(f"\n매니페스트: {root / MANIFEST_NAME}  ({len(manifest)}개, {total_gb:.2f} GB)")

    if args.dry_run:
        print("\n--dry-run — 업로드하지 않았다.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        repo_id=args.repo,
        repo_type=args.repo_type,
        private=True,       # 항상 private 로 만든다. 공개는 의도적으로만.
        exist_ok=True,
    )
    dest = args.path_in_repo or "(루트)"
    print(f"\n업로드 → {args.repo} ({args.repo_type}, private) : {dest}")
    api.upload_folder(
        folder_path=str(root),
        path_in_repo=args.path_in_repo,
        repo_id=args.repo,
        repo_type=args.repo_type,
        commit_message=args.message,
    )
    print("업로드 완료.")
    print(f"\n다른 인스턴스에서 받은 뒤 반드시 확인할 것:")
    print(f"  python scripts/upload_hub.py --verify <받은경로>")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, help="업로드할 디렉토리")
    parser.add_argument("--repo", help="예: myuser/vla-pick-rlds")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--message", default="milestone upload")
    parser.add_argument("--dry-run", action="store_true",
                        help="해시/매니페스트만 만들고 업로드하지 않는다")
    parser.add_argument("--verify", type=Path, help="매니페스트로 무결성만 확인")
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="repo 안의 하위 경로 (예: steps-17500). 한 repo 에 체크포인트를 여러 개 "
             "둘 때 필요하다 — 생략하면 루트에 올라가 기존 파일을 덮어쓴다.",
    )
    args = parser.parse_args()

    if args.verify:
        return cmd_verify(args.verify)
    if not args.path or (not args.repo and not args.dry_run):
        parser.error("--path 와 --repo 가 필요하다 (--dry-run 이면 --repo 생략 가능).")
    return cmd_upload(args)


if __name__ == "__main__":
    sys.exit(main())
