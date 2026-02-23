"""
prepare_data.py
===============
YOLO 형식의 탐지 레이블을 읽어 이미지별 위험도(소/중/대)를 산정하고
학습용 CSV 파일을 생성합니다.

위험도 판정 기준:
  - 소 (0): 탐지된 다크패턴 0~1개
  - 중 (1): 탐지된 다크패턴 2~4개 이고, 고위험 클래스(2,3) 없음
  - 대 (2): 탐지된 다크패턴 5개 이상 OR 고위험 클래스(2,3) 포함

사용법:
  cd risk_classifier
  python prepare_data.py
"""

# -*- coding: utf-8 -*-
import os
import sys
import csv
from pathlib import Path
from collections import Counter

# Windows 터미널 유니코드 출력 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent  # lumos/

SPLITS = {
    "train": (BASE_DIR / "train" / "images", BASE_DIR / "train" / "labels"),
    "valid": (BASE_DIR / "valid" / "images", BASE_DIR / "valid" / "labels"),
    "test":  (BASE_DIR / "test"  / "images", BASE_DIR / "test"  / "labels"),
}

OUTPUT_DIR = Path(__file__).parent  # risk_classifier/

# 위험도 판정 기준
HIGH_RISK_CLASSES = {2, 3}   # 고위험 클래스 ID
LOW_MAX    = 3               # 탐지 수 0~3  -> 소(0)
MEDIUM_MAX = 9               # 탐지 수 4~9 (고위험 없음) -> 중(1)
# 10개 이상 또는 고위험 클래스 포함 -> 대(2)

RISK_LABELS = {0: "소", 1: "중", 2: "대"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────
# 위험도 산정 함수
# ─────────────────────────────────────────────
def compute_risk(label_path: Path) -> int:
    """
    YOLO 레이블 파일을 읽어 위험도(0=소, 1=중, 2=대)를 반환합니다.
    레이블 파일이 없거나 비어 있으면 탐지 없음(소)으로 처리.
    """
    if not label_path.exists() or label_path.stat().st_size == 0:
        return 0  # 탐지 없음 → 소

    detections = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    detections.append(class_id)
    except Exception:
        return 0  # 파싱 오류 → 소

    n = len(detections)
    has_high_risk = any(cls in HIGH_RISK_CLASSES for cls in detections)

    if n <= LOW_MAX:
        return 0  # 소
    elif n <= MEDIUM_MAX and not has_high_risk:
        return 1  # 중
    else:
        return 2  # 대


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def process_split(split_name: str, img_dir: Path, lbl_dir: Path):
    """한 split(train/valid/test)을 처리하여 CSV를 생성합니다."""
    print(f"\n{'='*50}")
    print(f"  {split_name.upper()} 처리 중...")
    print(f"  이미지 : {img_dir}")
    print(f"  레이블 : {lbl_dir}")

    if not img_dir.exists():
        print(f"  ⚠️  이미지 폴더가 없습니다: {img_dir}")
        return

    rows = []
    risk_counter = Counter()

    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )

    for img_path in image_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        risk = compute_risk(lbl_path)
        risk_counter[risk] += 1

        # CSV에는 절대 경로 대신 프로젝트 루트 기준 상대 경로 저장
        rel_path = img_path.relative_to(BASE_DIR).as_posix()
        rows.append({
            "image_path": str(img_path.resolve()),
            "rel_path": rel_path,
            "risk": risk,
            "risk_label": RISK_LABELS[risk],
        })

    # CSV 저장
    out_csv = OUTPUT_DIR / f"{split_name}_labels.csv"
    fieldnames = ["image_path", "rel_path", "risk", "risk_label"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 통계 출력
    total = len(rows)
    print(f"\n  [완료] {total}개 이미지 -> {out_csv.name}")
    print(f"  +-----------------------------+")
    print(f"  |  위험도  |  수   |   비율   |")
    print(f"  +-----------------------------+")
    for risk_id, label in RISK_LABELS.items():
        cnt = risk_counter[risk_id]
        pct = cnt / total * 100 if total > 0 else 0
        print(f"  |   {label} ({risk_id})  | {cnt:4d}  | {pct:6.1f}%  |")
    print(f"  +-----------------------------+")

    return rows


def main():
    print("\n" + "="*60)
    print("  다크패턴 위험도 데이터 준비 스크립트")
    print("="*60)
    print(f"\n  기준 디렉터리: {BASE_DIR}")
    print(f"  출력 디렉터리: {OUTPUT_DIR}")
    print(f"\n  위험도 기준:")
    print(f"    소 (0): 탐지 수 0~{LOW_MAX}개")
    print(f"    중 (1): 탐지 수 {LOW_MAX+1}~{MEDIUM_MAX}개 (고위험 클래스 {HIGH_RISK_CLASSES} 제외)")
    print(f"    대 (2): 탐지 수 {MEDIUM_MAX+1}개 이상 OR 고위험 클래스 포함")

    all_stats = {}
    for split_name, (img_dir, lbl_dir) in SPLITS.items():
        rows = process_split(split_name, img_dir, lbl_dir)
        if rows:
            all_stats[split_name] = len(rows)

    print("\n" + "="*60)
    print("  전체 요약")
    print("="*60)
    for split, cnt in all_stats.items():
        print(f"  {split:6s}: {cnt:5d}개")

    total = sum(all_stats.values())
    print(f"  {'합계':6s}: {total:5d}개")
    print("\n  [성공] CSV 파일 생성 완료!")
    print("  -> train_labels.csv, valid_labels.csv, test_labels.csv")
    print("\n  다음 단계: python train_classifier.py")


if __name__ == "__main__":
    main()
