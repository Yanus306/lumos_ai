"""
predict_risk.py
===============
학습된 위험도 분류 모델로 새 이미지의 위험도(소/중/대)를 예측합니다.

사용법:
  cd risk_classifier

  # 단일 이미지
  python predict_risk.py --source ../test/images/kream1.png

  # 폴더 전체
  python predict_risk.py --source ../test/images/

  # 신뢰도 임계값 지정 (기본 0.0)
  python predict_risk.py --source ../test/images/ --conf 0.5

  # 특정 모델 지정
  python predict_risk.py --model runs/risk_classifier_v1/best.pt --source ../test/images/
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from train_classifier import build_model


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
RISK_NAMES  = {0: "소 (Low)", 1: "중 (Medium)", 2: "대 (High)"}
RISK_COLORS = {0: "\033[92m", 1: "\033[93m", 2: "\033[91m"}  # 초록/노랑/빨강
RESET       = "\033[0m"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# 예측 함수
# ─────────────────────────────────────────────
def load_model(model_path: str, device: torch.device):
    """저장된 체크포인트에서 모델을 불러옵니다."""
    ckpt = torch.load(model_path, map_location=device)
    model_name = ckpt.get("model_name", "efficientnet_b0")
    model = build_model(model_name, num_classes=3, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  모델 로드 완료: {model_name}  (checkpoint: {model_path})")
    if "val_acc" in ckpt:
        print(f"  검증 정확도 (학습 시): {ckpt['val_acc']:.2f}%")
    return model


@torch.no_grad()
def predict_image(model, img_path: str, device: torch.device):
    """단일 이미지에 대한 위험도 예측 결과를 반환합니다."""
    img = Image.open(img_path).convert("RGB")
    tensor = VAL_TRANSFORM(img).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred   = int(probs.argmax())
    return pred, probs


def collect_images(source: str):
    """source가 파일이면 [파일], 디렉터리면 이미지 파일 목록을 반환합니다."""
    p = Path(source)
    if p.is_file():
        return [p]
    elif p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    else:
        raise ValueError(f"존재하지 않는 경로: {source}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main(args):
    script_dir = Path(__file__).parent

    # 기본 모델 경로 자동 탐색
    if args.model is None:
        candidates = sorted(script_dir.glob("runs/*/best.pt"), reverse=True)
        if not candidates:
            raise FileNotFoundError(
                "학습된 모델을 찾을 수 없습니다.\n"
                "먼저 학습을 실행하세요: python train_classifier.py"
            )
        args.model = str(candidates[0])
        print(f"  [자동 탐색] 모델: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  다크패턴 위험도 예측")
    print(f"{'='*60}")
    print(f"  디바이스: {device}")
    print(f"  입력    : {args.source}")

    model  = load_model(args.model, device)
    images = collect_images(args.source)

    if not images:
        print("  ⚠️  이미지 파일이 없습니다.")
        return

    print(f"  이미지  : {len(images)}개\n")

    # 결과 저장을 위한 출력 디렉터리
    out_dir = script_dir / "predictions"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "results.csv"

    results = []
    risk_counter = {0: 0, 1: 0, 2: 0}

    print(f"  {'파일명':<40}  {'위험도':<16}  P(소)  P(중)  P(대)")
    print(f"  {'-'*80}")

    for img_path in images:
        pred, probs = predict_image(model, str(img_path), device)
        risk_label  = RISK_NAMES[pred]
        skip        = probs.max() < args.conf

        if not skip:
            risk_counter[pred] += 1

        color = RISK_COLORS[pred]
        fname = img_path.name
        if len(fname) > 38:
            fname = "..." + fname[-35:]

        print(f"  {fname:<40}  "
              f"{color}{risk_label:<16}{RESET}  "
              f"{probs[0]:.3f}  {probs[1]:.3f}  {probs[2]:.3f}"
              + (" [스킵: 신뢰도 낮음]" if skip else ""))

        results.append({
            "image_path": str(img_path),
            "image_name": img_path.name,
            "risk":       pred,
            "risk_label": risk_label,
            "prob_소":    f"{probs[0]:.4f}",
            "prob_중":    f"{probs[1]:.4f}",
            "prob_대":    f"{probs[2]:.4f}",
            "confidence": f"{probs.max():.4f}",
        })

    # CSV 저장
    fieldnames = ["image_name", "image_path", "risk", "risk_label",
                  "prob_소", "prob_중", "prob_대", "confidence"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # 요약
    total = len(images)
    print(f"\n{'='*60}")
    print(f"  예측 완료 ({total}개 이미지)")
    print(f"{'='*60}")
    print(f"  {'위험도':<12}  {'수':>5}  {'비율':>7}")
    print(f"  {'-'*28}")
    for risk_id, name in RISK_NAMES.items():
        cnt = risk_counter[risk_id]
        pct = cnt / total * 100 if total > 0 else 0
        color = RISK_COLORS[risk_id]
        print(f"  {color}{name:<12}{RESET}  {cnt:>5}  {pct:>6.1f}%")
    print(f"\n  결과 CSV: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다크패턴 위험도 예측")

    parser.add_argument("--source", type=str, required=True,
                        help="예측할 이미지 경로 또는 폴더")
    parser.add_argument("--model",  type=str, default=None,
                        help="모델 경로 (기본: runs/ 에서 자동 탐색)")
    parser.add_argument("--conf",   type=float, default=0.0,
                        help="최소 신뢰도 임계값 (기본: 0.0)")

    args = parser.parse_args()
    main(args)
