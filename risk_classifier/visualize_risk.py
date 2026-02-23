"""
visualize_risk.py
=================
테스트 이미지에 위험도 예측 결과를 오버레이하여 이미지 파일로 저장합니다.
세 위험도(소/중/대)에서 각 클래스별 대표 이미지를 선정해 시각화합니다.

사용법:
  cd risk_classifier
  python visualize_risk.py
  python visualize_risk.py --n 12 --source ../test/images/
"""

import argparse
import csv
import sys
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from train_classifier import build_model

# Windows stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
RISK_NAMES  = {0: "소 (Low)", 1: "중 (Medium)", 2: "대 (High)"}
RISK_COLORS_RGB = {
    0: (52, 199, 89),    # 초록
    1: (255, 149, 0),    # 주황
    2: (255, 59,  48),   # 빨강
}
BG_COLORS = {
    0: (220, 255, 230),
    1: (255, 245, 220),
    2: (255, 220, 218),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# 예측
# ─────────────────────────────────────────────
def load_model(model_path: str, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    model_name = ckpt.get("model_name", "efficientnet_b0")
    model = build_model(model_name, num_classes=3, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt.get("val_acc", 0)


@torch.no_grad()
def predict_image(model, img_path: str, device: torch.device):
    img = Image.open(img_path).convert("RGB")
    tensor = VAL_TRANSFORM(img).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred = int(probs.argmax())
    return pred, probs, img


# ─────────────────────────────────────────────
# 단일 이미지 카드 생성
# ─────────────────────────────────────────────
CARD_W = 400
IMG_H  = 300
BAR_H  = 110
CARD_H = IMG_H + BAR_H

def make_card(img: Image.Image, fname: str, pred: int, probs) -> Image.Image:
    """이미지 + 위험도 정보를 담은 카드 이미지를 생성합니다."""
    card = Image.new("RGB", (CARD_W, CARD_H), BG_COLORS[pred])
    draw = ImageDraw.Draw(card)

    # 이미지 붙이기 (비율 유지)
    img_copy = img.copy()
    img_copy.thumbnail((CARD_W, IMG_H), Image.LANCZOS)
    x_off = (CARD_W - img_copy.width) // 2
    card.paste(img_copy, (x_off, 0))

    # 위험도 배지
    color = RISK_COLORS_RGB[pred]
    badge_txt = RISK_NAMES[pred]
    # 배지 배경
    badge_x1, badge_y1 = 8, IMG_H + 8
    badge_x2, badge_y2 = CARD_W - 8, IMG_H + 42
    draw.rounded_rectangle([badge_x1, badge_y1, badge_x2, badge_y2],
                            radius=10, fill=color)

    # 폰트 설정 (시스템 한글 폰트 시도)
    font_large = None
    font_small = None
    font_tiny  = None
    for font_path in [
        "C:/Windows/Fonts/malgun.ttf",      # 맑은 고딕
        "C:/Windows/Fonts/gulim.ttc",       # 굴림
        "C:/Windows/Fonts/NanumGothic.ttf", # 나눔고딕
        "C:/Windows/Fonts/arial.ttf",
    ]:
        try:
            font_large = ImageFont.truetype(font_path, 22)
            font_small = ImageFont.truetype(font_path, 15)
            font_tiny  = ImageFont.truetype(font_path, 12)
            break
        except Exception:
            continue
    if font_large is None:
        font_large = font_small = font_tiny = ImageFont.load_default()

    # 배지 텍스트 (중앙 정렬)
    bbox = draw.textbbox((0, 0), badge_txt, font=font_large)
    tw = bbox[2] - bbox[0]
    draw.text(((CARD_W - tw) // 2, badge_y1 + 6), badge_txt,
              fill="white", font=font_large)

    # 막대 그래프 (소/중/대)
    bar_labels = ["소", "중", "대"]
    bar_colors = [RISK_COLORS_RGB[i] for i in range(3)]
    bar_y_start = IMG_H + 50
    bar_max_w   = CARD_W - 120
    bar_h_px    = 14

    for i, (lbl, col, p) in enumerate(zip(bar_labels, bar_colors, probs)):
        y = bar_y_start + i * 20
        # 레이블
        draw.text((10, y - 1), lbl, fill=(80, 80, 80), font=font_small)
        # 배경 바
        draw.rounded_rectangle([45, y, 45 + bar_max_w, y + bar_h_px],
                                radius=4, fill=(210, 210, 210))
        # 값 바
        fill_w = max(int(bar_max_w * p), 4)
        draw.rounded_rectangle([45, y, 45 + fill_w, y + bar_h_px],
                                radius=4, fill=col)
        # 확률 텍스트
        pct_txt = f"{p*100:.1f}%"
        draw.text((45 + bar_max_w + 6, y - 1), pct_txt,
                  fill=(60, 60, 60), font=font_tiny)

    # 파일명 (하단)
    short_name = fname if len(fname) <= 38 else "..." + fname[-35:]
    draw.text((8, CARD_H - 16), short_name, fill=(120, 120, 120), font=font_tiny)

    return card


# ─────────────────────────────────────────────
# 그리드 합치기
# ─────────────────────────────────────────────
def make_grid(cards, cols=3, padding=12, bg=(240, 240, 242)):
    rows = (len(cards) + cols - 1) // cols
    W = cols * CARD_W + (cols + 1) * padding
    H = rows * CARD_H + (rows + 1) * padding
    grid = Image.new("RGB", (W, H), bg)
    for idx, card in enumerate(cards):
        r, c = divmod(idx, cols)
        x = padding + c * (CARD_W + padding)
        y = padding + r * (CARD_H + padding)
        grid.paste(card, (x, y))
    return grid


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main(args):
    script_dir = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 자동 탐색
    if args.model is None:
        candidates = sorted(script_dir.glob("runs/*/best.pt"), reverse=True)
        if not candidates:
            raise FileNotFoundError("학습된 모델 없음. python train_classifier.py 먼저 실행하세요.")
        args.model = str(candidates[0])

    print(f"\n모델 로드 중: {args.model}")
    model, val_acc = load_model(args.model, device)
    print(f"검증 정확도 (학습시): {val_acc:.2f}%")

    # 이미지 목록
    src = Path(args.source)
    all_images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not all_images:
        print("이미지 없음")
        return

    # n개 랜덤 선택
    n = min(args.n, len(all_images))
    random.seed(args.seed)
    chosen = random.sample(all_images, n)
    print(f"\n{len(all_images)}개 중 {n}개 이미지 선택 (seed={args.seed})\n")

    # 예측 및 카드 생성
    cards = []
    results = []
    risk_cnt = {0: 0, 1: 0, 2: 0}

    for img_path in chosen:
        pred, probs, pil_img = predict_image(model, str(img_path), device)
        risk_cnt[pred] += 1
        card = make_card(pil_img, img_path.name, pred, probs)
        cards.append(card)
        results.append((img_path.name, pred, probs))
        print(f"  {img_path.name:<50}  [{RISK_NAMES[pred]}]  "
              f"P={probs[pred]*100:.1f}%")

    # 그리드 저장
    out_dir = script_dir / "visualizations"
    out_dir.mkdir(exist_ok=True)
    grid = make_grid(cards, cols=args.cols)
    out_path = out_dir / "risk_predictions.png"
    grid.save(out_path)

    # 요약
    print(f"\n결과 요약")
    print(f"  소  : {risk_cnt[0]}개")
    print(f"  중  : {risk_cnt[1]}개")
    print(f"  대  : {risk_cnt[2]}개")
    print(f"\n이미지 저장: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="위험도 예측 시각화")
    parser.add_argument("--source", type=str, default="../test/images/")
    parser.add_argument("--model",  type=str, default=None)
    parser.add_argument("--n",      type=int, default=9,  help="시각화할 이미지 수 (기본 9)")
    parser.add_argument("--cols",   type=int, default=3,  help="그리드 열 수")
    parser.add_argument("--seed",   type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()
    main(args)
