"""
train_classifier.py
===================
EfficientNet-B0 기반 다크패턴 위험도 분류기(소/중/대) 학습 스크립트

사용법:
  cd risk_classifier
  python train_classifier.py                          # 기본 설정으로 학습
  python train_classifier.py --epochs 30 --batch 8   # 커스텀 설정
  python train_classifier.py --model resnet50         # ResNet50 사용

결과는 risk_classifier/runs/<실행명>/ 에 저장됩니다.
"""

import argparse
import os
import sys
import csv
import time
from pathlib import Path

# Windows 터미널 유니코드 출력 설정
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


# ─────────────────────────────────────────────
# 데이터셋 클래스
# ─────────────────────────────────────────────
class RiskDataset(Dataset):
    """CSV 파일에서 이미지 경로와 위험도 레이블을 읽어오는 Dataset."""

    RISK_LABELS = {0: "소", 1: "중", 2: "대"}

    def __init__(self, csv_path: str, transform=None):
        self.transform = transform
        self.samples = []

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"\nCSV 파일을 찾을 수 없습니다: {csv_path}\n"
                "먼저 prepare_data.py를 실행하세요:\n"
                "  python prepare_data.py"
            )

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = Path(row["image_path"])
                risk = int(row["risk"])
                if img_path.exists():
                    self.samples.append((str(img_path), risk))

        if len(self.samples) == 0:
            raise ValueError(f"유효한 이미지가 없습니다: {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, risk = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, risk


# ─────────────────────────────────────────────
# 모델 생성
# ─────────────────────────────────────────────
def build_model(model_name: str, num_classes: int = 3, pretrained: bool = True):
    """사전학습된 모델을 불러와 분류 헤드를 교체합니다."""
    weights_param = "pretrained" if pretrained else None

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}\n"
                         "선택 가능: efficientnet_b0, efficientnet_b2, resnet50, resnet18, mobilenet_v3_small")

    return model


# ─────────────────────────────────────────────
# 학습 함수
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total * 100


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * 3
    class_total   = [0] * 3

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += imgs.size(0)

        for c in range(3):
            mask = labels == c
            class_correct[c] += (predicted[mask] == c).sum().item()
            class_total[c]   += mask.sum().item()

    acc = correct / total * 100 if total > 0 else 0
    class_acc = [
        class_correct[c] / class_total[c] * 100 if class_total[c] > 0 else 0
        for c in range(3)
    ]
    return total_loss / total, acc, class_acc


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main(args):
    RISK_NAMES = ["소 (Low)", "중 (Medium)", "대 (High)"]

    # 경로 설정
    script_dir = Path(__file__).parent
    train_csv  = script_dir / "train_labels.csv"
    valid_csv  = script_dir / "valid_labels.csv"
    run_dir    = script_dir / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 디바이스
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  다크패턴 위험도 분류기 학습")
    print(f"{'='*60}")
    print(f"  디바이스  : {device}")
    print(f"  모델      : {args.model}")
    print(f"  에포크    : {args.epochs}")
    print(f"  배치 크기 : {args.batch}")
    print(f"  학습률    : {args.lr}")
    print(f"  저장 경로 : {run_dir}")

    # 데이터 transforms
    IMG_SIZE = 224
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # 데이터셋 & 로더
    print(f"\n  CSV 로딩 중...")
    train_ds = RiskDataset(train_csv, transform=train_tf)
    valid_ds = RiskDataset(valid_csv, transform=val_tf)

    print(f"  학습 샘플: {len(train_ds):,}개")
    print(f"  검증 샘플: {len(valid_ds):,}개")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )

    # 클래스 가중치 계산 (불균형 데이터 대응)
    from collections import Counter
    label_counter = Counter(s[1] for s in train_ds.samples)
    total_samples = sum(label_counter.values())
    class_weights = torch.tensor([
        total_samples / (3 * max(label_counter.get(i, 1), 1))
        for i in range(3)
    ], dtype=torch.float32).to(device)
    print(f"\n  클래스 가중치: {class_weights.cpu().numpy()}")

    # 모델 & 손실함수 & 옵티마이저
    print(f"\n  모델 초기화 중 (pretrained={not args.no_pretrain})...")
    model = build_model(args.model, num_classes=3, pretrained=not args.no_pretrain)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 학습 로그
    log_path = run_dir / "training_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc",
                         "val_acc_소", "val_acc_중", "val_acc_대", "lr"])

    best_val_acc = 0.0
    best_epoch   = 0

    print(f"\n  학습 시작!\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, class_acc = evaluate(
            model, valid_loader, criterion, device)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"  Epoch [{epoch:3d}/{args.epochs}] "
              f"loss={train_loss:.4f} acc={train_acc:.1f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}%  "
              f"[소:{class_acc[0]:.0f}% 중:{class_acc[1]:.0f}% 대:{class_acc[2]:.0f}%]  "
              f"({elapsed:.1f}s)")

        # 로그 저장
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                             f"{val_loss:.4f}", f"{val_acc:.2f}",
                             f"{class_acc[0]:.2f}", f"{class_acc[1]:.2f}", f"{class_acc[2]:.2f}",
                             f"{lr_now:.6f}"])

        # 최고 모델 저장
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_path    = run_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_acc": class_acc,
            }, best_path)
            print(f"  [BEST] 최고 모델 저장! (val_acc={val_acc:.1f}%)")

        # 마지막 모델 저장 (매 epoch)
        torch.save({
            "epoch": epoch,
            "model_name": args.model,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
        }, run_dir / "last.pt")

    print(f"\n{'='*60}")
    print(f"  학습 완료!")
    print(f"  최고 성능: epoch {best_epoch}, val_acc={best_val_acc:.2f}%")
    print(f"  저장 위치: {run_dir}")
    print(f"    - best.pt: 최고 성능 모델")
    print(f"    - last.pt: 마지막 에포크 모델")
    print(f"    - training_log.csv: 에포크별 성능 기록")
    print(f"\n  다음 단계: python predict_risk.py --model {run_dir / 'best.pt'} --source ../test/images/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다크패턴 위험도 분류기 학습")

    parser.add_argument("--model",      type=str,   default="efficientnet_b0",
                        help="모델 이름 (efficientnet_b0, efficientnet_b2, resnet50, resnet18, mobilenet_v3_small)")
    parser.add_argument("--epochs",     type=int,   default=30,
                        help="학습 에포크 수 (기본: 30)")
    parser.add_argument("--batch",      type=int,   default=16,
                        help="배치 크기 (기본: 16, GPU 메모리 부족 시 8이나 4로 낮추세요)")
    parser.add_argument("--lr",         type=float, default=1e-4,
                        help="초기 학습률 (기본: 0.0001)")
    parser.add_argument("--workers",    type=int,   default=0,
                        help="데이터 로딩 worker 수 (Windows는 0 권장)")
    parser.add_argument("--run-name",   type=str,   default="risk_classifier_v1",
                        help="실행 이름 (결과 폴더명)")
    parser.add_argument("--no-pretrain", action="store_true",
                        help="사전학습 가중치 사용 안 함")

    args = parser.parse_args()
    main(args)
