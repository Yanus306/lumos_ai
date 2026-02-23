# 다크패턴 위험도 분류기

기존 YOLOv8 탐지 데이터셋을 재활용해 이미지 단위로 **위험도(소/중/대)** 를 분류하는 AI 모델입니다.

## 📁 폴더 구조

```
risk_classifier/
├── prepare_data.py      # YOLO 레이블 → 위험도 CSV 변환
├── train_classifier.py  # EfficientNet-B0 분류기 학습
├── predict_risk.py      # 학습된 모델로 위험도 예측
├── data.yaml            # 데이터셋 설정
├── train_labels.csv     # (생성됨) 학습 레이블
├── valid_labels.csv     # (생성됨) 검증 레이블
├── test_labels.csv      # (생성됨) 테스트 레이블
└── runs/
    └── risk_classifier_v1/
        ├── best.pt          # 최고 성능 모델
        ├── last.pt          # 마지막 에포크 모델
        └── training_log.csv # 에포크별 성능 기록
```

## 🎯 위험도 기준

| 위험도 | 설명 | 기준 |
|--------|------|------|
| **소 (Low)** | 안전 | 탐지된 다크패턴 0~1개 |
| **중 (Medium)** | 주의 | 탐지 2~4개 (고위험 패턴 없음) |
| **대 (High)** | 위험 | 탐지 5개 이상 또는 고위험 패턴(클래스 2,3) 포함 |

> 고위험 클래스 기준은 `prepare_data.py`의 `HIGH_RISK_CLASSES` 변수에서 조정할 수 있습니다.

## 🚀 사용법

### 1단계: 데이터 준비

```bash
cd risk_classifier
python prepare_data.py
```

YOLO 레이블을 분석해 `train_labels.csv`, `valid_labels.csv`, `test_labels.csv`를 생성합니다.

### 2단계: 모델 학습

```bash
# 기본 설정 (EfficientNet-B0, 30 에포크)
python train_classifier.py

# 옵션 조정
python train_classifier.py --epochs 50 --batch 8 --lr 5e-5
python train_classifier.py --model resnet50
```

**옵션:**
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `efficientnet_b0` | 모델 (efficientnet_b0/b2, resnet18/50, mobilenet_v3_small) |
| `--epochs` | `30` | 학습 에포크 수 |
| `--batch` | `16` | 배치 크기 |
| `--lr` | `1e-4` | 학습률 |
| `--run-name` | `risk_classifier_v1` | 결과 저장 폴더명 |

### 3단계: 위험도 예측

```bash
# 폴더 전체 예측
python predict_risk.py --source ../test/images/

# 단일 이미지
python predict_risk.py --source ../test/images/example.png

# 신뢰도 낮은 결과 필터링
python predict_risk.py --source ../test/images/ --conf 0.6
```

결과는 `predictions/results.csv`에 저장됩니다.

## ⚙️ 문제 해결

**GPU 메모리 부족:**
```bash
python train_classifier.py --batch 4
```

**학습이 느림 (CPU):**
```bash
python train_classifier.py --model mobilenet_v3_small --epochs 20
```

**클래스 불균형 심함:**  
`prepare_data.py`의 `LOW_MAX`, `MEDIUM_MAX` 임계값을 조정해 세 클래스 비율을 균등하게 맞추세요.
