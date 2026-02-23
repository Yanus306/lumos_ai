# 다크패턴 탐지 AI 모델

YOLOv8 기반의 웹/앱 UI 다크패턴 자동 탐지 시스템

## 📋 개요

이 프로젝트는 웹사이트나 애플리케이션의 UI 스크린샷에서 다크패턴을 자동으로 탐지하는 AI 모델입니다. YOLOv8 객체 탐지 모델을 활용하여 실시간으로 4가지 유형의 다크패턴을 식별합니다.

## 🎯 탐지 가능한 다크패턴

- **dark_pattern_0**: 첫 번째 유형의 다크패턴
- **dark_pattern_1**: 두 번째 유형의 다크패턴
- **dark_pattern_2**: 세 번째 유형의 다크패턴
- **dark_pattern_3**: 네 번째 유형의 다크패턴

> **참고**: 실제 프로젝트에서는 각 클래스명을 구체적인 다크패턴 유형으로 변경하세요.  
> 예: `misleading_button`, `hidden_costs`, `forced_continuity`, `urgency_pressure`

## 📊 데이터셋 구조

```
lumos/
├── train/
│   ├── images/    # 1,594 장의 학습 이미지
│   └── labels/    # YOLO 형식 레이블
├── valid/
│   ├── images/    # 185 장의 검증 이미지
│   └── labels/
├── test/
│   ├── images/    # 183 장의 테스트 이미지
│   └── labels/
├── data.yaml      # 데이터셋 설정
├── train.py       # 모델 학습
├── validate.py    # 모델 검증
└── predict.py     # 예측 및 시각화
```

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 학습

```bash
python train.py
```

**학습 옵션:**
- 기본적으로 100 에포크 학습
- GPU가 있으면 자동으로 사용
- 학습 결과는 `runs/detect/dark_pattern_train/` 폴더에 저장
- 최고 성능 모델: `runs/detect/dark_pattern_train/weights/best.pt`

**예상 학습 시간:**
- GPU (RTX 3080): ~2-3시간
- GPU (RTX 4090): ~1-1.5시간
- CPU: ~10-15시간 (권장하지 않음)

### 3. 모델 검증

```bash
# 테스트 데이터셋으로 평가
python validate.py

# 특정 모델 평가
python validate.py --model runs/detect/dark_pattern_train/weights/best.pt

# 검증 데이터셋 사용
python validate.py --split val
```

### 4. 다크패턴 탐지 (예측)

```bash
# 단일 이미지 예측
python predict.py --source test/images/example.jpg

# 폴더 전체 예측
python predict.py --source test/images/

# 신뢰도 임계값 조정
python predict.py --source test/images/ --conf 0.5

# 결과 표시 및 저장
python predict.py --source test/images/ --show --save
```

## 📈 성능 메트릭

학습 후 다음 메트릭으로 모델 성능을 평가합니다:

- **mAP50**: IoU 0.5에서의 Mean Average Precision
- **mAP50-95**: IoU 0.5~0.95 범위의 mAP
- **Precision**: 정밀도 (탐지한 것 중 실제 다크패턴 비율)
- **Recall**: 재현율 (실제 다크패턴 중 탐지한 비율)

## 🎨 시각화 결과

학습 중 자동으로 생성되는 시각화:

- `results.png`: 학습/검증 손실 및 메트릭 곡선
- `confusion_matrix.png`: 클래스별 혼동 행렬
- `PR_curve.png`: Precision-Recall 곡선
- `F1_curve.png`: F1 스코어 곡선

예측 결과:
- 바운딩 박스가 표시된 이미지
- 클래스명 및 신뢰도 점수 표시

## ⚙️ 고급 설정

### 학습 하이퍼파라미터 조정

`train.py` 파일에서 다음 파라미터를 수정할 수 있습니다:

```python
epochs=100        # 학습 에포크 수
imgsz=640        # 입력 이미지 크기
batch=16         # 배치 크기 (GPU 메모리에 따라 조정)
lr0=0.01         # 초기 학습률
patience=50      # Early stopping patience
```

### 다른 YOLOv8 모델 사용

```python
# train.py에서 모델 변경
model = YOLO('yolov8n.pt')  # nano (가장 빠름, 가벼움)
model = YOLO('yolov8s.pt')  # small
model = YOLO('yolov8m.pt')  # medium
model = YOLO('yolov8l.pt')  # large
model = YOLO('yolov8x.pt')  # xlarge (가장 정확, 무거움)
```

### 신뢰도 임계값 조정

```bash
# 더 엄격한 탐지 (False Positive 감소)
python predict.py --source image.jpg --conf 0.5

# 더 관대한 탐지 (Recall 향상)
python predict.py --source image.jpg --conf 0.15
```

## 🔧 문제 해결

### GPU 메모리 부족 (CUDA Out of Memory)

```python
# train.py에서 배치 크기 줄이기
batch=8  # 또는 4
```

### 학습이 너무 느림

- GPU가 활성화되어 있는지 확인: `torch.cuda.is_available()`
- 더 작은 모델 사용: `yolov8n.pt`
- 이미지 크기 줄이기: `imgsz=416`

### 과적합 (Overfitting)

- 더 많은 데이터 증강 활용
- Early stopping patience 줄이기
- Dropout이나 regularization 강화

## 📝 라이센스

이 프로젝트는 학습 및 연구 목적으로 개발되었습니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

## 📧 문의

프로젝트에 대한 질문이나 피드백이 있으시면 언제든지 연락주세요.
