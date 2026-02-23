# 백엔드 개발자를 위한 다크패턴 탐지 시스템 통합 가이드

안녕하세요! 본 문서는 다크패턴 탐지 AI 모델을 백엔드 시스템(FastAPI, Flask, Spring 등)에 통합하기 위해 제공되는 가이드입니다. 

제공해드린 폴더에는 다크패턴을 화면에서 찾아내는 **객체 탐지 모델**과 찾아낸 패턴의 **위험도를 판별하는 분류 모델**, 총 2가지의 AI 모델 가중치가 포함되어 있습니다.

---

## 📁 전달된 파일 목록과 역할 설명

### 1. `yolov8_dark_pattern.pt` (위치 탐지 모델)
* **어떤 모델인가요?**: 사용자가 캡처한 웹/앱 화면 전체 이미지 속에서 **다크패턴 요소(버튼, 배너, 텍스트 등)가 어디에 위치해 있는지**를 찾아주는 YOLOv8 기반 객체 탐지 모델입니다.
* **무엇을 반환하나요?**: 이미지 내 다크패턴 타겟들의 바운딩 박스(Bounding Box) 좌표 `[x1, y1, x2, y2]`와 어떤 유형의 패턴인지 클래스 정보를 반환합니다.

### 2. `efficientnet_risk.pt` (위험도 분류 모델)
* **어떤 모델인가요?**: 1번 모델(YOLO)이 찾아낸 네모난 영역의 다크패턴 이미지를 바탕으로, **해당 다크패턴이 사용자에게 미치는 위험도 등급(소/중/대)**을 판별해 주는 EfficientNet-B0 기반 이미지 분류 모델입니다.
* **무엇을 반환하나요?**: 각각의 소/중/대 위험도에 대한 모델의 확신 퍼센티지(확률값)와 최종 결정된 기준 위험도를 반환합니다.

### 3. `risk_inference.py` (분류 모델 실행 도우미)
* **어떤 파일인가요?**: `efficientnet_risk.pt` 모델의 구조는 파이썬 코드로 정의되어야 풀리기 때문에, 백엔드 서버에서 이 모델을 띄워 이미지 위험도를 추론할 수 있도록 미리 작성해둔 **파이썬 샘플/코어 코드**입니다.
* **어떻게 쓰나요?**: 파이썬 기반 서버를 구축하신다면 이 코드를 프로젝트에 복사해 넣고, `predict_dark_pattern_risk` 함수를 호출하시기만 하면 됩니다.

### 4. `test_models.py` & `test_result.png` (통합 테스트 스크립트)
* 두 모델이 어떻게 결합하여 작동하는지 시각적으로 보여주고 테스트하기 위한 샘플 코드 파일과 그 결과값 이미지입니다!

---


### 1. 웹 화면 -> 다크패턴 객체 탐지 (YOLOv8)

YOLOv8 모델은 `ultralytics` 패키지만 설치하시면 몇 줄의 코드로 바로 작동합니다.

**설치:**
```bash
pip install ultralytics
```

**예제 코드:**
```python
from ultralytics import YOLO

# 서버 시작 시 메모리에 1회만 로드
yolo_model = YOLO("yolov8_dark_pattern.pt")

def detect_dark_patterns(image_path_or_bytes):
    # confidence 임계값을 0.2 ~ 0.5 사이로 튜닝하여 사용하세요.
    results = yolo_model(image_path_or_bytes, conf=0.25)
    
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist() # 픽셀 좌표값 [가로시작, 세로시작, 가로끝, 세로끝]
            conf = float(box.conf[0])            # 신뢰도 점수 (0~1)
            cls_idx = int(box.cls[0])            # 탐지된 클래스 인덱스
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_index": cls_idx
            })
    return detections
```

### 2. 탐지된 객체 위치 자르기 -> 위험도 분류 (EfficientNet)

탐지된 좌표값을 토대로 원본 모바일/웹 이미지를 잘라낸 뒤(Cropping), 제공해드린 `risk_inference.py` 의 모델 추론 함수를 호출합니다.

**설치:**
```bash
# 위험도 모델 구동을 위해 필요한 패키지
pip install torch torchvision Pillow
```

**예제 코드:**
```python
# 제공해드린 파일에서 예측 함수 임포트
from risk_inference import predict_dark_pattern_risk
from PIL import Image

# 1. 원본 이미지 로드
original_image = Image.open(image_path_or_bytes).convert("RGB")

for det in detections:
    x1, y1, x2, y2 = det["bbox"]
    
    # 2. YOLO에서 찾은 바운딩 박스를 통대로 이미지를 잘라냄
    cropped_img = original_image.crop((x1, y1, x2, y2))
    
    # 3. 잘라낸 이미지를 위험도 모델에 통과시킴
    risk_result = predict_dark_pattern_risk(cropped_img)
    
    # 결과 예시:
    # {
    #   "risk_level": "대", 
    #   "risk_index": 2, 
    #   "probabilities": {"소": 0.05, "중": 0.1, "대": 0.85}
    # }
```

## 🔄 프론트엔드 연동 반환(Response) 포맷 추천

위의 과정을 거쳐 백엔드에서 생성된 최종 데이터를 프론트엔드 API로 응답할 때의 JSON 포맷 예시입니다. 이 포맷을 기준으로 프론트엔드 팀과 화면 표시(네모칸 그리기) 협의를 추천드립니다.

```json
{
  "status": "success",
  "data": {
    "image_id": "req-12345",
    "detections": [
      {
        "bbox": [100.2, 50.5, 300.0, 150.0],
        "pattern_type": "dark_pattern_1",
        "yolo_confidence": 0.88,
        "risk": {
          "level": "대",
          "score": 0.95
        }
      },
      {
        "bbox": [400.0, 600.0, 500.0, 700.0],
        "pattern_type": "dark_pattern_3",
        "yolo_confidence": 0.65,
        "risk": {
          "level": "중",
          "score": 0.55
        }
      }
    ]
  }
}
```
