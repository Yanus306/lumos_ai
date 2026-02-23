import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# =====================================================================
# 1. 모델 아키텍처 정의 (EfficientNet-B0 기반 위험도 분류 모델)
# =====================================================================
def build_risk_model(model_name: str = "efficientnet_b0", num_classes: int = 3):
    """
    학습 시 사용한 모델 아키텍처와 동일하게 모델을 생성합니다.
    """
    if model_name == "efficientnet_b0":
        # 사전학습 가중치 없이 껍데기만 생성 (어차피 pt 파일로 덮어씌움)
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        # 분류기 마지막 레이어를 클래스 수(3)에 맞게 변경 (소, 중, 대)
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"해당 시스템은 {model_name} 아키텍처만 지원합니다.")
    return model

# =====================================================================
# 2. 이미지 전처리 파이프라인
# =====================================================================
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# =====================================================================
# 3. 모델 로드 (서버 기동 시 1회만 실행하는 것을 권장)
# =====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RISK_MODEL_PATH = "./efficientnet_risk.pt"  # 현재 폴더에 있는 pt 파일 경로

# 체크포인트 로드
ckpt = torch.load(RISK_MODEL_PATH, map_location=DEVICE)
model_name = ckpt.get("model_name", "efficientnet_b0")

# 모델 생성 및 가중치 삽입
risk_model = build_risk_model(model_name=model_name, num_classes=3)
risk_model.load_state_dict(ckpt["model_state_dict"])
risk_model.to(DEVICE)
risk_model.eval()  # 추론 모드로 변경

print(f"[INFO] 모델 로드 완료: {DEVICE} 환경에서 실행 중입니다.")

# =====================================================================
# 4. 추론 함수 (API 엔드포인트에서 호출)
# =====================================================================
@torch.no_grad()
def predict_dark_pattern_risk(image_path_or_file):
    """
    이미지를 입력받아 다크패턴 위험도(소/중/대)와 각 확률을 반환합니다.
    
    Args:
        image_path_or_file: 이미지 파일 경로 또는 FastAPI/Flask의 UploadFile 객체 (바이트스트림)
        
    Returns:
        dict: 예측 결과 (예: {"risk_level": "대", "risk_index": 2, "probabilities": {"소": 0.05, "중": 0.1, "대": 0.85}})
    """
    # 이미지 로드 (RGB 포맷으로 변환)
    img = Image.open(image_path_or_file).convert("RGB")
    
    # 전처리 및 텐서 변환 (Batch 차원 추가: [1, 3, 224, 224])
    tensor = VAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    
    # 모델 예측
    logits = risk_model(tensor)
    
    # Softmax를 통과시켜 확률값으로 변환
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    
    # 가장 높은 확률을 가진 클래스 인덱스 (0: 소, 1: 중, 2: 대)
    pred_idx = int(probs.argmax())
    
    risk_labels = {0: "소", 1: "중", 2: "대"}
    
    return {
        "risk_level": risk_labels[pred_idx],
        "risk_index": pred_idx,
        "probabilities": {
            "소": float(probs[0]),
            "중": float(probs[1]),
            "대": float(probs[2])
        }
    }

# =====================================================================
# 사용 예시 (테스트용)
# =====================================================================
if __name__ == "__main__":
    # 백엔드 개발자 테스트용 코드 (실제 서비스 시 제거하거나 별도로 분리하세요)
    # test.jpg를 현재 폴더에 놓고 실행해보세요.
    test_image = "test.jpg"
    import os
    if os.path.exists(test_image):
        result = predict_dark_pattern_risk(test_image)
        print(f"예측 결과: {result}")
    else:
        print(f"테스트용 이미지가 없습니다: {test_image}")
