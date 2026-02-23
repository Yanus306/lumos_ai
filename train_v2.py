from ultralytics import YOLO
import torch

def main():
    # GPU 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        # VRAM 확인 코드 추가 가능
    
    # YOLOv8 Small 모델 초기화 (Nano보다 성능 우수, 속도는 다소 느림)
    # GTX 1650 (4GB) 환경을 고려하여 배치 사이즈 조정 필요
    model = YOLO('yolov8s.pt') 
    
    # 학습 하이퍼파라미터 설정
    # Nano 모델 대비 파라미터 수가 약 3배 많음 (3.2M -> 11.2M)
    results = model.train(
        data='data.yaml',           # 데이터셋 설정 파일
        epochs=150,                 # 에포크 증가 (더 복잡한 모델이라 더 오래 학습 권장)
        imgsz=640,                  # 입력 이미지 크기 (VRAM 부족 시 512로 감소 고려)
        batch=8,                    # 배치 크기 감소 (16 -> 8, VRAM 4GB 고려)
        device=device,              # 사용할 디바이스
        workers=4,                  # 데이터 로딩 워커 수
        project='runs/detect',      # 결과 저장 프로젝트 폴더
        name='dark_pattern_s_train', # 실험 이름 (s 모델)
        exist_ok=True,              # 기존 폴더 덮어쓰기 허용 여부
        pretrained=True,            # 사전 학습된 가중치 사용
        optimizer='AdamW',          # 최적화 알고리즘
        lr0=0.001,                  # 초기 학습률
        patience=20,                # 조기 종료 (20 에포크 동안 개선 없으면 중단)
        
        # 데이터 증강 (Augmentation) 강화
        degrees=10.0,               # 회전 (+/- 10도)
        translate=0.1,              # 이동 (+/- 10%)
        scale=0.5,                  # 크기 조절 (+/- 50%)
        flipud=0.0,                 # 상하 반전 (안함)
        fliplr=0.5,                 # 좌우 반전 (50%)
        mosaic=1.0,                 # Mosaic 증강 (100%)
        mixup=0.1,                  # Mixup 증강 (10% 추가)
    )
    
    print("YOLOv8 Small 모델 학습 완료!")
    
    # 검증 실행
    metrics = model.val()
    print(f"Validation mAP50: {metrics.box.map50}")

if __name__ == '__main__':
    main()
