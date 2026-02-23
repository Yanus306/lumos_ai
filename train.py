"""
다크패턴 탐지 YOLOv8 모델 학습 스크립트

사용법:
    python train.py
    
옵션 조정이 필요한 경우 아래 하이퍼파라미터를 수정하세요.
"""

from ultralytics import YOLO
import torch
import os

def main():
    # GPU 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # YOLOv8 모델 초기화 (nano 버전으로 시작)
    # 더 큰 모델이 필요한 경우: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    model = YOLO('yolov8n.pt')
    
    # 학습 하이퍼파라미터
    results = model.train(
        data='data.yaml',           # 데이터셋 설정 파일
        epochs=100,                 # 학습 에포크 수
        imgsz=640,                  # 입력 이미지 크기
        batch=16,                   # 배치 크기 (GPU 메모리에 따라 조정)
        device=device,              # 사용할 디바이스
        workers=8,                  # 데이터 로더 워커 수
        project='runs/detect',      # 결과 저장 폴더
        name='dark_pattern_train',  # 실험 이름
        exist_ok=True,              # 기존 폴더 덮어쓰기 허용
        patience=50,                # Early stopping patience
        save=True,                  # 체크포인트 저장
        save_period=10,             # 10 에포크마다 체크포인트 저장
        
        # 데이터 증강 설정
        hsv_h=0.015,               # HSV Hue 증강
        hsv_s=0.7,                 # HSV Saturation 증강
        hsv_v=0.4,                 # HSV Value 증강
        degrees=0.0,               # 회전 (UI 스크린샷은 보통 회전 안 함)
        translate=0.1,             # 이동
        scale=0.5,                 # 스케일
        shear=0.0,                 # 전단 변환
        perspective=0.0,           # 원근 변환
        flipud=0.0,                # 상하 반전 (UI는 보통 안 함)
        fliplr=0.5,                # 좌우 반전
        mosaic=1.0,                # Mosaic 증강
        mixup=0.0,                 # MixUp 증강
        
        # 최적화 설정
        optimizer='auto',          # 옵티마이저 (auto, SGD, Adam, AdamW)
        lr0=0.01,                  # 초기 학습률
        lrf=0.01,                  # 최종 학습률 (lr0 * lrf)
        momentum=0.937,            # SGD momentum/Adam beta1
        weight_decay=0.0005,       # 가중치 감쇠
        warmup_epochs=3.0,         # Warmup 에포크
        warmup_momentum=0.8,       # Warmup momentum
        warmup_bias_lr=0.1,        # Warmup bias 학습률
        
        # 기타 설정
        box=7.5,                   # Box loss gain
        cls=0.5,                   # Class loss gain
        dfl=1.5,                   # DFL loss gain
        plots=True,                # 학습 플롯 저장
        verbose=True,              # 상세 출력
    )
    
    # 학습 완료 메시지
    print("\n" + "="*60)
    print("학습이 완료되었습니다!")
    print("="*60)
    print(f"최고 성능 모델: runs/detect/dark_pattern_train/weights/best.pt")
    print(f"마지막 모델: runs/detect/dark_pattern_train/weights/last.pt")
    print(f"학습 결과: runs/detect/dark_pattern_train/")
    print("="*60)
    
    # 최종 성능 메트릭 출력
    print("\n최종 검증 성능:")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

if __name__ == '__main__':
    main()
