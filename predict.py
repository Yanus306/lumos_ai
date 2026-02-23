"""
다크패턴 탐지 예측 및 시각화 스크립트

사용법:
    # 단일 이미지 예측
    python predict.py --source test/images/example.jpg
    
    # 폴더 전체 예측
    python predict.py --source test/images/
    
    # 웹캠 실시간 예측
    python predict.py --source 0
"""

from ultralytics import YOLO
import argparse
import os
import cv2
from pathlib import Path

def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='다크패턴 탐지 예측')
    parser.add_argument('--model', type=str,
                       default='runs/detect/dark_pattern_train/weights/best.pt',
                       help='사용할 모델 경로')
    parser.add_argument('--source', type=str, required=False,
                       help='예측할 이미지/폴더/비디오 경로 또는 웹캠 (0)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='추론 이미지 크기')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='신뢰도 임계값 (0.0-1.0)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='NMS IoU 임계값')
    parser.add_argument('--max-det', type=int, default=300,
                       help='이미지당 최대 탐지 수')
    parser.add_argument('--save', action='store_true', default=True,
                       help='결과 이미지 저장')
    parser.add_argument('--save-txt', action='store_true',
                       help='결과를 텍스트 파일로 저장')
    parser.add_argument('--save-conf', action='store_true',
                       help='신뢰도 점수 저장')
    parser.add_argument('--save-crop', action='store_true',
                       help='예측 박스 영역 크롭하여 저장')
    parser.add_argument('--show', action='store_true',
                       help='결과 이미지 표시')
    parser.add_argument('--show-labels', action='store_true', default=True,
                       help='라벨 표시')
    parser.add_argument('--show-conf', action='store_true', default=True,
                       help='신뢰도 표시')
    parser.add_argument('--line-width', type=int, default=2,
                       help='바운딩 박스 선 두께')
    
    args = parser.parse_args()
    
    # 모델 파일 존재 확인
    if not os.path.exists(args.model):
        print(f"오류: 모델 파일을 찾을 수 없습니다: {args.model}")
        print("먼저 train.py를 실행하여 모델을 학습시키세요.")
        return
    
    # source가 없으면 테스트 이미지 하나 사용
    if args.source is None:
        test_images = list(Path('test/images').glob('*.jpg'))
        if not test_images:
            print("오류: test/images/ 폴더에 이미지가 없습니다.")
            print("--source 옵션으로 이미지 경로를 지정하세요.")
            return
        args.source = str(test_images[0])
        print(f"source가 지정되지 않아 테스트 이미지를 사용합니다: {args.source}")
    
    print("="*60)
    print("다크패턴 탐지 예측")
    print("="*60)
    print(f"모델: {args.model}")
    print(f"Source: {args.source}")
    print(f"신뢰도 임계값: {args.conf}")
    print("="*60 + "\n")
    
    # 모델 로드
    model = YOLO(args.model)
    
    # 예측 실행
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        show=args.show,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        line_width=args.line_width,
        project='runs/detect',
        name='predict',
        exist_ok=True,
        verbose=True,
    )
    
    # 결과 요약
    print("\n" + "="*60)
    print("예측 결과 요약")
    print("="*60)
    
    total_detections = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    class_names = {
        0: 'dark_pattern_0',
        1: 'dark_pattern_1', 
        2: 'dark_pattern_2',
        3: 'dark_pattern_3'
    }
    
    for i, result in enumerate(results):
        num_boxes = len(result.boxes)
        total_detections += num_boxes
        
        print(f"\n이미지 {i+1}: {num_boxes}개의 다크패턴 탐지됨")
        
        if num_boxes > 0:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_counts[cls] += 1
                print(f"  - {class_names[cls]}: {conf:.2f}")
    
    print("\n클래스별 탐지 수:")
    for cls_id, count in class_counts.items():
        if count > 0:
            print(f"  {class_names[cls_id]}: {count}개")
    
    print(f"\n총 {total_detections}개의 다크패턴이 탐지되었습니다.")
    
    if args.save:
        print(f"\n결과 이미지가 저장되었습니다: runs/detect/predict/")
    
    print("="*60)

if __name__ == '__main__':
    main()
