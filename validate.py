"""
다크패턴 탐지 모델 검증 스크립트

사용법:
    python validate.py
    python validate.py --model runs/detect/dark_pattern_train/weights/best.pt
"""

from ultralytics import YOLO
import argparse
import os

def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='다크패턴 탐지 모델 검증')
    parser.add_argument('--model', type=str, 
                       default='runs/detect/dark_pattern_train/weights/best.pt',
                       help='검증할 모델 경로')
    parser.add_argument('--data', type=str, default='data.yaml',
                       help='데이터셋 설정 파일')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='검증할 데이터 분할')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='이미지 크기')
    parser.add_argument('--batch', type=int, default=16,
                       help='배치 크기')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='신뢰도 임계값')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='NMS IoU 임계값')
    parser.add_argument('--save-json', action='store_true',
                       help='COCO JSON 형식으로 결과 저장')
    parser.add_argument('--save-hybrid', action='store_true',
                       help='라벨과 예측을 함께 저장')
    
    args = parser.parse_args()
    
    # 모델 파일 존재 확인
    if not os.path.exists(args.model):
        print(f"오류: 모델 파일을 찾을 수 없습니다: {args.model}")
        print("먼저 train.py를 실행하여 모델을 학습시키세요.")
        return
    
    print("="*60)
    print("다크패턴 탐지 모델 검증")
    print("="*60)
    print(f"모델: {args.model}")
    print(f"데이터: {args.data}")
    print(f"분할: {args.split}")
    print(f"신뢰도 임계값: {args.conf}")
    print("="*60 + "\n")
    
    # 모델 로드
    model = YOLO(args.model)
    
    # 검증 실행
    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        save_json=args.save_json,
        save_hybrid=args.save_hybrid,
        plots=True,
        verbose=True,
    )
    
    # 결과 출력
    print("\n" + "="*60)
    print("검증 결과")
    print("="*60)
    
    # 전체 성능 메트릭
    print("\n전체 성능:")
    print(f"  mAP50      : {results.box.map50:.4f}")
    print(f"  mAP50-95   : {results.box.map:.4f}")
    print(f"  Precision  : {results.box.mp:.4f}")
    print(f"  Recall     : {results.box.mr:.4f}")
    
    # 클래스별 성능
    print("\n클래스별 성능:")
    class_names = ['dark_pattern_0', 'dark_pattern_1', 'dark_pattern_2', 'dark_pattern_3']
    
    if hasattr(results.box, 'maps') and results.box.maps is not None:
        print(f"  {'클래스':<20} {'AP50':<10} {'AP50-95':<10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10}")
        for i, (name, ap50, ap) in enumerate(zip(class_names, 
                                                   results.box.ap50, 
                                                   results.box.ap)):
            print(f"  {name:<20} {ap50:.4f}     {ap:.4f}")
    
    # 혼동 행렬 확인
    print("\n혼동 행렬이 생성되었습니다.")
    print("결과 폴더에서 confusion_matrix.png를 확인하세요.")
    
    print("\n" + "="*60)
    print("검증 완료!")
    print("="*60)

if __name__ == '__main__':
    main()
