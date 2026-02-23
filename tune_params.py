"""
파라미터 튜닝 스크립트
다양한 conf, iou 조합으로 모델을 평가하여 최적의 설정을 찾습니다.
"""
from ultralytics import YOLO
import pandas as pd
import torch

def main():
    model_path = 'runs/detect/runs/detect/dark_pattern_train/weights/best.pt'
    model = YOLO(model_path)
    
    # 테스트할 파라미터 조합
    # conf: 신뢰도 임계값 (높으면 정밀도 상승, 재현율 하락)
    # iou: NMS 임계값 (높으면 겹친 박스 허용)
    params = [
        {'conf': 0.25, 'iou': 0.45}, # 기본값
        {'conf': 0.15, 'iou': 0.45}, # Recall 중시
        {'conf': 0.40, 'iou': 0.45}, # Precision 중시
        {'conf': 0.25, 'iou': 0.60}, # 겹친 객체 허용
        {'conf': 0.30, 'iou': 0.50}, # 균형
        {'conf': 0.50, 'iou': 0.45}, # 고신뢰도만
    ]
    
    results = []
    
    print("="*60)
    print("파라미터 튜닝 시작")
    print("="*60)
    
    for p in params:
        print(f"\n테스트 중: conf={p['conf']}, iou={p['iou']}")
        
        # 검증 실행 (verbose=False로 상세 출력 생략)
        metrics = model.val(
            data='data.yaml',
            split='test',
            conf=p['conf'],
            iou=p['iou'],
            verbose=False,
            plots=False
        )
        
        # 필요한 메트릭 추출
        # map50: mAP@0.5
        # map: mAP@0.5:0.95
        # mp: mean Precision
        # mr: mean Recall
        
        res = {
            'conf': p['conf'],
            'iou': p['iou'],
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'Precision': metrics.box.mp,
            'Recall': metrics.box.mr,
            'F1': 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-16)
        }
        
        results.append(res)
        print(f"  -> mAP50: {res['mAP50']:.4f}, F1: {res['F1']:.4f}")

    # 결과 정리
    print("\n" + "="*60)
    print("튜닝 결과 요약")
    print("="*60)
    
    # F1 스코어 기준 정렬
    sorted_results = sorted(results, key=lambda x: x['F1'], reverse=True)
    
    print(f"{'Conf':<10} {'IoU':<10} {'mAP50':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 65)
    
    for r in sorted_results:
        print(f"{r['conf']:<10} {r['iou']:<10} {r['mAP50']:.4f}     {r['F1']:.4f}     {r['Precision']:.4f}      {r['Recall']:.4f}")
        
    best = sorted_results[0]
    print("\n" + "="*60)
    print(f"추천 설정: conf={best['conf']}, iou={best['iou']}")
    print("="*60)

if __name__ == '__main__':
    main()
