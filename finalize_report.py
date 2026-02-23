import os
import re
import shutil
from pathlib import Path
from ultralytics import YOLO

def update_report():
    print("Small 모델 학습 결과 업데이트 중...")
    
    # 모델 경로 확인 (가장 최근 학습 결과)
    # 중첩된 runs/detect/runs/detect 구조도 확인
    possible_roots = [
        'runs/detect',
        'runs/detect/runs/detect'
    ]
    
    s_train_dirs = []
    
    s_train_dirs = []
    
    for root in possible_roots:
        if os.path.exists(root):
            subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            # 'dark_pattern_s_train'이 포함된 폴더 찾기
            found = [d for d in subdirs if 'dark_pattern_s_train' in d]
            s_train_dirs.extend(found)
    
    # weights/best.pt가 존재하는 폴더만 필터링
    valid_dirs = []
    for d in s_train_dirs:
        if os.path.exists(os.path.join(d, 'weights', 'best.pt')):
            valid_dirs.append(d)

    if not valid_dirs:
        print("유효한 Small 모델 학습 폴더(weights/best.pt 포함)를 찾을 수 없습니다.")
        return

    # mtime 기준으로 정렬하여 가장 최신 폴더 선택
    latest_dir = max(valid_dirs, key=os.path.getmtime)
    best_pt = os.path.join(latest_dir, 'weights', 'best.pt')
    
    if not os.path.exists(best_pt):
        print(f"모델 파일이 없습니다: {best_pt}")
        return
        
    print(f"모델 로드: {best_pt}")
    model = YOLO(best_pt)
    
    # 검증 실행
    metrics = model.val(split='test')
    
    # 결과 이미지 저장 경로 (metrics.save_dir은 Path 객체)
    save_dir = metrics.save_dir
    print(f"결과 이미지 저장 경로: {save_dir}")

    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr
    f1 = 2 * (precision * recall) / (precision + recall + 1e-16)
    
    print(f"성능 측정 완료: mAP50={map50:.3f}, F1={f1:.3f}")
    
    # report.md 업데이트
    artifact_dir = r'C:\Users\chlqh\.gemini\antigravity\brain\b2a1117d-6528-43eb-8ed0-a973a777b8f2'
    report_path = os.path.join(artifact_dir, 'report.md')
    
    # 이미지 복사
    images_to_copy = ['confusion_matrix.png', 'BoxF1_curve.png', 'val_batch0_pred.jpg']
    
    for img_name in images_to_copy:
        src = save_dir / img_name
        dst = os.path.join(artifact_dir, img_name)
        if src.exists():
            shutil.copy(src, dst)
            print(f"이미지 복사 완료: {img_name}")
        else:
            print(f"이미지 없음: {img_name}")
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Small 모델 행 업데이트
    # | **YOLOv8 Small** | 고성능 | *(진행 중)* | ...
    
    new_line = f"| **YOLOv8 Small** | 고성능 | **{map50:.3f}** | {map50_95:.3f} | {precision:.3f} | {recall:.3f} | {f1:.3f} | **완료** ✅ |"
    
    # 정규표현식으로 해당 라인 찾아서 교체
    pattern = r"\| \*\*YOLOv8 Small\*\* \| 고성능 \| .*? \| .*? \| .*? \| .*? \| .*? \| .*? \|"
    updated_content = re.sub(pattern, new_line, content)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
        
    print("report.md 업데이트 완료!")

if __name__ == "__main__":
    update_report()
