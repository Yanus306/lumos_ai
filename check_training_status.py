import os
import sys
import time

def check_status():
    # 학습 완료 기준: runs/detect/dark_pattern_s_train*/weights/best.pt 존재 여부
    # 단, 학습 중에도 best.pt는 생성될 수 있으므로, train.py 프로세스가 종료되었는지 확인하거나
    # results.png가 최종적으로 업데이트되었는지 확인해야 함.
    # 가장 확실한 건 해당 폴더에 'args.yaml'과 'results.csv'가 있고 파일 크기가 변하지 않을 때.
    
    runs_dir = 'runs/detect'
    if not os.path.exists(runs_dir):
        sys.exit(1)
        
    subdirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    s_train_dirs = [d for d in subdirs if 'dark_pattern_s_train' in d]
    
    if not s_train_dirs:
        sys.exit(1)

    latest_dir = max(s_train_dirs, key=os.path.getmtime)
    
    # 학습 완료 시 생성되는 파일들
    # best.pt, last.pt, results.png, confusion_matrix.png 등
    # 가장 확실한 건 lock 파일이 없거나(파일 락 구현 필요), 
    # 여기서는 간단히 'confusion_matrix.png'가 생성되었는지로 판단 (학습 완료 시 생성됨)
    
    confusion_matrix = os.path.join(latest_dir, 'confusion_matrix.png')
    
    if os.path.exists(confusion_matrix):
        print("학습 완료 확인됨!")
        sys.exit(0) # 완료
    else:
        # print("아직 학습 중...")
        sys.exit(1) # 미완료

if __name__ == "__main__":
    check_status()
