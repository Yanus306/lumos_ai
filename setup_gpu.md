# GPU 학습 설정 가이드

## 1단계: Python 3.12 설치

### 다운로드
**Python 3.12.8 다운로드 (권장)**:

👉 **직접 다운로드 링크**: 
- Windows 64-bit: https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe
- 또는 공식 사이트: https://www.python.org/downloads/

**중요**: 설치 시 "Add Python to PATH" 체크박스를 **반드시** 선택하세요!

### 설치 확인
새 터미널을 열고 확인:
```bash
python --version
# Python 3.12.x 가 나와야 합니다
```

## 2단계: 필요한 패키지 설치

### 방법 A: 자동 설치 (권장)
```bash
cd c:\YANUS\lumos
setup_gpu.bat
```
이 스크립트가 자동으로 모든 패키지를 설치하고 GPU 인식을 확인합니다.

### 방법 B: 수동 설치

#### PyTorch CUDA 버전 설치
```bash
cd c:\YANUS\lumos

# PyTorch와 torchvision CUDA 12.1 버전 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 다른 패키지 설치
```bash
pip install ultralytics opencv-python matplotlib seaborn pillow pyyaml
```

### GPU 인식 확인
```bash
python -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**출력 예시**:
```
CUDA 사용 가능: True
GPU: NVIDIA GeForce GTX 1650
```

## 3단계: GPU 학습 시작

```bash
cd c:\YANUS\lumos
python train.py
```

학습 시작 시 다음과 같이 GPU가 표시되어야 합니다:
```
Using device: cuda
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
```

`GPU_mem` 컬럼에 GPU 메모리 사용량이 표시되면 GPU로 학습되고 있는 것입니다!

## 예상 학습 시간

- **GTX 1650 GPU**: 약 1-3시간 (100 에포크)
- **CPU**: 약 10-15시간

## 문제 해결

### CUDA를 인식하지 못하는 경우
1. NVIDIA 드라이버가 최신인지 확인
2. Python을 완전히 종료하고 다시 시작
3. PyTorch 재설치:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### GPU 메모리 부족 에러
`train.py` 파일에서 배치 크기를 줄이세요:
```python
batch=8  # 또는 4
```
