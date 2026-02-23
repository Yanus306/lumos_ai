@echo off
echo ====================================
echo GPU 학습 환경 설정 스크립트
echo ====================================
echo.

echo [1/3] Python 버전 확인 중...
python --version
echo.

echo [2/3] PyTorch CUDA 버전 설치 중...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo.

echo [3/3] Ultralytics 및 기타 패키지 설치 중...
python -m pip install ultralytics opencv-python matplotlib seaborn pillow pyyaml
echo.

echo ====================================
echo GPU 인식 테스트
echo ====================================
python -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available()); print('CUDA 버전:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU 이름:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo.

echo ====================================
echo 설정 완료!
echo ====================================
echo.
echo 다음 명령으로 GPU 학습을 시작하세요:
echo   python train.py
echo.
pause
