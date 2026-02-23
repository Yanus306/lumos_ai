@echo off
echo ====================================
echo Python 3.12로 PyTorch CUDA 설치
echo ====================================
echo.

set PYTHON312=C:\Users\chlqh\AppData\Local\Programs\Python\Python312\python.exe

echo [1/3] Python 버전 확인...
%PYTHON312% --version
echo.

echo [2/3] PyTorch CUDA 버전 설치 중...
%PYTHON312% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo.

echo [3/3] Ultralytics 및 기타 패키지 설치 중...
%PYTHON312% -m pip install ultralytics opencv-python matplotlib seaborn pillow pyyaml
echo.

echo ====================================
echo GPU 인식 테스트
echo ====================================
%PYTHON312% -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo.

echo ====================================
echo 설정 완료!
echo ====================================
pause
