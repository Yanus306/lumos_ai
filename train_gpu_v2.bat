@echo off
REM 성능 향상 버전 (Small 모델) 학습 스크립트
set PYTHON312=C:\Users\chlqh\AppData\Local\Programs\Python\Python312\python.exe

echo ====================================
echo YOLOv8 Small 모델(성능 향상 버전) 학습 시작
echo ====================================
echo 주의: Nano 모델보다 학습 시간이 약 2-3배 더 소요됩니다.
echo Python: %PYTHON312%
echo.

%PYTHON312% train_v2.py
pause
