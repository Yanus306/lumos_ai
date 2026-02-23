@echo off
REM GPU 학습 스크립트 - Python 3.12 사용
set PYTHON312=C:\Users\chlqh\AppData\Local\Programs\Python\Python312\python.exe

echo ====================================
echo GPU로 YOLOv8 모델 학습 시작
echo ====================================
echo Python: %PYTHON312%
echo.

%PYTHON312% train.py
