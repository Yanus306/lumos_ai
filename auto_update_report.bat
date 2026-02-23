@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo [자동 모니터링] 학습이 완료되면 리포트를 업데이트합니다.
echo ========================================================
echo.
echo 현재 Small 모델 학습 진행 중... (완료 대기 중)
echo 창을 닫지 말고 그대로 두세요.
echo.

:LOOP
REM 1분(60초) 대기
timeout /t 60 /nobreak >nul

REM Small 모델 학습 완료 파일(best.pt) 확인
REM 폴더명이 가변적이므로 runs/detect 내의 최신 dark_pattern_s_train 폴더를 찾아야 함
REM 여기서는 간단하게 가장 최근에 생성된 폴더 내의 best.pt를 찾도록 파이썬 스크립트에 위임하거나
REM 파이썬 스크립트 자체를 반복 실행하여 완료 여부를 체크하게 하는 것이 나음

REM finalize_report.py를 약간 수정하여 "학습 완료 체크" 기능 추가
python check_training_status.py
if errorlevel 1 (
    REM 아직 완료 안 됨 (exit code 1)
    goto LOOP
) else (
    REM 완료 됨 (exit code 0)
    echo.
    echo 학습 완료 감지! 리포트 업데이트를 시작합니다.
    python finalize_report.py
    echo.
    echo 모든 작업이 완료되었습니다. 창을 닫아도 좋습니다.
    pause
    exit
)
