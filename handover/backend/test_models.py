import os
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# risk_inference.py 에서 위험도 예측 함수 가져오기
from risk_inference import predict_dark_pattern_risk

def create_visual_test():
    print("\n" + "="*50)
    print(" 🛠️ 전달용 AI 모델 역할 검증 테스트 시작")
    print("="*50)

    # 1. 테스트 이미지 로드
    test_image_dir = "../../test/images/"
    if not os.path.exists(test_image_dir):
        print(f"❌ 오류: 테스트 이미지 폴더를 찾을 수 없습니다 ({test_image_dir})")
        return

    images = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.png'))]
    if not images:
        print("❌ 오류: 테스트 이미지가 없습니다.")
        return

    # 첫 번째 이미지 선택
    test_image_path = os.path.join(test_image_dir, images[0])
    print(f"\n[준비] 테스트 화면 로드: {images[0]}")
    original_image = Image.open(test_image_path).convert("RGB")
    
    # 시각화용 이미지 복사 및 그리기 도구 준비
    draw_image = original_image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # 폰트 설정 (기본 폰트 사용)
    try:
        # 윈도우 맑은 고딕 시도
        font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 16)
    except:
        font = ImageFont.load_default()

    # ====================================================================
    # 역할 1: YOLO 객체 탐지 모델 (다크패턴 위치 찾기)
    # ====================================================================
    print("\n[역할 1 검증] YOLO 모델 작동 (목적: 다크패턴 화면상 위치 찾기)")
    try:
        yolo_model = YOLO("yolov8_dark_pattern.pt")
        print(" -> ✅ YOLO 모델 로드 성공")
    except Exception as e:
        print(f" -> ❌ YOLO 모델 로드 실패: {e}")
        return

    print(f" -> 🔍 이미지에서 요소 탐색 중...")
    results = yolo_model(test_image_path, conf=0.25, verbose=False)
    
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_name = yolo_model.names[int(box.cls[0])]
            detections.append({"bbox": [x1, y1, x2, y2], "confidence": conf, "class_name": class_name})

    if not detections:
        print(" -> ⚠️ 다크패턴이 탐지되지 않았습니다. 다른 이미지로 재시도해주세요.")
        return

    print(f" -> ✅ 총 {len(detections)}개의 요소 위치 좌표(Bounding Box) 반환 완료")

    # ====================================================================
    # 역할 2: EfficientNet 위험도 분류 모델 (위험도 수치 분석)
    # ====================================================================
    print("\n[역할 2 검증] EfficientNet 분류 모델 작동 (목적: 요소별 위험도 분석)")
    print(" -> 찾아낸 요소 이미지를 각각 잘라내어(Crop) 분류 모델에 전달합니다.")

    risk_colors = {"소": "green", "중": "orange", "대": "red"}

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        
        # 1. 자르기
        crop_width, crop_height = max(1, x2 - x1), max(1, y2 - y1)
        cropped_img = original_image.crop((x1, y1, x2, y2))
        
        temp_crop_path = f"temp_crop.jpg"
        cropped_img.save(temp_crop_path)
        
        # 2. 위험도 분석
        try:
            risk_result = predict_dark_pattern_risk(temp_crop_path)
            risk_level = risk_result['risk_level']
            risk_score = risk_result['probabilities'][risk_level] * 100
            
            print(f"   - 요소 {i+1} [{crop_width}x{crop_height}px] 분석 완료 -> 결과: 위험도 '{risk_level}' ({risk_score:.1f}%)")
            
            # 3. 브라우저/앱 화면에 데이터 시각화 시뮬레이션
            # 박스 그리기
            color = risk_colors.get(risk_level, "blue")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 라벨 텍스트 배경 및 텍스트 쓰기
            label_text = f"{det['class_name']} | 위험:{risk_level}({risk_score:.0f}%)"
            
            # Pillow 최신/구버전 호환용 text 크기 구하기
            if hasattr(draw, 'textbbox'):
                bbox = draw.textbbox((x1, y1), label_text, font=font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            else:
                text_w, text_h = draw.textsize(label_text, font=font)
                
            draw.rectangle([x1, y1-text_h-4, x1+text_w, y1], fill=color)
            draw.text((x1, y1-text_h-4), label_text, fill="white", font=font)

        except Exception as e:
             print(f"   - ❌ 요소 {i+1} 위험도 분석 실패: {e}")

        # 임시 파일 삭제
        if os.path.exists(temp_crop_path):
            os.remove(temp_crop_path)

    # 결과물 저장
    output_path = "test_result.png"
    draw_image.save(output_path)
    print("\n" + "="*50)
    print(f" ✅ 검증 완료!")
    print(f" ✅ 시각화 결과가 저장되었습니다: handover/backend/{output_path}")
    print("="*50)

if __name__ == "__main__":
    create_visual_test()
