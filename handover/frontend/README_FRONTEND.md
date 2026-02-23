# 프론트엔드 개발자를 위한 다크패턴 탐지 명세 가이드

본 문서는 만들어질 백엔드 API와 프론트엔드(웹 프론트엔드 또는 크롬 익스텐션)가 주고받을 데이터 포맷(명세)의 기준안을 제시합니다.

---

## 🧩 [핵심] 크롬 익스텐션 개발 플로우 (Real-time Detection)

실시간 다크패턴 탐지 및 차단 익스텐션을 개발할 때의 권장 워크플로우입니다.

**1. 화면 캡처 (Chrome Extension - Background/Content Script)**
*   사용자가 웹페이지에 접속하거나 스크롤을 멈췄을 때, `chrome.tabs.captureVisibleTab` API 등을 활용하여 현재 뷰포트(화면)를 이미지(Base64 또는 Blob)로 캡처합니다.

**2. API 전송 (Chrome Extension -> Backend)**
*   캡처된 이미지를 백엔드 AI 서버(API)로 POST 전송합니다.

**3. AI 분석 및 응답 (Backend -> Chrome Extension)**
*   백엔드 서버는 AI 모델을 돌려 화면 내 다크패턴의 좌표(`bbox`)와 위험도(`risk`)를 계산하여 JSON 형태로 응답합니다. (아래 📝 **API 데이터 명세** 참고)

**4. 화면 차단 및 렌더링 (Chrome Extension - Content Script)**
*   응답받은 좌표 데이터를 바탕으로 현재 사용자 웹페이지의 DOM 요소 위에 시각적 오버레이를 씌웁니다.
*   **탐지(Highlight)**: 의심되는 요소에 빨간색 점선 테두리(`border`)를 주입합니다.
*   **차단(Block)**: 위험도(대) 요소일 경우, 해당 좌표와 일치하는 DOM 요소 위에 `z-index`가 아주 높은 반투명 `div` 박스를 덮어씌우고, `pointer-events: none` 이나 `e.preventDefault()`를 통해 사용자의 클릭을 원천 차단합니다.

---

## 📝 API 데이터 명세 (제안)

백엔드 서버와 이런 형식으로 데이터를 주고받기로 합의하시면 개발이 수월합니다.

### 📤 1. Request (익스텐션 -> 백엔드)

*   **Endpoint**: `POST /api/v1/detect-dark-patterns`
*   **Content-Type**: `multipart/form-data` 또는 `application/json` (Base64 전송 시)
*   **Body**:
    *   `image`: `<File>` (캡처된 화면 이미지)

### 📥 2. Response (백엔드 -> 익스텐션)

*   **Status Code**: `200 OK`
*   **Content-Type**: `application/json`

```json
{
  "status": "success",
  "data": {
    "image_id": "req-12345",
    "detections": [
      {
        "bbox": [100.2, 50.5, 300.0, 150.0],  // [x1, y1, x2, y2] (좌상단, 우하단 픽셀 좌표)
        "pattern_type": "dark_pattern_1",     // 탐지된 다크패턴 유형 이름
        "yolo_confidence": 0.88,              // 모델이 다크패턴이라고 확신하는 정도 (0~1)
        "risk": {
          "level": "대",                      // 위험도 등급 ("소", "중", "대")
          "score": 0.95                       // 다크패턴이 해당 등급일 확률 (0~1)
        }
      },
      {
        "bbox": [400.0, 600.0, 500.0, 700.0],
        "pattern_type": "dark_pattern_3",
        "yolo_confidence": 0.65,
        "risk": {
          "level": "중",
          "score": 0.55
        }
      }
    ]
  }
}
```

---

## 🎨 UI 렌더링 꿀팁 

크롬 익스텐션 Content Script에서 결과물(네모 박스)을 그릴 때 참고하세요.

*   **절대 좌표 계산**:
    *   서버에서 받은 `bbox` `[x1, y1, x2, y2]`는 캡처했던 이미지 크기 기준입니다. 
    *   박스의 크기를 `width = x2 - x1`, `height = y2 - y1` 로 잡고, 화면 내 위치는 `position: absolute; top: y1px; left: x1px;` 로 설정하세요.
*   **위험도별 색상 테마**:
    *   🟢 `소 (Low)` -> `#34C759` (초록 테두리 살짝)
    *   🟡 `중 (Medium)` -> `#FF9500` (주황색 테두리 및 ⚠️ 경고 아이콘 추가)
    *   🔴 `대 (High)` -> `#FF3B30` (빨간색 반투명 레이어로 덮고 클릭 차단)
*   **반응형 대응**:
    *   사용자가 브라우저 창 크기를 줄이거나 늘리면 좌표가 틀어질 수 있습니다. 창 크기(`resize` 이벤트)나 스크롤(`scroll` 이벤트)이 발생할 때마다 박스 위치를 DOM 요소에 맞춰 재계산해주거나, 기존 박스를 지우고 다시 캡처-분석 사이클을 돌리는 것이 안전합니다.
