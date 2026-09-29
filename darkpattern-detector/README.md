# Dark Pattern Detector - Backend Integration Guide

## Overview

E-commerce 텍스트에서 다크패턴을 탐지하는 AI 모델입니다.
RoBERTa-base를 파인튜닝하여 7가지 다크패턴 유형을 분류합니다.

## Model Performance

| Task | Accuracy | F1 Score | AUC |
| :--- | :---: | :---: | :---: |
| Binary (Dark Pattern vs Normal) | **96.5%** | **96.5%** | **98.4%** |
| Multi-class (7 types + Normal) | **95.8%** | **95.0%** | - |

## Directory Structure

```
darkpattern-detector/
├── model/
│   ├── binary/                    # 이진 분류 모델 (다크패턴 O/X)
│   │   ├── model.safetensors      # 모델 가중치 (~499MB)
│   │   ├── config.json            # 모델 설정
│   │   ├── tokenizer.json         # 토크나이저
│   │   ├── vocab.json             # 어휘 사전
│   │   ├── merges.txt             # BPE 병합 규칙
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── test_metrics.json      # 테스트 성능 지표
│   └── multiclass/                # 다중 분류 모델 (7가지 유형)
│       └── (동일 구조)
├── detector.py                    # 추론 모듈 (이 파일만 import)
├── requirements.txt               # 의존성 (torch, transformers)
└── README.md                      # 이 문서
```

## Setup

```bash
pip install -r requirements.txt
```

> **GPU 권장**: CUDA 지원 GPU가 있으면 추론 속도가 ~10배 빠릅니다.
> CPU에서도 동작하지만 첫 로딩에 ~10초, 추론에 텍스트당 ~100ms 소요됩니다.

## Quick Start

```python
from detector import DarkPatternDetector

# 모델 로드 (첫 호출 시 ~3초 소요)
detector = DarkPatternDetector(model_dir="./model")

# 단일 텍스트 판별
result = detector.predict("Hurry! Only 2 left in stock")
print(result)
```

**Output:**
```json
{
  "text": "Hurry! Only 2 left in stock",
  "is_dark_pattern": true,
  "probability": 0.999,
  "category": "Scarcity",
  "category_id": 1,
  "description_en": "Fake scarcity / low stock pressure",
  "description_kr": "허위 희소성 / 재고 부족 압박"
}
```

## API Reference

### `DarkPatternDetector(model_dir, device, confidence_threshold)`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_dir` | str | `"./model"` | 모델 디렉토리 경로 |
| `device` | str/None | `None` | `"cuda"`, `"cpu"`, 또는 `None` (자동 감지) |
| `confidence_threshold` | float | `0.75` | 다크패턴 판정 최소 확률 (0.0~1.0) |

### `detector.predict(text: str) -> dict`

단일 텍스트를 분석합니다.

**Returns:**

| Key | Type | Description |
| :--- | :--- | :--- |
| `text` | str | 입력 텍스트 |
| `is_dark_pattern` | bool | 다크패턴 여부 |
| `probability` | float | 다크패턴 확률 (0.0~1.0) |
| `category` | str | 카테고리명 (영문) |
| `category_id` | int | 카테고리 ID (0~7) |
| `description_en` | str | 영문 설명 |
| `description_kr` | str | 한국어 설명 |

### `detector.predict_batch(texts: List[str]) -> List[dict]`

여러 텍스트를 한 번에 분석합니다.

```python
results = detector.predict_batch([
    "Only 2 left in stock",
    "Add to Cart",
    "1,142 people viewed this",
])
dark_patterns = [r for r in results if r["is_dark_pattern"]]
```

### `detector.get_model_info() -> dict`

모델 메타정보를 반환합니다.

### `detector.get_categories() -> dict`

전체 카테고리 정의(설명, 예시 포함)를 반환합니다.

## Category Reference

| ID | Category | 한국어 | Example |
| :---: | :--- | :--- | :--- |
| 0 | Not Dark Pattern | 정상 | "Add to Cart" |
| 1 | Scarcity | 허위 희소성 | "Only 2 left in stock" |
| 2 | Social Proof | 허위 사회적 증거 | "1,142 people viewed this" |
| 3 | Urgency | 허위 긴급성 | "FLASH SALE - LIMITED TIME" |
| 4 | Misdirection | 확인수치/트릭 질문 | "No thanks, I hate saving money" |
| 5 | Obstruction | 취소 방해 | "Call to cancel subscription" |
| 6 | Sneaking | 숨겨진 비용 | "Shipping insurance added" |
| 7 | Forced Action | 강제 가입 | "Create account to continue" |

## Backend Integration Example (FastAPI)

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from detector import DarkPatternDetector

app = FastAPI()
detector = DarkPatternDetector(model_dir="./model")

class TextRequest(BaseModel):
    texts: List[str]

@app.post("/detect")
async def detect_dark_patterns(request: TextRequest):
    results = detector.predict_batch(request.texts)
    return {
        "total": len(results),
        "dark_patterns": [r for r in results if r["is_dark_pattern"]],
        "all_results": results,
    }
```

## Backend Integration Example (Flask)

```python
from flask import Flask, request, jsonify
from detector import DarkPatternDetector

app = Flask(__name__)
detector = DarkPatternDetector(model_dir="./model")

@app.route("/detect", methods=["POST"])
def detect():
    texts = request.json.get("texts", [])
    results = detector.predict_batch(texts)
    return jsonify({
        "total": len(results),
        "dark_patterns": [r for r in results if r["is_dark_pattern"]],
        "all_results": results,
    })
```

## Notes

- 모델은 **영문 텍스트**에 최적화되어 있습니다 (학습 데이터가 영문 이커머스 기반)
- `confidence_threshold`를 높이면 정밀도(Precision) 증가, 낮추면 재현율(Recall) 증가
- GPU VRAM 최소 2GB 필요 (각 모델 ~500MB × 2)
- CPU 환경에서도 동작하지만 응답 시간이 느려집니다
