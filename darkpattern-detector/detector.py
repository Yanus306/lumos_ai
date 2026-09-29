"""
Dark Pattern Detector - Inference Module

Backend integration module for dark pattern detection.
Load this module and use DarkPatternDetector class.

Requirements:
  pip install torch transformers

Example:
  from detector import DarkPatternDetector
  
  detector = DarkPatternDetector(model_dir="./model")
  result = detector.predict("Hurry! Only 2 left in stock")
  print(result)
  # {'text': 'Hurry! Only 2 left in stock', 'is_dark_pattern': True,
  #  'probability': 0.999, 'category': 'Scarcity', ...}
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Category mappings
CATEGORIES = {
    0: "Not Dark Pattern",
    1: "Scarcity",
    2: "Social Proof",
    3: "Urgency",
    4: "Misdirection",
    5: "Obstruction",
    6: "Sneaking",
    7: "Forced Action",
}

CATEGORY_DESCRIPTIONS = {
    "Scarcity": {
        "en": "Fake scarcity / low stock pressure",
        "kr": "허위 희소성 / 재고 부족 압박",
        "examples": ["Hurry! Only 2 left", "Limited Availability", "ONLY 8 LEFT"],
    },
    "Social Proof": {
        "en": "Fake social proof / activity notifications",
        "kr": "허위 사회적 증거 / 활동 알림",
        "examples": ["1,142 people viewed this", "24 sold in last hour"],
    },
    "Urgency": {
        "en": "Fake urgency / countdown timers",
        "kr": "허위 긴급성 / 카운트다운 타이머",
        "examples": ["FLASH SALE | LIMITED TIME ONLY", "Deal ends in 2:00:00"],
    },
    "Misdirection": {
        "en": "Confirmshaming / trick questions / visual interference",
        "kr": "확인수치(Confirmshaming) / 트릭 질문",
        "examples": ["No thanks, I hate saving money", "I don't want to save"],
    },
    "Obstruction": {
        "en": "Hard to cancel / roach motel",
        "kr": "취소 방해 / 로치 모텔",
        "examples": ["Call customer service to cancel"],
    },
    "Sneaking": {
        "en": "Hidden costs / sneak into basket",
        "kr": "숨겨진 비용 / 몰래 장바구니 추가",
        "examples": ["Shipping insurance added", "Protection plan included"],
    },
    "Forced Action": {
        "en": "Forced registration / enrollment",
        "kr": "강제 가입 / 등록 강요",
        "examples": ["Create an account to continue"],
    },
}


class DarkPatternDetector:
    """
    Dark Pattern detection model using fine-tuned RoBERTa-base.
    
    Two models:
      - Binary: Dark Pattern vs Normal (Accuracy: 96.5%, F1: 96.5%, AUC: 98.4%)
      - Multiclass: 7 dark pattern types + Normal (Accuracy: 95.8%, F1: 95.0%)
    
    Args:
        model_dir: Path to the directory containing 'binary/' and 'multiclass/' subdirs
        device: 'cuda', 'cpu', or None (auto-detect)
        confidence_threshold: Minimum probability to classify as dark pattern (default: 0.75)
    """

    def __init__(
        self,
        model_dir: str = "./model",
        device: Optional[str] = None,
        confidence_threshold: float = 0.75,
    ):
        self.model_dir = Path(model_dir)
        self.confidence_threshold = confidence_threshold

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        binary_dir = self.model_dir / "binary"
        multi_dir = self.model_dir / "multiclass"

        if not binary_dir.exists():
            raise FileNotFoundError(f"Binary model not found: {binary_dir}")
        if not multi_dir.exists():
            raise FileNotFoundError(f"Multiclass model not found: {multi_dir}")

        # Load binary model
        self.binary_tokenizer = AutoTokenizer.from_pretrained(str(binary_dir))
        self.binary_model = AutoModelForSequenceClassification.from_pretrained(
            str(binary_dir)
        ).to(self.device)
        self.binary_model.eval()

        # Load multiclass model
        self.multi_tokenizer = AutoTokenizer.from_pretrained(str(multi_dir))
        self.multi_model = AutoModelForSequenceClassification.from_pretrained(
            str(multi_dir)
        ).to(self.device)
        self.multi_model.eval()

    def predict(self, text: str) -> Dict:
        """
        Predict whether a single text is a dark pattern.
        
        Args:
            text: Input text string
            
        Returns:
            dict with keys:
                - text (str): Original input
                - is_dark_pattern (bool): Whether it's a dark pattern
                - probability (float): Confidence score 0.0~1.0
                - category (str): Dark pattern category name
                - category_id (int): Category integer ID
                - description_en (str): English description
                - description_kr (str): Korean description
        """
        # Binary classification
        inputs = self.binary_tokenizer(
            text, truncation=True, padding="max_length",
            max_length=64, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.binary_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            dark_prob = probs[0][1].item()

        # Multiclass classification (only if binary positive)
        category = "Not Dark Pattern"
        category_id = 0

        if pred == 1 and dark_prob >= self.confidence_threshold:
            multi_inputs = self.multi_tokenizer(
                text, truncation=True, padding="max_length",
                max_length=64, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                multi_outputs = self.multi_model(**multi_inputs)
                multi_probs = torch.softmax(multi_outputs.logits, dim=-1)
                multi_pred = torch.argmax(multi_probs, dim=-1).item()

            category = CATEGORIES.get(multi_pred, "Unknown")
            category_id = multi_pred

            if category == "Not Dark Pattern":
                pred = 0
                dark_prob = 1.0 - dark_prob

        is_dark = pred == 1 and dark_prob >= self.confidence_threshold
        cat_info = CATEGORY_DESCRIPTIONS.get(category, {})

        return {
            "text": text,
            "is_dark_pattern": is_dark,
            "probability": round(dark_prob, 4),
            "category": category if is_dark else "Not Dark Pattern",
            "category_id": category_id if is_dark else 0,
            "description_en": cat_info.get("en", "") if is_dark else "",
            "description_kr": cat_info.get("kr", "") if is_dark else "",
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict multiple texts at once.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of prediction dicts (same format as predict())
        """
        return [self.predict(t) for t in texts]

    def get_categories(self) -> Dict:
        """Return all category definitions."""
        return CATEGORY_DESCRIPTIONS

    def get_model_info(self) -> Dict:
        """Return model metadata."""
        return {
            "base_model": "roberta-base",
            "max_length": 64,
            "binary_accuracy": 0.965,
            "binary_f1": 0.965,
            "binary_auc": 0.984,
            "multiclass_accuracy": 0.958,
            "multiclass_f1": 0.950,
            "categories": list(CATEGORIES.values()),
            "confidence_threshold": self.confidence_threshold,
            "device": str(self.device),
        }


# ── Quick test when run directly ──
if __name__ == "__main__":
    print("Loading Dark Pattern Detector...")
    detector = DarkPatternDetector(model_dir="./model")
    
    test_texts = [
        "Hurry! Only 2 left in stock",
        "Add to Cart",
        "1,142 people viewed this recently",
        "No thanks, I hate saving money",
        "Free Shipping on orders over $50",
        "LIMITED TIME OFFER - 50% OFF",
        "Only 3 left - order soon",
    ]

    print(f"\nModel Info: {detector.get_model_info()}\n")

    for text in test_texts:
        result = detector.predict(text)
        if result["is_dark_pattern"]:
            print(f"  [!] {result['category']} ({result['probability']:.1%}) -> \"{text}\"")
        else:
            print(f"  [OK] Normal ({1-result['probability']:.1%}) -> \"{text}\"")
