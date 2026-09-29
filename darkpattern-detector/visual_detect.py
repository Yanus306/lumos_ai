"""
Dark Pattern Visual Detector (Selenium Chrome Browser)

Opens a real Chrome browser, loads the website, analyzes texts with
the trained RoBERTa AI model, and visually highlights detected dark patterns
with glowing red borders and informative floating badges on the live page!

Usage:
  python visual_detect.py https://www.shopmissa.com
  python visual_detect.py https://colourpop.com
  python visual_detect.py https://www.aliexpress.com
"""

import sys
import time
import json
import argparse
from pathlib import Path

import torch
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Import our trained detector
sys.path.append(str(Path(r"D:\dark\darkpattern-detector")))
from detect import DarkPatternDetector, is_noise, CONFIDENCE_THRESHOLD


def run_visual_detection(url: str, threshold: float = 0.75, device: str = None):
    print("=" * 65)
    print("  👀 다크패턴 실시간 시각화 브라우저 실행")
    print(f"  🔗 대상 URL: {url}")
    print(f"  🎯 탐지 임계값: {threshold}")
    print("=" * 65)

    # 1. AI 모델 로드
    print("\n[1/3] AI 모델 로딩 중...")
    detector = DarkPatternDetector(device=device)
    # Update threshold
    import detect
    detect.CONFIDENCE_THRESHOLD = threshold

    # 2. 크롬 브라우저 실행
    print("\n[2/3] 크롬 브라우저 실행 및 웹사이트 접속 중...")
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    # Realistic User-Agent
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(url)
        print("  ⏳ 웹페이지 로딩 대기 중 (4초)...")
        time.sleep(4)

        # 3. 브라우저 내 텍스트 엘리먼트 추출 스크립트 실행
        print("\n[3/3] 웹페이지 텍스트 분석 및 다크패턴 탐지 중...")
        
        # JS to collect leaf or meaningful text elements and tag them with data-lumos-id
        collect_script = """
        const candidates = [];
        const seenTexts = new Set();
        let idCounter = 0;

        // Tags to inspect
        const tags = ['button', 'a', 'span', 'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'label', 'strong', 'em', 'b', 'figcaption'];
        
        for (const tag of tags) {
            const elements = document.getElementsByTagName(tag);
            for (const el of elements) {
                // Check if element is visible
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
                if (rect.width === 0 && rect.height === 0) continue;

                // Check text length and direct content
                let text = el.innerText ? el.innerText.trim() : '';
                if (!text || text.length < 4 || text.length > 200) continue;

                // Avoid duplicate texts on the same level
                const textKey = text.toLowerCase();
                if (seenTexts.has(textKey)) continue;

                // Don't grab big container divs that have multiple long paragraphs
                if (el.children.length > 5) continue;

                seenTexts.add(textKey);
                const elemId = 'lumos-target-' + (idCounter++);
                el.setAttribute('data-lumos-id', elemId);

                candidates.push({
                    id: elemId,
                    text: text
                });
            }
        }
        return candidates;
        """

        candidates = driver.execute_script(collect_script)
        print(f"  🔍 분석 대상 화면 텍스트: {len(candidates)}개 발견")

        # Python AI 모델로 다크패턴 분석
        dark_pattern_hits = []
        batch_size = 32

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            texts = [c["text"] for c in batch]
            
            # Predict
            results = detector.predict_batch(texts)

            for cand, res in zip(batch, results):
                if res["is_dark_pattern"] and not is_noise(res["text"]):
                    dark_pattern_hits.append({
                        "id": cand["id"],
                        "text": res["text"][:100],
                        "category": res["category"],
                        "category_kr": res["description_kr"],
                        "probability": round(res["dark_pattern_probability"] * 100, 1)
                    })

        print(f"\n{'='*65}")
        print(f"  🚨 총 {len(dark_pattern_hits)}개의 다크패턴이 탐지되었습니다!")
        print(f"{'='*65}")

        for idx, hit in enumerate(dark_pattern_hits, 1):
            print(f"  {idx}. [{hit['category']}] ({hit['probability']}%) - \"{hit['text']}\"")
            print(f"     -> {hit['category_kr']}")

        # 4. 브라우저 화면에 빨간색 강조 테두리 및 뱃지 주입 (Highlight Injection)
        inject_highlight_script = """
        const hits = arguments[0];

        // 1. Inject Stylesheet
        const style = document.createElement('style');
        style.innerHTML = `
            .lumos-dark-box {
                outline: 3px solid #ff0055 !important;
                box-shadow: 0 0 20px rgba(255, 0, 85, 0.8) !important;
                position: relative !important;
                transition: all 0.3s ease !important;
            }
            .lumos-badge {
                position: absolute !important;
                top: -12px !important;
                left: 0px !important;
                background: linear-gradient(135deg, #ff0055, #ff5500) !important;
                color: #ffffff !important;
                font-size: 11px !important;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
                font-weight: 800 !important;
                padding: 3px 8px !important;
                border-radius: 12px !important;
                box-shadow: 0 4px 10px rgba(0,0,0,0.4) !important;
                z-index: 9999999 !important;
                pointer-events: none !important;
                white-space: nowrap !important;
                letter-spacing: 0.5px !important;
            }
            #lumos-dashboard {
                position: fixed !important;
                bottom: 20px !important;
                right: 20px !important;
                background: rgba(18, 18, 24, 0.95) !important;
                color: #ffffff !important;
                border: 2px solid #ff0055 !important;
                box-shadow: 0 10px 30px rgba(0,0,0,0.6), 0 0 15px rgba(255,0,85,0.4) !important;
                border-radius: 16px !important;
                padding: 16px 20px !important;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
                z-index: 99999999 !important;
                max-width: 340px !important;
                backdrop-filter: blur(10px) !important;
            }
            .lumos-btn {
                background: #ff0055 !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 6px 12px !important;
                font-size: 12px !important;
                font-weight: bold !important;
                cursor: pointer !important;
                margin-top: 8px !important;
                width: 100% !important;
            }
            .lumos-btn:hover {
                background: #ff3377 !important;
            }
        `;
        document.head.appendChild(style);

        // 2. Highlight Elements
        hits.forEach((hit, idx) => {
            const el = document.querySelector('[data-lumos-id="' + hit.id + '"]');
            if (el) {
                el.classList.add('lumos-dark-box');

                // Create floating badge
                const badge = document.createElement('span');
                badge.className = 'lumos-badge';
                badge.innerText = '🚨 ' + hit.category + ' (' + hit.probability + '%)';
                
                // Position relative parent if static
                const computed = window.getComputedStyle(el);
                if (computed.position === 'static') {
                    el.style.position = 'relative';
                }
                el.appendChild(badge);
            }
        });

        // 3. Inject Floating Dashboard
        const dash = document.createElement('div');
        dash.id = 'lumos-dashboard';
        dash.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-weight:bold; font-size:15px; color:#ff3377;">🛡️ Lumos AI 탐지기</span>
                <span style="background:#ff0055; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:bold;">LIVE</span>
            </div>
            <div style="font-size:13px; color:#e0e0e0; margin-bottom:6px;">
                다크패턴 총 <b style="color:#ff5555; font-size:16px;">` + hits.length + `</b>개 발견됨
            </div>
            <div style="font-size:11px; color:#888; line-height:1.4;">
                화면에서 <b style="color:#ff0055;">빨간색 테두리</b>와 <b style="color:#ff5500;">🚨 라벨</b>이 붙은 요소를 확인하세요!
            </div>
            <button class="lumos-btn" id="lumos-next-btn">첫 번째 다크패턴으로 이동 ▶</button>
        `;
        document.body.appendChild(dash);

        // Add scroll to next dark pattern interaction
        let currentIndex = 0;
        document.getElementById('lumos-next-btn').addEventListener('click', () => {
            if (hits.length === 0) return;
            const targetHit = hits[currentIndex % hits.length];
            const targetEl = document.querySelector('[data-lumos-id="' + targetHit.id + '"]');
            if (targetEl) {
                targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
                // Flash animation
                targetEl.style.transform = 'scale(1.05)';
                setTimeout(() => { targetEl.style.transform = 'scale(1)'; }, 400);
            }
            currentIndex++;
            document.getElementById('lumos-next-btn').innerText = 
                '다음 다크패턴 (' + ((currentIndex % hits.length) + 1) + '/' + hits.length + ') 이동 ▶';
        });
        """

        driver.execute_script(inject_highlight_script, dark_pattern_hits)
        print("\n✨ 웹페이지 화면에 빨간색 강조 박스와 대시보드가 주입되었습니다!")
        print("👉 브라우저 오른쪽 아래 [다음 다크패턴 이동 ▶] 버튼을 누르면 해당 위치로 자동 스크롤됩니다.")
        print("\n" + "=" * 65)
        print("  💡 브라우저 창을 직접 보며 자유롭게 확인하세요.")
        print("  종료하려면 이 터미널에서 [Enter] 키를 누르세요.")
        print("=" * 65 + "\n")

        # Keep browser open until user presses Enter in terminal
        input()

    finally:
        print("브라우저를 종료합니다.")
        driver.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lumos AI - 실시간 다크패턴 시각화 브라우저")
    parser.add_argument("url", nargs="?", default="https://www.shopmissa.com", help="분석할 웹사이트 URL")
    parser.add_argument("--threshold", "-t", type=float, default=0.75, help="다크패턴 판정 임계값 (0.0~1.0)")
    parser.add_argument("--device", "-d", default=None, help="실행 디바이스 (cuda 또는 cpu)")

    args = parser.parse_args()
    run_visual_detection(args.url, threshold=args.threshold, device=args.device)
