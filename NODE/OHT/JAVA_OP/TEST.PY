#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
화면 캡처 → OCR로 Java 코드 추출
- 화면에 보이는 코드를 캡처해서 텍스트로 변환
- 복사 불가능한 환경에서 사용

설치:
    pip install pillow pytesseract pyautogui

Tesseract 설치 필요:
    Windows: https://github.com/UB-Mannheim/tesseract/wiki
    Mac: brew install tesseract
    Linux: sudo apt install tesseract-ocr
"""

import os
import sys
from datetime import datetime

try:
    import pyautogui
    from PIL import Image
    import pytesseract
except ImportError:
    print("필요한 라이브러리 설치:")
    print("  pip install pillow pytesseract pyautogui")
    sys.exit(1)

# Windows의 경우 Tesseract 경로 설정
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

JAVA_FILES = [
    "CnvEdge.java",
    "DataService.java",
    "DataSet.java",
    "DijkstraVhlRouteFind.java",
    "LongEdge.java",
    "OhtMsgWorkerRunnable.java",
    "OhtUdpListener.java",
    "RailEdge.java",
    "Station.java",
    "StkRmEdge.java",
    "TrafficBatch.java",
    "TransferEdge.java",
    "Vhl.java",
]

OUTPUT_FILE = "oht_java_sources.py"


def capture_screen():
    """전체 화면 캡처"""
    screenshot = pyautogui.screenshot()
    return screenshot


def capture_region():
    """영역 지정 캡처"""
    print("\n  마우스로 영역 지정:")
    print("  - 좌상단 모서리로 이동 후 Enter")
    input()
    x1, y1 = pyautogui.position()
    print(f"  좌상단: ({x1}, {y1})")
    
    print("  - 우하단 모서리로 이동 후 Enter")
    input()
    x2, y2 = pyautogui.position()
    print(f"  우하단: ({x2}, {y2})")
    
    # 캡처
    width = x2 - x1
    height = y2 - y1
    screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
    return screenshot


def ocr_image(image):
    """이미지에서 텍스트 추출"""
    # OCR 설정 (코드 인식에 최적화)
    config = '--psm 6 -c preserve_interword_spaces=1'
    text = pytesseract.image_to_string(image, config=config)
    return text


def save_sources(sources):
    """소스 파일로 저장"""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("# " + "=" * 68 + "\n")
        f.write(f"# OHT Java 소스 코드 모음 (OCR 추출)\n")
        f.write(f"# 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# " + "=" * 68 + "\n\n")
        
        for file_name, source in sources.items():
            f.write("\n# " + "=" * 68 + "\n")
            f.write(f"# {file_name}\n")
            f.write("# " + "=" * 68 + "\n")
            f.write('"""\n')
            if source.strip():
                f.write(source)
                if not source.endswith('\n'):
                    f.write('\n')
            else:
                f.write("(캡처 없음)\n")
            f.write('"""\n')
    
    print(f"\n  저장 완료: {OUTPUT_FILE}")


def main():
    print("=" * 70)
    print("  화면 캡처 → OCR Java 코드 추출")
    print("=" * 70)
    print("\n  사용법:")
    print("  1. Java 코드를 화면에 띄워놓기")
    print("  2. 캡처할 영역 지정")
    print("  3. OCR로 텍스트 추출")
    print("  4. 파일로 저장 → 복사 가능!")
    print("\n  Enter 누르면 시작...")
    input()
    
    sources = {}
    
    for i, file_name in enumerate(JAVA_FILES, 1):
        print("\n" + "=" * 70)
        print(f"  [{i}/{len(JAVA_FILES)}] {file_name}")
        print("=" * 70)
        print("\n  해당 파일을 화면에 띄워주세요.")
        print("  [1] 영역 캡처  [2] 전체 화면  [s] 건너뛰기  [q] 종료")
        
        choice = input("  선택: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 's':
            sources[file_name] = ""
            print("  → 건너뜀")
            continue
        elif choice == '2':
            print("\n  3초 후 전체 화면 캡처...")
            import time
            time.sleep(3)
            image = capture_screen()
        else:
            image = capture_region()
        
        # OCR 실행
        print("\n  OCR 처리 중...")
        text = ocr_image(image)
        
        sources[file_name] = text
        line_count = len(text.split('\n'))
        print(f"  → {line_count}줄 추출됨")
        
        # 미리보기
        print("\n  [미리보기 (처음 10줄)]")
        print("-" * 50)
        for line in text.split('\n')[:10]:
            print(f"  {line}")
        print("-" * 50)
        
        # 중간 저장
        save_sources(sources)
    
    # 완료
    print("\n" + "=" * 70)
    print("  완료!")
    print("=" * 70)
    for file_name, source in sources.items():
        if source.strip():
            line_count = len(source.split('\n'))
            print(f"  ✓ {file_name}: {line_count}줄")
        else:
            print(f"  - {file_name}: (건너뜀)")
    print("=" * 70)
    print(f"\n  저장 파일: {OUTPUT_FILE}")
    print("  이 파일에서 복사 가능!")


if __name__ == "__main__":
    main()