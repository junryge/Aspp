"""
웹페이지 소스 추출기
"""

import requests
from datetime import datetime
import os
import re

# ============================================
# 여기에 URL 입력
# ============================================
URL = "https://www.naver.com"
# ============================================

def fetch_page(url, timeout=10):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return response.text
    except requests.RequestException as e:
        print(f"[ERROR] {e}")
        return None

def save_html(content, url, output_dir="output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    clean_name = re.sub(r'[^\w\-]', '_', url.replace('https://', '').replace('http://', ''))[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{clean_name}_{timestamp}.html"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filename

if __name__ == "__main__":
    print(f"[INFO] URL: {URL}")
    
    html = fetch_page(URL)
    
    if html:
        saved_path = save_html(html, URL)
        print(f"[OK] 저장완료: {saved_path}")
        print(f"[INFO] 길이: {len(html):,} bytes")
        print("\n미리보기:\n" + "="*50)
        print(html[:2000])