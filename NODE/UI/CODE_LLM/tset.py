#!/usr/bin/env python3
"""HCP 서버 사용 가능한 모델 목록 조회"""

import requests
import json
import os

# 토큰 로드
token_paths = ["token.txt", "../token.txt", os.path.expanduser("~/token.txt")]
API_TOKEN = None

for path in token_paths:
    if os.path.exists(path):
        with open(path, "r") as f:
            API_TOKEN = f.read().strip()
        print(f"✅ 토큰 로드: {path}")
        break

if not API_TOKEN:
    print("❌ token.txt 파일을 찾을 수 없습니다!")
    exit(1)

# HCP 서버 모델 목록 조회
url = "http://hcp.llm.skhynix.com/v1/models"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

print(f"\n🔍 모델 목록 조회 중... ({url})")
print("-" * 50)

try:
    response = requests.get(url, headers=headers, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"✅ 성공! (Status: {response.status_code})\n")
        
        # 모델 목록 출력
        if "data" in data:
            models = data["data"]
            print(f"📋 사용 가능한 모델 ({len(models)}개):\n")
            for i, model in enumerate(models, 1):
                model_id = model.get("id", "unknown")
                print(f"  {i}. {model_id}")
        else:
            print("응답 데이터:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(f"❌ 오류: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ 요청 실패: {e}")