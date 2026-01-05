#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SK Hynix LLM API 모델 목록 확인
"""

import requests
import os

# 토큰 로드
TOKEN_PATH = "token.txt"
API_BASE = "http://dev.assistant.llm.skhynix.com"

def load_token():
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "r") as f:
            return f.read().strip()
    return None

def check_models():
    token = load_token()
    if not token:
        print("❌ token.txt 파일이 없습니다.")
        return
    
    print(f"✅ 토큰 로드 완료: {token[:20]}...")
    print(f"🔗 API: {API_BASE}")
    print("=" * 60)
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # 1. /v1/models 엔드포인트 시도
    endpoints = [
        "/v1/models",
        "/models",
        "/v1/model/list",
        "/api/models"
    ]
    
    for endpoint in endpoints:
        url = f"{API_BASE}{endpoint}"
        print(f"\n🔍 시도: {url}")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"   상태: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ 성공!")
                print("-" * 60)
                
                # OpenAI 형식
                if "data" in data:
                    print("📋 사용 가능한 모델:")
                    for model in data["data"]:
                        model_id = model.get("id", model.get("name", "unknown"))
                        print(f"   • {model_id}")
                # 리스트 형식
                elif isinstance(data, list):
                    print("📋 사용 가능한 모델:")
                    for model in data:
                        if isinstance(model, str):
                            print(f"   • {model}")
                        elif isinstance(model, dict):
                            print(f"   • {model.get('id', model.get('name', model))}")
                # 기타
                else:
                    print("📋 응답 데이터:")
                    import json
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                
                return
            else:
                print(f"   응답: {response.text[:200]}")
                
        except Exception as e:
            print(f"   ❌ 오류: {e}")
    
    # 2. 현재 설정된 모델로 테스트
    print("\n" + "=" * 60)
    print("🧪 현재 모델 테스트 (Qwen3-Coder-30B-A3B-Instruct)")
    
    test_url = f"{API_BASE}/v1/chat/completions"
    test_data = {
        "model": "Qwen3-Coder-30B-A3B-Instruct",
        "messages": [{"role": "user", "content": "안녕"}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(test_url, headers=headers, json=test_data, timeout=30)
        print(f"   상태: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ 모델 작동 확인!")
            result = response.json()
            if "choices" in result:
                print(f"   응답: {result['choices'][0]['message']['content'][:50]}...")
        else:
            print(f"   응답: {response.text[:300]}")
            
            # 오류 메시지에서 모델 목록 힌트 찾기
            if "model" in response.text.lower():
                print("\n💡 오류 메시지에서 모델 정보 확인:")
                print(response.text)
                
    except Exception as e:
        print(f"   ❌ 오류: {e}")

if __name__ == "__main__":
    check_models()