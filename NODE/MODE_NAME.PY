#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hynix LLM API 모델 정보 확인
"""

import requests

# API 엔드포인트들
ENDPOINTS = {
    "dev": "http://dev.assistant.llm.skhynix.com",
    "prod": "http://summary.llm.skhynix.com",
    "main": "http://assistant.llm.skhynix.com",
}

def load_token(path: str = "token.txt") -> str:
    with open(path, "r") as f:
        return f.read().strip()

def get_models(base_url: str, token: str):
    """GET /v1/models 호출"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        resp = requests.get(f"{base_url}/v1/models", headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    token = load_token()
    print(f"토큰 로드 완료: {token[:10]}...\n")
    
    for name, url in ENDPOINTS.items():
        print(f"=== {name.upper()} ({url}) ===")
        result = get_models(url, token)
        
        if "error" in result:
            print(f"  ❌ 오류: {result['error']}")
        else:
            # OpenAI 호환 형식: {"data": [{"id": "model-name", ...}]}
            models = result.get("data", result)
            if isinstance(models, list):
                for m in models:
                    model_id = m.get("id", m) if isinstance(m, dict) else m
                    print(f"  ✅ {model_id}")
            else:
                print(f"  응답: {result}")
        print()

if __name__ == "__main__":
    main()