#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hynix LLM API 모델 정보 확인
"""

import requests
import json

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
        
        # 에러 체크
        if isinstance(result, dict) and "error" in result:
            print(f"  ❌ 오류: {result['error']}")
            continue
        
        # 리스트로 바로 오는 경우
        if isinstance(result, list):
            models = result
        # {"data": [...]} 형식인 경우
        elif isinstance(result, dict):
            models = result.get("data", [result])
        else:
            print(f"  알 수 없는 형식: {type(result)}")
            continue
        
        # 모델 출력
        for m in models:
            if isinstance(m, dict):
                model_id = m.get("id", m.get("model", "unknown"))
                owned_by = m.get("owned_by", "")
                print(f"  ✅ {model_id}" + (f" ({owned_by})" if owned_by else ""))
            else:
                print(f"  ✅ {m}")
        
        # 전체 응답도 출력 (디버깅용)
        print(f"\n  [RAW] {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")
        print()

if __name__ == "__main__":
    main()