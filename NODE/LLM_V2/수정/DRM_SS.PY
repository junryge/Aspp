"""
SK하이닉스 DRM Excel Export API 호출 스크립트
- TOKEN.TXT 파일에 API 토큰 기입 필요
- 운영 URL: https://workplace.skhynix.com/api/common/v1/drm/excel/sheetParse
"""

import requests
import json
from flask import Flask, request
import threading
import time
import os

# ============== 설정 ==============
API_URL_SINGLE = "https://workplace.skhynix.com/api/common/v1/drm/excel/parse"
API_URL_MULTI = "https://workplace.skhynix.com/api/common/v1/drm/excel/sheetParse"
CDN_BASE = "https://cdn.skhynix.com"

# 토큰 파일 경로
TOKEN_FILE = "TOKEN.TXT"

# 콜백 서버 설정
CALLBACK_PORT = 5000
CALLBACK_HOST = "0.0.0.0"

result_data = None


# ============== 토큰 로드 ==============
def load_token():
    """TOKEN.TXT에서 토큰 읽기"""
    if not os.path.exists(TOKEN_FILE):
        print(f"❌ {TOKEN_FILE} 파일이 없습니다!")
        print(f"   {TOKEN_FILE} 파일 생성 후 API 토큰을 입력하세요.")
        return None
    
    with open(TOKEN_FILE, 'r', encoding='utf-8') as f:
        token = f.read().strip()
    
    if not token:
        print(f"❌ {TOKEN_FILE} 파일이 비어있습니다!")
        return None
    
    print(f"✅ 토큰 로드 완료 ({len(token)}자)")
    return token


# ============== 콜백 서버 ==============
app = Flask(__name__)

@app.route('/callback', methods=['POST'])
def callback():
    """API 결과 수신"""
    global result_data
    result_data = request.json
    
    print("\n" + "="*60)
    print("✅ DRM 해제 결과 수신!")
    print("="*60)
    print(f"코드: {result_data.get('code')}")
    print(f"상태: {result_data.get('parseStatus')} (1=등록, 2=변환중, 3=성공, 9=실패)")
    
    if result_data.get('parseStatus') == 3:
        headers = result_data.get('headers') or result_data.get('heasers', [])
        datas = result_data.get('datas', [])
        
        print(f"헤더: {headers}")
        print(f"데이터 행수: {len(datas)}")
        
        # JSON 저장
        with open('drm_result.json', 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\n📁 저장: drm_result.json")
        
    elif result_data.get('parseStatus') == 9:
        print(f"❌ 에러: {result_data.get('errorMsg')}")
    
    return {'status': 'received'}


def start_callback_server():
    """콜백 서버 시작"""
    app.run(host=CALLBACK_HOST, port=CALLBACK_PORT, threaded=True, use_reloader=False)


# ============== API 호출 ==============
def parse_drm_excel(
    token: str,
    file_seq: str,
    file_name: str,
    file_ext: str = "xlsx",
    file_size: int = 0,
    header_type: int = 1,
    header_row: int = 1,
    start_row: int = 2,
    callback_url: str = None,
    request_user: str = ""
):
    """단일 시트 파싱"""
    if callback_url is None:
        callback_url = f"http://localhost:{CALLBACK_PORT}/callback"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    payload = {
        "file": {
            "down": f"{CDN_BASE}/down/file/{file_seq}",
            "fileExt": file_ext,
            "fileName": file_name,
            "fileSize": file_size,
            "fileType": 1,
            "link": f"{CDN_BASE}/img/icon/{file_ext}",
            "seq": file_seq,
            "size": file_size
        },
        "header": {
            "headerType": header_type,
            "headerRow": header_row,
            "headerNames": []
        },
        "startRow": start_row,
        "returnType": "json",
        "callbackUrl": callback_url,
        "requestUser": request_user
    }
    
    print(f"\n📤 API 호출: {API_URL_SINGLE}")
    print(f"   파일: {file_name} (seq: {file_seq})")
    
    response = requests.post(API_URL_SINGLE, json=payload, headers=headers)
    print(f"   응답: {response.status_code}")
    
    return response


def parse_drm_excel_multi(
    token: str,
    file_seq: str,
    file_name: str,
    sheets_config: list,
    file_ext: str = "xlsx",
    file_size: int = 0,
    callback_url: str = None,
    request_user: str = ""
):
    """
    다중 시트 파싱
    
    sheets_config 예시:
    [
        {"sheet_index": 0, "header_row": 1, "start_row": 2, "calculate": True},
        {"sheet_index": 1, "header_row": 1, "start_row": 2, "calculate": False},
    ]
    """
    if callback_url is None:
        callback_url = f"http://localhost:{CALLBACK_PORT}/callback"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    sheets = []
    for cfg in sheets_config:
        sheet = {
            "header": {
                "headerType": cfg.get("header_type", 1),
                "headerRow": cfg.get("header_row", 1),
                "headerNames": cfg.get("header_names", []),
                "parseType": cfg.get("parse_type", 1),
                "sheetIndex": cfg.get("sheet_index", 0)
            },
            "calculate": cfg.get("calculate", False),
            "startRow": cfg.get("start_row", 2)
        }
        sheets.append(sheet)
    
    payload = {
        "file": {
            "down": f"{CDN_BASE}/down/file/{file_seq}",
            "fileExt": file_ext,
            "fileName": file_name,
            "fileSize": file_size,
            "fileType": 1,
            "link": f"{CDN_BASE}/img/icon/{file_ext}",
            "seq": file_seq,
            "size": file_size
        },
        "sheets": sheets,
        "returnType": "json",
        "callbackUrl": callback_url,
        "requestUser": request_user
    }
    
    print(f"\n📤 다중시트 API 호출: {API_URL_MULTI}")
    print(f"   파일: {file_name} (seq: {file_seq})")
    print(f"   시트 수: {len(sheets)}")
    
    response = requests.post(API_URL_MULTI, json=payload, headers=headers)
    print(f"   응답: {response.status_code}")
    
    return response


# ============== 메인 ==============
if __name__ == "__main__":
    print("="*60)
    print("SK하이닉스 DRM Excel Export API")
    print("="*60)
    
    # 1. 토큰 로드
    token = load_token()
    if not token:
        exit(1)
    
    # 2. 콜백 서버 시작
    print(f"\n🚀 콜백 서버 시작 (port {CALLBACK_PORT})...")
    server_thread = threading.Thread(target=start_callback_server, daemon=True)
    server_thread.start()
    time.sleep(1)
    
    # 3. 설정값 (수정 필요)
    print("\n" + "-"*60)
    print("⚠️  아래 값들을 실제 값으로 변경하세요")
    print("-"*60)
    
    FILE_SEQ = "YOUR_FILE_SEQ"       # CDN 업로드 후 받은 seq
    FILE_NAME = "your_file.xlsx"     # 파일명
    FILE_SIZE = 0                    # 파일 크기
    REQUEST_USER = "X0000000"        # 사원번호
    
    # 콜백 URL (사내망에서 접근 가능한 주소로 변경)
    CALLBACK_URL = f"http://localhost:{CALLBACK_PORT}/callback"
    
    if FILE_SEQ == "YOUR_FILE_SEQ":
        print("\n❌ FILE_SEQ를 실제 값으로 변경 후 실행하세요!")
        print("\n[사용법]")
        print("1. TOKEN.TXT에 API 토큰 입력")
        print("2. CDN에 Excel 파일 업로드 → seq 값 획득")
        print("3. FILE_SEQ, FILE_NAME, REQUEST_USER 수정")
        print("4. CALLBACK_URL을 사내망 접근 가능한 주소로 변경")
        print("5. 스크립트 실행")
    else:
        # 단일 시트 호출 예시
        response = parse_drm_excel(
            token=token,
            file_seq=FILE_SEQ,
            file_name=FILE_NAME,
            file_size=FILE_SIZE,
            header_type=1,
            header_row=1,
            start_row=2,
            callback_url=CALLBACK_URL,
            request_user=REQUEST_USER
        )
        
        print(f"\n응답: {response.text}")
        
        # 결과 대기
        print("\n⏳ 결과 대기 중... (Ctrl+C 종료)")
        try:
            while result_data is None:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n종료")