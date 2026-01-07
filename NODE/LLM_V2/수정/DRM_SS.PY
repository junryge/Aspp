"""
SK하이닉스 DRM 해제 → 로컬 저장
- CSV, Excel, 다중시트 지원
- TOKEN.TXT에 API 토큰 입력
"""

import requests
import json
from flask import Flask, request
import threading
import time
import os
import csv
import pandas as pd

# ============== 설정 ==============
API_URL_SINGLE = "https://workplace.skhynix.com/api/common/v1/drm/excel/parse"
API_URL_MULTI = "https://workplace.skhynix.com/api/common/v1/drm/excel/sheetParse"
CDN_BASE = "https://cdn.skhynix.com"

TOKEN_FILE = "TOKEN.TXT"
OUTPUT_DIR = "./output"  # 저장 폴더

CALLBACK_PORT = 5000
CALLBACK_HOST = "0.0.0.0"

result_data = None
current_file_info = {}


# ============== 토큰 로드 ==============
def load_token():
    if not os.path.exists(TOKEN_FILE):
        print(f"❌ {TOKEN_FILE} 파일이 없습니다!")
        return None
    with open(TOKEN_FILE, 'r', encoding='utf-8') as f:
        token = f.read().strip()
    if not token:
        print(f"❌ {TOKEN_FILE} 파일이 비어있습니다!")
        return None
    print(f"✅ 토큰 로드 완료")
    return token


# ============== 파일 저장 함수 ==============
def save_to_csv(headers, datas, filepath):
    """CSV로 저장"""
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(headers)
        writer.writerows(datas)
    print(f"📁 CSV 저장: {filepath}")


def save_to_excel(headers, datas, filepath):
    """Excel로 저장"""
    df = pd.DataFrame(datas, columns=headers if headers else None)
    df.to_excel(filepath, index=False, engine='openpyxl')
    print(f"📁 Excel 저장: {filepath}")


def save_to_excel_multi(sheets_data, filepath):
    """다중 시트 Excel 저장"""
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        for i, sheet in enumerate(sheets_data):
            headers = sheet.get('headers', [])
            datas = sheet.get('datas', [])
            sheet_name = sheet.get('name', f'Sheet{i+1}')
            df = pd.DataFrame(datas, columns=headers if headers else None)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"📁 다중시트 Excel 저장: {filepath}")


def save_result(result, original_filename):
    """결과를 원본 형식으로 저장"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    headers = result.get('headers') or result.get('heasers', [])
    datas = result.get('datas', [])
    
    name, ext = os.path.splitext(original_filename)
    ext = ext.lower()
    
    output_path = os.path.join(OUTPUT_DIR, original_filename)
    
    if ext == '.csv':
        save_to_csv(headers, datas, output_path)
    elif ext in ['.xlsx', '.xls']:
        if isinstance(datas, list) and len(datas) > 0 and isinstance(datas[0], dict) and 'sheet' in datas[0]:
            sheets_data = []
            headers_list = headers if isinstance(headers, list) and len(headers) > 0 and isinstance(headers[0], dict) else []
            for i, d in enumerate(datas):
                sheet_headers = headers_list[i].get('header', []) if i < len(headers_list) else []
                sheets_data.append({
                    'headers': sheet_headers,
                    'datas': d.get('sheet', []),
                    'name': f'Sheet{i+1}'
                })
            save_to_excel_multi(sheets_data, output_path)
        else:
            save_to_excel(headers, datas, output_path)
    else:
        csv_path = os.path.join(OUTPUT_DIR, name + '.csv')
        save_to_csv(headers, datas, csv_path)
    
    json_path = os.path.join(OUTPUT_DIR, name + '_raw.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"📁 JSON 백업: {json_path}")


# ============== 콜백 서버 ==============
app = Flask(__name__)

@app.route('/callback', methods=['POST'])
def callback():
    global result_data
    result_data = request.json
    
    print("\n" + "="*60)
    print("✅ DRM 해제 결과 수신!")
    print("="*60)
    
    status = result_data.get('parseStatus')
    
    if status == 3:
        original_filename = current_file_info.get('filename', 'output.csv')
        save_result(result_data, original_filename)
        print("\n✅ 저장 완료!")
    elif status == 9:
        print(f"❌ 에러: {result_data.get('errorMsg')}")
    else:
        print(f"⏳ 진행중... (status={status})")
    
    return {'status': 'received'}


def start_callback_server():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host=CALLBACK_HOST, port=CALLBACK_PORT, threaded=True, use_reloader=False)


# ============== API 호출 ==============
def parse_drm_file(
    token: str,
    file_seq: str,
    file_name: str,
    file_ext: str = None,
    file_size: int = 0,
    header_type: int = 1,
    header_row: int = 1,
    start_row: int = 2,
    callback_url: str = None,
    request_user: str = ""
):
    global current_file_info
    
    if file_ext is None:
        file_ext = os.path.splitext(file_name)[1].lstrip('.').lower()
        if not file_ext:
            file_ext = 'xlsx'
    
    current_file_info = {'filename': file_name, 'ext': file_ext}
    
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
    
    print(f"\n📤 API 호출중...")
    print(f"   파일: {file_name}")
    print(f"   SEQ: {file_seq}")
    
    response = requests.post(API_URL_SINGLE, json=payload, headers=headers)
    print(f"   응답: {response.status_code}")
    
    return response


# ============== 메인 ==============
if __name__ == "__main__":
    print("="*60)
    print("SK하이닉스 DRM 해제 → 로컬 저장")
    print("="*60)
    
    token = load_token()
    if not token:
        exit(1)
    
    print(f"\n🚀 콜백 서버 시작 (port {CALLBACK_PORT})...")
    server_thread = threading.Thread(target=start_callback_server, daemon=True)
    server_thread.start()
    time.sleep(1)
    
    # ========================================
    # ⚠️ 여기 수정!
    # ========================================
    FILE_SEQ = "YOUR_FILE_SEQ"           # CDN에서 받은 seq
    FILE_NAME = "APS_20250105.CSV"       # 원본 파일명
    FILE_SIZE = 0
    REQUEST_USER = "X0000000"            # 사원번호
    CALLBACK_URL = f"http://localhost:{CALLBACK_PORT}/callback"
    # ========================================
    
    if FILE_SEQ == "YOUR_FILE_SEQ":
        print("\n❌ FILE_SEQ를 입력하세요!")
        print("\n[사용법]")
        print("1. TOKEN.TXT → API 토큰")
        print("2. FILE_SEQ → CDN seq 값")
        print("3. FILE_NAME → 원본 파일명")
        print("4. 실행 → ./output 폴더에 저장")
    else:
        response = parse_drm_file(
            token=token,
            file_seq=FILE_SEQ,
            file_name=FILE_NAME,
            file_size=FILE_SIZE,
            callback_url=CALLBACK_URL,
            request_user=REQUEST_USER
        )
        
        if response.status_code == 200:
            print("\n⏳ 결과 대기중...")
            try:
                while result_data is None:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n종료")
        else:
            print(f"❌ 오류: {response.text}")