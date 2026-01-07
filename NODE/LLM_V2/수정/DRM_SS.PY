"""
# SK하이닉스 DRM 해제 설정
# 아래 값들을 입력하세요

CDN_KEY=여기에_CDN키_입력
DRM_TOKEN=여기에_DRM토큰_입력
EMP_NO=여기에_사번_입력


SK하이닉스 DRM 해제 통합 도구
로컬 파일 → CDN 업로드 → DRM 해제 → CSV/Excel 저장

사용법: python drm_parser.py 파일경로
예시: python drm_parser.py APS_20250105.CSV
"""

import requests
import json
from flask import Flask, request as flask_request
import threading
import time
import os
import csv
import sys

# ============== 설정 ==============
CDN_URL = "https://cdn.skhynix.com"
CDN_UPLOAD_API = f"{CDN_URL}/api/pub/v1/upload"
CDN_DELETE_API = f"{CDN_URL}/api/pub/v1/upload/del/seq"

DRM_API_SINGLE = "https://workplace.skhynix.com/api/common/v1/drm/excel/parse"
DRM_API_MULTI = "https://workplace.skhynix.com/api/common/v1/drm/excel/sheetParse"

CONFIG_FILE = "CONFIG.TXT"
OUTPUT_DIR = "./output"

CALLBACK_PORT = 5000
result_data = None
current_file_info = {}


# ============== 설정 로드 ==============
def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ {CONFIG_FILE} 없음!")
        return None
    
    config = {}
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    
    required = ['CDN_KEY', 'DRM_TOKEN', 'EMP_NO']
    missing = [k for k in required if not config.get(k) or config.get(k).startswith('여기에')]
    
    if missing:
        print(f"❌ CONFIG.TXT 미입력: {missing}")
        return None
    
    print("✅ 설정 로드 완료")
    return config


# ============== CDN 업로드 ==============
def upload_to_cdn(filepath, cdn_key, emp_no):
    if not os.path.exists(filepath):
        print(f"❌ 파일 없음: {filepath}")
        return None
    
    filename = os.path.basename(filepath)
    filesize = os.path.getsize(filepath)
    
    print(f"\n📤 [1/3] CDN 업로드 중...")
    print(f"   파일: {filename} ({filesize:,} bytes)")
    
    with open(filepath, 'rb') as f:
        files = {'file': (filename, f)}
        params = {'key': cdn_key, 'empNo': emp_no}
        response = requests.post(CDN_UPLOAD_API, files=files, params=params)
    
    if response.status_code != 200:
        print(f"❌ CDN 실패: {response.status_code} - {response.text}")
        return None
    
    result = response.json()
    if result.get('code') != 300:
        print(f"❌ CDN 에러: {result}")
        return None
    
    print(f"✅ CDN 업로드 성공! SEQ: {result.get('seq')}")
    return result


# ============== CDN 삭제 ==============
def delete_from_cdn(seq, cdn_key, emp_no):
    """CDN에서 파일 삭제 (정리용)"""
    payload = {'seq': seq, 'empNo': emp_no, 'key': cdn_key}
    try:
        requests.post(CDN_DELETE_API, json=payload)
        print(f"🗑️ CDN 파일 삭제 완료")
    except:
        pass


# ============== 파일 저장 ==============
def save_to_csv(headers, datas, filepath):
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(headers)
        writer.writerows(datas)
    print(f"📁 저장: {filepath}")


def save_to_excel(headers, datas, filepath):
    try:
        import pandas as pd
        df = pd.DataFrame(datas, columns=headers if headers else None)
        df.to_excel(filepath, index=False, engine='openpyxl')
        print(f"📁 저장: {filepath}")
    except ImportError:
        csv_path = filepath.rsplit('.', 1)[0] + '.csv'
        save_to_csv(headers, datas, csv_path)
        print("⚠️ pandas 없어서 CSV로 저장됨")


def save_result(result, original_filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    headers = result.get('headers') or result.get('heasers', [])
    datas = result.get('datas', [])
    
    name, ext = os.path.splitext(original_filename)
    ext = ext.lower()
    output_path = os.path.join(OUTPUT_DIR, original_filename)
    
    # 다중 시트 체크
    is_multi = isinstance(datas, list) and len(datas) > 0 and isinstance(datas[0], dict) and 'sheet' in datas[0]
    
    if ext == '.csv':
        save_to_csv(headers, datas, output_path)
    elif ext in ['.xlsx', '.xls']:
        if is_multi:
            # 다중 시트 Excel
            try:
                import pandas as pd
                headers_list = headers if isinstance(headers, list) and isinstance(headers[0], dict) else []
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    for i, d in enumerate(datas):
                        sheet_headers = headers_list[i].get('header', []) if i < len(headers_list) else []
                        sheet_data = d.get('sheet', [])
                        df = pd.DataFrame(sheet_data, columns=sheet_headers if sheet_headers else None)
                        df.to_excel(writer, sheet_name=f'Sheet{i+1}', index=False)
                print(f"📁 저장: {output_path}")
            except ImportError:
                csv_path = os.path.join(OUTPUT_DIR, name + '.csv')
                save_to_csv(headers, datas, csv_path)
        else:
            save_to_excel(headers, datas, output_path)
    else:
        csv_path = os.path.join(OUTPUT_DIR, name + '.csv')
        save_to_csv(headers, datas, csv_path)
    
    # JSON 백업
    json_path = os.path.join(OUTPUT_DIR, name + '_raw.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ============== 콜백 서버 ==============
app = Flask(__name__)

@app.route('/callback', methods=['POST'])
def callback():
    global result_data
    result_data = flask_request.json
    
    status = result_data.get('parseStatus')
    if status == 3:
        print(f"\n✅ [3/3] DRM 해제 성공!")
        save_result(result_data, current_file_info.get('filename', 'output.csv'))
    elif status == 9:
        print(f"\n❌ DRM 해제 실패: {result_data.get('errorMsg')}")
    
    return {'status': 'received'}


def start_callback_server():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=CALLBACK_PORT, threaded=True, use_reloader=False)


# ============== DRM 해제 요청 ==============
def request_drm_parse(cdn_result, drm_token, emp_no, callback_url):
    global current_file_info
    
    filename = cdn_result.get('fileName')
    seq = cdn_result.get('seq')
    file_ext = cdn_result.get('fileExt', 'xlsx')
    file_size = cdn_result.get('fileSize', 0)
    
    current_file_info = {'filename': filename, 'seq': seq}
    
    print(f"\n🔓 [2/3] DRM 해제 요청 중...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {drm_token}"
    }
    
    payload = {
        "file": {
            "down": cdn_result.get('down'),
            "fileExt": file_ext,
            "fileName": filename,
            "fileSize": file_size,
            "fileType": cdn_result.get('fileType', 1),
            "link": cdn_result.get('link', ''),
            "seq": seq,
            "size": file_size
        },
        "header": {
            "headerType": 1,
            "headerRow": 1,
            "headerNames": []
        },
        "startRow": 2,
        "returnType": "json",
        "callbackUrl": callback_url,
        "requestUser": emp_no
    }
    
    response = requests.post(DRM_API_SINGLE, json=payload, headers=headers)
    
    if response.status_code == 200:
        print(f"✅ DRM 요청 접수됨, 결과 대기중...")
    else:
        print(f"❌ DRM 요청 실패: {response.status_code} - {response.text}")
    
    return response


# ============== 메인 ==============
def main():
    print("="*60)
    print("SK하이닉스 DRM 해제 도구")
    print("="*60)
    
    # 파일 경로 확인
    if len(sys.argv) < 2:
        print("\n사용법: python drm_parser.py 파일경로")
        print("예시: python drm_parser.py APS_20250105.CSV")
        return
    
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"❌ 파일 없음: {filepath}")
        return
    
    # 설정 로드
    config = load_config()
    if not config:
        return
    
    cdn_key = config['CDN_KEY']
    drm_token = config['DRM_TOKEN']
    emp_no = config['EMP_NO']
    
    # 콜백 서버 시작
    print(f"\n🚀 콜백 서버 시작 (port {CALLBACK_PORT})...")
    server_thread = threading.Thread(target=start_callback_server, daemon=True)
    server_thread.start()
    time.sleep(1)
    
    # 콜백 URL (사내망 IP로 변경 필요시 여기 수정)
    callback_url = f"http://localhost:{CALLBACK_PORT}/callback"
    
    # 1. CDN 업로드
    cdn_result = upload_to_cdn(filepath, cdn_key, emp_no)
    if not cdn_result:
        return
    
    seq = cdn_result.get('seq')
    
    try:
        # 2. DRM 해제 요청
        response = request_drm_parse(cdn_result, drm_token, emp_no, callback_url)
        
        if response.status_code != 200:
            return
        
        # 3. 결과 대기
        global result_data
        timeout = 120
        start = time.time()
        
        while result_data is None and (time.time() - start) < timeout:
            time.sleep(1)
        
        if result_data is None:
            print(f"\n⏰ 타임아웃! 콜백 URL 확인 필요")
            print(f"   현재 설정: {callback_url}")
            print(f"   사내망에서 접근 가능한 IP:PORT로 변경하세요")
        else:
            print(f"\n✅ 완료! 결과: {OUTPUT_DIR}/")
    
    finally:
        # CDN 파일 삭제 (정리)
        delete_from_cdn(seq, cdn_key, emp_no)


if __name__ == "__main__":
    main()