"""
AMP Map 데이터 추출 스크립트
사용법: python amp_data_fetch.py
출력: amp_map_data.json (같은 폴더)
"""
import pymysql
import json
import os
from datetime import datetime

# DB 설정
DB_CONFIG = {
    'host': '10.32.72.48',
    'port': 3306,
    'user': 'root',
    'password': 'test',
    'database': 'amp',
    'charset': 'utf8mb4'
}

TABLES = ['location', 'node', 'hostlocation', 'bufferlocation']

def fetch():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    data = {}
    for t in TABLES:
        try:
            cursor.execute(f"SELECT * FROM `{t}`")
            data[t] = cursor.fetchall()
            print(f"  {t}: {len(data[t])}건")
        except Exception as e:
            print(f"  {t}: 오류 - {e}")
            data[t] = []

    cursor.close()
    conn.close()
    return data

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AMP 데이터 추출 시작")
    data = fetch()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'amp_map_data.json')
    f = open(out, 'w', encoding='utf-8')
    json.dump(data, f, ensure_ascii=False, default=str)
    f.close()

    size = os.path.getsize(out) / 1024 / 1024
    print(f"  저장: {out} ({size:.1f}MB)")
    print("  완료")

if __name__ == '__main__':
    main()
