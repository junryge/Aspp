"""
CNV Bridge Map 데이터 추출 스크립트
- amp.cnvinformation: P4ACV5R01 전체 경로 (859건)
- config_bridge.conveyor: P4ACV5R01 포트 이름 (29건)
사용법: python cnv_data_fetch.py
출력: cnv_map_data.json
"""
import pymysql
import json
import os
from datetime import datetime

DB_CONFIG = {
    'host': '10.32.72.48',
    'port': 3306,
    'user': 'root',
    'password': 'test',
    'charset': 'utf8mb4'
}

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CNV 데이터 추출 시작\n")

    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    data = {}

    cursor.execute("SELECT * FROM amp.cnvinformation WHERE EqpID='P4ACV5R01'")
    data['cnv'] = cursor.fetchall()
    print(f"  amp.cnvinformation: {len(data['cnv'])}건")

    cursor.execute("SELECT * FROM config_bridge.conveyor WHERE CnvHostID LIKE 'P4ACV5R01%%'")
    data['port'] = cursor.fetchall()
    print(f"  config_bridge.conveyor: {len(data['port'])}건")

    cursor.close()
    conn.close()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnv_map_data.json')
    f = open(out, 'w', encoding='utf-8')
    json.dump(data, f, ensure_ascii=False, default=str)
    f.close()

    size = os.path.getsize(out) / 1024 / 1024
    print(f"\n  저장: {out} ({size:.1f}MB)")
    print("  완료")

if __name__ == '__main__':
    main()
