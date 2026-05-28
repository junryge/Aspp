#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
운영 채팅 로그 파서 → CSV 변환

SK하이닉스 이천 AMHS 운영팀 채팅 로그를 분석해 이벤트를 추출.
알고리즘 경보 CSV (검증결과_02_S3전체_*.csv 등) 와 시간축을 맞춰
실제 운영 이벤트와의 상관관계를 분석할 수 있게 함.

사용법:
    python3 운영로그_파서.py <운영로그.txt>
    python3 운영로그_파서.py <운영로그.txt> <출력.csv>

입력 형식:
    작성자,조직,YYYY-MM-DD HH:MM:SS
    (메시지)

    작성자,조직,YYYY-MM-DD HH:MM:SS
    (메시지)
    ...

출력 CSV 컬럼:
    datetime, date, time, author, org, event_type, keyword, message_summary, raw_message
"""

import csv
import re
import sys
from collections import Counter
from datetime import datetime


# 이벤트 유형별 키워드 매핑 (우선순위 순)
EVENT_PATTERNS = [
    # 1순위 — 가장 의미있는 이벤트
    ('DEADLOCK_SIGNAL',    ['정체', '몰림', '밀림', '밀리는', '밀려', '증가하고 있', 'Queue 증가', 'QUE 증가', 'Que 증가']),
    ('BRIDGE_ERROR',       ['Bridge OHT Error', 'Bridge OHT 발생', 'bridge 이상', 'Bridge 정체']),
    ('MLUD_ISSUE',         ['MLUD', 'Mlud', 'mlud', 'M16HUBTOM14MANUAL']),

    # 2순위 — 대응 액션
    ('CAPA_CHANGE_1',      ['MAX CAPA "1"', 'MAX CAPA 1', 'Max Capa 1', 'MAXCAPA 1', 'Capa 1로', '"1"로 변경']),
    ('CAPA_CHANGE_50',     ['MAX CAPA 50', 'Max Capa 50', 'MAXCAPA 50', 'Capa 50']),
    ('CAPA_CHANGE_3',      ['MAX CAPA "3"', 'MAX CAPA 3', 'Max Capa 3', 'MAXCAPA 3']),
    ('CAPA_RESTORE',       ['원복', '원위치']),

    # 3순위 — 장비 상태
    ('LIFTER_DOWN',        ['Lifter', 'lifter', 'LIFTER', '리프터']),
    ('ERROR_OCCURRED',     ['Error 발생', 'Error발생', 'Err 발생', 'Error가 발생', 'ERROR 발생', 'Alarm 발생']),
    ('ERROR_RECOVERED',    ['Error 조치', 'Error조치', 'Err 조치', 'Error 처리', 'Error Clear', 'Error 복구', 'ERROR 처리', 'ERROR 조치', 'Error 해결']),
    ('PORT_CLOSE',         ['AI Close', 'AI close', 'AI CLOSE', 'PORT Close', 'Port Close', 'IN Close', 'Port 차단', 'PORT 차단', '차단']),
    ('PORT_OPEN',          ['AI Open', 'AI open', 'AI OPEN', 'PORT Open', 'Port Open', 'IN Open', 'IN OPEN']),

    # 4순위 — 작업/공지
    ('MAINTENANCE',        ['작업', '교체', '점검', 'H/T', 'Handy Stop', 'HT-STOP', 'HT STOP']),
    ('TEACHING',           ['Teaching', 'teaching', 'TEACHING']),
    ('CABLE_REPLACE',      ['Cable 교체', 'cable 교체', 'LD 교체']),

    # 5순위 — 자동 알림
    ('ALERT_BOT',          ['통합알림센터', 'Intelligent Bot', 'LOW_ALARM', 'HIGH_ALARM', 'CRITICAL_ALARM']),
    ('Q_OVER',             ['Q-OVER', 'Q OVER', 'T-OVER', 'T OVER', '반송 지연', '반송지연']),
]


HEADER_RE = re.compile(
    r'^(?P<author>[^,]+),(?P<org>[^,]+),(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*$'
)


def classify(text):
    """메시지 본문으로 event_type 판정. 매칭 안되면 'OTHER'."""
    matched = []
    for ev_type, keywords in EVENT_PATTERNS:
        for kw in keywords:
            if kw in text:
                matched.append((ev_type, kw))
                break
    if not matched:
        return 'OTHER', ''
    # 우선순위 첫번째 선택
    return matched[0][0], matched[0][1]


def truncate(text, n=80):
    text = re.sub(r'\s+', ' ', text).strip()
    return text if len(text) <= n else text[:n-1] + '…'


def strip_attachments(text):
    """이미지/URL 등 제거."""
    text = re.sub(r'<<이미지>>[^\n]*', '', text)
    text = re.sub(r'<<파일>>[^\n]*', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    return text


def parse_log(path):
    """로그 파일 → 이벤트 리스트"""
    events = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        m = HEADER_RE.match(line)
        if not m:
            i += 1
            continue

        author = m['author'].strip()
        org = m['org'].strip()
        ts_str = m['ts']
        try:
            dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            i += 1
            continue

        # 다음 헤더가 나오기 전까지 모두 메시지 본문
        body_lines = []
        i += 1
        while i < len(lines):
            if HEADER_RE.match(lines[i]):
                break
            body_lines.append(lines[i].rstrip())
            i += 1
        body = '\n'.join(body_lines).strip()
        body = strip_attachments(body)
        if not body:
            continue  # 본문 없으면 스킵 (이미지만 있는 경우 등)

        event_type, keyword = classify(body)
        events.append({
            'datetime': dt.strftime('%Y-%m-%d %H:%M'),
            'date': dt.strftime('%Y-%m-%d'),
            'time': dt.strftime('%H:%M'),
            'author': author,
            'org': org,
            'event_type': event_type,
            'keyword': keyword,
            'message_summary': truncate(body, 100),
            'raw_message': truncate(body, 500),
        })

    events.sort(key=lambda e: e['datetime'])
    return events


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else f'운영로그_파싱_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

    print(f'📂 파싱 중: {inp}')
    events = parse_log(inp)
    if not events:
        print('❌ 이벤트 파싱 실패 — 로그 포맷 확인')
        sys.exit(1)

    # CSV 저장
    with open(out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(events[0].keys()))
        w.writeheader()
        w.writerows(events)

    print(f'✅ CSV 저장: {out} ({len(events)}건)')
    print()

    # 이벤트 타입별 통계
    counter = Counter(e['event_type'] for e in events)
    print('📊 이벤트 타입별 분포 (상위 15):')
    for t, c in counter.most_common(15):
        pct = 100.0 * c / len(events)
        print(f'    {t:<22} {c:>5}건  ({pct:.1f}%)')

    # 날짜별 "의미있는" 이벤트 빈도 (OTHER 제외)
    meaningful = [e for e in events if e['event_type'] not in ('OTHER', 'ALERT_BOT')]
    date_counter = Counter(e['date'] for e in meaningful)
    print()
    print(f'📅 이벤트 집중일 TOP 10 (OTHER/ALERT_BOT 제외):')
    for d, c in date_counter.most_common(10):
        print(f'    {d}  {c}건')


if __name__ == '__main__':
    main()
