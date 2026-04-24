#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python3 3단계_경보_검증_스크립트.py test.csv \

    --ops-log 운영로그.txt \
3단계 데드락 경보 룰 검증 스크립트 (단독 실행)

사용법:
    python3 3단계_경보_검증_스크립트.py <STAR_CSV_경로> [--ops-log <운영로그.txt>] [데드락시각 ...]

운영로그 옵션:
    --ops-log 운영로그.txt  (선택, 제공 시 각 CSV 에 운영이벤트 컬럼 추가)
    → 3단계 경보가 실제 운영 이슈와 얼마나 일치하는지 자동 검증.
    → '가설 입증' 용: 알고리즘이 실제 데드락 징후를 포착했다는 증거 확보.

데드락 시각 포맷 (혼용 가능):
    14:00                    # 시:분만 (CSV 1개 처리 시 그 날짜로 가정)
    20260421 14:00           # 날짜 + 시:분 (일주일 치 일괄 처리 시 필수)
    2026-04-21 14:00         # 하이픈 포맷
    2026-04-21T14:00         # ISO
    20260421_14:00           # 언더스코어 구분

예:
    # 1개 파일 + 시각
    python3 3단계_경보_검증_스크립트.py STAR_20260421.csv 14:00

    # 여러 파일 (일주일) + 날짜별 데드락
    python3 3단계_경보_검증_스크립트.py "STAR_*.csv" \\
        "20260421 14:00" "20260421 16:30" "20260423 09:15"

    # 데드락 시각 모를 때 (탐지만)
    python3 3단계_경보_검증_스크립트.py "STAR_*.csv"

출력:
    - 터미널: 날짜별 단계 전환 타임라인 + 종합 precision/recall
    - CSV 3종 자동 저장:
        · 검증결과_이벤트_YYYYMMDD_HHMMSS.csv     (모든 단계 전환 이벤트 + TP/FP 분류)
        · 검증결과_파일별요약_YYYYMMDD_HHMMSS.csv (파일별 S3/TP/FN/FP 개수)
        · 검증결과_종합지표_YYYYMMDD_HHMMSS.csv   (precision/recall/평균선행시간)
      ※ UTF-8 BOM 포함 (엑셀 한글 깨짐 방지)

룰 정의 (M16A_BR 검증):
    R-A' AVGTOTAL1MIN ≥ 9분이 10분창 1회 이상
    R-B  M14→M16 큐 30분간 +100 이상
    R-C' 리프터 합 감소 + 역증가 2개 이상

    1단계 = R-A' 2회+
    2단계 = R-B 발동
    3단계 = R-A' AND R-B AND R-C' (AND 조건 / 데드락 10분 전 목표)
"""

import csv
import sys
import os
import glob
from datetime import datetime, timedelta
from collections import defaultdict


# 리프터 ID 10개 (M16A_BR 기준)
LIFTER_IDS = [
    '6ABL6011', '6ABL6012', '6ABL6021', '6ABL6022',
    '6ABL6031', '6ABL6032', '6ABL0111', '6ABL0112',
    '6ABL0121', '6ABL0122',
]


# ────────────────────────────────────────────────────────────
# 운영 채팅 로그 파싱 (선택적, --ops-log 제공 시)
# ────────────────────────────────────────────────────────────
import re as _re

_OPS_EVENT_PATTERNS = [
    ('DEADLOCK_SIGNAL',  ['정체', '몰림', '밀림', '밀리는', '밀려', '증가하고 있', 'Queue 증가', 'QUE 증가', 'Que 증가', '쌓이고']),
    ('BRIDGE_ERROR',     ['Bridge OHT Error', 'Bridge OHT 발생', 'bridge 이상', 'Bridge 정체']),
    ('MLUD_ISSUE',       ['MLUD', 'Mlud', 'mlud']),
    ('CAPA_CHANGE_1',    ['MAX CAPA "1"', 'MAX CAPA 1', 'Max Capa 1', 'MAXCAPA 1', 'Capa 1로', '"1"로 변경']),
    ('CAPA_CHANGE_50',   ['MAX CAPA 50', 'Max Capa 50', 'MAXCAPA 50', 'Capa 50']),
    ('CAPA_CHANGE_3',    ['MAX CAPA "3"', 'MAX CAPA 3', 'Max Capa 3', 'MAXCAPA 3']),
    ('CAPA_RESTORE',     ['원복', '원위치']),
    ('LIFTER_DOWN',      ['Lifter', 'lifter', 'LIFTER', '리프터']),
    ('ERROR_OCCURRED',   ['Error 발생', 'Error발생', 'Err 발생', 'ERROR 발생', 'Alarm 발생']),
    ('ERROR_RECOVERED',  ['Error 조치', 'Error조치', 'Err 조치', 'Error 처리', 'Error Clear', 'ERROR 처리', 'ERROR 조치']),
    ('PORT_CLOSE',       ['AI Close', 'AI close', 'AI CLOSE', 'PORT Close', 'Port Close', 'Port 차단', 'PORT 차단']),
    ('PORT_OPEN',        ['AI Open', 'AI open', 'AI OPEN', 'PORT Open', 'Port Open']),
    ('MAINTENANCE',      ['작업', '교체', '점검', 'H/T', 'Handy Stop', 'HT-STOP', 'HT STOP']),
    ('ALERT_BOT',        ['통합알림센터', 'Intelligent Bot', 'LOW_ALARM', 'HIGH_ALARM']),
]

_OPS_HEADER_RE = _re.compile(
    r'^(?P<author>[^,]+),(?P<org>[^,]+),(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*$'
)


def _ops_classify(text):
    for ev_type, keywords in _OPS_EVENT_PATTERNS:
        for kw in keywords:
            if kw in text:
                return ev_type, kw
    return 'OTHER', ''


def _ops_strip(text):
    text = _re.sub(r'<<이미지>>[^\n]*', '', text)
    text = _re.sub(r'<<파일>>[^\n]*', '', text)
    text = _re.sub(r'http[s]?://\S+', '', text)
    return text


def _ops_truncate(text, n=80):
    text = _re.sub(r'\s+', ' ', text).strip()
    return text if len(text) <= n else text[:n-1] + '…'


def parse_ops_log(path):
    """운영 채팅 로그 → [{dt, author, org, event_type, keyword, summary}, ...]"""
    ops = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except (IOError, OSError) as e:
        print(f'  ⚠️  운영로그 읽기 실패: {e}')
        return ops

    i = 0
    while i < len(lines):
        m = _OPS_HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue
        author = m['author'].strip()
        org = m['org'].strip()
        try:
            dt = datetime.strptime(m['ts'], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            i += 1
            continue
        body_lines = []
        i += 1
        while i < len(lines) and not _OPS_HEADER_RE.match(lines[i]):
            body_lines.append(lines[i].rstrip())
            i += 1
        body = _ops_strip('\n'.join(body_lines).strip())
        if not body:
            continue
        ev_type, kw = _ops_classify(body)
        ops.append({
            'dt': dt,
            'author': author,
            'org': org,
            'event_type': ev_type,
            'keyword': kw,
            'summary': _ops_truncate(body, 60),
        })
    ops.sort(key=lambda x: x['dt'])
    return ops


def find_nearby_ops(query_dt, ops_list, window_min=30):
    """query_dt ±window_min 범위의 운영 이벤트 반환 (ALERT_BOT 제외)"""
    if not ops_list or query_dt is None:
        return []
    lo = query_dt - timedelta(minutes=window_min)
    hi = query_dt + timedelta(minutes=window_min)
    result = []
    for op in ops_list:
        if op['dt'] < lo:
            continue
        if op['dt'] > hi:
            break
        if op['event_type'] == 'ALERT_BOT':
            continue
        result.append(op)
    return result


def summarize_ops(nearby_ops, query_dt):
    """근처 운영 이벤트 요약 → (count, types_str, top_msg_str)"""
    if not nearby_ops:
        return 0, '', ''
    type_counts = {}
    for op in nearby_ops:
        type_counts[op['event_type']] = type_counts.get(op['event_type'], 0) + 1
    types_str = ', '.join(f'{t}({c})' for t, c in sorted(type_counts.items(), key=lambda x: -x[1]))
    # 가장 가까운 시각의 이벤트 1개
    closest = min(nearby_ops, key=lambda x: abs((x['dt'] - query_dt).total_seconds()))
    diff_min = (closest['dt'] - query_dt).total_seconds() / 60.0
    sign = '+' if diff_min >= 0 else ''
    top_msg = f"[{sign}{diff_min:.0f}분] {closest['author']}: {closest['summary']}"
    return len(nearby_ops), types_str, top_msg


def safe_float(v):
    try:
        return float(v) if v not in (None, '', 'null') else None
    except (ValueError, TypeError):
        return None


def safe_int(v):
    try:
        return int(float(v)) if v not in (None, '', 'null') else None
    except (ValueError, TypeError):
        return None


_parse_time_failed_samples = []

def parse_time(s):
    """CRT_TM 파싱 — 다양한 포맷 대응"""
    if not s:
        return None
    raw = s
    s = s.strip().strip('"').strip("'")
    # Timezone suffix 제거
    if 'T' in s and '+' in s.split('T')[1]:
        s = s.split('+')[0]
    if s.endswith('Z'):
        s = s[:-1]
    # 밀리초 제거
    if '.' in s:
        s = s.split('.')[0]
    # Unix timestamp (숫자만)
    if s.isdigit() and len(s) in (10, 13):
        try:
            ts = int(s)
            if len(s) == 13:
                ts //= 1000
            return datetime.fromtimestamp(ts)
        except (ValueError, OSError):
            pass
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M',
                '%Y/%m/%d %H:%M', '%Y.%m.%d %H:%M:%S',
                '%Y%m%d%H%M%S', '%Y%m%d %H%M%S',
                '%Y%m%d %H:%M:%S', '%Y%m%d %H:%M',
                '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
                '%d-%m-%Y %H:%M:%S'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    if len(_parse_time_failed_samples) < 3:
        _parse_time_failed_samples.append(raw)
    return None


def parse_deadlock_arg(s, fallback_date=None):
    """데드락 시각 인자 파싱 — 날짜 포함 / 시각만 모두 지원.

    반환: datetime or None
    """
    if not s:
        return None
    s = s.strip().replace('_', ' ').replace('T', ' ')

    # 날짜+시각 포맷 우선
    for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d %H:%M', '%Y%m%d %H:%M', '%Y%m%d %H:%M:%S'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    # 시각만 (HH:MM) — fallback_date 필요
    try:
        hh, mm = map(int, s.split(':')[:2])
        if fallback_date is None:
            return None  # 날짜 미상
        return datetime.combine(fallback_date, datetime.min.time()).replace(hour=hh, minute=mm)
    except (ValueError, AttributeError):
        return None


def detect_prefix(fieldnames):
    """STAR 컬럼 prefix 자동 감지 (M16HUB / M14 / 기타)"""
    anchor = '.QUE.ALL.CURRENTQCNT'
    for col in fieldnames or []:
        if col and col.endswith(anchor):
            return col[:-len(anchor)]
    return None


def load_star(filepath):
    """STAR CSV 로드 → [(time, star_dict), ...]"""
    timeline = []
    missing_cols = defaultdict(int)
    total = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        prefix = detect_prefix(reader.fieldnames)
        if not prefix:
            print(f"❌ {filepath}: prefix 감지 실패 (CURRENTQCNT 컬럼 없음)")
            return [], None

        def C(suffix):
            return f"{prefix}{suffix}"

        # 핵심 컬럼 존재 여부 사전 체크
        required = {
            'avgtotal1min': C('.QUE.TIME.AVGTOTALTIME1MIN'),
            'm14_to_m16':   C('.QUE.M14TOM16.MESCURRENTQCNT'),
        }
        for k, col in required.items():
            if col not in (reader.fieldnames or []):
                missing_cols[k] += 1

        for row in reader:
            total += 1
            t = parse_time(row.get('CRT_TM', ''))
            if not t:
                continue
            star = {
                'avgtotal1min': safe_float(row.get(C('.QUE.TIME.AVGTOTALTIME1MIN'))),
                'm14_to_m16':   safe_int(row.get(C('.QUE.M14TOM16.MESCURRENTQCNT'))),
                'queue_total':  safe_int(row.get(C('.QUE.ALL.CURRENTQCNT'))),
                'lft_list': {
                    lid: safe_int(row.get(C(f'.LFT.{lid}.TOTAL_CURRENTQCNT')))
                    for lid in LIFTER_IDS
                },
            }
            # 리프터 None 제거 (해당 FAB 에 없으면 빈 dict)
            star['lft_list'] = {k: v for k, v in star['lft_list'].items() if v is not None}
            timeline.append((t, star))

    timeline.sort(key=lambda x: x[0])

    if missing_cols:
        print(f"  ⚠️  누락 컬럼: {list(missing_cols.keys())} (해당 룰 평가 불가)")

    return timeline, prefix


def evaluate_timeline(timeline):
    """3단계 룰 평가 → [{time, stage, label, reason, rules_detail}, ...]"""
    events = []
    if not timeline:
        return events

    t1_hist = []
    m14_hist = []
    lft_hist = []

    last_logged_stage = -1
    last_s3_time = None

    for t, star in timeline:
        if star['avgtotal1min'] is not None:
            t1_hist.append(star['avgtotal1min'])
        if star['m14_to_m16'] is not None and star['m14_to_m16'] > 0:
            m14_hist.append(star['m14_to_m16'])
        if star['lft_list']:
            lft_hist.append(dict(star['lft_list']))

        # R-A'
        recent_t1 = t1_hist[-10:]
        ra_count = sum(1 for v in recent_t1 if v >= 9.0)
        ra_value = recent_t1[-1] if recent_t1 else None
        ra_trig = ra_count >= 1

        # R-B
        rb_diff = 0
        rb_trig = False
        if len(m14_hist) >= 31:
            rb_diff = m14_hist[-1] - m14_hist[-31]
            rb_trig = rb_diff >= 100

        # R-C'
        rc_trend = 0
        rev_count = 0
        rev_lids = []
        rc_trig = False
        if len(lft_hist) >= 21:
            now_l = lft_hist[-1]
            prev_l = lft_hist[-21]
            rc_trend = sum(now_l.values()) - sum(prev_l.values())
            for lid in now_l:
                if now_l[lid] > prev_l.get(lid, 0):
                    rev_lids.append(lid)
                    rev_count += 1
            rc_trig = rc_trend < 0 and rev_count >= 2

        s1 = ra_count >= 2
        s2 = rb_trig
        s3 = ra_trig and rb_trig and rc_trig
        stage = 3 if s3 else (2 if s2 else (1 if s1 else 0))

        record = False
        reason = ''
        if stage > last_logged_stage and stage > 0:
            record = True
            if stage == 1:
                reason = f'1MIN ≥9분이 {ra_count}회'
            elif stage == 2:
                reason = f'M14→M16 +{rb_diff} (30분간)'
            elif stage == 3:
                reason = (f'AND 만족 (1MIN {ra_value:.2f}, '
                          f'M14→M16 +{rb_diff}, 역증가 {rev_count}개)')
            last_logged_stage = stage
            if stage == 3:
                last_s3_time = t
        elif stage == 3 and last_s3_time is not None:
            diff_min = (t - last_s3_time).total_seconds() / 60.0
            if diff_min >= 10:
                record = True
                reason = f'재발동 (역증가 {rev_count}개)'
                last_s3_time = t
        elif stage == 0 and last_logged_stage >= 1:
            record = True
            reason = '정상화'
            last_logged_stage = 0
            last_s3_time = None

        if record:
            events.append({
                'time': t,
                'stage': stage,
                'reason': reason,
                'ra_value': ra_value,
                'rb_diff': rb_diff,
                'rev_count': rev_count,
            })

    return events


def build_incidents(events):
    """3단계 이벤트를 "사건 단위" 로 묶음.

    1개 사건 = [신규 AND 만족] + 뒤따르는 [재발동 ...] 의 연속. '정상화' 나
    다른 신규 S3 를 만나면 사건 종료.

    반환: [{start, end, refire_count, max_1min, max_rb_diff, max_rev, severity}, ...]
    """
    incidents = []
    cur = None

    def _finalize(c, end_time):
        if not c:
            return
        c['end'] = end_time
        dur = (c['end'] - c['start']).total_seconds() / 60.0
        c['duration_min'] = round(dur, 1)
        # ★ 등급 판정
        rc = c['refire_count']
        m1 = c['max_1min'] or 0
        if rc >= 4 or m1 >= 20:
            c['severity'] = '★★★'
        elif rc >= 2 or m1 >= 15:
            c['severity'] = '★★'
        elif rc >= 1 or m1 >= 10:
            c['severity'] = '★'
        else:
            c['severity'] = '-'
        incidents.append(c)

    for e in events:
        if e['stage'] == 3 and e['reason'].startswith('AND 만족'):
            # 신규 S3 → 이전 사건 종료, 새 사건 시작
            _finalize(cur, e['time'])
            cur = {
                'start': e['time'],
                'end': e['time'],
                'refire_count': 0,
                'max_1min': e.get('ra_value') or 0,
                'max_rb_diff': e.get('rb_diff') or 0,
                'max_rev': e.get('rev_count') or 0,
                'start_reason': e['reason'],
            }
        elif e['stage'] == 3 and e['reason'].startswith('재발동'):
            if cur is not None:
                cur['refire_count'] += 1
                cur['end'] = e['time']
                cur['max_rev'] = max(cur['max_rev'], e.get('rev_count') or 0)
        elif e['stage'] == 0 and cur is not None:
            # 정상화 → 사건 종료
            _finalize(cur, e['time'])
            cur = None

    # 파일 끝난 시점에 열려있는 사건 종료
    if cur is not None and events:
        _finalize(cur, events[-1]['time'])

    return incidents


def report(filepath, events, deadlock_datetimes=None):
    """결과 출력.

    deadlock_datetimes: 이 파일 날짜와 매칭되는 데드락 datetime 리스트 (이미 필터됨)
    """
    deadlock_datetimes = deadlock_datetimes or []
    print()
    print('═' * 78)
    print(f'📁 {os.path.basename(filepath)}')
    print('═' * 78)

    if not events:
        print('  ✅ 어느 단계도 발동 안 함 (완전 정상 / 또는 데이터 부족)')
        return [], [], []

    icons = {0: '✅', 1: '🔔', 2: '⚠️', 3: '🚨'}
    names = {0: '정상화', 1: '1단계 조기경보', 2: '2단계 주의보', 3: '3단계 ⭐확정'}

    distinct_dates = sorted({e['time'].date() for e in events})
    multi_day = len(distinct_dates) > 1
    if multi_day:
        print(f'\n  📅 기간: {distinct_dates[0]} ~ {distinct_dates[-1]} ({len(distinct_dates)}일)')
    else:
        print(f'\n  📅 날짜: {distinct_dates[0]}')
    print(f'  📊 단계 전환 타임라인 ({len(events)}건)')
    print(f'  {"-"*74}')
    for e in events:
        ts = e['time'].strftime('%Y-%m-%d %H:%M') if multi_day else e['time'].strftime('%H:%M')
        print(f"  {ts}  {icons[e['stage']]} {names[e['stage']]:<18} {e['reason']}")

    # 통계
    counts = defaultdict(int)
    for e in events:
        counts[e['stage']] += 1
    print(f'\n  📈 단계별 발동 횟수')
    for stage in (1, 2, 3):
        print(f'    {names[stage]:<18}: {counts[stage]}회')

    # 사건 단위 묶음 + ★ 등급
    incidents = build_incidents(events)
    if incidents:
        new_count = len(incidents)
        refire_total = sum(i['refire_count'] for i in incidents)
        print(f'\n  🧩 사건 단위 분류: {new_count}개 사건 (신규 {new_count} + 재발동 {refire_total} = {counts[3]})')
        print(f'  {"-"*74}')
        # ★ 등급 높은 순 → 시각 순
        sev_rank = {'★★★': 3, '★★': 2, '★': 1, '-': 0}
        sorted_inc = sorted(incidents, key=lambda i: (-sev_rank[i['severity']], i['start']))
        for i in sorted_inc:
            stamp = i['start'].strftime('%Y-%m-%d %H:%M')
            one_min = i['max_1min']
            dur = f"{i['duration_min']:.0f}분" if i['duration_min'] > 0 else '단발'
            refire = f"재발동 {i['refire_count']}회" if i['refire_count'] else '단발'
            print(f"  {i['severity']:<4}  {stamp}  1MIN {one_min:>5.2f}  M14→M16 +{i['max_rb_diff']:<3}  {refire:<10}  {dur} 지속")

    tp_list, fn_list, fp_list = [], [], []

    # 데드락 ground truth 매칭
    if deadlock_datetimes:
        print(f'\n  🎯 이 날짜의 데드락 발생 시각 매칭 ({len(deadlock_datetimes)}건)')
        s3_events = [e for e in events if e['stage'] == 3]

        for dt in deadlock_datetimes:
            # 가장 가까운 3단계 발동 (이전 1시간 내)
            closest = None
            min_diff = None
            for s3 in s3_events:
                diff_sec = (dt - s3['time']).total_seconds()
                if 0 <= diff_sec <= 3600:
                    if min_diff is None or diff_sec < min_diff:
                        min_diff = diff_sec
                        closest = s3

            if closest:
                lead_min = min_diff / 60.0
                mark = '✅' if 5 <= lead_min <= 30 else '⚠️'
                print(f'    {mark} 데드락 {dt.strftime("%Y-%m-%d %H:%M")} ← '
                      f'3단계 {closest["time"].strftime("%H:%M")} (선행 {lead_min:.0f}분)')
                tp_list.append((dt, closest, lead_min))
            else:
                print(f'    ❌ 데드락 {dt.strftime("%Y-%m-%d %H:%M")}: 직전 1시간 내 3단계 없음 (FN)')
                fn_list.append(dt)

        # False positive 추정
        for s3 in s3_events:
            matched = False
            for dt in deadlock_datetimes:
                diff = (dt - s3['time']).total_seconds()
                if -600 <= diff <= 1800:  # 데드락 -10분 ~ +30분
                    matched = True
                    break
            if not matched:
                fp_list.append(s3)

        if fp_list:
            print(f'\n  ⚠️  False positive 후보 ({len(fp_list)}건, 데드락 시각과 무관한 3단계)')
            for e in fp_list:
                print(f'    {e["time"].strftime("%H:%M")} - {e["reason"]}')

    return tp_list, fn_list, fp_list


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pattern = sys.argv[1]
    raw_args = sys.argv[2:]

    # --ops-log <path> 파싱
    ops_log_path = None
    raw_deadlock_args = []
    idx = 0
    while idx < len(raw_args):
        a = raw_args[idx]
        if a == '--ops-log' and idx + 1 < len(raw_args):
            ops_log_path = raw_args[idx + 1]
            idx += 2
            continue
        if a.startswith('--ops-log='):
            ops_log_path = a.split('=', 1)[1]
            idx += 1
            continue
        raw_deadlock_args.append(a)
        idx += 1

    # 운영 로그 로드 (선택적)
    ops_list = []
    if ops_log_path:
        print(f'\n📜 운영 로그 로드: {ops_log_path}')
        ops_list = parse_ops_log(ops_log_path)
        print(f'   → {len(ops_list)}건 파싱')

    files = sorted(glob.glob(pattern)) if any(c in pattern for c in '*?[]') else [pattern]
    if not files:
        print(f'❌ 파일 없음: {pattern}')
        sys.exit(1)

    # 데드락 인자 1차 파싱
    dl_dated = []
    dl_time_only = []
    for arg in raw_deadlock_args:
        parsed = parse_deadlock_arg(arg)
        if parsed is not None:
            dl_dated.append(parsed)
        elif ':' in arg:
            dl_time_only.append(arg)
        else:
            print(f'  ⚠️  해석 불가 인자 무시: {arg!r}')

    print(f'\n🔍 검증 시작 — 파일 {len(files)}개')
    if dl_dated:
        print(f'   데드락 시각 (날짜 포함, {len(dl_dated)}건):')
        for dt in dl_dated:
            print(f'     · {dt.strftime("%Y-%m-%d %H:%M")}')
    if dl_time_only:
        print(f'   데드락 시각 (날짜 없음, {len(dl_time_only)}건): {dl_time_only}')

    all_summary = []
    total_tp, total_fn, total_fp = [], [], []
    all_events_rows = []      # CSV 출력용 (전체 단계 전환 이벤트)
    all_incidents_rows = []   # CSV 출력용 (사건 단위)

    for fp in files:
        if not os.path.exists(fp):
            print(f'⚠️  스킵 (없음): {fp}')
            continue

        timeline, prefix = load_star(fp)
        if not timeline:
            print(f'⚠️  스킵 (데이터 없음): {fp}')
            continue

        dates_in_file = sorted({t.date() for t, _ in timeline})
        first_ts = timeline[0][0].strftime('%Y-%m-%d %H:%M:%S')
        last_ts = timeline[-1][0].strftime('%Y-%m-%d %H:%M:%S')
        span_info = f'{dates_in_file[0]} ~ {dates_in_file[-1]} ({len(dates_in_file)}일)' if len(dates_in_file) > 1 else str(dates_in_file[0])
        print(f'\n📥 {os.path.basename(fp)} — prefix={prefix}, 날짜={span_info}, {len(timeline)}행')
        print(f'    CRT_TM 범위: {first_ts} ~ {last_ts}')
        # 하위 호환: report 에 쓸 대표 날짜 (다일 파일일 땐 전체 범위)
        file_date = dates_in_file[0] if len(dates_in_file) == 1 else None
        if _parse_time_failed_samples:
            print(f'    ⚠️  시각 파싱 실패 샘플: {_parse_time_failed_samples[:3]}')
            _parse_time_failed_samples.clear()

        file_deadlocks = [dt for dt in dl_dated if dt.date() == file_date]
        for ts in dl_time_only:
            dt = parse_deadlock_arg(ts, fallback_date=file_date)
            if dt is not None:
                file_deadlocks.append(dt)

        events = evaluate_timeline(timeline)
        tp, fn, fp_list = report(fp, events, file_deadlocks)

        s3_count = sum(1 for e in events if e['stage'] == 3)
        date_label = span_info  # 다일 파일이면 'YYYY-MM-DD ~ YYYY-MM-DD (N일)'
        all_summary.append((os.path.basename(fp), date_label, len(events), s3_count,
                            len(tp), len(fn), len(fp_list)))
        total_tp.extend(tp)
        total_fn.extend(fn)
        total_fp.extend(fp_list)

        # CSV 행 구성: 각 이벤트 + 분류
        fp_times = {e['time'] for e in fp_list}
        tp_times = {closest['time']: (dl_dt, lead) for dl_dt, closest, lead in tp}

        for e in events:
            classification = '-'
            matched_deadlock = ''
            lead_min = ''
            if e['stage'] == 3:
                if e['time'] in tp_times:
                    dl_dt, lead = tp_times[e['time']]
                    classification = 'TP'
                    matched_deadlock = dl_dt.strftime('%Y-%m-%d %H:%M')
                    lead_min = f'{lead:.1f}'
                elif e['time'] in fp_times:
                    classification = 'FP'
            nearby = find_nearby_ops(e['time'], ops_list, 30) if ops_list else []
            op_cnt, op_types, op_top = summarize_ops(nearby, e['time'])
            all_events_rows.append({
                'file': os.path.basename(fp),
                'date': e['time'].strftime('%Y-%m-%d'),
                'time': e['time'].strftime('%H:%M'),
                'datetime': e['time'].strftime('%Y-%m-%d %H:%M'),
                'stage': e['stage'],
                'stage_name': {0:'정상화', 1:'1단계 조기경보', 2:'2단계 주의보', 3:'3단계 확정'}[e['stage']],
                'reason': e['reason'],
                'classification': classification,
                'matched_deadlock': matched_deadlock,
                'lead_minutes': lead_min,
                'ops_count_30min': op_cnt,
                'ops_event_types': op_types,
                'ops_top_message': op_top,
            })

        # 사건 단위 CSV (사건 시작 ~ 종료 구간에 걸친 운영 이벤트 집계)
        file_incidents = build_incidents(events)
        for i in file_incidents:
            # 사건 기간 ±30분 윈도우
            win_ops = []
            if ops_list:
                lo = i['start'] - timedelta(minutes=30)
                hi = i['end'] + timedelta(minutes=30)
                for op in ops_list:
                    if op['dt'] < lo:
                        continue
                    if op['dt'] > hi:
                        break
                    if op['event_type'] == 'ALERT_BOT':
                        continue
                    win_ops.append(op)
            op_cnt = len(win_ops)
            op_types_ct = {}
            for op in win_ops:
                op_types_ct[op['event_type']] = op_types_ct.get(op['event_type'], 0) + 1
            op_types_str = ', '.join(f'{t}({c})' for t, c in sorted(op_types_ct.items(), key=lambda x: -x[1]))
            op_msgs = ' | '.join(f"{o['dt'].strftime('%H:%M')} {o['author']}: {o['summary']}" for o in win_ops[:3])
            # 가설 검증 플래그
            has_deadlock_signal = any(o['event_type'] in ('DEADLOCK_SIGNAL', 'BRIDGE_ERROR', 'ERROR_OCCURRED') for o in win_ops)
            has_operator_action = any(o['event_type'].startswith('CAPA_CHANGE') or o['event_type'].startswith('PORT_') for o in win_ops)
            verdict = (
                '✅ 운영 이슈 일치' if has_deadlock_signal else
                '⚠️ 대응만 있음' if has_operator_action else
                '❓ 운영 로그 無' if ops_list else '-'
            )
            all_incidents_rows.append({
                'file': os.path.basename(fp),
                'date': i['start'].strftime('%Y-%m-%d'),
                'start_time': i['start'].strftime('%H:%M'),
                'end_time': i['end'].strftime('%H:%M'),
                'duration_min': i['duration_min'],
                'refire_count': i['refire_count'],
                'max_1min': round(i['max_1min'], 2),
                'max_m14_diff': i['max_rb_diff'],
                'max_reverse_lifters': i['max_rev'],
                'severity': i['severity'],
                'ops_count_window': op_cnt,
                'ops_event_types': op_types_str,
                'ops_sample_messages': op_msgs,
                'verdict': verdict,
            })

    # 종합 요약 + CSV 저장
    if all_summary:
        print('\n' + '═' * 78)
        print('📊 전체 요약')
        print('═' * 78)
        print(f'{"파일":<45} {"날짜":<12} {"전환":>5} {"S3":>4} {"TP":>4} {"FN":>4} {"FP":>4}')
        print('-' * 78)
        for name, d, ev, s3, tp, fn, fp_ct in all_summary:
            print(f'{name[:45]:<45} {d:<12} {ev:>5} {s3:>4} {tp:>4} {fn:>4} {fp_ct:>4}')

        n_tp, n_fn, n_fp = len(total_tp), len(total_fn), len(total_fp)
        metrics = {}
        if n_tp + n_fn + n_fp > 0:
            print()
            print(f'  실제 데드락: {n_tp + n_fn}건')
            print(f'  3단계 적중 (TP): {n_tp}건')
            print(f'  3단계 놓침 (FN): {n_fn}건')
            print(f'  False positive (FP): {n_fp}건')
            metrics['total_deadlocks'] = n_tp + n_fn
            metrics['TP'] = n_tp
            metrics['FN'] = n_fn
            metrics['FP'] = n_fp
            if n_tp + n_fp > 0:
                prec = 100.0 * n_tp / (n_tp + n_fp)
                print(f'  Precision: {prec:.1f}%  ({n_tp}/{n_tp+n_fp})')
                metrics['precision_pct'] = f'{prec:.1f}'
            if n_tp + n_fn > 0:
                rec = 100.0 * n_tp / (n_tp + n_fn)
                print(f'  Recall:    {rec:.1f}%  ({n_tp}/{n_tp+n_fn})')
                metrics['recall_pct'] = f'{rec:.1f}'
            if total_tp:
                avg_lead = sum(lead for _, _, lead in total_tp) / len(total_tp)
                print(f'  평균 선행시간: {avg_lead:.1f}분')
                metrics['avg_lead_minutes'] = f'{avg_lead:.1f}'

        # === 운영 로그 대비 가설 검증 (ops log 제공 시) ===
        if ops_list and all_incidents_rows:
            n_inc = len(all_incidents_rows)
            n_match = sum(1 for r in all_incidents_rows if r.get('verdict', '').startswith('✅'))
            n_action = sum(1 for r in all_incidents_rows if r.get('verdict', '').startswith('⚠️'))
            n_noop = sum(1 for r in all_incidents_rows if r.get('verdict', '').startswith('❓'))
            high = [r for r in all_incidents_rows if r['severity'] in ('★★★', '★★')]
            n_high = len(high)
            n_high_match = sum(1 for r in high if r.get('verdict', '').startswith('✅'))
            print()
            print(f'🎯 가설 검증 (운영 로그 대비):')
            print(f'   전체 사건 {n_inc}개 중')
            print(f'     ✅ 운영 이슈 직접 일치 : {n_match}건 ({100.0*n_match/n_inc:.0f}%)')
            print(f'     ⚠️ 운영자 대응만 감지   : {n_action}건 ({100.0*n_action/n_inc:.0f}%)')
            print(f'     ❓ 운영 로그에 근거 無  : {n_noop}건 ({100.0*n_noop/n_inc:.0f}%)')
            if n_high > 0:
                print(f'   ★★ 이상 사건 {n_high}개 중 운영 이슈 일치: {n_high_match}건 ({100.0*n_high_match/n_high:.0f}%)')
            metrics['incidents_total'] = n_inc
            metrics['incidents_match_ops_issue'] = n_match
            metrics['incidents_match_operator_action'] = n_action
            metrics['incidents_no_ops_evidence'] = n_noop
            metrics['incidents_high_severity'] = n_high
            metrics['incidents_high_match_pct'] = f'{100.0*n_high_match/n_high:.1f}' if n_high else '0'
            metrics['hypothesis_precision_pct'] = f'{100.0*(n_match+n_action)/n_inc:.1f}' if n_inc else '0'

        # === CSV 6종 출력 (각 용도별 분리) ===
        ts_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_events  = f'검증결과_01_전체이벤트_{ts_suffix}.csv'
        out_s3_all  = f'검증결과_02_S3전체_{ts_suffix}.csv'
        out_s3_new  = f'검증결과_03_S3신규만_{ts_suffix}.csv'
        out_s3_ref  = f'검증결과_04_S3재발동만_{ts_suffix}.csv'
        out_incidents = f'검증결과_05_사건단위_{ts_suffix}.csv'
        out_summary = f'검증결과_06_파일별요약_{ts_suffix}.csv'
        out_metrics = f'검증결과_07_종합지표_{ts_suffix}.csv'

        def write_csv(path, rows, header=None):
            if not rows:
                return
            with open(path, 'w', encoding='utf-8-sig', newline='') as f:
                if header:
                    w = csv.writer(f)
                    w.writerows([header] + rows)
                else:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

        # 1. 전체 이벤트 (모든 단계 전환)
        write_csv(out_events, all_events_rows)

        # 2/3/4. S3 관련 분리
        s3_all  = [r for r in all_events_rows if r['stage'] == 3]
        s3_new  = [r for r in s3_all if 'AND 만족' in r['reason']]
        s3_ref  = [r for r in s3_all if '재발동' in r['reason']]
        write_csv(out_s3_all, s3_all)
        write_csv(out_s3_new, s3_new)
        write_csv(out_s3_ref, s3_ref)

        # 5. 사건 단위 CSV (★ 등급 내림차순)
        if all_incidents_rows:
            sev_rank = {'★★★': 3, '★★': 2, '★': 1, '-': 0}
            all_incidents_rows.sort(key=lambda r: (-sev_rank.get(r['severity'], 0), r['date'], r['start_time']))
            write_csv(out_incidents, all_incidents_rows)

        # 6. 파일별 요약
        write_csv(out_summary, [list(row) for row in all_summary],
                  header=['file', 'date', 'transitions', 'S3_count', 'TP', 'FN', 'FP'])

        # 7. 종합 지표
        if metrics:
            write_csv(out_metrics, [[k, v] for k, v in metrics.items()],
                      header=['metric', 'value'])

        print('\n💾 CSV 저장 (용도별 분리):')
        if all_events_rows: print(f'   01 · {out_events}  ({len(all_events_rows)} 전체 이벤트)')
        if s3_all:  print(f'   02 · {out_s3_all}  ({len(s3_all)} S3 전체)')
        if s3_new:  print(f'   03 · {out_s3_new}  ({len(s3_new)} S3 신규만)')
        if s3_ref:  print(f'   04 · {out_s3_ref}  ({len(s3_ref)} S3 재발동만)')
        if all_incidents_rows:
            high = sum(1 for r in all_incidents_rows if r["severity"] in ("★★★", "★★"))
            print(f'   05 · {out_incidents}  ({len(all_incidents_rows)} 사건, ★★ 이상 {high}건)')
        print(f'   06 · {out_summary}  ({len(all_summary)} 파일)')
        if metrics:
            print(f'   07 · {out_metrics}  ({len(metrics)} 지표)')
        print(f'\n👉 이 CSV들을 전달해주세요.')


if __name__ == '__main__':
    main()
