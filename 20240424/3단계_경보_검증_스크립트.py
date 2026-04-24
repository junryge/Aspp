#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3단계 데드락 경보 룰 검증 스크립트 (단독 실행)

사용법:
    python3 3단계_경보_검증_스크립트.py <STAR_CSV_경로> [데드락시각 ...]

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
            })

    return events


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
    raw_deadlock_args = sys.argv[2:]

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
    all_events_rows = []  # CSV 출력용 (전체 단계 전환 이벤트)

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

        # === CSV 3종 출력 ===
        ts_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_events = f'검증결과_이벤트_{ts_suffix}.csv'
        out_summary = f'검증결과_파일별요약_{ts_suffix}.csv'
        out_metrics = f'검증결과_종합지표_{ts_suffix}.csv'

        # 1. 이벤트 CSV
        if all_events_rows:
            with open(out_events, 'w', encoding='utf-8-sig', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(all_events_rows[0].keys()))
                w.writeheader()
                w.writerows(all_events_rows)

        # 2. 파일별 요약 CSV
        with open(out_summary, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['file', 'date', 'transitions', 'S3_count', 'TP', 'FN', 'FP'])
            for row in all_summary:
                w.writerow(row)

        # 3. 종합 지표 CSV
        if metrics:
            with open(out_metrics, 'w', encoding='utf-8-sig', newline='') as f:
                w = csv.writer(f)
                w.writerow(['metric', 'value'])
                for k, v in metrics.items():
                    w.writerow([k, v])

        print('\n💾 CSV 저장:')
        if all_events_rows:
            print(f'   · {out_events}  ({len(all_events_rows)} 이벤트)')
        print(f'   · {out_summary}  ({len(all_summary)} 파일)')
        if metrics:
            print(f'   · {out_metrics}  ({len(metrics)} 지표)')
        print(f'\n👉 이 3개 CSV 를 전달해주세요.')


if __name__ == '__main__':
    main()
