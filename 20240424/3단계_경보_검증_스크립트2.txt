#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3단계 데드락 룰베이스 사건 단위 추출 스크립트

목적:
  운영 환경 시뮬레이션. STAR CSV 한 줄씩 처리하면서 과거 90분 데이터만 보고
  3단계 데드락 위험 사건을 사건 단위로 추출해 단일 CSV 출력.

기존 `3단계_경보_검증_스크립트.py` 와 차이:
  - 검증 스크립트: 전체 CSV 일괄 분석 (10개 CSV 출력, 분석/검증용)
  - 본 스크립트: 슬라이딩 윈도우 (1개 CSV 출력, 운영 시뮬용)

룰 (3차 검증된 것 — 변경 없음):
  R-A'         : 1MIN ≥ 9분이 10분창 1회+
  ra_sustained : 1MIN ≥ 6분이 5분창 3회+
  R-B          : M14→M16 +100/30분
  rb_fast      : M14→M16 +30/10분
  R-C'         : 리프터 합 감소 + 역증가 2개+

  S1 = R-A' 2회+ OR ra_sustained
  S2 = R-B OR rb_fast
  S3 = R-A' AND R-B AND R-C'  (불변 — 04-21 검증 보호)

사용법:
    python3 3단계_룰베이스_사건단위.py STAR.csv
    python3 3단계_룰베이스_사건단위.py STAR.csv -o my_사건.csv

출력:
    사건단위_<YYYYMMDD_HHMMSS>.csv (또는 -o 지정 경로)
"""

import csv
import os
import sys
from collections import deque
from datetime import datetime, timedelta


# ====== 상수 ======
WINDOW_MIN = 90  # 슬라이딩 윈도우 길이 (분)

LIFTER_IDS = [
    '6ABL6011', '6ABL6012', '6ABL6021', '6ABL6022',
    '6ABL6031', '6ABL6032', '6ABL0111', '6ABL0112',
    '6ABL0121', '6ABL0122',
]

# 룰 임계치 (검증된 값)
TH_RA_VALUE = 9.0     # R-A' 절대 임계 (분)
TH_RA_SUSTAINED_VALUE = 6.0
TH_RA_SUSTAINED_COUNT = 3   # 5분창 안 3회
TH_RB_DIFF_30 = 100   # R-B 30분 +100
TH_RB_DIFF_10 = 30    # R-B FAST 10분 +30
TH_RC_REVERSE = 2     # R-C' 역증가 2개+

# 사건 종료 판단: S3 신규/재발동 사이 간격
INCIDENT_END_GAP_MIN = 10
# predict_time 추적 윈도우: S3 직전 60분 내 최초 S1/S2 시각
PREDICT_LOOKBACK_MIN = 60


# ====== 유틸 ======
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


def parse_time(s):
    if not s:
        return None
    s = s.strip().strip('"').strip("'")
    if 'T' in s and '+' in s.split('T')[1]:
        s = s.split('+')[0]
    if s.endswith('Z'):
        s = s[:-1]
    if '.' in s:
        s = s.split('.')[0]
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M',
                '%Y%m%d%H%M%S', '%Y%m%d %H%M%S'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def detect_prefix(fieldnames):
    anchor = '.QUE.ALL.CURRENTQCNT'
    for col in fieldnames or []:
        if col and col.endswith(anchor):
            return col[:-len(anchor)]
    return None


# ====== STAR 로드 (한 줄씩 yield) ======
def iter_star_rows(filepath):
    """STAR CSV → (datetime, star_dict) 제너레이터"""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        prefix = detect_prefix(reader.fieldnames)
        if not prefix:
            print(f"❌ {filepath}: prefix 감지 실패")
            return

        def C(suffix):
            return f"{prefix}{suffix}"

        for row in reader:
            t = parse_time(row.get('CRT_TM', ''))
            if not t:
                continue
            star = {
                'avgtotal1min': safe_float(row.get(C('.QUE.TIME.AVGTOTALTIME1MIN'))),
                'm14_to_m16':   safe_int(row.get(C('.QUE.M14TOM16.MESCURRENTQCNT'))),
                'lft_list': {
                    lid: safe_int(row.get(C(f'.LFT.{lid}.TOTAL_CURRENTQCNT')))
                    for lid in LIFTER_IDS
                },
            }
            star['lft_list'] = {k: v for k, v in star['lft_list'].items() if v is not None}
            yield t, star, prefix


# ====== 룰 평가 (90분 윈도우 데이터 기반) ======
def evaluate_rules(t1_window, m14_window, lft_window):
    """
    슬라이딩 윈도우 deque 들로부터 S1/S2/S3 평가.

    반환: (s1, s2, s3, ctx)
      ctx = {
        'ra_count', 'ra_value', 'ra_sustained',
        'rb_diff', 'rb_fast', 'rb_diff_10',
        'rc_trend', 'rev_count', 'rev_lids',
      }
    """
    t1_list = list(t1_window)
    m14_list = list(m14_window)
    lft_list = list(lft_window)

    # ── R-A' (절대) ── 최근 10개 (10분) 중 9분 이상 회수
    recent_t1 = t1_list[-10:]
    ra_count = sum(1 for v in recent_t1 if v is not None and v >= TH_RA_VALUE)
    ra_value = recent_t1[-1] if recent_t1 else None
    ra_trig = ra_count >= 1

    # ── R-A' SUSTAINED ── 최근 5개 중 6분 이상 3회+
    ra_sustained = False
    if len(recent_t1) >= 5:
        last5 = [v for v in recent_t1[-5:] if v is not None]
        if len(last5) >= 3:
            ra_sustained = sum(1 for v in last5 if v >= TH_RA_SUSTAINED_VALUE) >= TH_RA_SUSTAINED_COUNT

    # ── R-B ── 30분 전 대비 +100 (m14_window 길이 31 이상 필요)
    rb_diff = 0
    rb_trig = False
    if len(m14_list) >= 31 and m14_list[-1] is not None and m14_list[-31] is not None:
        rb_diff = m14_list[-1] - m14_list[-31]
        rb_trig = rb_diff >= TH_RB_DIFF_30

    # ── R-B FAST ── 10분 전 대비 +30
    rb_fast = False
    rb_diff_10 = 0
    if len(m14_list) >= 11 and m14_list[-1] is not None and m14_list[-11] is not None:
        rb_diff_10 = m14_list[-1] - m14_list[-11]
        rb_fast = rb_diff_10 >= TH_RB_DIFF_10

    # ── R-C' ── 20분 전 대비 합 감소 + 역증가 2개+
    rc_trend = 0
    rev_count = 0
    rev_lids = []
    rc_trig = False
    if len(lft_list) >= 21 and lft_list[-1] and lft_list[-21]:
        now_l = lft_list[-1]
        prev_l = lft_list[-21]
        rc_trend = sum(now_l.values()) - sum(prev_l.values())
        for lid in now_l:
            if now_l[lid] > prev_l.get(lid, 0):
                rev_lids.append(lid)
                rev_count += 1
        rc_trig = rc_trend < 0 and rev_count >= TH_RC_REVERSE

    # ── Stage ──
    s1 = (ra_count >= 2) or ra_sustained
    s2 = rb_trig or rb_fast
    s3 = ra_trig and rb_trig and rc_trig

    ctx = {
        'ra_count': ra_count,
        'ra_value': ra_value,
        'ra_sustained': ra_sustained,
        'ra_trig': ra_trig,
        'rb_diff': rb_diff,
        'rb_diff_10': rb_diff_10,
        'rb_fast': rb_fast,
        'rb_trig': rb_trig,
        'rc_trend': rc_trend,
        'rc_trig': rc_trig,
        'rev_count': rev_count,
        'rev_lids': rev_lids,
    }
    return s1, s2, s3, ctx


# ====== 사건 추적 FSM ======
class IncidentTracker:
    """
    상태:
      - 'IDLE'        : 활성 사건 없음
      - 'IN_INCIDENT' : 사건 진행 중

    전이:
      IDLE        + s3 신규     → IN_INCIDENT
      IN_INCIDENT + s3 (10분+)  → 같은 사건 재발동
      IN_INCIDENT + 10분 무 s3  → IDLE (사건 종료)
    """
    def __init__(self):
        self.state = 'IDLE'
        self.current = None
        self.incidents = []
        self.early_signals = deque(maxlen=PREDICT_LOOKBACK_MIN)
        # 발동 이벤트 타임라인 (단계 전환 시점 기록)
        self.events = []          # [{time, stage, prev_stage, reason}]
        self.last_stage = 0

    def _record_event(self, t, stage, ctx):
        if stage == self.last_stage:
            return
        if stage == 1:
            if ctx.get('ra_count', 0) >= 2:
                reason = f"1MIN ≥9분이 {ctx['ra_count']}회"
            elif ctx.get('ra_sustained'):
                reason = "1MIN ≥6분 지속"
            else:
                reason = "1단계 발동"
        elif stage == 2:
            if ctx.get('rb_diff', 0) >= 100:
                reason = f"M14→M16 +{ctx['rb_diff']} (30분간)"
            elif ctx.get('rb_fast'):
                reason = f"M14→M16 +{ctx.get('rb_diff_10', 0)} (10분간 fast)"
            else:
                reason = "2단계 발동"
        elif stage == 3:
            reason = (f"AND 만족 (1MIN {ctx.get('ra_value') or 0:.2f}, "
                      f"M14→M16 +{ctx.get('rb_diff', 0)}, "
                      f"역증가 {ctx.get('rev_count', 0)}개)")
        else:
            reason = "정상화"
        self.events.append({
            'time': t,
            'stage': stage,
            'prev_stage': self.last_stage,
            'reason': reason,
        })
        self.last_stage = stage

    def update(self, t, s1, s2, s3, ctx):
        if s1 or s2:
            self.early_signals.append(t)

        # 단계 변환 기록 (S1/S2/S3/정상화 전환마다 events 에 추가)
        cur_stage = 3 if s3 else (2 if s2 else (1 if s1 else 0))
        self._record_event(t, cur_stage, ctx)

        if self.state == 'IDLE':
            if s3:
                self._start_new(t, ctx)
        else:  # IN_INCIDENT
            if s3:
                last_s3 = self.current['last_s3_time']
                if (t - last_s3).total_seconds() / 60.0 >= INCIDENT_END_GAP_MIN:
                    self.current['refire_count'] += 1
                self._update_current(t, ctx)
            else:
                last_s3 = self.current['last_s3_time']
                if (t - last_s3).total_seconds() / 60.0 >= INCIDENT_END_GAP_MIN:
                    self._end_current(t)

    def _start_new(self, t, ctx):
        cutoff = t - timedelta(minutes=PREDICT_LOOKBACK_MIN)
        early = [x for x in self.early_signals if x >= cutoff]
        predict_time = min(early) if early else t

        self.current = {
            'predict_time': predict_time,
            'start_time': t,
            'last_s3_time': t,
            'end_time': t,
            'refire_count': 0,
            'max_1min': ctx['ra_value'] or 0,
            'max_rb_diff': ctx['rb_diff'] or 0,
            'max_rev': ctx['rev_count'] or 0,
            'rev_lids_union': set(ctx.get('rev_lids') or []),
        }
        self.state = 'IN_INCIDENT'

    def _update_current(self, t, ctx):
        c = self.current
        c['last_s3_time'] = t
        c['end_time'] = t
        if ctx['ra_value'] and ctx['ra_value'] > c['max_1min']:
            c['max_1min'] = ctx['ra_value']
        if ctx['rb_diff'] and ctx['rb_diff'] > c['max_rb_diff']:
            c['max_rb_diff'] = ctx['rb_diff']
        if ctx['rev_count'] and ctx['rev_count'] > c['max_rev']:
            c['max_rev'] = ctx['rev_count']
        c['rev_lids_union'].update(ctx.get('rev_lids') or [])

    def _end_current(self, t):
        c = self.current
        c['end_time'] = c['last_s3_time']
        self.incidents.append(c)
        self.current = None
        self.state = 'IDLE'

    def finalize(self, last_t):
        if self.state == 'IN_INCIDENT':
            self.current['end_time'] = self.current['last_s3_time']
            self.incidents.append(self.current)
            self.current = None
            self.state = 'IDLE'


# ====== 사건 → CSV 행 변환 ======
def _classify_severity(refire_count, max_1min):
    """★/★★/★★★/-"""
    if refire_count >= 4 or max_1min >= 20:
        return '★★★'
    elif refire_count >= 2 or max_1min >= 15:
        return '★★'
    elif refire_count >= 1 or max_1min >= 10:
        return '★'
    return '-'


def _build_explanation(max_1min, max_rb_diff, max_rev):
    """primary_cause, contrib_breakdown, anomaly_explanation 계산"""
    ra_val = float(max_1min or 0)
    rb_val = int(max_rb_diff or 0)
    rc_val = int(max_rev or 0)

    ra_score = round(100 * (ra_val / 9.0 - 1), 1) if ra_val >= 9.0 else 0
    rb_score = round(100 * (rb_val / 100.0 - 1), 1) if rb_val >= 100 else 0
    rc_score = round(100 * (rc_val / 2.0 - 1), 1) if rc_val >= 2 else 0

    contrib = [
        ("R-A' 반송시간", ra_score, f"{ra_val:.2f}분 (기준 9분)"),
        ("R-B FAB큐",    rb_score, f"+{rb_val} (기준 +100)"),
        ("R-C' 리프터",  rc_score, f"{rc_val}개 역증가 (기준 2개)"),
    ]
    contrib.sort(key=lambda x: -x[1])

    primary_cause = contrib[0][0] if contrib[0][1] > 0 else '기준 미달'
    breakdown = ' | '.join(f"{name} {desc}" for name, _, desc in contrib)

    if contrib[0][1] > 50:
        impact = '매우 강함'
    elif contrib[0][1] > 20:
        impact = '강함'
    elif contrib[0][1] > 0:
        impact = '보통'
    else:
        impact = '약함'

    parts = []
    if ra_score > 0:
        parts.append(f"반송시간 {ra_val:.1f}분")
    if rb_score > 0:
        parts.append(f"FAB간 큐 +{rb_val}")
    if rc_score > 0:
        parts.append(f"리프터 역증가 {rc_val}개")

    if not parts:
        explanation = '3단계 조건 일부만 부분 충족'
    else:
        explanation = f"{primary_cause} 주도 ({impact}): " + ", ".join(parts)

    return primary_cause, breakdown, explanation


def incident_to_row(c, file_name):
    duration_min = round((c['end_time'] - c['start_time']).total_seconds() / 60.0, 1)
    lead_min = round((c['start_time'] - c['predict_time']).total_seconds() / 60.0)
    primary_cause, breakdown, explanation = _build_explanation(
        c['max_1min'], c['max_rb_diff'], c['max_rev']
    )
    early_warning = (
        f"{c['predict_time'].strftime('%H:%M')} 1·2단계 발동 → "
        f"{c['start_time'].strftime('%H:%M')} 3단계 확정 ({lead_min}분 먼저 인지)"
    )
    return {
        'file': file_name,
        'date': c['start_time'].strftime('%Y-%m-%d'),
        'predict_time': c['predict_time'].strftime('%H:%M'),
        'start_time': c['start_time'].strftime('%H:%M'),
        'end_time': c['end_time'].strftime('%H:%M'),
        'lead_min': lead_min,
        'duration_min': duration_min,
        'refire_count': c['refire_count'],
        'max_1min': round(c['max_1min'], 2),
        'max_m14_diff': c['max_rb_diff'],
        'max_reverse_lifters': c['max_rev'],
        'primary_cause': primary_cause,
        'contrib_breakdown': breakdown,
        'anomaly_explanation': explanation,
        'early_warning': early_warning,
    }


# ====== 메인 ======
def process(input_csv, output_csv=None):
    if output_csv is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_csv = f'사건단위_{ts}.csv'

    # 90분 슬라이딩 윈도우
    t1_window = deque(maxlen=WINDOW_MIN)
    m14_window = deque(maxlen=WINDOW_MIN)
    lft_window = deque(maxlen=WINDOW_MIN)

    tracker = IncidentTracker()
    last_t = None
    file_name = os.path.basename(input_csv)
    rows_processed = 0
    rules_evaluated = 0

    print(f'📥 입력: {input_csv}')
    print(f'   윈도우: 과거 {WINDOW_MIN}분')

    for t, star, prefix in iter_star_rows(input_csv):
        # ① 윈도우 push (None 도 그대로 — 평가 시 None 체크)
        t1_window.append(star.get('avgtotal1min'))
        m14_window.append(star.get('m14_to_m16'))
        lft_window.append(star.get('lft_list') or {})
        rows_processed += 1
        last_t = t

        # ② 90분 안 채워지면 룰 평가 보류
        if len(t1_window) < 31:  # R-B 평가 최소 (30분 전 비교 위해 31개)
            continue

        # ③ 룰 평가
        s1, s2, s3, ctx = evaluate_rules(t1_window, m14_window, lft_window)
        rules_evaluated += 1

        # ④ FSM 갱신
        tracker.update(t, s1, s2, s3, ctx)

    # 마지막 미해소 사건 강제 종료
    tracker.finalize(last_t)

    # ── CSV 출력 ──
    rows = [incident_to_row(c, file_name) for c in tracker.incidents]

    if not rows:
        print(f'\n⚠️ 사건 0건 — CSV 빈 헤더만 출력')
        rows = [{
            'file': '', 'date': '', 'predict_time': '', 'start_time': '',
            'end_time': '', 'lead_min': '', 'duration_min': '', 'refire_count': '',
            'max_1min': '', 'max_m14_diff': '', 'max_reverse_lifters': '',
            'primary_cause': '',
            'contrib_breakdown': '', 'anomaly_explanation': '', 'early_warning': '',
        }]
        header_only = True
    else:
        header_only = False

    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        if not header_only:
            rows.sort(key=lambda r: (r['date'], r['start_time']))
            w.writerows(rows)

    # 발동 이벤트 타임라인 CSV (S1/S2/S3/정상화 매 전환 시점)
    events_csv = output_csv.replace('.csv', '_발동이벤트.csv')
    if not events_csv.endswith('.csv'):
        events_csv = output_csv + '_발동이벤트.csv'
    stage_label = {0: '정상', 1: '1단계 조기경보', 2: '2단계 주의보', 3: '3단계 ⭐확정'}
    with open(events_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'datetime', 'date', 'time', 'stage', 'stage_name',
                    'prev_stage', 'transition', 'reason'])
        for ev in tracker.events:
            t_str = ev['time'].strftime('%Y-%m-%d %H:%M')
            d_str = ev['time'].strftime('%Y-%m-%d')
            hm = ev['time'].strftime('%H:%M')
            transition = f"{ev['prev_stage']}→{ev['stage']}"
            w.writerow([file_name, t_str, d_str, hm, ev['stage'],
                        stage_label.get(ev['stage'], ''), ev['prev_stage'],
                        transition, ev['reason']])

    print()
    print(f'📊 처리 결과')
    print(f'   행 처리:    {rows_processed:,} 행')
    print(f'   룰 평가:    {rules_evaluated:,} 회 (90분 채워진 후)')
    print(f'   사건 추출:  {len(tracker.incidents)} 건')
    print(f'   발동 전환:  {len(tracker.events)} 회 (S1/S2/S3/정상화 합산)')
    print()
    print(f'💾 출력:')
    print(f'   · {output_csv}                (사건 단위)')
    print(f'   · {events_csv}  (S1/S2/S3 발동 타임라인)')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = None

    # -o <path> 옵션 파싱
    if '-o' in sys.argv:
        idx = sys.argv.index('-o')
        if idx + 1 < len(sys.argv):
            output_csv = sys.argv[idx + 1]

    if not os.path.exists(input_csv):
        print(f'❌ 파일 없음: {input_csv}')
        sys.exit(1)

    process(input_csv, output_csv)


if __name__ == '__main__':
    main()
