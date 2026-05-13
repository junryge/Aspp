#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3단계 데드락 룰베이스 실시간 예측기
======================================
수집기가 매분 00초에 ./predict/M16A_HUBROOM_PR.csv 덮어쓰면
본 스크립트는 매분 05초에 깨어나 마지막 행 1개를 새로 처리해서
./predict_tobe/ 폴더에 날짜별 CSV로 append.

기본값:
  · 입력: ./predict/M16A_HUBROOM_PR.csv
  · 출력: ./predict_tobe/
  · 주기: 매분 00초 동기 (수집기와 5초 offset)

룰 (3차 검증):
  R-A'         : 1MIN ≥ 9분이 10분창 1회+
  ra_sustained : 1MIN ≥ 6분이 5분창 3회+
  R-B          : M14→M16 +100/30분
  rb_fast      : M14→M16 +30/10분
  R-C'         : 리프터 합 감소 + 역증가 2개+
  R-D          : FAB저장률 ≥25% (5/7 7시 정체 검증)

  S1 = R-A' 2회+ OR ra_sustained
  S2 = R-B OR rb_fast
  S3 = R-A' AND R-C' AND (R-B OR R-D)

출력:
  ./predict_tobe/<YYYYMMDD>_발동이벤트.csv  (매분 1행, 이벤트 없는 분도 기록)
  ./predict_tobe/<YYYYMMDD>_사건단위.csv    (S3 종료 시 1건)

사용법:
    # 실시간 (기본 — 매분 동기, 추천)
    python3 3단계_룰베이스_사건단위.py --watch

    # 일괄 (현재 CSV 한 번만 처리)
    python3 3단계_룰베이스_사건단위.py

    # 경로 지정
    python3 3단계_룰베이스_사건단위.py path/to/INPUT.csv -o ./predict_tobe --watch
"""

import csv
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path


# ====== 기본 경로 ======
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = BASE_DIR / "predict" / "M16A_HUBROOM_PR.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "predict_tobe"

SYNC_OFFSET_SEC = 5
INTERVAL_SEC = 60


# ====== 상수 ======
WINDOW_MIN = 90

LIFTER_IDS = [
    '6ABL6011', '6ABL6012', '6ABL6021', '6ABL6022',
    '6ABL6031', '6ABL6032', '6ABL0111', '6ABL0112',
    '6ABL0121', '6ABL0122',
]

TH_RA_VALUE = 9.0
TH_RA_SUSTAINED_VALUE = 6.0
TH_RA_SUSTAINED_COUNT = 3
TH_RB_DIFF_30 = 100
TH_RB_DIFF_10 = 30
TH_RC_REVERSE = 2
TH_RD_FABSTORAGE = 25.0
TH_RD_7F_HUB_ALT = 20

INCIDENT_END_GAP_MIN = 10
PREDICT_LOOKBACK_MIN = 60


# ====== 로깅 ======
def setup_logger(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("predictor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(out_dir / "predictor.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


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


# ====== STAR 로드 ======
def iter_star_rows(filepath):
    """STAR CSV → (datetime, star_dict, prefix) 제너레이터.
    수집기가 utf-8-sig BOM 붙임. 인코딩 폴백."""
    enc_list = ['utf-8-sig', 'utf-8', 'cp949']
    last_err = None
    for enc in enc_list:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                reader = csv.DictReader(f)
                prefix = detect_prefix(reader.fieldnames)
                if not prefix:
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
                        'fabstorage_ratio':  safe_float(row.get(C('.STRATE.ALL.FABSTORAGERATIO'))),
                        'm14b_aotransdelay': safe_float(row.get('M14B.QUE.ABN.AOTRANSDELAY')),
                        'm14b_oht_util':     safe_float(row.get('M14B.QUE.OHT.OHTUTIL')),
                        'm14b_4abld122':     safe_int(row.get('M14B.LFT.4ABLD122.TOTAL_CURRENTQCNT')),
                        'm14b_avgtotal1min': safe_float(row.get('M14B.QUE.TIME.AVGTOTALTIME1MIN')),
                        'm14b_7f_to_hub':    safe_int(row.get('M14B.QUE.ALL.7F_TO_HUB_JOB')),
                        'm14b_7f_to_hub_alt': safe_int(row.get('M14B.QUE.ALL.7F_TO_HUB_JOB_ALT')),
                        'm14_htstop':        safe_int(row.get('M14.OHT.STATECNT.HTSTOP')),
                        'm14_congested':     safe_int(row.get('M14.OHT.STATECNT.CONGESTED')),
                        'm14_abnormal':      safe_int(row.get('M14.OHT.STATECNT.ABNORMAL')),
                        'm16pkt_aotransdelay': safe_float(row.get('M16_PKT.QUE.ABN.AOTRANSDELAY')),
                        'm16wt_aotransdelay':  safe_float(row.get('M16_WT.QUE.ABN.AOTRANSDELAY')),
                    }
                    star['lft_list'] = {k: v for k, v in star['lft_list'].items() if v is not None}
                    yield t, star, prefix
            return
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if last_err:
        raise last_err


# ====== 룰 평가 ======
def evaluate_rules(t1_window, m14_window, lft_window, v3_window=None):
    t1_list = list(t1_window)
    m14_list = list(m14_window)
    lft_list = list(lft_window)

    recent_t1 = t1_list[-10:]
    ra_count = sum(1 for v in recent_t1 if v is not None and v >= TH_RA_VALUE)
    ra_value = recent_t1[-1] if recent_t1 else None
    ra_trig = ra_count >= 1

    ra_sustained = False
    if len(recent_t1) >= 5:
        last5 = [v for v in recent_t1[-5:] if v is not None]
        if len(last5) >= 3:
            ra_sustained = sum(1 for v in last5 if v >= TH_RA_SUSTAINED_VALUE) >= TH_RA_SUSTAINED_COUNT

    rb_diff = 0
    rb_trig = False
    if len(m14_list) >= 31 and m14_list[-1] is not None and m14_list[-31] is not None:
        rb_diff = m14_list[-1] - m14_list[-31]
        rb_trig = rb_diff >= TH_RB_DIFF_30

    rb_fast = False
    rb_diff_10 = 0
    if len(m14_list) >= 11 and m14_list[-1] is not None and m14_list[-11] is not None:
        rb_diff_10 = m14_list[-1] - m14_list[-11]
        rb_fast = rb_diff_10 >= TH_RB_DIFF_10

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

    rd_fabstorage = 0
    rd_7f_alt = 0
    rd_trig = False
    if v3_window:
        latest = v3_window[-1] if v3_window else {}
        if latest:
            rd_fabstorage = latest.get('fabstorage_ratio') or 0
            rd_7f_alt = latest.get('m14b_7f_to_hub_alt') or 0
            rd_trig = rd_fabstorage >= TH_RD_FABSTORAGE

    s1 = (ra_count >= 2) or ra_sustained
    s2 = rb_trig or rb_fast
    s3 = ra_trig and rc_trig and (rb_trig or rd_trig)

    ctx = {
        'ra_count': ra_count, 'ra_value': ra_value, 'ra_sustained': ra_sustained,
        'ra_trig': ra_trig,
        'rb_diff': rb_diff, 'rb_diff_10': rb_diff_10,
        'rb_fast': rb_fast, 'rb_trig': rb_trig,
        'rc_trend': rc_trend, 'rc_trig': rc_trig,
        'rev_count': rev_count, 'rev_lids': rev_lids,
        'rd_fabstorage': rd_fabstorage, 'rd_7f_alt': rd_7f_alt, 'rd_trig': rd_trig,
    }
    return s1, s2, s3, ctx


# ====== 사건 추적 FSM ======
class IncidentTracker:
    def __init__(self):
        self.state = 'IDLE'
        self.current = None
        self.incidents = []
        self.early_signals = deque(maxlen=PREDICT_LOOKBACK_MIN)
        self.events = []
        self.last_stage = 0

    def _record_event(self, t, stage, ctx):
        is_transition = stage != self.last_stage
        if stage == 0:
            reason = "이벤트 없음"
        elif stage == 1:
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
            if ctx.get('rd_trig') and not (ctx.get('rb_trig') and ctx.get('rc_trig')):
                reason = (f"R-D FAB저장률 정체 (FABSTORAGE={ctx.get('rd_fabstorage', 0):.1f}%, "
                          f"1MIN {ctx.get('ra_value') or 0:.2f})")
            else:
                reason = (f"AND 만족 (1MIN {ctx.get('ra_value') or 0:.2f}, "
                          f"M14→M16 +{ctx.get('rb_diff', 0)}, "
                          f"역증가 {ctx.get('rev_count', 0)}개)")
        else:
            reason = "이벤트 없음"
        self.events.append({
            'time': t, 'stage': stage, 'prev_stage': self.last_stage,
            'reason': reason, 'is_transition': is_transition,
            'ra_value': ctx.get('ra_value'), 'ra_count': ctx.get('ra_count', 0),
            'ra_sustained': bool(ctx.get('ra_sustained')),
            'rb_diff': ctx.get('rb_diff', 0), 'rb_diff_10': ctx.get('rb_diff_10', 0),
            'rb_fast': bool(ctx.get('rb_fast')), 'rb_trig': bool(ctx.get('rb_trig')),
            'rc_trend': ctx.get('rc_trend', 0),
            'rev_count': ctx.get('rev_count', 0),
            'rev_lids': list(ctx.get('rev_lids') or []),
            'rd_fabstorage': ctx.get('rd_fabstorage', 0),
            'rd_7f_alt': ctx.get('rd_7f_alt', 0),
            'rd_trig': bool(ctx.get('rd_trig')),
        })
        self.last_stage = stage

    def update(self, t, s1, s2, s3, ctx):
        if s1 or s2:
            self.early_signals.append(t)
        cur_stage = 3 if s3 else (2 if s2 else (1 if s1 else 0))
        self._record_event(t, cur_stage, ctx)

        if self.state == 'IDLE':
            if s3:
                self._start_new(t, ctx)
        else:
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
            'predict_time': predict_time, 'start_time': t,
            'last_s3_time': t, 'end_time': t, 'refire_count': 0,
            'max_1min': ctx['ra_value'] or 0, 'max_rb_diff': ctx['rb_diff'] or 0,
            'max_rev': ctx['rev_count'] or 0,
            'rev_lids_union': set(ctx.get('rev_lids') or []),
            'max_rd_fabstorage': ctx.get('rd_fabstorage', 0) or 0,
            'max_rd_7f_alt': ctx.get('rd_7f_alt', 0) or 0,
            'rd_triggered': bool(ctx.get('rd_trig')),
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
        rd_f = ctx.get('rd_fabstorage', 0) or 0
        rd_a = ctx.get('rd_7f_alt', 0) or 0
        if rd_f > c.get('max_rd_fabstorage', 0):
            c['max_rd_fabstorage'] = rd_f
        if rd_a > c.get('max_rd_7f_alt', 0):
            c['max_rd_7f_alt'] = rd_a
        if ctx.get('rd_trig'):
            c['rd_triggered'] = True

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
def _build_explanation(max_1min, max_rb_diff, max_rev):
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
    explanation = f"{primary_cause} 주도 ({impact}): " + ", ".join(parts) if parts else '3단계 조건 일부만 부분 충족'
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
    severity = '확정' if (c['max_rb_diff'] or 0) >= 100 else '주의'
    return {
        'file': file_name,
        'date': c['start_time'].strftime('%Y-%m-%d'),
        'severity': severity,
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
        'relation': build_incident_relation(c),
    }


# ====== CSV 헤더 ======
STAGE_LABEL = {0: '이벤트없음', 1: '1단계 조기경보', 2: '2단계 주의보', 3: '3단계 ⭐확정'}

SRC_COL_RA = 'QUE.TIME.AVGTOTALTIME1MIN'
SRC_COL_RB = 'QUE.M14TOM16.MESCURRENTQCNT'
SRC_COL_RC_TPL = 'LFT.{lid}.TOTAL_CURRENTQCNT'

EVENT_FIELDS = [
    'file', 'datetime', 'date', 'time',
    'stage', 'stage_name', 'prev_stage', 'transition', 'reason', 'relation',
]
INCIDENT_FIELDS = [
    'file', 'date', 'severity', 'predict_time', 'start_time', 'end_time',
    'lead_min', 'duration_min', 'refire_count',
    'max_1min', 'max_m14_diff', 'max_reverse_lifters',
    'primary_cause', 'contrib_breakdown', 'anomaly_explanation', 'early_warning',
    'relation',
]


def build_event_relation(ev):
    if ev['stage'] == 0:
        return ''
    ra_val = ev.get('ra_value')
    ra_cnt = ev.get('ra_count', 0) or 0
    ra_sus = bool(ev.get('ra_sustained'))
    ra_trig = ra_cnt >= 1 or ra_sus
    rb_trig = bool(ev.get('rb_trig'))
    rb_fast = bool(ev.get('rb_fast'))
    rb_any = rb_trig or rb_fast
    rev_lids = ev.get('rev_lids') or []
    rev_n = ev.get('rev_count', 0) or 0
    rc_trend = ev.get('rc_trend', 0) or 0
    rc_trig = (rc_trend < 0) and (rev_n >= 2)

    ra_flag = 'Y' if ra_trig else 'N'
    ra_val_s = f"{ra_val:.2f}분" if ra_val is not None else 'N/A'
    ra_part = f"[R-A' {ra_flag}] {SRC_COL_RA}={ra_val_s} (≥9분 10분창 {ra_cnt}회"
    ra_part += ", 지속Y)" if ra_sus else ", 지속N)"

    rb_flag = 'Y' if rb_any else 'N'
    rb_part = (f"[R-B {rb_flag}] {SRC_COL_RB} 30분Δ={ev.get('rb_diff', 0)} (≥100), "
               f"10분Δ={ev.get('rb_diff_10', 0)} (≥30 fast)")

    rc_flag = 'Y' if rc_trig else 'N'
    if rev_lids:
        lid_cols = ', '.join(SRC_COL_RC_TPL.format(lid=l) for l in rev_lids)
        rc_part = f"[R-C' {rc_flag}] 역증가 {rev_n}개 (≥2): {lid_cols}; trend={rc_trend}"
    else:
        rc_part = f"[R-C' {rc_flag}] 역증가 0개 (≥2 필요); trend={rc_trend}"

    return f"{ra_part} | {rb_part} | {rc_part}"


def build_incident_relation(c):
    parts = []
    if c.get('max_1min'):
        parts.append(f"{SRC_COL_RA} max={float(c['max_1min']):.2f}분 (기준 9.0분)")
    if c.get('max_rb_diff'):
        parts.append(f"{SRC_COL_RB} 최대증가 +{c['max_rb_diff']} (기준 +100)")
    rev = sorted(c.get('rev_lids_union') or [])
    if rev:
        lid_cols = ', '.join(SRC_COL_RC_TPL.format(lid=l) for l in rev)
        parts.append(f"역증가 LFT({len(rev)}): {lid_cols}")
    return ' | '.join(parts) if parts else ''


def event_to_row(ev, file_name):
    t_str = ev['time'].strftime('%Y-%m-%d %H:%M')
    d_str = ev['time'].strftime('%Y-%m-%d')
    hm = ev['time'].strftime('%H:%M')
    stage_name = STAGE_LABEL.get(ev['stage'], '')
    if ev['stage'] == 0:
        return [file_name, t_str, d_str, hm, '', stage_name, '', '', '', '']
    transition = f"{ev['prev_stage']}→{ev['stage']}" if ev.get('is_transition') else ''
    relation = build_event_relation(ev)
    return [
        file_name, t_str, d_str, hm,
        ev['stage'], stage_name, ev['prev_stage'], transition, ev['reason'], relation,
    ]


def append_rows_csv(path, fields, rows):
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(fields)
        for r in rows:
            w.writerow(r)


def append_event_row(out_dir, ev, file_name):
    ymd = ev['time'].strftime('%Y%m%d')
    path = os.path.join(out_dir, f'{ymd}_발동이벤트.csv')
    append_rows_csv(path, EVENT_FIELDS, [event_to_row(ev, file_name)])
    return path


def append_incident_row(out_dir, incident, file_name):
    ymd = incident['start_time'].strftime('%Y%m%d')
    path = os.path.join(out_dir, f'{ymd}_사건단위.csv')
    row = incident_to_row(incident, file_name)
    append_rows_csv(path, INCIDENT_FIELDS, [[row[k] for k in INCIDENT_FIELDS]])
    return path


# ====== Predictor: 매분 호출되는 처리기 ======
class Predictor:
    def __init__(self, input_csv: Path, out_dir: Path, logger):
        self.input_csv = Path(input_csv)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.file_name = self.input_csv.name

        self.t1_window = deque(maxlen=WINDOW_MIN)
        self.m14_window = deque(maxlen=WINDOW_MIN)
        self.lft_window = deque(maxlen=WINDOW_MIN)
        self.v3_window = deque(maxlen=WINDOW_MIN)

        self.tracker = IncidentTracker()
        self.last_t = None
        self.last_event_count = 0
        self.last_incident_count = 0

    def tick(self):
        """1회 처리. 새 행만 append. 처리 행 수 반환."""
        if not self.input_csv.exists():
            self.logger.warning(f"입력 CSV 없음: {self.input_csv}")
            return 0

        new_rows = 0
        try:
            for t, star, _prefix in iter_star_rows(self.input_csv):
                if self.last_t is not None and t <= self.last_t:
                    continue

                self.t1_window.append(star.get('avgtotal1min'))
                self.m14_window.append(star.get('m14_to_m16'))
                self.lft_window.append(star.get('lft_list') or {})
                self.v3_window.append({k: star.get(k) for k in (
                    'fabstorage_ratio',
                    'm14b_aotransdelay', 'm14b_oht_util', 'm14b_4abld122',
                    'm14b_avgtotal1min', 'm14b_7f_to_hub', 'm14b_7f_to_hub_alt',
                    'm14_htstop', 'm14_congested', 'm14_abnormal',
                    'm16pkt_aotransdelay', 'm16wt_aotransdelay',
                )})
                self.last_t = t
                new_rows += 1

                if len(self.t1_window) < 31:
                    continue

                s1, s2, s3, ctx = evaluate_rules(
                    self.t1_window, self.m14_window, self.lft_window, self.v3_window
                )
                self.tracker.update(t, s1, s2, s3, ctx)

                while self.last_event_count < len(self.tracker.events):
                    ev = self.tracker.events[self.last_event_count]
                    append_event_row(self.out_dir, ev, self.file_name)
                    self.last_event_count += 1
                    if ev['stage'] >= 1:
                        self.logger.info(
                            f"  ▶ {ev['time'].strftime('%H:%M')} "
                            f"단계{ev['stage']} ({STAGE_LABEL[ev['stage']]}) — {ev['reason']}"
                        )

                while self.last_incident_count < len(self.tracker.incidents):
                    inc = self.tracker.incidents[self.last_incident_count]
                    p = append_incident_row(self.out_dir, inc, self.file_name)
                    self.last_incident_count += 1
                    self.logger.info(
                        f"  ★ 사건 종료 → {inc['start_time'].strftime('%H:%M')}~"
                        f"{inc['end_time'].strftime('%H:%M')} 저장: {os.path.basename(p)}"
                    )

        except Exception as e:
            self.logger.exception(f"tick 오류: {e}")

        return new_rows

    def finalize(self):
        self.tracker.finalize(self.last_t)
        while self.last_incident_count < len(self.tracker.incidents):
            inc = self.tracker.incidents[self.last_incident_count]
            append_incident_row(self.out_dir, inc, self.file_name)
            self.last_incident_count += 1


# ====== 동기 sleep ======
def sleep_until_next_minute(offset_sec=SYNC_OFFSET_SEC):
    """다음 분 (00초 + offset_sec) 까지 대기."""
    now = time.time()
    sec_in_min = now % 60
    wait = (60 - sec_in_min) + offset_sec
    if wait > 60:
        wait -= 60
    if wait < 0.05:
        wait += 60
    time.sleep(wait)


# ====== 모드 함수 ======
def run_once(input_csv: Path, out_dir: Path, logger):
    logger.info("=" * 60)
    logger.info("일괄 처리 모드 (1회)")
    logger.info(f"  INPUT : {input_csv}")
    logger.info(f"  OUTPUT: {out_dir}")
    logger.info("=" * 60)
    p = Predictor(input_csv, out_dir, logger)
    n = p.tick()
    p.finalize()
    logger.info(f"처리 완료: {n}행, 이벤트 {p.last_event_count}건, 사건 {p.last_incident_count}건")


def run_watch(input_csv: Path, out_dir: Path, logger):
    logger.info("=" * 60)
    logger.info("실시간 감시 모드 (매분 00초 + 5초 offset)")
    logger.info(f"  INPUT : {input_csv}")
    logger.info(f"  OUTPUT: {out_dir}")
    logger.info(f"  주기  : {INTERVAL_SEC}초")
    logger.info("=" * 60)

    p = Predictor(input_csv, out_dir, logger)

    logger.info("[INIT] 시작 시점 윈도우 채우기...")
    n0 = p.tick()
    logger.info(f"[INIT] 초기 {n0}행 처리. 다음 분 동기 대기 시작...")

    try:
        while True:
            sleep_until_next_minute(SYNC_OFFSET_SEC)
            cycle_start = time.time()
            n = p.tick()
            elapsed = time.time() - cycle_start
            now_s = datetime.now().strftime('%H:%M:%S')
            if n > 0:
                logger.info(f"[{now_s}] tick: 신규 {n}행 처리, {elapsed:.2f}s")
            else:
                logger.info(f"[{now_s}] tick: 신규 행 없음 (CSV 미갱신?), {elapsed:.2f}s")
    except KeyboardInterrupt:
        logger.info("사용자 중단 (Ctrl+C)")
        p.finalize()
        logger.info("종료")


def main():
    input_csv = DEFAULT_INPUT_CSV
    out_dir = DEFAULT_OUTPUT_DIR
    watch_mode = False

    args = sys.argv[1:]
    i = 0
    if i < len(args) and not args[i].startswith('-'):
        input_csv = Path(args[i])
        i += 1
    while i < len(args):
        a = args[i]
        if a == '-o' and i + 1 < len(args):
            out_dir = Path(args[i + 1])
            i += 2
        elif a == '--watch':
            watch_mode = True
            i += 1
        elif a in ('-h', '--help'):
            print(__doc__)
            return
        else:
            i += 1

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)

    if not Path(input_csv).exists() and not watch_mode:
        logger.error(f"입력 CSV 없음: {input_csv}")
        sys.exit(1)

    if watch_mode:
        run_watch(Path(input_csv), out_dir, logger)
    else:
        run_once(Path(input_csv), out_dir, logger)


if __name__ == '__main__':
    main()
