#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한줄분석 — 발동이벤트.csv 의 '데이터 한 줄' 을 통째로 붙여넣으면 정밀 분석

reason 파싱(추정) 이 아니라, 그 행의 '모든 컬럼 실제값' 으로 정확히 분석:
  · 점수 구성 (layer1/flow/sla/sorter/mc)
  · 영역 × 룰별 점수 기여 (pts 컬럼)
  · 발동 룰 → 원본 컬럼명 + 실측값 + 임계 비교

사용:
  python 한줄분석.py            ← 실행 후 데이터 한 줄 붙여넣고 Enter
  (헤더 줄은 필요 없음 — 코드에 내장. 데이터 행만 붙여넣으면 됨)
"""
import csv
import io
import sys

# ── 발동이벤트.csv 헤더 (hubroom_predictor.py EVENT_FIELDS 와 동일 순서) ──
HEADER = [
    'file', 'datetime', 'date', 'time',
    'stage', 'stage_name', 'prev_stage', 'transition',
    'incident_state', 'continuity_min', 'refire_count', 'predicted_fault_type',
    'unified_risk_score', 'unified_risk_level', 'hot_area', 'hot_score',
    'affected_areas', 'propagation_chain', 'flow_signals', 'maxcapa_signals',
    'M16HUB_score', 'M14_score', 'M14B_score', 'M16A_score', 'M16B_score',
    'M16_score', 'M16_PKT_score', 'M16_WT_score',
    'M16HUB_signals', 'M14_signals', 'M14B_signals', 'M16A_signals', 'M16B_signals',
    'M16HUB_ra', 'M14_ra', 'M14B_ra', 'M16A_ra', 'M16B_ra',
    'M16HUB_rb_diff30', 'M14_rb_diff30', 'M14B_rb_diff30', 'M16A_rb_diff30', 'M16B_rb_diff30',
    'M16HUB_rd_fab', 'M16HUB_stb_util', 'M16HUB_rev_count', 'M16HUB_rev_lids',
    'sla_M14', 'sla_M14B', 'sla_M16A', 'sla_M16B', 'sla_M16HUB',
    'sorter_M14', 'sorter_M14B', 'sorter_M16A', 'sorter_M16B', 'sorter_M16HUB',
    'reason',
    'M16HUB_pts_RA', 'M16HUB_pts_RA_sus', 'M16HUB_pts_RB', 'M16HUB_pts_RB_fast',
    'M16HUB_pts_RC', 'M16HUB_pts_RD', 'M16HUB_pts_SLA', 'M16HUB_pts_SORT', 'M16HUB_pts_MAXCAPA',
    'M14_pts_RA', 'M14_pts_RA_sus', 'M14_pts_RB', 'M14_pts_RB_fast',
    'M14_pts_RC', 'M14_pts_RD', 'M14_pts_SLA', 'M14_pts_SORT', 'M14_pts_MAXCAPA',
    'M14B_pts_RA', 'M14B_pts_RA_sus', 'M14B_pts_RB', 'M14B_pts_RB_fast',
    'M14B_pts_RC', 'M14B_pts_RD', 'M14B_pts_SLA', 'M14B_pts_SORT', 'M14B_pts_MAXCAPA',
    'M16A_pts_RA', 'M16A_pts_RA_sus', 'M16A_pts_RB', 'M16A_pts_RB_fast',
    'M16A_pts_RC', 'M16A_pts_RD', 'M16A_pts_SLA', 'M16A_pts_SORT', 'M16A_pts_MAXCAPA',
    'M16B_pts_RA', 'M16B_pts_RA_sus', 'M16B_pts_RB', 'M16B_pts_RB_fast',
    'M16B_pts_RC', 'M16B_pts_RD', 'M16B_pts_SLA', 'M16B_pts_SORT', 'M16B_pts_MAXCAPA',
    'layer1_total', 'flow_score', 'sla_score_total', 'sorter_score_total', 'mc_score_total',
    'M16HUB_ra_count', 'M14_ra_count', 'M14B_ra_count', 'M16A_ra_count', 'M16B_ra_count',
    'M16HUB_rb_diff10', 'M14_rb_diff10', 'M14B_rb_diff10', 'M16A_rb_diff10', 'M16B_rb_diff10',
    'M16HUB_rc_trend', 'M14_cnv_skew',
    'M14_rd_oht', 'M14B_rd_oht', 'M16A_rd_oht', 'M16B_rd_oht',
    'M16HUB_sla_cnt', 'M14_sla_cnt', 'M16A_sla_cnt', 'M16B_sla_cnt',
    'M16A_sorter_fail', 'M16B_sorter_fail',
    'M16HUB_score_raw', 'M14_score_raw', 'M14B_score_raw', 'M16A_score_raw', 'M16B_score_raw',
]

AREAS5 = ['M16HUB', 'M14', 'M14B', 'M16A', 'M16B']
RULES = ['RA', 'RA_sus', 'RB', 'RB_fast', 'RC', 'RD', 'SLA', 'SORT', 'MAXCAPA']
RULE_KR = {
    'RA': '반송지연', 'RA_sus': '반송지연지속', 'RB': '큐누적', 'RB_fast': '큐급증',
    'RC': '리프터역증가', 'RD': 'FAB/STB/OHT', 'SLA': 'SLA초과',
    'SORT': 'Sorter대기', 'MAXCAPA': 'MAXCAPA변경',
}
# 룰 → 원본 컬럼명
RA_COL = {'M16HUB':'M16HUB.QUE.TIME.AVGTOTALTIME1MIN','M14':'M14.QUE.LOAD.AVGLOADTIME1MIN',
          'M14B':'M14B.QUE.TIME.AVGTOTALTIME1MIN','M16A':'M16A.QUE.LOAD.AVGLOADTIME1MIN',
          'M16B':'M16B.QUE.LOAD.AVGLOADTIME1MIN'}
RB_COL = {'M16HUB':'M16HUB.QUE.ALL.CURRENTQCNT','M14':'M14.QUE.ALL.HUB_TO_M14_JOB',
          'M14B':'M14B.QUE.ALL.HUB_TO_M14B_JOB','M16A':'M16A.QUE.ALL.6F_TO_HUB_JOB',
          'M16B':'M16B.QUE.ALL.10F_TO_HUB_JOB'}
SLA_COL = {a:f'{a}.QUE.ALL.TRANSPORT4MINOVERRATIO' for a in AREAS5}
SORT_COL = {a:f'{a}.SORTER.ABN.SORTERWAITCOUNTOVER' for a in AREAS5}
OHT_COL = {a:f'{a}.QUE.OHT.OHTUTIL' for a in AREAS5}

# 임계 (thresholds.json v6 기본값)
TH_RA = {'M16HUB':9.0,'M14':3.3,'M14B':5.0,'M16A':3.2,'M16B':3.5}
TH_RB = {'M16HUB':100,'M14':80,'M14B':150,'M16A':84,'M16B':32}
TH_SLA = {'M16HUB':5.0,'M14':25.45,'M16A':14.05,'M16B':22.05}
TH_SORTER = {'M14':148,'M14B':109,'M16A':180,'M16B':90,'M16HUB':30}
TH_FAB, TH_STB, TH_OHT, TH_RC = 25.75, 99.3, 95.0, 4


def _mark(val, th):
    """실측 vs 임계 판정 마크."""
    if th is None or th == 0:
        return ""
    if val >= th:
        ratio = val / th if th > 0 else 0
        return f"  🔴 초과 ({ratio:.1f}배)"
    return f"  🟢 정상 (임계의 {val/th*100:.0f}%)"


def f(r, k):
    v = (r.get(k) or '').strip()
    try: return float(v)
    except: return 0


def _verdict(r):
    """점수 → 정상/문제 한 줄 판정."""
    sc = f(r, 'unified_risk_score')
    cont = f(r, 'continuity_min')
    cont_s = f" / {int(cont)}분째 지속" if cont >= 1 else ""
    if sc >= 220:   return f"⚫ 발동 — 점수 {int(sc)}{cont_s}"
    if sc >= 160:   return f"🔴 위험 — 점수 {int(sc)}{cont_s}"
    if sc >= 130:   return f"🟠 경계 — 점수 {int(sc)}{cont_s}"
    if sc >= 120:   return f"🟡 주의 — 점수 {int(sc)}{cont_s}"
    if sc >= 100:   return f"🔵 관심 — 점수 {int(sc)}{cont_s}"
    return f"🟢 정상 — 점수 {int(sc)} (알람 X)"


def analyze(r):
    print("\n" + _verdict(r))
    print("=" * 74)
    print(f"  {r.get('datetime','?')} | 점수 {r.get('unified_risk_score','?')} "
          f"[{r.get('unified_risk_level','?')}] | hot={r.get('hot_area','?')} "
          f"| 유형 {r.get('predicted_fault_type','')}")
    print(f"  사건상태 {r.get('incident_state','')} | 지속 {r.get('continuity_min','')}분 "
          f"| 재발 {r.get('refire_count','')}회 | 영향영역 {r.get('affected_areas','')}")
    print("=" * 74)

    # 점수 구성
    l1,fl,sl,so,mc = (f(r,'layer1_total'),f(r,'flow_score'),f(r,'sla_score_total'),
                      f(r,'sorter_score_total'),f(r,'mc_score_total'))
    print(f"\n  [점수 구성] {l1+fl+sl+so+mc:.0f} = 영역합 {l1:.0f} + 흐름 {fl:.0f} "
          f"+ SLA {sl:.0f} + Sorter {so:.0f} + MAXCAPA {mc:.0f}")

    # 영역×룰 점수표
    print(f"\n  [영역 × 룰 점수]  {'영역':<7}", end='')
    for rule in RULES: print(f"{rule:>8}", end='')
    print(f"{'합':>5}")
    print("  " + "-" * 88)
    for a in AREAS5:
        vals = [f(r, f'{a}_pts_{rule}') for rule in RULES]
        if sum(vals) == 0: continue
        print(f"  {'':<17}{a:<7}", end='')
        for v in vals: print(f"{(int(v) if v else '·'):>8}", end='')
        print(f"{sum(vals):>5.0f}")

    # 발동 룰 → 원본 컬럼 + 실측값 + 임계 + 판정마크
    print(f"\n  [발동 룰 → 원본 컬럼 / 실측값 / 임계 / 판정]")
    for a in AREAS5:
        lines = []
        if f(r,f'{a}_pts_RA')>0 or f(r,f'{a}_pts_RA_sus')>0:
            v=f(r,f'{a}_ra'); th=TH_RA[a]
            lines.append(f"반송지연  {RA_COL[a]} = {v:.2f}분 (임계 {th}){_mark(v,th)}")
        if f(r,f'{a}_pts_RB')>0 or f(r,f'{a}_pts_RB_fast')>0:
            v=f(r,f'{a}_rb_diff30'); th=TH_RB[a]
            lines.append(f"큐누적    {RB_COL[a]} = +{int(v)}/30분 (임계 {th}){_mark(v,th)}")
        if f(r,f'{a}_pts_RC')>0 and a=='M16HUB':
            v=f(r,'M16HUB_rev_count')
            lines.append(f"리프터역증가  역증가 {int(v)}개 (임계 {TH_RC}){_mark(v,TH_RC)} [{r.get('M16HUB_rev_lids','')}]")
        if f(r,f'{a}_pts_RD')>0:
            if a=='M16HUB':
                fab=f(r,'M16HUB_rd_fab'); stb=f(r,'M16HUB_stb_util')
                fab_m = "🔴 초과" if fab>=TH_FAB else "🟢 정상"
                stb_m = "🔴 초과" if stb>=TH_STB else "🟢 정상"
                lines.append(f"FAB저장   FABSTORAGERATIO = {fab:.1f}% (임계 {TH_FAB})  {fab_m}")
                lines.append(f"STB저장   STB.3F_STORAGE_UTIL = {stb:.1f}% (임계 {TH_STB})  {stb_m}")
            else:
                v=f(r,f'{a}_rd_oht')
                lines.append(f"OHT정체   {OHT_COL[a]} = {v:.1f}% (임계 {TH_OHT}){_mark(v,TH_OHT)}")
        if f(r,f'{a}_pts_SLA')>0:
            v=f(r,f'sla_{a}'); th=TH_SLA.get(a)
            lines.append(f"SLA초과   {SLA_COL[a]} = {v:.1f}% (임계 {th}){_mark(v,th) if th else ''}")
        if f(r,f'{a}_pts_SORT')>0:
            v=f(r,f'sorter_{a}'); th=TH_SORTER.get(a)
            lines.append(f"Sorter    {SORT_COL[a]} = {int(v)}LOT 대기 (임계 {th}){_mark(v,th) if th else ''}")
        if f(r,f'{a}_pts_MAXCAPA')>0:
            lines.append(f"MAXCAPA   {a}.QUE.CNV.3F_CNV_MAXCAPA 변경  🟡 운영자조치")
        if lines:
            print(f"\n   ▶ {a}")
            for ln in lines: print(f"      · {ln}")

    fs = r.get('flow_signals','')
    if fs: print(f"\n   ▶ 흐름(flow): {fs}")
    print(f"\n  ※ reason 원문: {r.get('reason','')[:120]}...")
    print()


# 구버전 헤더 (v6 신규 4컬럼 없는 131컬럼) — 자동 감지용
HEADER_OLD = [c for c in HEADER
              if c not in ('incident_state', 'continuity_min',
                           'refire_count', 'predicted_fault_type')]


def parse_line(line):
    line = line.strip()
    if not line: return None
    row = next(csv.reader(io.StringIO(line)))
    # 컬럼 수로 신/구버전 자동 감지
    if len(row) == len(HEADER):
        hdr = HEADER
    elif len(row) == len(HEADER_OLD):
        hdr = HEADER_OLD       # 구버전 (신규 4컬럼 없음)
    elif len(row) > len(HEADER):
        hdr = HEADER + [f'extra{i}' for i in range(len(row) - len(HEADER))]
    else:
        hdr = HEADER_OLD if len(row) <= len(HEADER_OLD) else HEADER
        row = row + [''] * (len(hdr) - len(row))
    return dict(zip(hdr, row))


def main():
    if len(sys.argv) >= 2:
        # 파일 경로면 그 안 데이터행 전부, 아니면 인자를 한 줄로
        import os
        if os.path.exists(sys.argv[1]):
            for ln in open(sys.argv[1], encoding='utf-8-sig'):
                if ln.startswith('file,') or not ln.strip(): continue
                r = parse_line(ln)
                if r: analyze(r)
            return
        r = parse_line(' '.join(sys.argv[1:]))
        if r: analyze(r)
        return
    print("=" * 74)
    print("  한줄분석 — 발동이벤트.csv 의 데이터 한 줄을 복사해서 붙여넣고 Enter")
    print("            (헤더는 빼고 데이터만 / 끝내려면 q 입력)")
    print("=" * 74)
    while True:
        try:
            line = input("\nCSV 데이터 한 줄 붙여넣기> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line.lower() in ('q','quit','exit'):
            print("종료."); break
        if line.startswith('file,'):
            print("  (헤더 줄임 — 데이터 한 줄만 붙여넣어줘)"); continue
        try:
            r = parse_line(line)
            analyze(r)
        except Exception as e:
            print(f"  파싱 실패: {e}")


if __name__ == '__main__':
    main()
