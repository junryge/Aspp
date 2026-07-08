#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
경보요약 — june_final.csv(매분 상세) → 사람이 볼 '경보 구간만' 압축 요약
====================================================================================
매분 수만 줄 상세는 근거용으로 남기고, 실제 경보 뜬 구간만 한 줄로 묶어 요약.
지평선별로 2개 파일 생성:
    경보요약_10분.csv  — 10분 최종등급 경보 구간
    경보요약_30분.csv  — 30분 최종등급 경보 구간

압축 규칙:
    - 등급(경계+)이 연속으로 뜨는 구간 = 한 줄 (시작~끝, 최고등급, 대표사유)
    - 정상(등급 공란)은 전부 버림
    - 5분 이내 끊김은 같은 사건으로 이어붙임(잔깜빡임 병합)

입력:
    --infile   xgb_비정상_infer 산출 (june_final.csv)
    --gap      끊김 병합 허용(분) 기본 5
    --outdir   출력 폴더 (기본 입력과 같은 폴더)

실행:
    python 경보요약.py --infile ./out_ml/june_final.csv
"""
import argparse
import csv
import os
from datetime import datetime, timedelta

ORD = {'': 0, '경계': 1, '위험': 2, '초위험': 3}


def summarize(rows, hz, gap):
    """hz='10' or '30'. 해당 지평선 최종등급 경보 구간을 묶어 요약 리스트 반환."""
    g_col = f'{hz}분_최종등급'
    p_col = f'{hz}분_확률%'
    tgt_col = f'{hz}분_예측시각'
    why_col = f'{hz}분_사유'
    out = []
    cur = None
    last_t = None
    for r in rows:
        t = datetime.strptime(r['datetime'][:16], '%Y-%m-%d %H:%M')
        g = (r.get(g_col) or '').strip()
        if g in ('경계', '위험', '초위험'):
            if cur and last_t is not None and (t - last_t) <= timedelta(minutes=gap + 1):
                # 이어짐 — 갱신
                cur['end'] = t
                if ORD[g] > ORD[cur['top']]:
                    cur['top'] = g
                    cur['why'] = (r.get(why_col) or '').strip()
                    cur['peak_p'] = r.get(p_col, '')
                    cur['peak_tgt'] = r.get(tgt_col, '')
            else:
                if cur:
                    out.append(cur)
                cur = {'start': t, 'end': t, 'top': g,
                       'why': (r.get(why_col) or '').strip(),
                       'peak_p': r.get(p_col, ''), 'peak_tgt': r.get(tgt_col, '')}
            last_t = t
        else:
            if cur and last_t is not None and (t - last_t) > timedelta(minutes=gap + 1):
                out.append(cur); cur = None
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infile', required=True)
    ap.add_argument('--gap', type=int, default=5)
    ap.add_argument('--outdir', default=None)
    a = ap.parse_args()

    with open(a.infile, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r['datetime'])
    outdir = a.outdir or (os.path.dirname(a.infile) or '.')

    for hz in ('10', '30'):
        segs = summarize(rows, hz, a.gap)
        fp = os.path.join(outdir, f'경보요약_{hz}분.csv')
        with open(fp, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['경보시작', '경보종료', '지속(분)', '최종등급',
                        f'예측대상시각({hz}분후)', '최고확률', '사유'])
            for s in segs:
                dur = int((s['end'] - s['start']).total_seconds() / 60) + 1
                w.writerow([s['start'].strftime('%Y-%m-%d %H:%M'),
                            s['end'].strftime('%H:%M'), dur, s['top'],
                            s['peak_tgt'], s['peak_p'], s['why']])
        n_high = sum(1 for s in segs if s['top'] in ('위험', '초위험'))
        print(f"[{hz}분] 경보 구간 {len(segs)}건 (위험+ {n_high}건) → {fp}")

    print("→ 사람은 이 요약 2개만 보면 됨. 상세는 june_final.csv (근거용).")


if __name__ == '__main__':
    main()
