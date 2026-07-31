#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 사건단위_합치기 — 날짜별 사건단위/발동이벤트 CSV 를 한 파일로 병합
# ====================================================================
# hubroom_predictor 가 날짜별로 뱉은 파일들을 고객 전달용 한 파일로 합친다.
#   20260701_사건단위.csv ... 20260730_사건단위.csv  →  2026년07월_사건단위.csv
#
# 사용법:
#   python 사건단위_합치기.py -i .\out_7
#       → out_7 안의 *_사건단위.csv 전부 합쳐 .\사건단위_합본.csv 생성
#
#   python 사건단위_합치기.py -i .\out_7 --kind 발동이벤트 -o .\7월_발동이벤트.csv
#   python 사건단위_합치기.py -i .\out_7 --from 20260701 --to 20260730
#
# 특징:
#   · 헤더는 첫 파일 기준. 파일마다 컬럼이 달라도 합집합으로 맞춰 빈칸 채움
#   · 날짜(파일명 YYYYMMDD) 순으로 정렬해서 이어붙임
#   · 원본 CSV 의 따옴표/줄바꿈 셀을 그대로 보존 (csv 모듈로 읽고 씀)
#   · 파일별 건수와 합계를 출력, 0건 파일도 표시
import argparse
import csv
import os
import re
import sys

csv.field_size_limit(10 ** 7)   # reason 등 긴 셀 대비


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--indir', required=True, help='날짜별 CSV 가 있는 폴더 (예: .\\out_7)')
    ap.add_argument('-o', '--out', default=None, help='출력 파일 (기본: <종류>_합본.csv)')
    ap.add_argument('--kind', default='사건단위', help='합칠 종류: 사건단위(기본) / 발동이벤트')
    ap.add_argument('--from', dest='d_from', default=None, help='시작 YYYYMMDD (포함)')
    ap.add_argument('--to', dest='d_to', default=None, help='종료 YYYYMMDD (포함)')
    a = ap.parse_args()

    if not os.path.isdir(a.indir):
        sys.exit(f'❌ 폴더 없음: {os.path.abspath(a.indir)}')

    files = []
    for f in os.listdir(a.indir):
        if not f.lower().endswith('.csv') or a.kind not in f:
            continue
        m = re.search(r'(\d{8})', f)
        if not m:
            continue
        ymd = m.group(1)
        if a.d_from and ymd < a.d_from:
            continue
        if a.d_to and ymd > a.d_to:
            continue
        files.append((ymd, os.path.join(a.indir, f)))
    files.sort()

    if not files:
        sys.exit(f'❌ {os.path.abspath(a.indir)} 안에 *{a.kind}*.csv 없음')

    out = a.out or f'{a.kind}_합본.csv'
    print(f'[합치기] {a.kind} · 대상 {len(files)}개 파일 → {out}\n')

    # 1차: 전체 컬럼 합집합 (파일마다 컬럼이 다를 수 있음)
    header, extra = [], []
    for ymd, fp in files:
        with open(fp, encoding='utf-8-sig', newline='') as f:
            cols = next(csv.reader(f), [])
        if not header:
            header = list(cols)
        else:
            for c in cols:
                if c not in header and c not in extra:
                    extra.append(c)
    if extra:
        print(f'  ℹ️ 뒤 파일에만 있던 컬럼 {len(extra)}개를 끝에 추가: {", ".join(extra[:6])}'
              + (' …' if len(extra) > 6 else ''))
    header += extra

    # 2차: 이어붙이기
    total, empty = 0, []
    with open(out, 'w', newline='', encoding='utf-8-sig') as fo:
        w = csv.DictWriter(fo, fieldnames=header, extrasaction='ignore')
        w.writeheader()
        for ymd, fp in files:
            with open(fp, encoding='utf-8-sig', newline='') as fi:
                rows = list(csv.DictReader(fi))
            for r in rows:
                w.writerow({c: (r.get(c) or '') for c in header})
            total += len(rows)
            print(f'  {"✅" if rows else "  "} {os.path.basename(fp):<34} {len(rows):>5}건')
            if not rows:
                empty.append(ymd)

    print(f'\n🎉 완료 — {len(files)}개 파일 · 총 {total}건 → {os.path.abspath(out)}')
    print(f'   컬럼 {len(header)}개')
    if empty:
        print(f'   ℹ️ 0건인 날짜: {", ".join(empty)}  (그날 사건이 없었던 것)')


if __name__ == '__main__':
    main()
