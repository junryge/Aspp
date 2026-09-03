#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 발동이벤트_컬럼맞추기 — 이미 쌓인 발동이벤트.csv 를 새 예측기의 컬럼 구성에 맞춘다
# ====================================================================
# 왜 필요한가
#   예측기는 파일이 있으면 헤더를 다시 쓰지 않고 행만 이어 붙인다(append).
#   그래서 컬럼이 늘어난 새 버전으로 교체하면, 그날 이미 쌓인 파일은
#   헤더(예전 134칸)와 새 행(136칸)이 어긋난다.
#   이 도구로 헤더를 미리 새 구성으로 맞춰 두면 자정까지 기다리거나
#   파일을 지울 필요 없이 바로 교체할 수 있다.
#
#   · 새로 생긴 컬럼(pio_10min_cnt, pio_score 등)은 기존 행에 공란으로 넣는다
#   · 값은 하나도 건드리지 않는다 — 자리만 맞춘다
#   · 컬럼 순서가 예측기(EVENT_FIELDS)와 정확히 같아진다 (중간 삽입도 처리)
#   · 예측기가 안 만드는 컬럼(로그프레소·MAXCAPA·PIO 12컬럼 등)은 뒤에 그대로 남긴다
#   · 저장은 임시파일 → 원자 교체, 원본은 .bak 로 백업
#
# 사용법 (run_ml 을 멈춘 상태에서 하는 것이 가장 안전하다)
#   확인만:   python 발동이벤트_컬럼맞추기.py --event ../predict_tobe --check
#   맞추기:   python 발동이벤트_컬럼맞추기.py --event ../predict_tobe
#   한 파일:  python 발동이벤트_컬럼맞추기.py --event ../predict_tobe/20260903_발동이벤트.csv
#   여러 날:  python 발동이벤트_컬럼맞추기.py --event ../predict_tobe --days 3
import argparse, csv, os, re, shutil, sys, unicodedata
from datetime import datetime, timedelta

csv.field_size_limit(10 ** 7)
EVENT_KEY = '발동이벤트'


def is_event_csv(name):
    if not name.lower().endswith('.csv'):
        return False
    n = unicodedata.normalize('NFC', name)
    return EVENT_KEY in n and '_M1' not in n


def load_fields():
    """예측기의 EVENT_FIELDS 를 그대로 가져온다 (한 곳에서만 정의되게)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.path.dirname(here), os.getcwd()):
        fp = os.path.join(d, 'hubroom_predictor.py')
        if os.path.exists(fp):
            import importlib.util
            spec = importlib.util.spec_from_file_location('hp', fp)
            m = importlib.util.module_from_spec(spec)
            sys.modules['hp'] = m
            spec.loader.exec_module(m)
            print(f'  [기준] {fp} · EVENT_FIELDS {len(m.EVENT_FIELDS)}컬럼')
            return list(m.EVENT_FIELDS)
    raise SystemExit('❌ hubroom_predictor.py 를 찾을 수 없습니다 (같은 폴더나 상위 폴더에 두세요)')


def pick(event, days=None, since=None):
    ev = (event or '').strip().strip('"').strip("'")
    if not os.path.isdir(ev):
        if os.path.exists(ev):
            return [ev]
        print(f'  ❌ 파일 없음: {os.path.abspath(ev)}')
        return []
    files = sorted(f for f in os.listdir(ev) if is_event_csv(f))
    if days:
        since = max(since or '', (datetime.now() - timedelta(days=days - 1)).strftime('%Y%m%d'))
    out = []
    for f in files:
        m = re.search(r'(\d{8})', f)
        if since and m and m.group(1) < since:
            continue
        out.append(os.path.join(ev, f))
    if not out:
        print(f'  ❌ 대상 없음: {os.path.abspath(ev)}')
    return out


def fix(fp, fields, check=False):
    with open(fp, encoding='utf-8-sig', newline='') as f:
        rd = csv.DictReader(f)
        header = list(rd.fieldnames or [])
        rows = list(rd)
    if not header:
        print(f'  ⚠️ 빈 파일: {os.path.basename(fp)}')
        return False
    extra = [c for c in header if c not in fields]      # 기입기들이 붙인 컬럼 (뒤에 유지)
    want = fields + extra
    missing = [c for c in fields if c not in header]
    if header == want:
        print(f'  ✅ {os.path.basename(fp)} — 이미 맞음 ({len(header)}컬럼)')
        return False
    print(f'  🔧 {os.path.basename(fp)} — {len(header)} → {len(want)}컬럼'
          + (f' · 추가 {len(missing)}개: {", ".join(missing[:6])}' + ('…' if len(missing) > 6 else '')
             if missing else ' · 순서 정렬'))
    if check:
        return True
    bak = fp + '.bak'
    shutil.copy2(fp, bak)
    tmp = fp + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=want, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({c: (r.get(c) if r.get(c) is not None else '') for c in want})
    os.replace(tmp, fp)
    print(f'     저장 완료 (원본 백업: {os.path.basename(bak)}) · {len(rows)}행')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--event', required=True, help='발동이벤트 CSV 또는 폴더')
    ap.add_argument('--days', type=int, default=None, help='최근 N일만 (오늘 포함)')
    ap.add_argument('--since', default=None, metavar='YYYYMMDD')
    ap.add_argument('--check', action='store_true', help='바꾸지 않고 확인만')
    a = ap.parse_args()

    print('=' * 60)
    print('발동이벤트 컬럼 맞추기' + (' (확인만)' if a.check else ''))
    print('=' * 60)
    fields = load_fields()
    files = pick(a.event, a.days, a.since)
    if not files:
        sys.exit(2)
    n = 0
    for fp in files:
        n += bool(fix(fp, fields, a.check))
    if a.check:
        print(f'\n🎉 확인 완료 — 손봐야 할 파일 {n}개 / {len(files)}개')
        if n:
            print('   맞추려면 --check 를 빼고 다시 실행하세요.')
    else:
        print(f'\n🎉 완료 — {n}개 파일 수정 / {len(files)}개 검사')
        if n:
            print('   이제 새 예측기로 교체해도 열이 어긋나지 않습니다.')


if __name__ == '__main__':
    main()
