#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dijkstra_analyzer.py — OHT Dijkstra 경로 탐색 검증 스크립트 (독립 실행)

데이터가 너무 커서 공유가 어려운 경우, 이 한 파일을 데이터 옆에 두고 실행하면
WORLD_SIM의 Dijkstra(LineCost) 알고리즘을 그대로 검증할 수 있습니다.

필요한 파일 (사용자 로컬):
  1) layout_cache.json   — 정적 레일 그래프 (OHT_MAP/cache/{FAB}_{PFX}_layout_cache.json)
  2) OHT CSV             — LOGPRESSO_OHT_DATA_*.csv 또는 LOGPRESSO_oht_data_*.csv
                           (zip 안에 있어도 됨: --oht foo.zip:LOGPRESSO_oht_data_*.csv)
  3) RAIL_CUT CSV        — (선택) LOGPRESSO_OHT_RAIL_CUT_*.csv

사용 예:
  python dijkstra_analyzer.py \
      --layout ../OHT_MAP/cache/M16A_BR_layout_cache.json \
      --oht    /path/20260421.zip:LOGPRESSO_oht_data_m16BR_20260421.csv \
      --railcut /path/LOGPRESSO_OHT_RAIL_CUT_20260421.csv \
      --sample 500 \
      --report report.json

표준 라이브러리만 사용. 메모리 절약을 위해 CSV는 한 줄씩 스트리밍 처리.
"""

import argparse
import csv
import heapq
import io
import json
import os
import random
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# LineCost 가중치 (config.py 와 동일)
# ============================================================
LINECOST_IDLE_VHL_PENALTY = 3000   # idle 차량 1대 +3,000 pulse
LINECOST_WORK_VHL_PENALTY = 5000   # work 차량 1대 +5,000 pulse
PULSE_PER_SECOND          = 89614  # 1초 ≈ 89,614 pulse (M14A 회귀치)


# ============================================================
# 1. 레이아웃 그래프 로딩
# ============================================================
def load_layout(path):
    """layout_cache.json → (nodes, edges_dist, adj_static)
       edges_dist : {(u,v): pulse}
       adj_static : {u: [v, v, ...]}
    """
    with open(path, 'r', encoding='utf-8') as f:
        L = json.load(f)
    nodes = {int(k): tuple(v) for k, v in L.get('nodes', {}).items()}
    edges_dist = {}
    for k, dist in L.get('edges', {}).items():
        a, b = k.split(',')
        edges_dist[(int(a), int(b))] = float(dist)
    adj_static = defaultdict(list)
    for k, lst in L.get('adj', {}).items():
        adj_static[int(k)] = [int(x) for x in lst]
    return nodes, edges_dist, adj_static


# ============================================================
# 2. OHT CSV 스트리밍 (zip 안의 파일도 지원)
# ============================================================
def open_csv_stream(spec):
    """spec: 'foo.csv' 또는 'archive.zip:inner.csv'."""
    if ':' in spec and spec.lower().split(':', 1)[0].endswith('.zip'):
        zp, inner = spec.split(':', 1)
        zf = zipfile.ZipFile(zp, 'r')
        names = zf.namelist()
        # 와일드카드 / 부분 매칭 허용
        rx = re.compile(re.escape(inner).replace(r'\*', '.*'))
        match = next((n for n in names if rx.search(n)), None)
        if match is None:
            raise FileNotFoundError(f"{inner} not in {zp}; have: {names}")
        raw = zf.open(match, 'r')
        return io.TextIOWrapper(raw, encoding='utf-8', newline='')
    return open(spec, 'r', encoding='utf-8', newline='')


def aggregate_oht_csv(spec, max_rows=None):
    """OHT CSV 1패스 스캔.
    반환: idle[(u,v)], busy[(u,v)], stats(dict), unique_addresses(set), od_pairs(list)
    """
    idle = Counter()
    busy = Counter()
    addrs = set()
    vehicles = set()
    status_hist = Counter()
    rows = 0
    skipped = 0
    od_pairs = []     # (vehicle, address, destination_station, _time)
    last_dest_by_v = {}  # vid -> destination

    with open_csv_stream(spec) as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        # 컬럼명이 raw / 파싱 두 가지 — 둘 다 지원
        col_addr = 'ADDRESS'        if 'ADDRESS'        in cols else None
        col_next = 'NEXT_ADDRESS'   if 'NEXT_ADDRESS'   in cols else None
        col_stat = 'STATUS'         if 'STATUS'         in cols else None
        col_vhl  = 'VEHICLE'        if 'VEHICLE'        in cols else None
        col_dest = 'DESTINATION'    if 'DESTINATION'    in cols else None
        col_time = '_time'          if '_time'          in cols else None
        if not (col_addr and col_next and col_stat):
            raise RuntimeError(f"필수 컬럼 누락. 발견: {cols}")

        for row in reader:
            rows += 1
            if max_rows and rows > max_rows:
                break
            try:
                a = int(row[col_addr])
                n = int(row[col_next])
            except (ValueError, TypeError):
                skipped += 1
                continue
            if a == 0 or n == 0 or a == n:
                skipped += 1
                continue

            st = (row.get(col_stat) or '').strip()
            status_hist[st] += 1
            if st == '0':
                idle[(a, n)] += 1
            else:
                busy[(a, n)] += 1

            addrs.add(a); addrs.add(n)
            if col_vhl:
                vehicles.add(row[col_vhl])
            if col_dest and col_vhl:
                vid = row[col_vhl]
                d = (row.get(col_dest) or '').strip()
                t = row.get(col_time) if col_time else None
                # 차량별 첫 출현(=가장 위 행) 기준 OD 한 건만 저장
                if d and vid not in last_dest_by_v:
                    last_dest_by_v[vid] = d
                    od_pairs.append((vid, a, d, t))

    stats = {
        'rows_scanned': rows,
        'rows_skipped': skipped,
        'unique_addresses': len(addrs),
        'unique_vehicles': len(vehicles),
        'unique_edges_with_traffic': len(set(idle) | set(busy)),
        'status_histogram': dict(status_hist.most_common()),
    }
    return idle, busy, stats, addrs, od_pairs


# ============================================================
# 3. RAIL_CUT CSV
# ============================================================
def load_railcut(path):
    """ABNORMAL 상태의 차단 엣지 집합 반환: {(u,v), ...}
       path 형식은 OHT 와 동일: 'foo.csv' 또는 'archive.zip:inner.csv'.
    """
    cut = set()
    if not path:
        return cut
    with open_csv_stream(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if (row.get('STATE') or '').upper() != 'ABNORMAL':
                continue
            lst = row.get('AFFECT_ADDR_LST') or ''
            for pair in lst.split(','):
                pair = pair.strip()
                if ':' not in pair:
                    continue
                a, b = pair.split(':', 1)
                try:
                    cut.add((int(a), int(b)))
                except ValueError:
                    pass
    return cut


# ============================================================
# 4. LineCost 그래프 빌드
# ============================================================
def build_linecost_graph(adj_static, edges_dist, idle, busy, cut,
                         w_idle=LINECOST_IDLE_VHL_PENALTY,
                         w_busy=LINECOST_WORK_VHL_PENALTY):
    g = defaultdict(list)
    used_edges = 0
    cut_hits = 0
    for u, neighbors in adj_static.items():
        for v in neighbors:
            if (u, v) in cut:
                cut_hits += 1
                continue
            d = edges_dist.get((u, v), 1.0)
            cost = max(d, 1.0) + idle[(u, v)] * w_idle + busy[(u, v)] * w_busy
            g[u].append((v, cost))
            used_edges += 1
    return g, used_edges, cut_hits


# ============================================================
# 5. Dijkstra
# ============================================================
def dijkstra(g, src, dst):
    """반환: (cost, path[list], visited_count, ms)"""
    t0 = time.perf_counter()
    if src == dst:
        return 0.0, [src], 0, 0.0
    dist = {src: 0.0}
    prev = {}
    pq = [(0.0, src)]
    visited = 0
    while pq:
        c, u = heapq.heappop(pq)
        if c > dist.get(u, float('inf')):
            continue
        visited += 1
        if u == dst:
            break
        for v, w in g.get(u, []):
            nd = c + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    ms = (time.perf_counter() - t0) * 1000.0
    if dst not in dist:
        return None, [], visited, ms
    path = [dst]
    while path[-1] in prev:
        path.append(prev[path[-1]])
    path.reverse()
    return dist[dst], path, visited, ms


# ============================================================
# 6. 그래프 자체 진단 (degree, 고립 노드, BFS 도달 컴포넌트)
# ============================================================
def graph_diagnostics(adj_static, edges_dist):
    in_deg = Counter()
    out_deg = Counter()
    for u, lst in adj_static.items():
        out_deg[u] += len(lst)
        for v in lst:
            in_deg[v] += 1
    all_nodes = set(adj_static) | set(in_deg)
    isolated = [n for n in all_nodes if in_deg[n] == 0 and out_deg[n] == 0]
    sources  = [n for n in all_nodes if in_deg[n] == 0 and out_deg[n] > 0]
    sinks    = [n for n in all_nodes if out_deg[n] == 0 and in_deg[n] > 0]
    dists = list(edges_dist.values())
    return {
        'nodes_total': len(all_nodes),
        'edges_total': sum(out_deg.values()),
        'isolated_nodes': len(isolated),
        'pure_sources_no_incoming': len(sources),
        'pure_sinks_no_outgoing': len(sinks),
        'out_degree_max': max(out_deg.values()) if out_deg else 0,
        'out_degree_avg': (sum(out_deg.values()) / len(out_deg)) if out_deg else 0,
        'edge_dist_min': min(dists) if dists else 0,
        'edge_dist_max': max(dists) if dists else 0,
        'edge_dist_avg': (sum(dists) / len(dists)) if dists else 0,
    }


# ============================================================
# 7. OD 샘플링 + Dijkstra 검증
# ============================================================
def percentile(values, p):
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def verify_paths(g, samples, label="OD"):
    reach = 0
    fail = 0
    times_ms = []
    hops = []
    costs = []
    examples_ok = []
    examples_fail = []
    for src, dst in samples:
        cost, path, _, ms = dijkstra(g, src, dst)
        times_ms.append(ms)
        if cost is None:
            fail += 1
            if len(examples_fail) < 5:
                examples_fail.append((src, dst))
            continue
        reach += 1
        hops.append(len(path))
        costs.append(cost)
        if len(examples_ok) < 3:
            examples_ok.append({
                'src': src, 'dst': dst, 'hops': len(path),
                'cost_pulse': round(cost, 1),
                'eta_sec': round(cost / PULSE_PER_SECOND, 2),
                'path_head': path[:10],
                'path_tail': path[-5:] if len(path) > 10 else [],
                'query_ms': round(ms, 3),
            })
    n = max(1, len(samples))
    return {
        'label': label,
        'pairs_tested': len(samples),
        'reachable': reach,
        'unreachable': fail,
        'reachability_pct': round(100.0 * reach / n, 2),
        'query_ms_avg': round(sum(times_ms) / n, 3),
        'query_ms_p50': round(percentile(times_ms, 50), 3),
        'query_ms_p95': round(percentile(times_ms, 95), 3),
        'query_ms_max': round(max(times_ms) if times_ms else 0, 3),
        'hops_avg':  round(sum(hops) / max(1, len(hops)), 1),
        'hops_p50':  percentile(hops, 50),
        'hops_p95':  percentile(hops, 95),
        'cost_p50_pulse': round(percentile(costs, 50), 0),
        'cost_p95_pulse': round(percentile(costs, 95), 0),
        'eta_p50_sec': round(percentile(costs, 50) / PULSE_PER_SECOND, 2),
        'examples_ok': examples_ok,
        'examples_fail': examples_fail,
    }


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="OHT Dijkstra(LineCost) 경로 탐색 검증")
    ap.add_argument('--layout',  required=True, help='layout_cache.json 경로')
    ap.add_argument('--oht',     required=True, help='OHT CSV 또는 archive.zip:inner.csv')
    ap.add_argument('--railcut', default=None,  help='RAIL_CUT CSV (선택)')
    ap.add_argument('--sample',  type=int, default=200, help='임의 OD 샘플 수 (기본 200)')
    ap.add_argument('--max-rows', type=int, default=None, help='OHT CSV 최대 스캔 행수')
    ap.add_argument('--od',      action='append', default=[],
                    help='수동 OD 지정: --od 9390,3143  (여러 번 가능)')
    ap.add_argument('--seed',    type=int, default=42)
    ap.add_argument('--report',  default=None, help='JSON 결과 저장 경로')
    ap.add_argument('--w-idle',  type=float, default=LINECOST_IDLE_VHL_PENALTY)
    ap.add_argument('--w-busy',  type=float, default=LINECOST_WORK_VHL_PENALTY)
    args = ap.parse_args()

    random.seed(args.seed)
    print("=" * 72)
    print(" OHT Dijkstra 경로 탐색 분석기")
    print("=" * 72)

    # ---- 1. layout
    t = time.perf_counter()
    nodes, edges_dist, adj_static = load_layout(args.layout)
    print(f"[1/5] 레이아웃 로드: nodes={len(nodes):,}  edges={len(edges_dist):,}  "
          f"({(time.perf_counter()-t)*1000:.0f} ms)")

    diag = graph_diagnostics(adj_static, edges_dist)
    print(f"      out-deg avg={diag['out_degree_avg']:.2f} max={diag['out_degree_max']}  "
          f"isolated={diag['isolated_nodes']}  sources={diag['pure_sources_no_incoming']}  "
          f"sinks={diag['pure_sinks_no_outgoing']}")
    print(f"      edge dist (pulse): avg={diag['edge_dist_avg']:.0f} "
          f"min={diag['edge_dist_min']:.0f}  max={diag['edge_dist_max']:.0f}")

    # ---- 2. OHT CSV
    t = time.perf_counter()
    idle, busy, oht_stats, addrs, od_pairs = aggregate_oht_csv(
        args.oht, max_rows=args.max_rows)
    print(f"[2/5] OHT CSV 집계: rows={oht_stats['rows_scanned']:,}  "
          f"vehicles={oht_stats['unique_vehicles']:,}  "
          f"unique_addr={oht_stats['unique_addresses']:,}  "
          f"({(time.perf_counter()-t)*1000:.0f} ms)")
    print(f"      STATUS hist: {oht_stats['status_histogram']}")
    print(f"      트래픽 있는 엣지: {oht_stats['unique_edges_with_traffic']:,} "
          f"(idle keys={len(idle):,}, busy keys={len(busy):,})")

    # ---- 3. RAIL_CUT
    cut = load_railcut(args.railcut)
    print(f"[3/5] RAIL_CUT: 차단 엣지 {len(cut):,}개"
          + (f"  (예: {list(cut)[:3]})" if cut else ""))

    # ---- 4. LineCost 그래프
    t = time.perf_counter()
    g, used_edges, cut_hits = build_linecost_graph(
        adj_static, edges_dist, idle, busy, cut,
        w_idle=args.w_idle, w_busy=args.w_busy)
    print(f"[4/5] LineCost 그래프: 사용 엣지={used_edges:,}  "
          f"차단 적용={cut_hits}  "
          f"(idle×{args.w_idle:.0f} + busy×{args.w_busy:.0f})  "
          f"({(time.perf_counter()-t)*1000:.0f} ms)")

    # ---- 5. Dijkstra 검증
    print(f"[5/5] Dijkstra 실행")
    report = {
        'layout_diag': diag,
        'oht_stats': oht_stats,
        'railcut_edges': len(cut),
        'graph_used_edges': used_edges,
        'weights': {'w_idle': args.w_idle, 'w_busy': args.w_busy},
        'sections': [],
    }

    # 5-1. 무작위 OD (관측된 ADDRESS 풀에서)
    layout_addrs = list(adj_static.keys())
    candidate = list(addrs & set(layout_addrs)) or layout_addrs
    n = min(args.sample, len(candidate) * (len(candidate) - 1))
    random_pairs = []
    seen = set()
    while len(random_pairs) < n and len(seen) < 5 * n:
        s = random.choice(candidate); d = random.choice(candidate)
        if s == d or (s, d) in seen:
            seen.add((s, d)); continue
        seen.add((s, d))
        random_pairs.append((s, d))
    if random_pairs:
        sec = verify_paths(g, random_pairs, label="random_OD")
        report['sections'].append(sec)
        print(f"  ▸ 무작위 OD {sec['pairs_tested']}건: "
              f"reach={sec['reachability_pct']}%  "
              f"avg={sec['query_ms_avg']}ms p95={sec['query_ms_p95']}ms  "
              f"hops p50={sec['hops_p50']} p95={sec['hops_p95']}")

    # 5-2. 사용자 지정 OD
    manual = []
    for od in args.od:
        try:
            s, d = od.split(',')
            manual.append((int(s), int(d)))
        except ValueError:
            print(f"  ! --od 형식 오류 무시: {od}")
    if manual:
        sec = verify_paths(g, manual, label="manual_OD")
        report['sections'].append(sec)
        print(f"  ▸ 수동 OD {sec['pairs_tested']}건: reach={sec['reachability_pct']}%")
        for ex in sec['examples_ok']:
            print(f"      OK  {ex['src']} → {ex['dst']}  hops={ex['hops']}  "
                  f"cost={ex['cost_pulse']:.0f} pulse  ETA={ex['eta_sec']}s  "
                  f"({ex['query_ms']} ms)")
        for fs, fd in sec['examples_fail']:
            print(f"      FAIL {fs} → {fd}  (도달 불가)")

    # 5-3. CSV 관측 OD (차량별 ADDRESS → DESTINATION 의 ADDRESS)
    # DESTINATION 컬럼은 stationId라 직접 매칭 불가 → 관측된 ADDRESS만 사용해
    # 차량별 (첫 ADDRESS, 시간상 마지막에 등장한 다른 ADDRESS)는 본 1패스에서 구하지 않음.
    # 대신 관측 ADDRESS 풀 내에서 '실제 데이터에 존재한 노드끼리'의 도달성만 검증.
    if od_pairs:
        # 각 차량의 첫 ADDRESS 끼리 연결 가능성 빠르게 점검
        firsts = list({a for _, a, _, _ in od_pairs})[:200]
        random.shuffle(firsts)
        observed_pairs = [(firsts[i], firsts[(i+1) % len(firsts)])
                          for i in range(min(args.sample, len(firsts)))]
        if observed_pairs:
            sec = verify_paths(g, observed_pairs, label="observed_addr_pairs")
            report['sections'].append(sec)
            print(f"  ▸ 관측 ADDR 쌍 {sec['pairs_tested']}건: "
                  f"reach={sec['reachability_pct']}%  "
                  f"avg={sec['query_ms_avg']}ms")

    # ---- 결과 저장
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n결과 저장: {args.report}")

    print("\n완료.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
