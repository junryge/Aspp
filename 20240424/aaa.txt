#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dijkstra_formula_check.py — Dijkstra 수식 검증 (공식 ↔ 코드 ↔ 결과)

수식이 맞는지 사람이 직접 확인할 수 있도록:
  1) 표준 Dijkstra 의사코드와 LineCost 공식 인쇄
  2) 실제 OD 한 건에 대해 매 EXTRACT-MIN / RELAX 단계를 트레이스
  3) 결과 경로의 비용을 처음부터 다시 합산해 d[t] 와 동일한지 검증
  4) 비음 가중치 조건 (w ≥ 0) 을 모든 엣지에 대해 확인

사용 예:
  python dijkstra_formula_check.py ^
      --layout cache\\M16A_BR_layout_cache.json ^
      --oht    DES\\20260429_UDP.CSV ^
      --src 2189 --dst 3052 ^
      --trace 30        (매 단계 로그 30개까지)
"""

import argparse
import heapq
import math
import sys

from dijkstra_analyzer import (
    load_layout, aggregate_oht_csv, load_railcut, build_linecost_graph,
    LINECOST_IDLE_VHL_PENALTY, LINECOST_WORK_VHL_PENALTY, PULSE_PER_SECOND,
)


def _formula_banner():
    return (
        "\n================================================================\n"
        " Dijkstra 표준 의사코드\n"
        "================================================================\n"
        "INIT:\n"
        "    d[s] = 0,  d[v] = +∞  (∀ v≠s)\n"
        "    π[v] = NULL\n"
        "    S = ∅,  Q = V\n"
        "\n"
        "LOOP:\n"
        "    while Q ≠ ∅:\n"
        "        u ← EXTRACT-MIN(Q)            # d 최소 노드\n"
        "        S ← S ∪ {u}\n"
        "        for each v ∈ Adj[u]:\n"
        "            RELAX(u, v, w):\n"
        "                if d[u] + w(u,v) < d[v]:\n"
        "                    d[v] ← d[u] + w(u,v)\n"
        "                    π[v] ← u\n"
        "\n"
        "PATH(s,t) = reverse([t, π[t], π[π[t]], ..., s])\n"
        "\n"
        "비음수 가중치 정리:\n"
        "    w(u,v) ≥ 0  ⇒  EXTRACT-MIN으로 뽑힌 u는 d[u] = δ(s,u) 확정\n"
        "\n"
        "================================================================\n"
        " LineCost 가중치 (현 시스템)\n"
        "================================================================\n"
        "w(u,v) = distance_pulse(u,v)\n"
        f"       + idle_count(u,v) × {LINECOST_IDLE_VHL_PENALTY}\n"
        f"       + busy_count(u,v) × {LINECOST_WORK_VHL_PENALTY}\n"
        "\n"
        f"시간 환산: T(sec) = w_total / {PULSE_PER_SECOND:,} pulse/s\n"
        "================================================================\n"
    )


def dijkstra_traced(adj, src, dst, trace_limit=30):
    """매 EXTRACT-MIN / RELAX 를 출력하면서 실행."""
    d = {src: 0.0}
    pi = {}
    pq = [(0.0, src)]
    extract_n = 0
    relax_n = 0
    update_n = 0
    print(f"\nINIT  : d[{src}]=0, push (0, {src})")
    while pq:
        cu, u = heapq.heappop(pq)
        if cu > d.get(u, math.inf):
            # stale
            continue
        extract_n += 1
        if extract_n <= trace_limit:
            print(f"EXTR  #{extract_n:>3}: u={u}  d[u]={cu:.1f}"
                  + ("  ← 도착!" if u == dst else ""))
        elif extract_n == trace_limit + 1:
            print("  ... (이후 EXTRACT 로그 생략)")
        if u == dst:
            return d, pi, extract_n, relax_n, update_n
        for v, w in adj.get(u, []):
            relax_n += 1
            nd = cu + w
            old = d.get(v, math.inf)
            if nd < old:
                d[v] = nd
                pi[v] = u
                heapq.heappush(pq, (nd, v))
                update_n += 1
                if extract_n <= trace_limit:
                    print(f"  RELAX: ({u}→{v})  w={w:.1f}"
                          f"  d[u]+w={nd:.1f}  <  d[v](old)={old if old<math.inf else '∞'}"
                          f"  ⇒ d[v]←{nd:.1f}, π[v]←{u}")
    return d, pi, extract_n, relax_n, update_n


def verify_path_sum(path, edge_weight):
    """경로 비용을 엣지 단위로 다시 합산하여 d[t]와 비교."""
    total = 0.0
    breakdown = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        w = edge_weight((u, v))
        total += w
        breakdown.append((u, v, w))
    return total, breakdown


def main():
    ap = argparse.ArgumentParser(description="Dijkstra 공식 검증기")
    ap.add_argument('--layout', required=True)
    ap.add_argument('--oht',    default=None,
                    help='UDP CSV (없으면 가중치 = 거리만)')
    ap.add_argument('--railcut', default=None)
    ap.add_argument('--max-rows', type=int, default=None)
    ap.add_argument('--w-idle', type=float, default=LINECOST_IDLE_VHL_PENALTY)
    ap.add_argument('--w-busy', type=float, default=LINECOST_WORK_VHL_PENALTY)
    ap.add_argument('--src',   type=int, required=True)
    ap.add_argument('--dst',   type=int, required=True)
    ap.add_argument('--trace', type=int, default=30,
                    help='매 단계 로그 출력 개수 (기본 30)')
    args = ap.parse_args()

    print(_formula_banner())

    # 1. 그래프 빌드
    print("[1] 정적 레이아웃 로드")
    nodes, edges_dist, adj_static = load_layout(args.layout)
    print(f"    nodes={len(nodes):,}  edges={len(edges_dist):,}")

    if args.oht:
        print("\n[2] UDP CSV 집계 (실측 idle/busy 카운트)")
        idle, busy, oht_stats, _, _ = aggregate_oht_csv(
            args.oht, max_rows=args.max_rows)
        print(f"    rows={oht_stats['rows_scanned']:,}  "
              f"vehicles={oht_stats['unique_vehicles']:,}")
        print(f"    STATUS hist: {oht_stats['status_histogram']}")
        cut = load_railcut(args.railcut)
        print(f"\n[3] RAIL_CUT 차단 엣지: {len(cut)}")
        adj, used, _ = build_linecost_graph(
            adj_static, edges_dist, idle, busy, cut,
            w_idle=args.w_idle, w_busy=args.w_busy)
        print(f"\n[4] LineCost 그래프 빌드: 엣지 {used:,}개")

        def edge_weight(uv):
            d = edges_dist.get(uv, 1.0)
            return max(d, 1.0) + idle[uv] * args.w_idle + busy[uv] * args.w_busy
    else:
        from collections import defaultdict
        adj = defaultdict(list)
        for u, lst in adj_static.items():
            for v in lst:
                adj[u].append((v, edges_dist.get((u, v), 1.0)))
        print("\n[2~4] UDP 미사용 — 가중치 = distance_pulse 만")

        def edge_weight(uv):
            return edges_dist.get(uv, 1.0)

    # 5. 비음수 조건 검증
    print(f"\n[5] 비음 가중치 조건 (w ≥ 0) 검증")
    weights = [w for nbrs in adj.values() for _, w in nbrs]
    neg = sum(1 for w in weights if w < 0)
    zero = sum(1 for w in weights if w == 0)
    print(f"    엣지 수={len(weights):,}  음수={neg}  0={zero}  "
          f"min={min(weights):.1f}  max={max(weights):.1f}")
    if neg > 0:
        print("    ❌ 음수 가중치 존재 → Dijkstra 부적합 (Bellman-Ford 필요)")
        return 1
    print("    ✅ 모든 가중치 ≥ 0 → Dijkstra 정확성 정리 적용 가능")

    # 6. 트레이스 실행
    print(f"\n[6] Dijkstra 실행 — src={args.src} → dst={args.dst}")
    print(f"    (트레이스 상한: {args.trace} 단계)")
    d, pi, n_extract, n_relax, n_update = dijkstra_traced(
        adj, args.src, args.dst, trace_limit=args.trace)

    if args.dst not in d:
        print(f"\n  ❌ {args.src} → {args.dst} 도달 불가")
        return 1

    # 7. 경로 복원
    path = [args.dst]
    while path[-1] in pi:
        path.append(pi[path[-1]])
    path.reverse()
    print(f"\n[7] 경로 복원 PATH(s,t) = reverse([t, π[t], ..., s])")
    print(f"    홉수 = {len(path)}")
    print(f"    처음 10홉: {path[:10]}")
    if len(path) > 15:
        print(f"    끝   5홉: {path[-5:]}")

    # 8. 검증: 경로 비용 재합산 == d[t]
    print(f"\n[8] 검증: 경로 비용 재합산 vs d[t]")
    total, bd = verify_path_sum(path, edge_weight)
    print(f"    Σ w(u,v) along PATH = {total:.1f}")
    print(f"    d[{args.dst}]       = {d[args.dst]:.1f}")
    diff = abs(total - d[args.dst])
    if diff < 1e-6:
        print(f"    ✅ 일치 (오차 {diff:.2e}) — 구현이 공식과 동일")
    else:
        print(f"    ❌ 불일치 (차이 {diff:.3f}) — 구현 버그 가능")

    # 9. 통계
    print(f"\n[9] 통계")
    print(f"    EXTRACT-MIN 횟수 = {n_extract}")
    print(f"    RELAX 시도        = {n_relax}")
    print(f"    d/π 갱신 횟수     = {n_update}")
    print(f"    d[t] = {d[args.dst]:.1f} pulse"
          f"  ≈ {d[args.dst]/PULSE_PER_SECOND:.2f} 초")

    # 10. 첫 5개·마지막 5개 엣지 단가 표시
    print(f"\n[10] 경로 엣지 단가 샘플")
    print(f"     {'from':>6} {'to':>6} {'w(u,v)':>12}")
    for u, v, w in bd[:5]:
        print(f"     {u:>6} {v:>6} {w:>12.1f}")
    if len(bd) > 10:
        print(f"     ... ({len(bd)-10} 홉 생략)")
    for u, v, w in bd[-5:]:
        print(f"     {u:>6} {v:>6} {w:>12.1f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
