#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OHT_3초_송출_테스트.py — 3초 tumbling 윈도우 집계 동작 확인용 독립 테스트

다른 코드를 건드리지 않고 여기 안에 SpeedForwarder 의 참고 구현을 두어
  - 합성 샘플 (인위적 timing) 으로 기본 동작 검증
  - 실데이터 (M16A_BR_DATA) 로 end-to-end 흐름 확인
두 가지 방식으로 "3초에 한 번 평균/최소/최대/카운트가 맞게 나오는가" 를 본다.

실행:
    cd WORD_MODEL/WORLD_SIM   # velocity_tracker import 를 위해
    python ../OHS_DATA_MD/OHT_3초_송출_테스트.py

독립 모드 (합성 샘플만):
    python ../OHS_DATA_MD/OHT_3초_송출_테스트.py --synthetic-only
"""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "M16A_BR_DATA"
HID_CSV = DATA_DIR / "LOGPRESSO_HID_INOUT_20260421.csv"
ZIP_PATH = DATA_DIR / "20260421.zip"
OHT_CSV_NAME = "LOGPRESSO_oht_data_m16BR_20260421.csv"
LAYOUT_JSON = HERE.parent / "OHT_MAP" / "cache" / "M16A_BR_layout_cache.json"


# ============================================================
# 참고 구현 — SpeedForwarder (독립 구현, 외부 의존 0)
# ============================================================
@dataclass
class SpeedForwarder:
    """3초 tumbling 윈도우 버퍼 + 평균/최소/최대/카운트 송출.

    실운영에선 WORLD_SIM 밑으로 이동시키지만, 테스트 파일이라 여기 둔다.
    """
    window_sec: float = 3.0
    _vhl: Dict[str, List[Tuple[datetime, float]]] = field(
        default_factory=lambda: defaultdict(list))
    _zone: Dict[Tuple[int, int], List[Tuple[datetime, float]]] = field(
        default_factory=lambda: defaultdict(list))
    _window_start: Optional[datetime] = None

    def record_vhl(self, t: datetime, vid: str, speed: Optional[float]) -> None:
        if speed is None:
            return
        if self._window_start is None:
            self._window_start = t
        self._vhl[vid].append((t, speed))

    def record_zone(self, t: datetime, from_hid: int, to_hid: int,
                    ffs: Optional[float]) -> None:
        if ffs is None:
            return
        if self._window_start is None:
            self._window_start = t
        self._zone[(from_hid, to_hid)].append((t, ffs))

    @staticmethod
    def _stats(samples: List[Tuple[datetime, float]]) -> dict:
        vals = [s for _, s in samples]
        return {
            "avg": round(sum(vals) / len(vals), 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "n":   len(vals),
        }

    def emit(self, now: datetime) -> Optional[dict]:
        """윈도우 경계 넘었으면 집계 반환·flush, 아니면 None."""
        if self._window_start is None:
            return None
        if (now - self._window_start).total_seconds() < self.window_sec:
            return None

        t_end = self._window_start + timedelta(seconds=self.window_sec)
        out = {
            "t_start": self._window_start.isoformat(),
            "t_end":   t_end.isoformat(),
            "vhl_speed":  {v: self._stats(s) for v, s in self._vhl.items() if s},
            "zone_speed": {f"{a}_{b}": self._stats(s)
                           for (a, b), s in self._zone.items() if s},
        }
        self._vhl.clear()
        self._zone.clear()
        self._window_start = t_end
        return out


# ============================================================
# Part 1 — 합성 샘플 테스트 (어떤 환경에서도 돌아감)
# ============================================================
def expect(name: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        sys.exit(1)


def test_synthetic() -> None:
    print("\n" + "=" * 70)
    print("Part 1. 합성 샘플 테스트 (기본 동작 검증)")
    print("=" * 70)

    fwd = SpeedForwarder(window_sec=3.0)
    t0 = datetime(2026, 4, 21, 14, 0, 0)

    # 첫 윈도우 [t0, t0+3): 차량 2대, ZONE 1곳
    fwd.record_vhl(t0,                            "BV0103", 200.0)
    fwd.record_vhl(t0 + timedelta(seconds=1.0),  "BV0103", 240.0)
    fwd.record_vhl(t0 + timedelta(seconds=2.5),  "BV0103", 260.0)
    fwd.record_vhl(t0 + timedelta(seconds=1.5),  "BV0085", 180.0)
    fwd.record_zone(t0 + timedelta(seconds=0.5), 19, 13, 152.0)
    fwd.record_zone(t0 + timedelta(seconds=1.7), 19, 13, 154.0)

    print("\n  → 경계 이전 emit 호출 (윈도우 안)")
    expect("t<window_end 에선 emit None 반환",
           fwd.emit(t0 + timedelta(seconds=2.9)) is None)

    print("\n  → 경계 도달 후 emit 호출")
    payload = fwd.emit(t0 + timedelta(seconds=3.0))
    assert payload is not None
    print("    payload =", json.dumps(payload, indent=6, ensure_ascii=False))

    bv = payload["vhl_speed"]["BV0103"]
    expect("BV0103 n=3",             bv["n"] == 3)
    expect("BV0103 min=200.0",        bv["min"] == 200.0)
    expect("BV0103 max=260.0",        bv["max"] == 260.0)
    expect("BV0103 avg≈233.333",      abs(bv["avg"] - 233.333) < 0.01,
           f"got {bv['avg']}")

    bv85 = payload["vhl_speed"]["BV0085"]
    expect("BV0085 n=1, avg=min=max", bv85["n"] == 1
           and bv85["avg"] == bv85["min"] == bv85["max"] == 180.0)

    zone = payload["zone_speed"]["19_13"]
    expect("ZONE 19_13 avg=153.0",    zone["avg"] == 153.0)
    expect("ZONE 19_13 n=2",          zone["n"] == 2)

    print("\n  → emit 후 두 번째 호출 (flush 확인)")
    expect("즉시 재 emit 은 None (버퍼 비었음)",
           fwd.emit(t0 + timedelta(seconds=3.0)) is None)

    # 두 번째 윈도우 [t0+3, t0+6)
    fwd.record_vhl(t0 + timedelta(seconds=4.0), "BV0103", 300.0)
    fwd.record_vhl(t0 + timedelta(seconds=5.5), "BV0103", 300.0)

    print("\n  → 두 번째 윈도우 emit")
    p2 = fwd.emit(t0 + timedelta(seconds=6.0))
    assert p2 is not None
    expect("두 번째 윈도우 t_start 가 첫 t_end 와 연속",
           p2["t_start"] == (t0 + timedelta(seconds=3.0)).isoformat())
    expect("두 번째 윈도우 avg=300.0",
           p2["vhl_speed"]["BV0103"]["avg"] == 300.0)
    expect("두 번째 윈도우에 ZONE 없음 (키 생략)",
           "zone_speed" in p2 and p2["zone_speed"] == {})

    # None 샘플은 무시
    print("\n  → None 샘플/결측 스킵 확인")
    fwd2 = SpeedForwarder(window_sec=3.0)
    fwd2.record_vhl(t0, "BV0001", None)                   # 무시 (window_start 설정 X)
    fwd2.record_vhl(t0 + timedelta(seconds=1), "BV0001", 100.0)  # window_start=t0+1
    fwd2.record_zone(t0, 1, 2, None)                      # 무시
    p3 = fwd2.emit(t0 + timedelta(seconds=4.0))           # window_start+3 = t0+4
    assert p3 is not None, f"expected payload at t0+4, got None"
    expect("None 스피드 제외 후 n=1",
           p3["vhl_speed"]["BV0001"]["n"] == 1)
    expect("None FFS 제외 후 zone 없음",
           p3["zone_speed"] == {})

    print("\n  ✅ Part 1 모든 단언 통과")


# ============================================================
# Part 2 — 실데이터 end-to-end 테스트
# ============================================================
def parse_time(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip().strip('"')
    for fmt in ("%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=None)
        except ValueError:
            continue
    return None


def load_layout() -> tuple[dict, dict]:
    with open(LAYOUT_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    edge_map = {tuple(map(int, k.split(","))): float(v)
                for k, v in d.get("edges", {}).items()}
    adj_map = {int(k): list(v) for k, v in d.get("adj", {}).items()}
    return edge_map, adj_map


def test_realdata(vid: str = "BV0103", max_windows: int = 5) -> None:
    print("\n" + "=" * 70)
    print(f"Part 2. 실데이터 end-to-end 테스트 (차량 {vid})")
    print("=" * 70)

    if not (HID_CSV.exists() and ZIP_PATH.exists() and LAYOUT_JSON.exists()):
        print("  [SKIP] 실데이터 파일 없음 — Part 2 생략")
        return

    # WORLD_SIM 의 VelocityTracker import
    sys.path.insert(0, str(HERE.parent / "WORLD_SIM"))
    from velocity_tracker import VelocityTracker

    edge_map, adj_map = load_layout()
    tracker = VelocityTracker(edge_dist_map=edge_map, adj_map=adj_map)
    fwd = SpeedForwarder(window_sec=3.0)

    # ── OHT raw: vid 행을 시간순으로 읽어 tracker + forwarder 에 주입
    oht_rows: List[Tuple[datetime, int, int, int]] = []
    with zipfile.ZipFile(str(ZIP_PATH), "r") as zf:
        with zf.open(OHT_CSV_NAME) as raw:
            text = (line.decode("utf-8", errors="replace") for line in raw)
            for row in csv.DictReader(text):
                if row.get("VEHICLE") != vid:
                    continue
                t = parse_time(row.get("_time", ""))
                if not t:
                    continue
                try:
                    oht_rows.append((t, int(row["ADDRESS"]),
                                     int(row["NEXT_ADDRESS"]),
                                     int(row["DISTANCE"])))
                except (ValueError, KeyError):
                    continue
    oht_rows.sort()

    # ── HID_INOUT: vid 행
    hid_rows: List[Tuple[datetime, int, int, float]] = []
    with open(HID_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("VHL_ID") != vid:
                continue
            t = parse_time(row.get("_time", ""))
            if not t:
                continue
            try:
                hid_rows.append((t, int(row["FROM_HIDID"]),
                                 int(row["TO_HIDID"]),
                                 float(row["FREE_FLOW_SPEED"])))
            except (ValueError, KeyError):
                continue
    hid_rows.sort()

    print(f"\n  데이터 로드: OHT raw {len(oht_rows)}행, HID {len(hid_rows)}행")

    if not oht_rows:
        print("  [SKIP] OHT raw 데이터 없음")
        return

    # ── 시간순 병합 진행. 매 샘플 직후 현재 시각 기준으로 emit 시도
    oi = hi = 0
    payloads: List[dict] = []

    def next_event_time() -> Optional[datetime]:
        candidates = []
        if oi < len(oht_rows):
            candidates.append(oht_rows[oi][0])
        if hi < len(hid_rows):
            candidates.append(hid_rows[hi][0])
        return min(candidates) if candidates else None

    while True:
        nt = next_event_time()
        if nt is None:
            break
        # 먼저 현재 시각으로 emit 체크
        payload = fwd.emit(nt)
        if payload:
            payloads.append(payload)
            if len(payloads) >= max_windows:
                break

        # 이벤트 처리
        if oi < len(oht_rows) and oht_rows[oi][0] == nt:
            t, addr, nxt, dist = oht_rows[oi]
            speed = tracker.update(vid, t, addr, nxt, dist)
            fwd.record_vhl(t, vid, speed)
            oi += 1
        elif hi < len(hid_rows) and hid_rows[hi][0] == nt:
            t, fr, to, ffs = hid_rows[hi]
            fwd.record_zone(t, fr, to, ffs)
            hi += 1

    # ── 출력
    print(f"\n  ▶ 앞 {len(payloads)} 개 3초 윈도우 payload:")
    for i, p in enumerate(payloads, 1):
        print(f"\n  [window {i}]  {p['t_start']}  →  {p['t_end']}")
        for v, st in p["vhl_speed"].items():
            print(f"    OHT {v}   avg={st['avg']:>7.2f}  "
                  f"min={st['min']:>7.2f}  max={st['max']:>7.2f}  "
                  f"(km/h avg={st['avg']*0.06:.2f})  n={st['n']}")
        for z, st in p["zone_speed"].items():
            print(f"    ZONE {z}  avg={st['avg']:>7.2f}  "
                  f"min={st['min']:>7.2f}  max={st['max']:>7.2f}  n={st['n']}")
        if not p["vhl_speed"] and not p["zone_speed"]:
            print("    (샘플 없음)")

    # ── 불변조건 체크
    print("\n  ▶ 불변조건 체크")
    total_windows = len(payloads)
    for p in payloads:
        dt = (datetime.fromisoformat(p["t_end"])
              - datetime.fromisoformat(p["t_start"])).total_seconds()
        expect(f"윈도우 길이 = 3.0s ({p['t_start']})", abs(dt - 3.0) < 1e-6)
        for v, st in p["vhl_speed"].items():
            expect(f"{v} min≤avg≤max", st["min"] <= st["avg"] <= st["max"])
            expect(f"{v} n≥1",         st["n"] >= 1)

    print(f"\n  ✅ Part 2 — {total_windows} 개 윈도우 모든 불변조건 통과")


# ============================================================
# Entry
# ============================================================
def main() -> None:
    synthetic_only = "--synthetic-only" in sys.argv

    print("3초 송출 집계 동작 테스트")
    print(f"  SpeedForwarder.window_sec = 3.0")

    test_synthetic()

    if not synthetic_only:
        test_realdata()

    print("\n" + "=" * 70)
    print("전체 테스트 통과 ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()
