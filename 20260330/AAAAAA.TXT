#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_sender.py - OHT 데이터 전송

사용법:
    python data_sender.py        → 메뉴 선택
    python data_sender.py play   → 1번: TXT 재생 (5분 이동)
    python data_sender.py paste  → 2번: 붙여넣기 모드

TXT 데이터 생성:
    python generate_5min_data.py → oht_5v_data.txt 생성 (sim_server 필요)
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

SERVER_URL = "http://localhost:10003"
INTERVAL   = 0.5
DEFAULT_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oht_5v_data.txt")


def send(messages):
    payload = json.dumps({"messages": messages}).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/api/oht-raw",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        return {"status": "fail", "error": str(e)}


def clean_line(line):
    line = line.strip()
    if line.startswith('"'):
        line = line.lstrip('"')
    if line.endswith('",') or line.endswith('"'):
        line = line.rstrip(',').rstrip('"')
    return line.strip()


# ============================================================
# 1번: TXT 파일 재생 모드 (5분 이동 데이터)
# ============================================================
def play_mode(filepath=None):
    """TXT에서 5분 이동 데이터를 읽어 0.5초 간격으로 전송"""
    if filepath is None:
        filepath = DEFAULT_DATA_FILE

    if not os.path.exists(filepath):
        print(f"오류: 파일 없음 → {filepath}")
        print(f"먼저 데이터를 생성하세요:")
        print(f"  python generate_5min_data.py")
        return

    # TXT 파일 파싱: 빈 줄로 구분된 배치(틱) 단위로 읽기
    print(f"데이터 파일: {filepath}")
    batches = []
    current_batch = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                # 빈 줄 = 배치 구분
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
            elif "2,OHT," in stripped:
                current_batch.append(stripped)
        # 마지막 배치
        if current_batch:
            batches.append(current_batch)

    if not batches:
        print("오류: OHT 메시지가 없습니다")
        return

    total_ticks = len(batches)
    vehicles_per_tick = len(batches[0])
    duration = total_ticks * INTERVAL
    print(f"  {vehicles_per_tick}대 × {total_ticks}틱 = {duration:.0f}초 ({duration/60:.1f}분)")
    print()
    print("재생 시작!")
    print("Ctrl+C 종료")
    print("-" * 50)

    start_time = time.time()
    for tick, batch in enumerate(batches):
        result = send(batch)
        elapsed = time.time() - start_time
        remaining = duration - elapsed

        if tick % 10 == 0 or tick < 3:
            # 첫 번째 메시지에서 차량 정보 추출
            f = batch[0].split(",")
            vid = f[2]
            cur = f[7]
            dist = f[8]
            nxt = f[9]
            status = result.get("status", "?")
            print(f"[{tick+1:3d}/{total_ticks}] {status} 남은시간 {int(remaining)}초  "
                  f"{vid}: {cur}→{nxt} dist={dist}")

        time.sleep(INTERVAL)

    print(f"\n재생 완료! {total_ticks}틱, {int(time.time()-start_time)}초")


# ============================================================
# 2번: 붙여넣기 모드
# ============================================================
def paste_mode():
    print("붙여넣기 모드: OHT 메시지 붙여넣고 빈 줄에서 엔터 → 전송!")
    print("q = 종료")
    print("-" * 50)

    count = 0
    while True:
        print(f"\n[{count+1}] 메시지 붙여넣기:")

        messages = []
        while True:
            try:
                line = input()
            except EOFError:
                return

            if line.strip().lower() == "q":
                return

            cleaned = clean_line(line)
            if not cleaned:
                break

            if "2,OHT," in cleaned:
                idx = cleaned.index("2,OHT,")
                messages.append(cleaned[idx:])

        if not messages:
            continue

        result = send(messages)
        count += 1

        status = result.get("status", "?")
        updated = result.get("updated", [])
        print(f"  {status} updated={updated} ({len(messages)}대)")
        for m in messages:
            f = m.split(",")
            print(f"    {f[2]}: node={f[7]} dist={f[8]} → {f[9]}")


# ============================================================
def main():
    print("=" * 50)
    print(f"OHT data_sender → {SERVER_URL}")
    print("=" * 50)

    if len(sys.argv) > 1 and sys.argv[1] == "play":
        fp = sys.argv[2] if len(sys.argv) > 2 else None
        play_mode(fp)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "paste":
        paste_mode()
        return

    print()
    print("  1. TXT 재생 (oht_5v_data.txt → 5분 이동)")
    print("  2. 붙여넣기 (직접 입력)")
    print()
    try:
        sel = input("선택 (1 or 2): ").strip()
    except EOFError:
        return

    print()
    if sel == "2":
        paste_mode()
    else:
        play_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n종료.")
