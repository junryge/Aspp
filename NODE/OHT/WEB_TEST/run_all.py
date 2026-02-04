#!/usr/bin/env python3
"""
MES/MCS/SECS-GEM/OHT 통합 테스트 실행

모든 서버를 한번에 시작합니다.
"""

import subprocess
import sys
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SERVERS = [
    {
        "name": "SECS/GEM Mock",
        "script": "secs_gem_mock.py",
        "port": 10012,
        "color": "\033[95m"  # Magenta
    },
    {
        "name": "OHT Simulator",
        "script": "simulator_server_3D_B_TEST.py",
        "port": 10003,
        "color": "\033[94m"  # Blue
    },
    {
        "name": "MCS Server",
        "script": "mcs_server.py",
        "port": 10011,
        "color": "\033[92m"  # Green
    },
    {
        "name": "MES Server",
        "script": "mes_server.py",
        "port": 10010,
        "color": "\033[93m"  # Yellow
    },
]

RESET = "\033[0m"

def main():
    print("=" * 60)
    print("  MES/MCS/SECS-GEM/OHT 통합 테스트 환경")
    print("=" * 60)
    print()
    print("  시작할 서버:")
    for s in SERVERS:
        print(f"    {s['color']}● {s['name']}{RESET}: http://localhost:{s['port']}")
    print()
    print("=" * 60)
    print()

    processes = []

    try:
        for server in SERVERS:
            script_path = os.path.join(SCRIPT_DIR, server["script"])
            print(f"{server['color']}[{server['name']}]{RESET} 시작 중... (Port {server['port']})")

            proc = subprocess.Popen(
                [sys.executable, script_path],
                cwd=SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True
            )
            processes.append((server, proc))
            time.sleep(2)  # 서버 시작 대기

        print()
        print("=" * 60)
        print("  모든 서버 시작 완료!")
        print("=" * 60)
        print()
        print("  접속 URL:")
        print(f"    \033[93m● MES (Transport 입력)\033[0m: http://localhost:10010")
        print(f"    \033[92m● MCS (배차 관리)\033[0m:     http://localhost:10011")
        print(f"    \033[95m● SECS/GEM (설비)\033[0m:     http://localhost:10012")
        print(f"    \033[94m● OHT Simulator\033[0m:       http://localhost:10003")
        print()
        print("  테스트 시나리오:")
        print("    1. MES (http://localhost:10010) 접속")
        print("    2. Transport 요청 입력 (From/To Station)")
        print("    3. MCS → OHT 배차 확인")
        print("    4. SECS/GEM Load/Unload 이벤트 확인")
        print()
        print("  종료: Ctrl+C")
        print("=" * 60)

        # 출력 모니터링
        while True:
            for server, proc in processes:
                line = proc.stdout.readline()
                if line:
                    print(f"{server['color']}[{server['name'][:3]}]{RESET} {line.rstrip()}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n서버 종료 중...")
        for server, proc in processes:
            proc.terminate()
            print(f"  {server['name']} 종료")

        print("모든 서버가 종료되었습니다.")

if __name__ == "__main__":
    main()