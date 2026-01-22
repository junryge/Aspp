#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Java 소스 코드 타이핑 → 파일 저장 프로그램
- 보안상 복사 불가한 Java 소스를 타이핑하면 파일로 저장
- 저장된 파일에서 복사 가능
"""

import os
from datetime import datetime

# 입력받을 Java 파일 목록
JAVA_FILES = [
    "CnvEdge.java",
    "DataService.java",
    "DataSet.java",
    "DijkstraVhlRouteFind.java",
    "LongEdge.java",
    "OhtMsgWorkerRunnable.java",
    "OhtUdpListener.java",
    "RailEdge.java",
    "Station.java",
    "StkRmEdge.java",
    "TrafficBatch.java",
    "TransferEdge.java",
    "Vhl.java",
]

OUTPUT_FILE = "oht_java_sources.py"


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(file_name, file_idx, total):
    print("=" * 70)
    print(f"  [{file_idx}/{total}] {file_name}")
    print("=" * 70)
    print("  타이핑하세요. 완료되면 빈 줄에서 Enter 두 번 또는 ':end' 입력")
    print("-" * 70)


def input_source(file_name, file_idx, total):
    """한 파일의 소스 입력받기"""
    clear_screen()
    print_header(file_name, file_idx, total)
    
    lines = []
    empty_count = 0
    line_num = 1
    
    while True:
        try:
            line = input(f"{line_num:4d}| ")
            
            # 종료 조건
            if line.strip().lower() == ':end':
                break
            if line == '':
                empty_count += 1
                if empty_count >= 2:
                    # 마지막 빈 줄 제거
                    while lines and lines[-1] == '':
                        lines.pop()
                    break
            else:
                empty_count = 0
            
            lines.append(line)
            line_num += 1
            
        except KeyboardInterrupt:
            print("\n중단됨")
            break
        except EOFError:
            break
    
    return '\n'.join(lines)


def save_all(sources):
    """모든 소스를 하나의 파일로 저장"""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("# " + "=" * 68 + "\n")
        f.write(f"# OHT Java 소스 코드 모음\n")
        f.write(f"# 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# " + "=" * 68 + "\n\n")
        
        for file_name, source in sources.items():
            f.write("\n# " + "=" * 68 + "\n")
            f.write(f"# {file_name}\n")
            f.write("# " + "=" * 68 + "\n")
            f.write('"""\n')
            if source.strip():
                f.write(source)
                if not source.endswith('\n'):
                    f.write('\n')
            else:
                f.write("(입력 없음)\n")
            f.write('"""\n')
    
    print(f"\n저장 완료: {OUTPUT_FILE}")


def main():
    clear_screen()
    print("=" * 70)
    print("  Java 소스 타이핑 → 파일 저장 프로그램")
    print("=" * 70)
    print(f"\n  입력할 파일: {len(JAVA_FILES)}개")
    for i, f in enumerate(JAVA_FILES, 1):
        print(f"    {i:2d}. {f}")
    print(f"\n  저장 위치: {OUTPUT_FILE}")
    print("\n  Enter 누르면 시작...")
    input()
    
    sources = {}
    
    for i, file_name in enumerate(JAVA_FILES, 1):
        source = input_source(file_name, i, len(JAVA_FILES))
        sources[file_name] = source
        
        if source.strip():
            line_count = len(source.split('\n'))
            print(f"\n  → {line_count}줄 입력됨")
        else:
            print(f"\n  → 건너뜀")
        
        # 중간 저장
        save_all(sources)
        
        if i < len(JAVA_FILES):
            print("\n  Enter 누르면 다음 파일...")
            input()
    
    # 최종 요약
    clear_screen()
    print("=" * 70)
    print("  완료!")
    print("=" * 70)
    for file_name, source in sources.items():
        if source.strip():
            line_count = len(source.split('\n'))
            print(f"  ✓ {file_name}: {line_count}줄")
        else:
            print(f"  - {file_name}: (건너뜀)")
    print("=" * 70)
    print(f"\n  저장 파일: {OUTPUT_FILE}")
    print("  이 파일을 열어서 복사하세요!")


if __name__ == "__main__":
    main()