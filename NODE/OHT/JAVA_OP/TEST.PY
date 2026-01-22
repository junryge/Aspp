#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Java 소스 코드 직접 입력 프로그램
- 보안상 복사 불가한 Java 소스를 직접 타이핑하여 입력
- 각 파일별로 입력받아 하나의 파일로 저장
"""

import os
from datetime import datetime

# 입력받을 Java 파일 목록
JAVA_FILES = [
    "TrafficBatch.java",
    "BranchTrafficBatch.java",
    "DataSet.java",
    "RailEdge.java",
    "AreaVhlCountBatch.java",
    "OhtMsgWorkerRunnable.java",
    "DataService.java",
    "OhtUdpListener.java",
    "DijkstraVhlRouteFind.java",
]


def print_header():
    """프로그램 헤더 출력"""
    print("=" * 70)
    print("  OHT Java 소스 코드 입력 프로그램")
    print("  - 각 파일별로 소스 코드를 직접 입력하세요")
    print("  - 입력 완료 후 'END' 입력하면 다음 파일로 넘어갑니다")
    print("  - 해당 파일 건너뛰려면 바로 'END' 입력")
    print("=" * 70)
    print()


def input_java_source(file_name: str) -> str:
    """
    하나의 Java 파일 소스 코드 입력받기
    """
    print("-" * 70)
    print(f"  [{file_name}] 소스 코드 입력")
    print("  (입력 완료 시 'END' 입력, 건너뛰기: 바로 'END')")
    print("-" * 70)
    
    lines = []
    line_num = 1
    
    while True:
        try:
            line = input(f"{line_num:4d} | ")
            
            if line.strip().upper() == "END":
                break
            
            lines.append(line)
            line_num += 1
            
        except KeyboardInterrupt:
            print("\n[Ctrl+C] 입력 중단")
            break
        except EOFError:
            break
    
    return "\n".join(lines)


def save_to_file(sources: dict, output_path: str):
    """
    모든 소스를 하나의 파일로 저장
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# " + "=" * 68 + "\n")
        f.write(f"# OHT Java 소스 코드 모음\n")
        f.write(f"# 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# " + "=" * 68 + "\n\n")
        
        for file_name, source in sources.items():
            f.write("\n")
            f.write("# " + "=" * 68 + "\n")
            f.write(f"# {file_name}\n")
            f.write("# " + "=" * 68 + "\n")
            f.write('"""\n')
            
            if source.strip():
                f.write(source)
                if not source.endswith("\n"):
                    f.write("\n")
            else:
                f.write("(입력 없음)\n")
            
            f.write('"""\n')
    
    print(f"\n[저장 완료] {output_path}")


def main():
    """메인 함수"""
    print_header()
    
    # 저장할 파일 경로
    output_path = input("저장할 파일 경로 (기본: oht_java_sources.py): ").strip()
    if not output_path:
        output_path = "oht_java_sources.py"
    
    print()
    
    # 각 파일별 소스 입력
    sources = {}
    
    for i, file_name in enumerate(JAVA_FILES, 1):
        print(f"\n[{i}/{len(JAVA_FILES)}]")
        source = input_java_source(file_name)
        sources[file_name] = source
        
        if source.strip():
            line_count = len(source.split("\n"))
            print(f"  → {line_count}줄 입력됨")
        else:
            print("  → 건너뜀")
    
    # 파일 저장
    save_to_file(sources, output_path)
    
    # 요약 출력
    print("\n" + "=" * 70)
    print("  입력 요약")
    print("=" * 70)
    for file_name, source in sources.items():
        if source.strip():
            line_count = len(source.split("\n"))
            print(f"  ✓ {file_name}: {line_count}줄")
        else:
            print(f"  - {file_name}: (건너뜀)")
    print("=" * 70)


if __name__ == "__main__":
    main()