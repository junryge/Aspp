#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Java 소스 파일 읽어서 하나의 .py 파일로 합치기
- 로컬의 Java 파일들을 읽어서 복사 가능한 파일 생성
"""

import os
from datetime import datetime

# 읽을 Java 파일 목록
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


def main():
    # Java 파일이 있는 폴더 경로 입력
    print("=" * 70)
    print("  Java 소스 파일 → .py 파일 변환")
    print("=" * 70)
    
    folder = input("\nJava 파일들이 있는 폴더 경로 입력: ").strip()
    
    if not folder:
        folder = "."  # 현재 폴더
    
    if not os.path.isdir(folder):
        print(f"폴더를 찾을 수 없습니다: {folder}")
        return
    
    # 출력 파일 열기
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        out.write("# " + "=" * 68 + "\n")
        out.write(f"# OHT Java 소스 코드 모음\n")
        out.write(f"# 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write("# " + "=" * 68 + "\n\n")
        
        for file_name in JAVA_FILES:
            file_path = os.path.join(folder, file_name)
            
            out.write("\n# " + "=" * 68 + "\n")
            out.write(f"# {file_name}\n")
            out.write("# " + "=" * 68 + "\n")
            out.write('"""\n')
            
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    out.write(content)
                    if not content.endswith('\n'):
                        out.write('\n')
                    print(f"  ✓ {file_name} 읽기 완료")
                except Exception as e:
                    out.write(f"(읽기 오류: {e})\n")
                    print(f"  ✗ {file_name} 오류: {e}")
            else:
                out.write("(파일 없음)\n")
                print(f"  - {file_name} 없음")
            
            out.write('"""\n')
    
    print("\n" + "=" * 70)
    print(f"  완료! 저장 파일: {OUTPUT_FILE}")
    print("  이 파일을 열어서 복사하세요!")
    print("=" * 70)


if __name__ == "__main__":
    main()