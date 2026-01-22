#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Java 소스 파일 → 한 줄로 변환 (보안 우회)
- 줄바꿈을 특수문자로 치환하여 Java 코드로 인식 못하게 함
- 나중에 다시 복원 가능
"""

import os
from datetime import datetime

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

# 줄바꿈 대체 문자 (나중에 복원용)
LINE_BREAK = "§§§"

OUTPUT_FILE = "oht_java_oneline.txt"
RESTORE_FILE = "oht_java_restored.py"


def convert_to_oneline(folder):
    """Java 파일들을 한 줄로 변환"""
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        for file_name in JAVA_FILES:
            file_path = os.path.join(folder, file_name)
            
            out.write(f"###FILE:{file_name}###")
            
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # 줄바꿈을 특수문자로 치환
                    oneline = content.replace('\n', LINE_BREAK)
                    out.write(oneline)
                    print(f"  ✓ {file_name}")
                except Exception as e:
                    out.write(f"ERROR:{e}")
                    print(f"  ✗ {file_name}: {e}")
            else:
                out.write("FILE_NOT_FOUND")
                print(f"  - {file_name} 없음")
            
            out.write("###END###")
    
    print(f"\n저장: {OUTPUT_FILE}")


def restore_from_oneline():
    """한 줄 파일을 원래 형태로 복원"""
    
    if not os.path.isfile(OUTPUT_FILE):
        print(f"파일 없음: {OUTPUT_FILE}")
        return
    
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(RESTORE_FILE, 'w', encoding='utf-8') as out:
        out.write("# " + "=" * 68 + "\n")
        out.write(f"# OHT Java 소스 코드 모음\n")
        out.write(f"# 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write("# " + "=" * 68 + "\n\n")
        
        # 각 파일 추출
        import re
        pattern = r'###FILE:(.+?)###(.+?)###END###'
        matches = re.findall(pattern, content)
        
        for file_name, file_content in matches:
            out.write("\n# " + "=" * 68 + "\n")
            out.write(f"# {file_name}\n")
            out.write("# " + "=" * 68 + "\n")
            out.write('"""\n')
            
            if file_content not in ["FILE_NOT_FOUND", ""] and not file_content.startswith("ERROR:"):
                # 줄바꿈 복원
                restored = file_content.replace(LINE_BREAK, '\n')
                out.write(restored)
                if not restored.endswith('\n'):
                    out.write('\n')
                print(f"  ✓ {file_name} 복원됨")
            else:
                out.write(f"({file_content})\n")
                print(f"  - {file_name}: {file_content}")
            
            out.write('"""\n')
    
    print(f"\n복원 완료: {RESTORE_FILE}")


def main():
    print("=" * 70)
    print("  Java 파일 → 한 줄 변환 (보안 우회)")
    print("=" * 70)
    print("\n  [1] 변환 (Java → 한 줄)")
    print("  [2] 복원 (한 줄 → 원본)")
    
    choice = input("\n  선택: ").strip()
    
    if choice == "1":
        folder = input("\n  Java 파일 폴더 경로: ").strip()
        if not folder:
            folder = "."
        print()
        convert_to_oneline(folder)
        print("\n  → 이제 oht_java_oneline.txt 파일 복사 가능!")
        print("  → 복사 후 [2] 복원 실행하면 원본 형태로!")
        
    elif choice == "2":
        print()
        restore_from_oneline()
        print("\n  → oht_java_restored.py 파일에서 코드 확인!")


if __name__ == "__main__":
    main()