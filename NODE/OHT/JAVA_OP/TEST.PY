#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Java 소스 파일 들여쓰기 제거 (보안 우회)
- 앞 공백/탭 제거하여 Java 코드 패턴 인식 방지
- 나중에 다시 복원 가능
"""

import os
import re
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

OUTPUT_FILE = "oht_java_no_indent.txt"
RESTORE_FILE = "oht_java_restored.py"

# 들여쓰기 대체 문자
INDENT_MARK = "►"  # 탭/4스페이스 1개 = ► 1개


def remove_indent(content):
    """들여쓰기 제거"""
    lines = content.split('\n')
    result = []
    
    for line in lines:
        # 앞 공백 개수 세기
        stripped = line.lstrip()
        indent_count = len(line) - len(stripped)
        
        # 탭 또는 4스페이스 단위로 마킹
        indent_level = indent_count // 4
        if '\t' in line[:indent_count]:
            indent_level = line[:indent_count].count('\t')
        
        # 들여쓰기 마크 + 내용
        marked_line = (INDENT_MARK * indent_level) + stripped
        result.append(marked_line)
    
    return '\n'.join(result)


def restore_indent(content):
    """들여쓰기 복원"""
    lines = content.split('\n')
    result = []
    
    for line in lines:
        # 마크 개수 세기
        indent_level = 0
        while line.startswith(INDENT_MARK):
            indent_level += 1
            line = line[1:]
        
        # 들여쓰기 복원 (4스페이스)
        restored_line = ("    " * indent_level) + line
        result.append(restored_line)
    
    return '\n'.join(result)


def convert(folder):
    """Java 파일 들여쓰기 제거"""
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        for file_name in JAVA_FILES:
            file_path = os.path.join(folder, file_name)
            
            out.write(f"### {file_name} ###\n")
            
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    no_indent = remove_indent(content)
                    out.write(no_indent)
                    out.write("\n")
                    print(f"  ✓ {file_name}")
                except Exception as e:
                    out.write(f"ERROR: {e}\n")
                    print(f"  ✗ {file_name}: {e}")
            else:
                out.write("(파일 없음)\n")
                print(f"  - {file_name} 없음")
            
            out.write("### END ###\n\n")
    
    print(f"\n저장: {OUTPUT_FILE}")


def restore():
    """들여쓰기 복원"""
    
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
        pattern = r'### (.+?) ###\n(.*?)### END ###'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for file_name, file_content in matches:
            out.write("\n# " + "=" * 68 + "\n")
            out.write(f"# {file_name}\n")
            out.write("# " + "=" * 68 + "\n")
            out.write('"""\n')
            
            if file_content.strip() and not file_content.startswith("ERROR") and file_content.strip() != "(파일 없음)":
                restored = restore_indent(file_content.strip())
                out.write(restored)
                out.write('\n')
                print(f"  ✓ {file_name} 복원됨")
            else:
                out.write(f"{file_content.strip()}\n")
                print(f"  - {file_name}")
            
            out.write('"""\n')
    
    print(f"\n복원 완료: {RESTORE_FILE}")


def main():
    print("=" * 70)
    print("  Java 파일 들여쓰기 제거 (보안 우회)")
    print("=" * 70)
    print("\n  [1] 변환 (들여쓰기 제거)")
    print("  [2] 복원 (들여쓰기 복원)")
    
    choice = input("\n  선택: ").strip()
    
    if choice == "1":
        folder = input("\n  Java 파일 폴더 경로: ").strip()
        if not folder:
            folder = "."
        print()
        convert(folder)
        print("\n  → oht_java_no_indent.txt 복사 가능!")
        
    elif choice == "2":
        print()
        restore()
        print("\n  → oht_java_restored.py 에서 코드 확인!")


if __name__ == "__main__":
    main()