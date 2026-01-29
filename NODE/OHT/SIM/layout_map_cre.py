#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
layout_map_cre.py - layout.xml에서 layout.html 생성

사용법:
    1. 모듈로 import:
       from layout_map_cre import ensure_layout_html
       ensure_layout_html(html_path, zip_path)

    2. 직접 실행:
       python layout_map_cre.py [layout.zip 경로] [출력 layout.html 경로]
"""

import os
import re
import json
import zipfile
from typing import Dict, List, Tuple


def parse_layout_xml_content(xml_content: str) -> Tuple[List[Dict], List[List]]:
    """
    layout.xml 내용을 파싱하여 노드와 연결 정보 추출 (라인 단위 파싱)

    Returns:
        nodes: [{'no': int, 'x': float, 'y': float, 'stations': []}, ...]
        connections: [[from_no, to_no], ...]
    """
    print("XML 파싱 시작...")

    nodes = {}
    connections = []

    current_addr = None
    current_addr_params = {}
    in_next_addr = False
    next_addr_params = {}

    lines = xml_content.split('\n')
    total_lines = len(lines)
    print(f"  총 라인 수: {total_lines:,}")

    for line_no, line in enumerate(lines):
        if line_no % 500000 == 0:
            print(f"  처리 중: {line_no:,}/{total_lines:,} ({line_no*100//total_lines}%)")

        line = line.strip()

        # Addr 그룹 시작
        if '<group name="Addr' in line and 'class=' in line and 'address.Addr"' in line:
            if current_addr is not None and 'address' in current_addr_params:
                addr_no = int(current_addr_params.get('address', 0))
                if addr_no > 0:
                    x = float(current_addr_params.get('draw-x', 0))
                    y = float(current_addr_params.get('draw-y', 0))
                    nodes[addr_no] = {'no': addr_no, 'x': round(x, 2), 'y': round(y, 2), 'stations': []}

            current_addr_params = {}
            current_addr = line
            in_next_addr = False
            continue

        # NextAddr 그룹 시작
        if '<group name="NextAddr' in line and 'class=' in line and 'NextAddr"' in line:
            in_next_addr = True
            next_addr_params = {}
            continue

        # NextAddr 그룹 종료
        if in_next_addr and '</group>' in line:
            if 'address' in current_addr_params and 'next-address' in next_addr_params:
                from_addr = int(current_addr_params.get('address', 0))
                try:
                    to_addr = int(next_addr_params.get('next-address', '0'))
                    if from_addr > 0 and to_addr > 0:
                        connections.append([from_addr, to_addr])
                except ValueError:
                    pass
            in_next_addr = False
            continue

        # 파라미터 파싱
        if '<param ' in line and 'key="' in line and 'value="' in line:
            key_match = re.search(r'key="([^"]+)"', line)
            value_match = re.search(r'value="([^"]*)"', line)

            if key_match and value_match:
                key = key_match.group(1)
                value = value_match.group(1)

                if in_next_addr:
                    next_addr_params[key] = value
                elif current_addr is not None:
                    current_addr_params[key] = value

    # 마지막 Addr 저장
    if current_addr is not None and 'address' in current_addr_params:
        addr_no = int(current_addr_params.get('address', 0))
        if addr_no > 0:
            x = float(current_addr_params.get('draw-x', 0))
            y = float(current_addr_params.get('draw-y', 0))
            nodes[addr_no] = {'no': addr_no, 'x': round(x, 2), 'y': round(y, 2), 'stations': []}

    print(f"  총 노드: {len(nodes)}개")
    print(f"  총 연결: {len(connections)}개")

    return list(nodes.values()), connections


def generate_layout_html(nodes: List[Dict], connections: List[List], output_path: str) -> None:
    """
    layout.html 파일 생성
    """
    print(f"layout.html 생성: {output_path}")

    nodes_json = json.dumps(nodes, ensure_ascii=False)
    connections_json = json.dumps(connections, ensure_ascii=False)

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OHT Layout Data</title>
    <script>
const A={nodes_json};
const C={connections_json};
    </script>
</head>
<body>
    <p>Nodes: {len(nodes)}, Connections: {len(connections)}</p>
</body>
</html>
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  완료: {os.path.getsize(output_path):,} bytes")


def ensure_layout_html(html_path: str, zip_path: str) -> None:
    """
    layout.html이 없거나 layout.zip보다 오래된 경우 자동 생성

    Args:
        html_path: 생성할 layout.html 경로
        zip_path: layout.xml이 들어있는 layout.zip 경로
    """
    need_generate = False

    # layout.html 존재 확인
    if not os.path.exists(html_path):
        print(f"layout.html이 없습니다. 자동 생성합니다...")
        need_generate = True
    elif os.path.exists(zip_path):
        # layout.zip이 더 최신인지 확인
        html_mtime = os.path.getmtime(html_path)
        zip_mtime = os.path.getmtime(zip_path)
        if zip_mtime > html_mtime:
            print(f"layout.zip이 더 최신입니다. layout.html을 재생성합니다...")
            need_generate = True

    if not need_generate:
        return

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"layout.zip을 찾을 수 없습니다: {zip_path}")

    print("=" * 60)
    print("layout.xml에서 layout.html 자동 생성")
    print("=" * 60)

    # ZIP에서 layout.xml 추출
    print("layout.xml 추출 중...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open('layout.xml') as f:
            xml_content = f.read().decode('utf-8')

    print(f"  XML 크기: {len(xml_content):,} bytes")

    # XML 파싱
    nodes, connections = parse_layout_xml_content(xml_content)

    # HTML 생성
    generate_layout_html(nodes, connections, html_path)

    print("=" * 60)


def main():
    """
    커맨드라인 실행용 메인 함수
    """
    import sys
    import pathlib

    # 기본 경로
    script_dir = pathlib.Path(__file__).parent.resolve()
    default_zip = str(script_dir / 'layout' / 'layout' / 'layout.zip')
    default_output = str(script_dir / 'layout' / 'layout' / 'layout.html')

    # 명령행 인자 처리
    zip_path = sys.argv[1] if len(sys.argv) > 1 else default_zip
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output

    print("=" * 60)
    print("layout_map_cre.py - layout.xml에서 layout.html 생성")
    print("=" * 60)
    print(f"입력: {zip_path}")
    print(f"출력: {output_path}")
    print()

    if not os.path.exists(zip_path):
        print(f"오류: ZIP 파일을 찾을 수 없습니다: {zip_path}")
        sys.exit(1)

    # ZIP에서 layout.xml 추출
    print("layout.xml 추출 중...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open('layout.xml') as f:
            xml_content = f.read().decode('utf-8')

    print(f"  XML 크기: {len(xml_content):,} bytes")

    # XML 파싱
    nodes, connections = parse_layout_xml_content(xml_content)

    # HTML 생성
    generate_layout_html(nodes, connections, output_path)

    print()
    print("완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()