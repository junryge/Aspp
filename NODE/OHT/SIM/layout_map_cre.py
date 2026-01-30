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


def ensure_layout_html(html_path: str, xml_path: str, zip_path: str = None) -> None:
    """
    layout.html이 없거나 layout.xml보다 오래된 경우 자동 생성

    Args:
        html_path: 생성할 layout.html 경로
        xml_path: layout.xml 파일 경로 (우선)
        zip_path: layout.xml이 들어있는 layout.zip 경로 (xml 없을 때 사용)
    """
    need_generate = False
    source_path = None
    source_mtime = 0

    # layout.xml 또는 layout.zip 중 존재하는 것 찾기
    if os.path.exists(xml_path):
        source_path = xml_path
        source_mtime = os.path.getmtime(xml_path)
        print(f"layout.xml 발견: {xml_path}")
    elif zip_path and os.path.exists(zip_path):
        source_path = zip_path
        source_mtime = os.path.getmtime(zip_path)
        print(f"layout.zip 발견: {zip_path}")
    else:
        raise FileNotFoundError(f"layout.xml 또는 layout.zip을 찾을 수 없습니다: {xml_path}, {zip_path}")

    # layout.html 존재 확인
    if not os.path.exists(html_path):
        print(f"layout.html이 없습니다. 자동 생성합니다...")
        need_generate = True
    else:
        # 소스 파일이 더 최신인지 확인
        html_mtime = os.path.getmtime(html_path)
        if source_mtime > html_mtime:
            print(f"소스 파일이 더 최신입니다. layout.html을 재생성합니다...")
            need_generate = True

    if not need_generate:
        return

    print("=" * 60)
    print("layout.xml에서 layout.html 자동 생성")
    print("=" * 60)

    # XML 내용 읽기 (xml 파일 우선, 없으면 zip에서 추출)
    if os.path.exists(xml_path):
        print(f"layout.xml 읽는 중: {xml_path}")
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
    else:
        print("layout.zip에서 XML 추출 중...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # ZIP 내 파일 목록 확인
            file_list = zf.namelist()
            print(f"  ZIP 내 파일 수: {len(file_list)}개")

            # XML 파일 찾기 (우선순위: layout/layout.xml > layout.xml > *.xml)
            xml_file = None
            for name in file_list:
                lower_name = name.lower()
                # 1순위: layout/layout.xml (실제 구조)
                if lower_name == 'layout/layout.xml':
                    xml_file = name
                    break
                # 2순위: layout.xml
                elif lower_name == 'layout.xml' and xml_file is None:
                    xml_file = name
                # 3순위: 아무 .xml 파일 (layout.xml이 없을 때만)
                elif lower_name.endswith('.xml') and 'layout' in lower_name and xml_file is None:
                    xml_file = name

            if not xml_file:
                # XML 파일 목록 출력
                xml_files = [f for f in file_list if f.lower().endswith('.xml')]
                raise FileNotFoundError(f"ZIP 파일 내에 layout.xml이 없습니다. XML 파일들: {xml_files}")

            print(f"  사용할 XML 파일: {xml_file}")
            with zf.open(xml_file) as f:
                xml_content = f.read().decode('utf-8')

    print(f"  XML 크기: {len(xml_content):,} bytes")

    # XML 파싱
    nodes, connections = parse_layout_xml_content(xml_content)

    # HTML 생성
    generate_layout_html(nodes, connections, html_path)

    print("=" * 60)


def get_fab_paths(script_dir, fab_name: str, layout_prefix: str = "A"):
    """
    FAB별 파일 경로를 반환

    실제 폴더 구조:
        MAP/{FAB}/{prefix}.layout.zip              <- ZIP 파일
        MAP/{FAB}/{prefix}.layout/layout/layout.xml <- 압축 해제된 XML
        MAP/{FAB}/{prefix}.layout.html             <- 생성할 HTML

    Args:
        script_dir: 스크립트 디렉토리
        fab_name: FAB 이름 (예: "M14A", "M16A", "M16B")
        layout_prefix: 레이아웃 파일 접두사 (예: "A", "BR", "E", "B")

    Returns:
        dict: 각 파일 경로를 담은 딕셔너리
    """
    map_base_dir = script_dir / "MAP"
    fab_dir = map_base_dir / fab_name
    prefix = layout_prefix.upper()

    # 압축 해제된 폴더 경로
    extracted_dir = fab_dir / f"{prefix}.layout"

    return {
        "layout_zip": str(fab_dir / f"{prefix}.layout.zip"),
        "layout_xml": str(extracted_dir / "layout" / "layout.xml"),
        "layout_html": str(fab_dir / f"{prefix}.layout.html"),
    }


def ensure_layout_html_for_fab(script_dir, fab_name: str, layout_prefix: str = "A"):
    """
    FAB별 layout.html 생성

    Args:
        script_dir: 스크립트 디렉토리
        fab_name: FAB 이름 (예: "M14", "M16")
        layout_prefix: 레이아웃 파일 접두사 (예: "A", "BR", "E")
    """
    paths = get_fab_paths(script_dir, fab_name, layout_prefix)
    ensure_layout_html(
        paths["layout_html"],
        paths["layout_xml"],
        paths["layout_zip"]
    )


def main():
    """
    커맨드라인 실행용 메인 함수

    사용법:
        python layout_map_cre.py [layout.zip 경로] [출력 layout.html 경로]
        python layout_map_cre.py --fab M14 --layout A
    """
    import sys
    import pathlib

    # 기본 경로
    script_dir = pathlib.Path(__file__).parent.resolve()

    # FAB 모드 확인
    if "--fab" in sys.argv:
        fab_idx = sys.argv.index("--fab")
        fab_name = sys.argv[fab_idx + 1] if fab_idx + 1 < len(sys.argv) else "M14"

        layout_prefix = "A"
        if "--layout" in sys.argv:
            layout_idx = sys.argv.index("--layout")
            layout_prefix = sys.argv[layout_idx + 1] if layout_idx + 1 < len(sys.argv) else "A"

        print("=" * 60)
        print(f"layout_map_cre.py - FAB 모드: {fab_name}, 레이아웃: {layout_prefix}")
        print("=" * 60)

        ensure_layout_html_for_fab(script_dir, fab_name, layout_prefix)
        return

    # 기존 모드 (직접 경로 지정)
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