#!/usr/bin/env python3
"""
Layout XML → JSON 변환기

parse_layout_xml.py
OHT 레이아웃 XML 파일을 파싱하여 3D 시각화용 JSON 생성
"""

import xml.etree.ElementTree as ET
import json
import zipfile
import os
import sys
from collections import defaultdict

def parse_layout_xml(xml_path_or_zip):
    """XML 파일 파싱하여 노드와 엣지 추출"""

    print(f"파싱 시작: {xml_path_or_zip}")

    # ZIP 파일이면 압축 해제
    if xml_path_or_zip.endswith('.zip'):
        with zipfile.ZipFile(xml_path_or_zip, 'r') as zf:
            xml_content = zf.read('layout.xml')
            root = ET.fromstring(xml_content)
    else:
        tree = ET.parse(xml_path_or_zip)
        root = tree.getroot()

    nodes = {}
    edges = []

    # AddrControl 내의 모든 Addr 그룹 찾기
    addr_control = root.find(".//group[@name='AddrControl']")
    if addr_control is None:
        print("AddrControl을 찾을 수 없습니다.")
        return nodes, edges

    addr_count = 0

    for addr_group in addr_control.findall("group"):
        if not addr_group.get('name', '').startswith('Addr'):
            continue

        addr_count += 1

        # 노드 정보 추출
        address = None
        draw_x = None
        draw_y = None
        cad_x = None
        cad_y = None

        for param in addr_group.findall("param"):
            key = param.get('key')
            value = param.get('value')

            if key == 'address':
                address = int(value)
            elif key == 'draw-x':
                draw_x = float(value)
            elif key == 'draw-y':
                draw_y = float(value)
            elif key == 'cad-x':
                cad_x = float(value)
            elif key == 'cad-y':
                cad_y = float(value)

        if address is not None and draw_x is not None and draw_y is not None:
            nodes[address] = {
                'id': address,
                'x': draw_x,
                'y': draw_y,
                'cad_x': cad_x,
                'cad_y': cad_y
            }

        # 다음 노드 연결 (엣지) 정보 추출
        for next_addr_group in addr_group.findall("group"):
            if not next_addr_group.get('name', '').startswith('NextAddr'):
                continue

            for param in next_addr_group.findall("param"):
                if param.get('key') == 'next-address':
                    next_address = int(param.get('value'))
                    if address is not None and next_address > 0:
                        edges.append({
                            'from': address,
                            'to': next_address
                        })

    print(f"파싱 완료: {len(nodes)} 노드, {len(edges)} 엣지")

    return nodes, edges

def save_json(nodes, edges, output_path):
    """JSON 파일로 저장"""

    # 좌표 정규화 (0-1000 범위로)
    if nodes:
        min_x = min(n['x'] for n in nodes.values())
        max_x = max(n['x'] for n in nodes.values())
        min_y = min(n['y'] for n in nodes.values())
        max_y = max(n['y'] for n in nodes.values())

        scale_x = 1000 / (max_x - min_x) if max_x != min_x else 1
        scale_y = 1000 / (max_y - min_y) if max_y != min_y else 1
        scale = min(scale_x, scale_y)  # 비율 유지

        for node in nodes.values():
            node['x'] = (node['x'] - min_x) * scale
            node['y'] = (node['y'] - min_y) * scale

    data = {
        'nodes': list(nodes.values()),
        'edges': edges,
        'stats': {
            'nodeCount': len(nodes),
            'edgeCount': len(edges)
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"JSON 저장: {output_path}")

def main():
    # 기본 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # layout.zip 경로
    zip_path = os.path.join(parent_dir, 'layout', 'layout', 'layout.zip')

    if not os.path.exists(zip_path):
        print(f"파일을 찾을 수 없습니다: {zip_path}")
        print("사용법: python parse_layout_xml.py [layout.zip 경로]")
        sys.exit(1)

    # 명령행 인자로 경로 지정 가능
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]

    # 파싱
    nodes, edges = parse_layout_xml(zip_path)

    # JSON 저장
    output_path = os.path.join(script_dir, 'layout_data.json')
    save_json(nodes, edges, output_path)

    print(f"\n완료! layout_data.json을 생성했습니다.")
    print("index.html을 브라우저에서 열어 3D로 확인하세요.")

if __name__ == '__main__':
    main()