#!/usr/bin/env python3
"""
Layout XML → OHT 3D HTML 변환기
실제 OHT 레일처럼 3D 파이프 형태로 시각화
"""

import xml.etree.ElementTree as ET
import json
import zipfile
import os
import sys

def parse_layout_xml(xml_path_or_zip):
    """XML 파일 파싱하여 노드와 엣지 추출"""

    print(f"파싱 시작: {xml_path_or_zip}")

    if xml_path_or_zip.endswith('.zip'):
        with zipfile.ZipFile(xml_path_or_zip, 'r') as zf:
            xml_content = zf.read('layout.xml')
            root = ET.fromstring(xml_content)
    else:
        tree = ET.parse(xml_path_or_zip)
        root = tree.getroot()

    nodes = {}
    edges = []

    addr_control = root.find(".//group[@name='AddrControl']")
    if addr_control is None:
        print("AddrControl을 찾을 수 없습니다.")
        return nodes, edges

    for addr_group in addr_control.findall("group"):
        if not addr_group.get('name', '').startswith('Addr'):
            continue

        address = None
        draw_x = None
        draw_y = None
        floor = 1  # 기본 층

        for param in addr_group.findall("param"):
            key = param.get('key')
            value = param.get('value')

            if key == 'address':
                address = int(value)
            elif key == 'draw-x':
                draw_x = float(value)
            elif key == 'draw-y':
                draw_y = float(value)
            elif key == 'floor' or key == 'level':
                try:
                    floor = int(value)
                except:
                    pass

        if address is not None and draw_x is not None and draw_y is not None:
            nodes[address] = {'id': address, 'x': draw_x, 'y': draw_y, 'floor': floor}

        for next_addr_group in addr_group.findall("group"):
            if not next_addr_group.get('name', '').startswith('NextAddr'):
                continue

            for param in next_addr_group.findall("param"):
                if param.get('key') == 'next-address':
                    next_address = int(param.get('value'))
                    if address is not None and next_address > 0:
                        edges.append({'from': address, 'to': next_address})

    print(f"파싱 완료: {len(nodes)} 노드, {len(edges)} 엣지")
    return nodes, edges

def normalize_coords(nodes):
    """좌표 정규화"""
    if not nodes:
        return

    min_x = min(n['x'] for n in nodes.values())
    max_x = max(n['x'] for n in nodes.values())
    min_y = min(n['y'] for n in nodes.values())
    max_y = max(n['y'] for n in nodes.values())

    range_x = max_x - min_x if max_x != min_x else 1
    range_y = max_y - min_y if max_y != min_y else 1
    scale = 800 / max(range_x, range_y)

    for node in nodes.values():
        node['x'] = (node['x'] - min_x) * scale
        node['y'] = (node['y'] - min_y) * scale

def generate_html(nodes, edges, output_path):
    """OHT 3D HTML 생성 - 2D 평면도 + 3D 뷰 전환"""

    data = {
        'nodes': list(nodes.values()),
        'edges': edges
    }
    json_data = json.dumps(data)

    html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>OHT Layout Viewer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #1a1a2e; overflow: hidden; font-family: 'Segoe UI', sans-serif; }}
        #container {{ width: 100vw; height: 100vh; }}
        #info {{
            position: fixed; top: 20px; left: 20px; color: #fff;
            background: rgba(0,0,0,0.85); padding: 20px; border-radius: 12px;
            font-size: 14px; z-index: 100; border: 1px solid #444;
        }}
        #info h1 {{ font-size: 22px; margin-bottom: 15px; color: #00d4ff; }}
        #info p {{ margin: 8px 0; }}
        .stat {{ color: #00ff88; font-weight: bold; }}
        #controls {{
            position: fixed; bottom: 20px; left: 20px; color: #ccc;
            background: rgba(0,0,0,0.85); padding: 15px; border-radius: 12px;
            font-size: 13px; border: 1px solid #444;
        }}
        #controls p {{ margin: 4px 0; }}
        #viewMode {{
            position: fixed; top: 20px; right: 20px;
            background: rgba(0,0,0,0.85); padding: 15px; border-radius: 12px;
            border: 1px solid #444; z-index: 100;
        }}
        #viewMode button {{
            background: #00d4ff; color: #000; border: none;
            padding: 10px 20px; margin: 5px; border-radius: 8px;
            cursor: pointer; font-size: 14px; font-weight: bold;
        }}
        #viewMode button:hover {{ background: #00a8cc; }}
        #viewMode button.active {{ background: #00ff88; }}
    </style>
</head>
<body>
    <div id="info">
        <h1>OHT Rail Layout</h1>
        <p>노드 (교차점): <span class="stat">{len(nodes):,}</span>개</p>
        <p>레일 (연결선): <span class="stat">{len(edges):,}</span>개</p>
    </div>
    <div id="viewMode">
        <button id="btn2D" class="active" onclick="setView2D()">2D 평면도</button>
        <button id="btn3D" onclick="setView3D()">3D 입체</button>
    </div>
    <div id="controls">
        <p>마우스 드래그: 회전/이동</p>
        <p>스크롤: 줌</p>
        <p>R키: 뷰 리셋</p>
    </div>
    <div id="container"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        const layoutData = {json_data};

        let scene, camera, renderer, controls;
        let centerX = 400, centerZ = 400;

        function init() {{
            const container = document.getElementById('container');

            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);

            camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 10000);

            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            container.appendChild(renderer.domElement);

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.1;

            // 조명
            scene.add(new THREE.AmbientLight(0xffffff, 0.8));
            const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
            dirLight.position.set(500, 500, 500);
            scene.add(dirLight);

            renderLayout();
            setView2D(); // 기본 2D 뷰

            window.addEventListener('resize', onWindowResize);
            window.addEventListener('keydown', (e) => {{
                if (e.key === 'r' || e.key === 'R') setView2D();
            }});

            animate();
        }}

        function renderLayout() {{
            const nodeMap = {{}};
            let minX = Infinity, maxX = -Infinity;
            let minZ = Infinity, maxZ = -Infinity;

            layoutData.nodes.forEach(node => {{
                nodeMap[node.id] = node;
                if (node.x < minX) minX = node.x;
                if (node.x > maxX) maxX = node.x;
                if (node.y < minZ) minZ = node.y;
                if (node.y > maxZ) maxZ = node.y;
            }});

            centerX = (minX + maxX) / 2;
            centerZ = (minZ + maxZ) / 2;

            // === 바닥 그리드 ===
            const gridSize = Math.max(maxX - minX, maxZ - minZ) + 200;
            const grid = new THREE.GridHelper(gridSize, 50, 0x333366, 0x222244);
            grid.position.set(centerX, -1, centerZ);
            scene.add(grid);

            // === 레일 (두꺼운 파란 선) ===
            layoutData.edges.forEach(edge => {{
                const fromNode = nodeMap[edge.from];
                const toNode = nodeMap[edge.to];

                if (fromNode && toNode) {{
                    // 메인 레일 - 두꺼운 박스 형태
                    const start = new THREE.Vector3(fromNode.x, 0, fromNode.y);
                    const end = new THREE.Vector3(toNode.x, 0, toNode.y);
                    const dir = new THREE.Vector3().subVectors(end, start);
                    const len = dir.length();

                    if (len > 0.5) {{
                        // 레일 박스 (넓고 납작하게)
                        const railGeo = new THREE.BoxGeometry(len, 3, 8);
                        const railMat = new THREE.MeshStandardMaterial({{
                            color: 0x00aaff,
                            emissive: 0x003366,
                            emissiveIntensity: 0.5
                        }});
                        const rail = new THREE.Mesh(railGeo, railMat);

                        const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
                        rail.position.copy(mid);
                        rail.rotation.y = Math.atan2(dir.x, dir.z);
                        scene.add(rail);

                        // 방향 화살표 (진행 방향 표시)
                        const arrowLen = Math.min(len * 0.3, 20);
                        const arrowGeo = new THREE.ConeGeometry(4, arrowLen, 4);
                        const arrowMat = new THREE.MeshStandardMaterial({{
                            color: 0xffff00,
                            emissive: 0x666600,
                            emissiveIntensity: 0.5
                        }});
                        const arrow = new THREE.Mesh(arrowGeo, arrowMat);
                        const arrowPos = new THREE.Vector3().lerpVectors(start, end, 0.7);
                        arrow.position.copy(arrowPos);
                        arrow.position.y = 5;
                        arrow.rotation.x = Math.PI / 2;
                        arrow.rotation.z = -Math.atan2(dir.x, dir.z);
                        scene.add(arrow);
                    }}
                }}
            }});

            // === 노드 (교차점 - 초록 박스) ===
            const nodeGeo = new THREE.BoxGeometry(12, 6, 12);
            const nodeMat = new THREE.MeshStandardMaterial({{
                color: 0x00ff88,
                emissive: 0x006633,
                emissiveIntensity: 0.5
            }});

            layoutData.nodes.forEach(node => {{
                const box = new THREE.Mesh(nodeGeo, nodeMat);
                box.position.set(node.x, 3, node.y);
                scene.add(box);
            }});

            console.log('레이아웃 렌더링 완료: ' + layoutData.nodes.length + ' 노드');
        }}

        function setView2D() {{
            // 위에서 내려다보는 2D 뷰
            const dist = 1200;
            camera.position.set(centerX, dist, centerZ);
            controls.target.set(centerX, 0, centerZ);
            controls.update();
            document.getElementById('btn2D').classList.add('active');
            document.getElementById('btn3D').classList.remove('active');
        }}

        function setView3D() {{
            // 비스듬히 보는 3D 뷰
            camera.position.set(centerX + 500, 400, centerZ + 500);
            controls.target.set(centerX, 0, centerZ);
            controls.update();
            document.getElementById('btn3D').classList.add('active');
            document.getElementById('btn2D').classList.remove('active');
        }}

        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}

        init();
    </script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"HTML 저장: {output_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
    else:
        xml_path = os.path.join(script_dir, 'layout.xml')
        if not os.path.exists(xml_path):
            xml_path = os.path.join(script_dir, 'layout.zip')

    if not os.path.exists(xml_path):
        print(f"파일을 찾을 수 없습니다: {xml_path}")
        print("사용법: python parse_layout_xml.py [layout.xml 또는 layout.zip]")
        sys.exit(1)

    nodes, edges = parse_layout_xml(xml_path)
    normalize_coords(nodes)

    output_path = os.path.join(script_dir, 'layout_3d.html')
    generate_html(nodes, edges, output_path)

    print(f"\n완료! layout_3d.html 더블클릭해서 열어!")

if __name__ == '__main__':
    main()