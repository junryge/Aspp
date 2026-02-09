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
    """OHT 3D HTML 생성"""

    data = {
        'nodes': list(nodes.values()),
        'edges': edges
    }
    json_data = json.dumps(data)

    html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>OHT Layout 3D Viewer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #0a0a1a; overflow: hidden; font-family: 'Segoe UI', sans-serif; }}
        #container {{ width: 100vw; height: 100vh; }}
        #info {{
            position: fixed; top: 20px; left: 20px; color: #fff;
            background: rgba(0,0,0,0.8); padding: 20px; border-radius: 12px;
            font-size: 14px; z-index: 100; border: 1px solid #333;
        }}
        #info h1 {{ font-size: 20px; margin-bottom: 15px; color: #00d4ff; }}
        #info p {{ margin: 8px 0; }}
        .stat {{ color: #00ff88; font-weight: bold; }}
        #controls {{
            position: fixed; bottom: 20px; left: 20px; color: #aaa;
            background: rgba(0,0,0,0.8); padding: 15px; border-radius: 12px;
            font-size: 12px; border: 1px solid #333;
        }}
        #legend {{
            position: fixed; top: 20px; right: 20px; color: #fff;
            background: rgba(0,0,0,0.8); padding: 15px; border-radius: 12px;
            font-size: 13px; border: 1px solid #333;
        }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; }}
        .legend-color {{ width: 20px; height: 10px; margin-right: 10px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div id="info">
        <h1>🏭 OHT Rail Layout</h1>
        <p>노드: <span class="stat">{len(nodes):,}</span>개</p>
        <p>레일: <span class="stat">{len(edges):,}</span>개</p>
        <p>FPS: <span id="fps" class="stat">0</span></p>
    </div>
    <div id="legend">
        <div class="legend-item"><div class="legend-color" style="background:#00d4ff;"></div>OHT 레일</div>
        <div class="legend-item"><div class="legend-color" style="background:#00ff88;"></div>노드 (분기점)</div>
        <div class="legend-item"><div class="legend-color" style="background:#ff6b6b;"></div>스테이션</div>
    </div>
    <div id="controls">
        <p>🖱️ 좌클릭 드래그: 회전</p>
        <p>🖱️ 우클릭 드래그: 이동</p>
        <p>🖱️ 스크롤: 줌</p>
        <p>⌨️ R: 뷰 리셋</p>
        <p>⌨️ T: 위에서 보기</p>
    </div>
    <div id="container"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        const layoutData = {json_data};

        let scene, camera, renderer, controls;
        let frameCount = 0, lastTime = performance.now();

        const RAIL_HEIGHT = 50;  // 레일 높이 (천장)
        const RAIL_RADIUS = 1.5; // 레일 파이프 두께

        function init() {{
            const container = document.getElementById('container');

            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0a1a);

            // Camera
            camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 1, 5000);
            camera.position.set(600, 400, 600);

            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.shadowMap.enabled = true;
            container.appendChild(renderer.domElement);

            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.target.set(400, RAIL_HEIGHT/2, 400);
            controls.maxPolarAngle = Math.PI * 0.9;

            // Lights
            scene.add(new THREE.AmbientLight(0x404060, 1));

            const dirLight = new THREE.DirectionalLight(0xffffff, 1);
            dirLight.position.set(500, 500, 500);
            dirLight.castShadow = true;
            scene.add(dirLight);

            const dirLight2 = new THREE.DirectionalLight(0x4488ff, 0.5);
            dirLight2.position.set(-500, 300, -500);
            scene.add(dirLight2);

            // Floor (공장 바닥)
            const floorGeo = new THREE.PlaneGeometry(1000, 1000);
            const floorMat = new THREE.MeshStandardMaterial({{
                color: 0x1a1a2e,
                roughness: 0.8,
                metalness: 0.2
            }});
            const floor = new THREE.Mesh(floorGeo, floorMat);
            floor.rotation.x = -Math.PI / 2;
            floor.position.set(400, 0, 400);
            floor.receiveShadow = true;
            scene.add(floor);

            // Grid
            const grid = new THREE.GridHelper(1000, 40, 0x333355, 0x222244);
            grid.position.set(400, 0.1, 400);
            scene.add(grid);

            // Render OHT Layout
            renderOHTLayout();

            // Events
            window.addEventListener('resize', onWindowResize);
            window.addEventListener('keydown', onKeyDown);

            animate();
        }}

        function renderOHTLayout() {{
            const nodeMap = {{}};
            layoutData.nodes.forEach(node => {{
                nodeMap[node.id] = node;
            }});

            // === 레일 (3D 파이프) ===
            const railMaterial = new THREE.MeshStandardMaterial({{
                color: 0x00d4ff,
                metalness: 0.8,
                roughness: 0.3,
                emissive: 0x004466,
                emissiveIntensity: 0.3
            }});

            layoutData.edges.forEach(edge => {{
                const fromNode = nodeMap[edge.from];
                const toNode = nodeMap[edge.to];

                if (fromNode && toNode) {{
                    const start = new THREE.Vector3(fromNode.x, RAIL_HEIGHT, fromNode.y);
                    const end = new THREE.Vector3(toNode.x, RAIL_HEIGHT, toNode.y);

                    const direction = new THREE.Vector3().subVectors(end, start);
                    const length = direction.length();

                    if (length > 0.1) {{
                        // 파이프 형태 레일
                        const railGeo = new THREE.CylinderGeometry(RAIL_RADIUS, RAIL_RADIUS, length, 8);
                        const rail = new THREE.Mesh(railGeo, railMaterial);

                        // 위치 및 회전
                        const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
                        rail.position.copy(midpoint);
                        rail.quaternion.setFromUnitVectors(
                            new THREE.Vector3(0, 1, 0),
                            direction.normalize()
                        );
                        rail.castShadow = true;
                        scene.add(rail);

                        // 지지대 (일정 간격마다)
                        if (length > 50) {{
                            const supportCount = Math.floor(length / 80);
                            for (let i = 1; i <= supportCount; i++) {{
                                const t = i / (supportCount + 1);
                                const pos = new THREE.Vector3().lerpVectors(start, end, t);
                                addSupport(pos);
                            }}
                        }}
                    }}
                }}
            }});

            // === 노드 (분기점/교차점) ===
            const nodeMaterial = new THREE.MeshStandardMaterial({{
                color: 0x00ff88,
                metalness: 0.6,
                roughness: 0.4,
                emissive: 0x00ff88,
                emissiveIntensity: 0.2
            }});

            const nodeGeo = new THREE.SphereGeometry(3, 16, 16);

            layoutData.nodes.forEach(node => {{
                const sphere = new THREE.Mesh(nodeGeo, nodeMaterial);
                sphere.position.set(node.x, RAIL_HEIGHT, node.y);
                sphere.castShadow = true;
                scene.add(sphere);
            }});

            console.log('OHT 레이아웃 렌더링 완료');
        }}

        function addSupport(pos) {{
            // 수직 지지대
            const supportMat = new THREE.MeshStandardMaterial({{
                color: 0x666688,
                metalness: 0.7,
                roughness: 0.5
            }});

            const supportGeo = new THREE.CylinderGeometry(0.8, 0.8, RAIL_HEIGHT, 6);
            const support = new THREE.Mesh(supportGeo, supportMat);
            support.position.set(pos.x, RAIL_HEIGHT / 2, pos.z);
            support.castShadow = true;
            scene.add(support);
        }}

        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        function onKeyDown(e) {{
            if (e.key === 'r' || e.key === 'R') {{
                camera.position.set(600, 400, 600);
                controls.target.set(400, RAIL_HEIGHT/2, 400);
            }}
            if (e.key === 't' || e.key === 'T') {{
                camera.position.set(400, 800, 400);
                controls.target.set(400, 0, 400);
            }}
        }}

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);

            frameCount++;
            const now = performance.now();
            if (now - lastTime >= 1000) {{
                document.getElementById('fps').textContent = frameCount;
                frameCount = 0;
                lastTime = now;
            }}
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