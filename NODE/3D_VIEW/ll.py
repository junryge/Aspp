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
    """OHT 3D HTML 생성 - 반도체 팹 천장 레일 시스템"""

    data = {
        'nodes': list(nodes.values()),
        'edges': edges
    }
    json_data = json.dumps(data)

    html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>OHT Rail System - 3D Viewer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #0d1117; overflow: hidden; font-family: 'Segoe UI', sans-serif; }}
        #container {{ width: 100vw; height: 100vh; }}
        #info {{
            position: fixed; top: 20px; left: 20px; color: #fff;
            background: rgba(0,0,0,0.9); padding: 20px; border-radius: 8px;
            font-size: 14px; z-index: 100; border: 1px solid #30363d;
        }}
        #info h1 {{ font-size: 18px; margin-bottom: 12px; color: #58a6ff; }}
        #info p {{ margin: 6px 0; color: #8b949e; }}
        .val {{ color: #7ee787; font-weight: bold; }}
        #controls {{
            position: fixed; bottom: 20px; left: 20px; color: #8b949e;
            background: rgba(0,0,0,0.9); padding: 12px 16px; border-radius: 8px;
            font-size: 12px; border: 1px solid #30363d;
        }}
    </style>
</head>
<body>
    <div id="info">
        <h1>OHT Rail System</h1>
        <p>Nodes: <span class="val">{len(nodes):,}</span></p>
        <p>Rails: <span class="val">{len(edges):,}</span></p>
    </div>
    <div id="controls">
        드래그: 회전 | 우클릭: 이동 | 스크롤: 줌 | R: 리셋
    </div>
    <div id="container"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        const data = {json_data};
        const RAIL_H = 80;  // 레일 높이 (천장)

        let scene, camera, renderer, controls, center;

        function init() {{
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0d1117);

            // 범위 계산
            let minX=Infinity, maxX=-Infinity, minZ=Infinity, maxZ=-Infinity;
            data.nodes.forEach(n => {{
                minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x);
                minZ = Math.min(minZ, n.y); maxZ = Math.max(maxZ, n.y);
            }});
            center = {{ x: (minX+maxX)/2, z: (minZ+maxZ)/2 }};
            const size = Math.max(maxX-minX, maxZ-minZ);

            // Camera
            camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 1, size*10);
            camera.position.set(center.x + size*0.6, RAIL_H + size*0.4, center.z + size*0.6);

            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(innerWidth, innerHeight);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            document.getElementById('container').appendChild(renderer.domElement);

            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.target.set(center.x, RAIL_H/2, center.z);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.update();

            // === 조명 ===
            scene.add(new THREE.AmbientLight(0xffffff, 0.4));

            const sun = new THREE.DirectionalLight(0xffffff, 0.8);
            sun.position.set(center.x + size, size, center.z + size);
            sun.castShadow = true;
            sun.shadow.mapSize.width = 2048;
            sun.shadow.mapSize.height = 2048;
            sun.shadow.camera.near = 1;
            sun.shadow.camera.far = size * 3;
            sun.shadow.camera.left = -size;
            sun.shadow.camera.right = size;
            sun.shadow.camera.top = size;
            sun.shadow.camera.bottom = -size;
            scene.add(sun);

            // === 공장 바닥 ===
            const floorGeo = new THREE.PlaneGeometry(size + 400, size + 400);
            const floorMat = new THREE.MeshStandardMaterial({{
                color: 0x1a1f2e,
                roughness: 0.9
            }});
            const floor = new THREE.Mesh(floorGeo, floorMat);
            floor.rotation.x = -Math.PI / 2;
            floor.position.set(center.x, 0, center.z);
            floor.receiveShadow = true;
            scene.add(floor);

            // 바닥 그리드
            const grid = new THREE.GridHelper(size + 400, 40, 0x2d333b, 0x21262d);
            grid.position.set(center.x, 0.5, center.z);
            scene.add(grid);

            // === 레일 렌더링 ===
            const nodeMap = {{}};
            data.nodes.forEach(n => nodeMap[n.id] = n);

            // 레일 재질 (알루미늄 느낌)
            const railMat = new THREE.MeshStandardMaterial({{
                color: 0xc0c0c0,
                metalness: 0.8,
                roughness: 0.3
            }});

            // 지지대 재질 (스틸)
            const supportMat = new THREE.MeshStandardMaterial({{
                color: 0x4a5568,
                metalness: 0.6,
                roughness: 0.4
            }});

            // 분기점 재질
            const junctionMat = new THREE.MeshStandardMaterial({{
                color: 0xf6ad55,
                metalness: 0.5,
                roughness: 0.4
            }});

            data.edges.forEach(edge => {{
                const from = nodeMap[edge.from];
                const to = nodeMap[edge.to];
                if (!from || !to) return;

                const start = new THREE.Vector3(from.x, RAIL_H, from.y);
                const end = new THREE.Vector3(to.x, RAIL_H, to.y);
                const dir = new THREE.Vector3().subVectors(end, start);
                const len = dir.length();
                if (len < 1) return;

                // I-빔 형태 레일 (상단 플랜지 + 웹 + 하단 플랜지)
                const railGroup = new THREE.Group();

                // 상단 플랜지 (OHT가 매달리는 부분)
                const topFlange = new THREE.Mesh(
                    new THREE.BoxGeometry(len, 2, 10),
                    railMat
                );
                topFlange.position.y = 4;
                topFlange.castShadow = true;
                railGroup.add(topFlange);

                // 웹 (수직 연결부)
                const web = new THREE.Mesh(
                    new THREE.BoxGeometry(len, 8, 2),
                    railMat
                );
                web.castShadow = true;
                railGroup.add(web);

                // 하단 플랜지
                const bottomFlange = new THREE.Mesh(
                    new THREE.BoxGeometry(len, 2, 8),
                    railMat
                );
                bottomFlange.position.y = -4;
                bottomFlange.castShadow = true;
                railGroup.add(bottomFlange);

                // 레일 위치/회전
                const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
                railGroup.position.copy(mid);
                railGroup.rotation.y = Math.atan2(dir.x, dir.z);
                scene.add(railGroup);

                // 지지대 (일정 간격)
                const supportInterval = 100;
                const numSupports = Math.floor(len / supportInterval);
                for (let i = 1; i <= numSupports; i++) {{
                    const t = i / (numSupports + 1);
                    const pos = new THREE.Vector3().lerpVectors(start, end, t);

                    // 수직 기둥
                    const pillar = new THREE.Mesh(
                        new THREE.CylinderGeometry(1.5, 2, RAIL_H - 6, 8),
                        supportMat
                    );
                    pillar.position.set(pos.x, (RAIL_H - 6) / 2, pos.z);
                    pillar.castShadow = true;
                    scene.add(pillar);

                    // 기둥 바닥 플레이트
                    const plate = new THREE.Mesh(
                        new THREE.CylinderGeometry(4, 4, 1, 8),
                        supportMat
                    );
                    plate.position.set(pos.x, 0.5, pos.z);
                    scene.add(plate);
                }}
            }});

            // === 분기점 (Junction) ===
            data.nodes.forEach(node => {{
                const junction = new THREE.Mesh(
                    new THREE.BoxGeometry(14, 10, 14),
                    junctionMat
                );
                junction.position.set(node.x, RAIL_H, node.y);
                junction.castShadow = true;
                scene.add(junction);
            }});

            // === OHT 차량 생성 ===
            createOHTVehicles();

            window.addEventListener('resize', () => {{
                camera.aspect = innerWidth / innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(innerWidth, innerHeight);
            }});

            window.addEventListener('keydown', e => {{
                if (e.key === 'r' || e.key === 'R') {{
                    camera.position.set(center.x + size*0.6, RAIL_H + size*0.4, center.z + size*0.6);
                    controls.target.set(center.x, RAIL_H/2, center.z);
                }}
            }});

            animate();
        }}

        // OHT 차량들
        const ohtVehicles = [];

        function createOHTVehicles() {{
            const nodeMap = {{}};
            data.nodes.forEach(n => nodeMap[n.id] = n);

            // 유효한 경로 수집
            const paths = [];
            data.edges.forEach(edge => {{
                const from = nodeMap[edge.from];
                const to = nodeMap[edge.to];
                if (from && to) {{
                    paths.push({{ from, to }});
                }}
            }});

            if (paths.length === 0) return;

            // OHT 차량 수 (레일 수에 비례)
            const numOHT = Math.min(Math.max(5, Math.floor(paths.length / 10)), 30);

            const ohtBodyMat = new THREE.MeshStandardMaterial({{
                color: 0x2196F3,
                metalness: 0.7,
                roughness: 0.3
            }});

            const ohtTopMat = new THREE.MeshStandardMaterial({{
                color: 0x1565C0,
                metalness: 0.6,
                roughness: 0.4
            }});

            const foupMat = new THREE.MeshStandardMaterial({{
                color: 0xeeeeee,
                metalness: 0.3,
                roughness: 0.5
            }});

            for (let i = 0; i < numOHT; i++) {{
                const oht = new THREE.Group();

                // 상부 (레일에 연결되는 부분)
                const top = new THREE.Mesh(
                    new THREE.BoxGeometry(8, 6, 6),
                    ohtTopMat
                );
                top.position.y = 0;
                oht.add(top);

                // 본체
                const body = new THREE.Mesh(
                    new THREE.BoxGeometry(12, 10, 10),
                    ohtBodyMat
                );
                body.position.y = -10;
                oht.add(body);

                // FOUP (웨이퍼 캐리어) - 50% 확률
                if (Math.random() > 0.5) {{
                    const foup = new THREE.Mesh(
                        new THREE.BoxGeometry(8, 12, 8),
                        foupMat
                    );
                    foup.position.y = -25;
                    oht.add(foup);
                }}

                // 랜덤 경로 선택
                const pathIdx = Math.floor(Math.random() * paths.length);
                const path = paths[pathIdx];

                oht.position.set(path.from.x, RAIL_H - 8, path.from.y);
                oht.castShadow = true;
                scene.add(oht);

                ohtVehicles.push({{
                    mesh: oht,
                    path: path,
                    progress: Math.random(),
                    speed: 0.002 + Math.random() * 0.003
                }});
            }}
        }}

        function updateOHT() {{
            const nodeMap = {{}};
            data.nodes.forEach(n => nodeMap[n.id] = n);

            const paths = [];
            data.edges.forEach(edge => {{
                const from = nodeMap[edge.from];
                const to = nodeMap[edge.to];
                if (from && to) paths.push({{ from, to }});
            }});

            ohtVehicles.forEach(oht => {{
                oht.progress += oht.speed;

                if (oht.progress >= 1) {{
                    oht.progress = 0;
                    // 새 경로 선택
                    const newPath = paths[Math.floor(Math.random() * paths.length)];
                    oht.path = newPath;
                }}

                // 위치 보간
                const from = oht.path.from;
                const to = oht.path.to;
                const x = from.x + (to.x - from.x) * oht.progress;
                const z = from.y + (to.y - from.y) * oht.progress;

                oht.mesh.position.x = x;
                oht.mesh.position.z = z;

                // 진행 방향으로 회전
                const angle = Math.atan2(to.x - from.x, to.y - from.y);
                oht.mesh.rotation.y = angle;
            }});
        }}

        function animate() {{
            requestAnimationFrame(animate);
            updateOHT();
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