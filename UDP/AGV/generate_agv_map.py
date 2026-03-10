#!/usr/bin/env python3
"""AGV MAP XML 데이터를 파싱하여 인터랙티브 HTML MAP을 생성하는 스크립트"""

import xml.etree.ElementTree as ET
import json
import os

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'I3AV0BMap.xml')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agv_map.html')


def parse_xml(xml_path):
    """XML 파일에서 NODE, EDGE 데이터를 파싱"""
    print(f"Loading XML: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # FORMAT_VERSION
    version = root.find('FORMAT_VERSION')
    version_text = version.text if version is not None else 'unknown'
    print(f"Format Version: {version_text}")

    # Parse NODES
    nodes = []
    equip_count = 0
    for node_el in root.findall('.//NODES/NODE'):
        node_id = int(node_el.find('ID').text)
        pos = node_el.find('Position')
        x = int(pos.find('X').text)
        y = int(pos.find('Y').text)

        eq = None
        eq_el = node_el.find('Equipment')
        if eq_el is not None:
            equip_count += 1
            eq = {
                'name': eq_el.findtext('EQ_NAME', ''),
                'type': eq_el.findtext('EQ_TYPE', ''),
                'slot': eq_el.findtext('EQ_SLOT', ''),
                'deg': eq_el.findtext('EQ_DEG', ''),
                'alias': eq_el.findtext('EQ_ALIAS', ''),
            }
        nodes.append({'id': node_id, 'x': x, 'y': y, 'eq': eq})

    print(f"Nodes: {len(nodes)} (Equipment: {equip_count})")

    # Parse EDGES
    edges = []
    for edge_el in root.findall('.//EDGES/EDGE'):
        nd_from = int(edge_el.find('ND_FROM').text)
        nd_to = int(edge_el.find('ND_TO').text)
        edges.append({'from': nd_from, 'to': nd_to})

    print(f"Edges: {len(edges)}")

    # EQ_TYPE 통계
    eq_types = {}
    for n in nodes:
        if n['eq']:
            t = n['eq']['type']
            eq_types[t] = eq_types.get(t, 0) + 1
    print(f"Equipment Types: {eq_types}")

    return {'version': version_text, 'nodes': nodes, 'edges': edges}


def generate_html(data, output_path):
    """파싱된 데이터로 인터랙티브 HTML 생성"""
    nodes_json = json.dumps(data['nodes'], separators=(',', ':'))
    edges_json = json.dumps(data['edges'], separators=(',', ':'))

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>AGV MAP - I3AV0BMap (v{data['version']})</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif; overflow: hidden; }}

#toolbar {{
  position: fixed; top: 0; left: 0; right: 0; z-index: 10;
  background: #16213e; padding: 8px 16px; display: flex; align-items: center; gap: 12px;
  border-bottom: 2px solid #0f3460; font-size: 13px; flex-wrap: wrap;
}}
#toolbar h1 {{ font-size: 16px; color: #e94560; margin-right: 12px; white-space: nowrap; }}
#toolbar .ver {{ color: #666; font-size: 11px; margin-right: 12px; }}
#toolbar label {{ color: #ccc; cursor: pointer; white-space: nowrap; }}
#toolbar input[type=checkbox] {{ margin-right: 3px; accent-color: #e94560; }}
#toolbar select, #toolbar input[type=text] {{
  background: #0f3460; color: #eee; border: 1px solid #555; padding: 4px 8px; border-radius: 4px; font-size: 12px;
}}
#toolbar button {{
  background: #e94560; color: #fff; border: none; padding: 5px 14px; border-radius: 4px; cursor: pointer; font-size: 12px;
}}
#toolbar button:hover {{ background: #c73652; }}
#toolbar button.secondary {{ background: #0f3460; }}
#toolbar button.secondary:hover {{ background: #1a4a8a; }}

#info {{
  position: fixed; bottom: 10px; left: 10px; z-index: 10;
  background: rgba(22,33,62,0.92); padding: 8px 14px; border-radius: 6px;
  font-size: 12px; color: #aaa; border: 1px solid #0f3460;
}}
#info b {{ color: #53d8fb; }}

#tooltip {{
  position: fixed; display: none; z-index: 20;
  background: rgba(15,52,96,0.96); padding: 12px 16px; border-radius: 8px;
  font-size: 12px; color: #eee; border: 1px solid #e94560; pointer-events: none;
  max-width: 300px; box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}}
#tooltip .tt-name {{ color: #e94560; font-weight: bold; font-size: 15px; margin-bottom: 4px; }}
#tooltip .tt-type {{ color: #53d8fb; font-size: 13px; margin-bottom: 6px; }}
#tooltip .tt-row {{ color: #bbb; margin: 2px 0; }}
#tooltip .tt-row span {{ color: #eee; }}

#legend {{
  position: fixed; top: 52px; right: 10px; z-index: 10;
  background: rgba(22,33,62,0.92); padding: 12px 16px; border-radius: 6px;
  font-size: 12px; border: 1px solid #0f3460;
}}
#legend .title {{ color: #e94560; font-weight: bold; margin-bottom: 6px; }}
#legend .item {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
#legend .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; flex-shrink: 0; }}
#legend .cnt {{ color: #666; margin-left: auto; }}

canvas {{ display: block; cursor: grab; }}
canvas:active {{ cursor: grabbing; }}

#loading {{
  position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
  font-size: 22px; color: #e94560; z-index: 100;
}}
</style>
</head>
<body>

<div id="toolbar">
  <h1>AGV MAP Viewer</h1>
  <span class="ver">v{data['version']}</span>
  <label><input type="checkbox" id="showEdges" checked> Edges</label>
  <label><input type="checkbox" id="showNodes" checked> Nodes</label>
  <label><input type="checkbox" id="showEquip" checked> Equipment</label>
  <label><input type="checkbox" id="showLabels"> Labels</label>
  <label><input type="checkbox" id="showDirection"> Direction</label>
  <select id="filterType">
    <option value="ALL">All Types</option>
    <option value="Sorter">Sorter</option>
    <option value="Chamber">Chamber</option>
    <option value="Buffer">Buffer</option>
    <option value="CHARGER">CHARGER</option>
    <option value="LIFT">LIFT</option>
  </select>
  <input type="text" id="searchEq" placeholder="Search EQ_NAME..." style="width:150px;">
  <button onclick="doSearch()">Search</button>
  <button class="secondary" onclick="resetView()">Reset View</button>
  <button class="secondary" onclick="fitEquipment()">Fit Equipment</button>
</div>

<div id="legend">
  <div class="title">Equipment Types</div>
</div>

<div id="tooltip"></div>
<div id="info">Loading...</div>
<div id="loading">Loading AGV MAP data...</div>

<canvas id="canvas"></canvas>

<script>
// ===== DATA (embedded from Python) =====
const rawNodes = {nodes_json};
const rawEdges = {edges_json};

// ===== CONSTANTS =====
const EQ_COLORS = {{
  'Sorter':  '#e94560',
  'Chamber': '#53d8fb',
  'Buffer':  '#4ecca3',
  'CHARGER': '#ffd700',
  'LIFT':    '#ff6b6b'
}};

// ===== STATE =====
let nodesById = {{}};
let equipNodes = [];
let offsetX = 0, offsetY = 0, scale = 1;
let dragging = false, dragSX = 0, dragSY = 0;
let hoveredNode = null;
let searchResults = [];

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// ===== INIT =====
function init() {{
  rawNodes.forEach(n => {{
    nodesById[n.id] = n;
    if (n.eq) equipNodes.push(n);
  }});

  // Build legend
  const typeCounts = {{}};
  equipNodes.forEach(n => {{
    typeCounts[n.eq.type] = (typeCounts[n.eq.type] || 0) + 1;
  }});
  const legend = document.getElementById('legend');
  Object.entries(typeCounts).sort((a, b) => b[1] - a[1]).forEach(([type, cnt]) => {{
    const color = EQ_COLORS[type] || '#888';
    legend.innerHTML += `<div class="item"><span class="dot" style="background:${{color}};"></span>${{type}}<span class="cnt">${{cnt}}</span></div>`;
  }});
  legend.innerHTML += `<div class="item"><span class="dot" style="background:#444;"></span>Node<span class="cnt">${{rawNodes.length - equipNodes.length}}</span></div>`;

  document.getElementById('loading').style.display = 'none';
  resizeCanvas();
  resetView();
  draw();
  updateInfo();
}}

function resizeCanvas() {{
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}}

window.addEventListener('resize', () => {{ resizeCanvas(); draw(); }});

// ===== VIEW CONTROL =====
function resetView() {{
  if (rawNodes.length === 0) return;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  rawNodes.forEach(n => {{
    if (n.x < minX) minX = n.x;
    if (n.x > maxX) maxX = n.x;
    if (n.y < minY) minY = n.y;
    if (n.y > maxY) maxY = n.y;
  }});
  fitBounds(minX, minY, maxX, maxY, 80);
}}

function fitEquipment() {{
  if (equipNodes.length === 0) return;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const filter = document.getElementById('filterType').value;
  const targets = filter === 'ALL' ? equipNodes : equipNodes.filter(n => n.eq.type === filter);
  if (targets.length === 0) return;
  targets.forEach(n => {{
    if (n.x < minX) minX = n.x;
    if (n.x > maxX) maxX = n.x;
    if (n.y < minY) minY = n.y;
    if (n.y > maxY) maxY = n.y;
  }});
  fitBounds(minX, minY, maxX, maxY, 100);
}}

function fitBounds(minX, minY, maxX, maxY, pad) {{
  const w = canvas.width - pad * 2;
  const h = canvas.height - pad * 2;
  const rX = (maxX - minX) || 1;
  const rY = (maxY - minY) || 1;
  scale = Math.min(w / rX, h / rY);
  offsetX = pad - minX * scale + (w - rX * scale) / 2;
  offsetY = pad - minY * scale + (h - rY * scale) / 2;
  draw();
  updateInfo();
}}

function worldToScreen(wx, wy) {{
  return [wx * scale + offsetX, wy * scale + offsetY];
}}

function screenToWorld(sx, sy) {{
  return [(sx - offsetX) / scale, (sy - offsetY) / scale];
}}

// ===== DRAWING =====
function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const showEdges = document.getElementById('showEdges').checked;
  const showNodes = document.getElementById('showNodes').checked;
  const showEquip = document.getElementById('showEquip').checked;
  const showLabels = document.getElementById('showLabels').checked;
  const showDir = document.getElementById('showDirection').checked;
  const filter = document.getElementById('filterType').value;
  const cw = canvas.width, ch = canvas.height;

  // Edges
  if (showEdges) {{
    const lw = Math.max(0.5, Math.min(2, scale * 100));
    ctx.lineWidth = lw;

    if (showDir) {{
      // Directional edges with arrows
      ctx.strokeStyle = 'rgba(80,100,140,0.4)';
      rawEdges.forEach(e => {{
        const nf = nodesById[e.from], nt = nodesById[e.to];
        if (!nf || !nt) return;
        const [x1, y1] = worldToScreen(nf.x, nf.y);
        const [x2, y2] = worldToScreen(nt.x, nt.y);
        if (Math.max(x1, x2) < -50 || Math.min(x1, x2) > cw + 50) return;
        if (Math.max(y1, y2) < -50 || Math.min(y1, y2) > ch + 50) return;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        // Arrow
        const arrowSize = Math.max(3, Math.min(8, scale * 500));
        const dx = x2 - x1, dy = y2 - y1;
        const len = Math.sqrt(dx * dx + dy * dy);
        if (len < 5) return;
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        const ux = dx / len, uy = dy / len;
        ctx.fillStyle = 'rgba(80,100,140,0.6)';
        ctx.beginPath();
        ctx.moveTo(mx + ux * arrowSize, my + uy * arrowSize);
        ctx.lineTo(mx - ux * arrowSize + uy * arrowSize * 0.5, my - uy * arrowSize - ux * arrowSize * 0.5);
        ctx.lineTo(mx - ux * arrowSize - uy * arrowSize * 0.5, my - uy * arrowSize + ux * arrowSize * 0.5);
        ctx.fill();
      }});
    }} else {{
      ctx.strokeStyle = 'rgba(80,100,140,0.3)';
      ctx.beginPath();
      rawEdges.forEach(e => {{
        const nf = nodesById[e.from], nt = nodesById[e.to];
        if (!nf || !nt) return;
        const [x1, y1] = worldToScreen(nf.x, nf.y);
        const [x2, y2] = worldToScreen(nt.x, nt.y);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }});
      ctx.stroke();
    }}
  }}

  // Plain nodes
  if (showNodes) {{
    const r = Math.max(1, Math.min(3.5, scale * 250));
    ctx.fillStyle = '#3a3a5a';
    rawNodes.forEach(n => {{
      if (n.eq) return;
      const [sx, sy] = worldToScreen(n.x, n.y);
      if (sx < -10 || sx > cw + 10 || sy < -10 || sy > ch + 10) return;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fill();
    }});
  }}

  // Equipment nodes
  if (showEquip) {{
    const r = Math.max(3, Math.min(12, scale * 700));
    const fontSize = Math.max(8, Math.min(13, scale * 900));

    equipNodes.forEach(n => {{
      if (filter !== 'ALL' && n.eq.type !== filter) return;
      const [sx, sy] = worldToScreen(n.x, n.y);
      if (sx < -20 || sx > cw + 20 || sy < -20 || sy > ch + 20) return;

      const color = EQ_COLORS[n.eq.type] || '#888';
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fill();

      // Hover highlight
      if (n === hoveredNode) {{
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(sx, sy, r + 3, 0, Math.PI * 2);
        ctx.stroke();
      }}

      // Labels
      if (showLabels && scale > 0.006) {{
        ctx.fillStyle = '#fff';
        ctx.font = `${{fontSize}}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(n.eq.name, sx, sy - r - 3);
        if (n.eq.slot) {{
          ctx.fillStyle = '#999';
          ctx.font = `${{Math.max(7, fontSize - 2)}}px sans-serif`;
          ctx.fillText(`[${{n.eq.slot}}]`, sx, sy + r + fontSize - 2);
        }}
      }}
    }});
  }}

  // Search highlights
  if (searchResults.length > 0) {{
    const r = Math.max(10, Math.min(22, scale * 1500));
    searchResults.forEach(n => {{
      const [sx, sy] = worldToScreen(n.x, n.y);
      // Pulsing ring
      ctx.strokeStyle = '#ffd700';
      ctx.lineWidth = 3;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#ffd700';
      ctx.font = 'bold 13px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(n.eq.name, sx, sy - r - 4);
    }});
  }}
}}

// ===== SEARCH =====
function doSearch() {{
  const val = document.getElementById('searchEq').value.trim().toUpperCase();
  if (!val) {{ searchResults = []; draw(); return; }}
  searchResults = equipNodes.filter(n => n.eq.name.toUpperCase().includes(val));
  if (searchResults.length > 0) {{
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    searchResults.forEach(n => {{
      if (n.x < minX) minX = n.x;
      if (n.x > maxX) maxX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.y > maxY) maxY = n.y;
    }});
    const pad = Math.max(5000, (maxX - minX) * 0.3, (maxY - minY) * 0.3);
    fitBounds(minX - pad, minY - pad, maxX + pad, maxY + pad, 60);
  }}
  draw();
}}

// ===== INFO =====
function updateInfo() {{
  const filter = document.getElementById('filterType').value;
  const visible = filter === 'ALL' ? equipNodes.length : equipNodes.filter(n => n.eq.type === filter).length;
  document.getElementById('info').innerHTML =
    `Nodes: <b>${{rawNodes.length}}</b> | Edges: <b>${{rawEdges.length}}</b> | ` +
    `Equipment: <b>${{visible}}</b> | Zoom: <b>${{(scale * 1000).toFixed(1)}}x</b>` +
    (searchResults.length ? ` | Found: <b style="color:#ffd700">${{searchResults.length}}</b>` : '');
}}

// ===== MOUSE =====
canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  const [wx, wy] = screenToWorld(e.clientX, e.clientY);
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  scale *= factor;
  offsetX = e.clientX - wx * scale;
  offsetY = e.clientY - wy * scale;
  draw(); updateInfo();
}}, {{ passive: false }});

canvas.addEventListener('mousedown', e => {{
  dragging = true;
  dragSX = e.clientX - offsetX;
  dragSY = e.clientY - offsetY;
}});

canvas.addEventListener('mousemove', e => {{
  if (dragging) {{
    offsetX = e.clientX - dragSX;
    offsetY = e.clientY - dragSY;
    draw(); updateInfo();
    return;
  }}
  const [wx, wy] = screenToWorld(e.clientX, e.clientY);
  const threshold = Math.max(400, 10 / scale);
  let closest = null, closestD = Infinity;
  equipNodes.forEach(n => {{
    const d = Math.hypot(n.x - wx, n.y - wy);
    if (d < threshold && d < closestD) {{ closestD = d; closest = n; }}
  }});

  if (closest !== hoveredNode) {{
    hoveredNode = closest;
    draw();
  }}

  const tt = document.getElementById('tooltip');
  if (closest) {{
    tt.style.display = 'block';
    tt.style.left = Math.min(e.clientX + 15, window.innerWidth - 310) + 'px';
    tt.style.top = (e.clientY - 10) + 'px';
    tt.innerHTML =
      `<div class="tt-name">${{closest.eq.name}}</div>` +
      `<div class="tt-type">${{closest.eq.type}}</div>` +
      `<div class="tt-row">Alias: <span>${{closest.eq.alias}}</span></div>` +
      `<div class="tt-row">Slot: <span>${{closest.eq.slot || '-'}}</span></div>` +
      `<div class="tt-row">Degree: <span>${{closest.eq.deg}}&deg;</span></div>` +
      `<div class="tt-row">Node ID: <span>${{closest.id}}</span></div>` +
      `<div class="tt-row">Position: <span>(${{closest.x}}, ${{closest.y}})</span></div>`;
  }} else {{
    tt.style.display = 'none';
  }}
}});

canvas.addEventListener('mouseup', () => {{ dragging = false; }});
canvas.addEventListener('mouseleave', () => {{
  dragging = false; hoveredNode = null;
  document.getElementById('tooltip').style.display = 'none';
  draw();
}});

// Controls
['showEdges','showNodes','showEquip','showLabels','showDirection'].forEach(id => {{
  document.getElementById(id).addEventListener('change', draw);
}});
document.getElementById('filterType').addEventListener('change', () => {{ draw(); updateInfo(); }});
document.getElementById('searchEq').addEventListener('keyup', e => {{ if (e.key === 'Enter') doSearch(); }});

// Start
init();
</script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\nHTML generated: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == '__main__':
    data = parse_xml(XML_PATH)
    generate_html(data, OUTPUT_PATH)
    print("Done!")
