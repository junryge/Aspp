
#!/usr/bin/env python3
"""
OHT Layout XML to HTML Visualizer
SK Hynix M14 OHT System Layout Visualization
"""

import xml.etree.ElementTree as ET
import json
import re
import sys
from collections import defaultdict

def parse_layout_xml(xml_file):
    """Parse layout.xml and extract addresses, connections, and stations"""

    print("Loading XML file... (this may take a while for large files)")

    addresses = {}  # addr_no -> {x, y, stations: [], next_addrs: []}
    stations = {}   # station_no -> {port_id, type, addr_no}
    connections = []  # [(from_addr, to_addr)]

    context = ET.iterparse(xml_file, events=('start', 'end'))

    current_addr = None
    current_addr_no = None
    current_station = None
    current_station_no = None
    current_next_addr = None

    addr_data = {}
    station_data = {}
    next_addr_data = {}
    count = 0

    for event, elem in context:
        if event == 'start':
            if elem.tag == 'group':
                name = elem.get('name', '')
                cls = elem.get('class', '')

                if 'Addr' in name and 'address.Addr' in cls and 'NextAddr' not in name:
                    match = re.search(r'Addr(\d+)', name)
                    if match:
                        current_addr_no = int(match.group(1))
                        addr_data = {'x': 0, 'y': 0, 'stations': [], 'next_addrs': []}
                        current_addr = True

                elif 'Station' in name and 'Station' in cls:
                    match = re.search(r'Station(\d+)', name)
                    if match:
                        current_station_no = int(match.group(1))
                        station_data = {'port_id': '', 'type': 0, 'addr_no': current_addr_no}
                        current_station = True

                elif 'NextAddr' in name and 'NextAddr' in cls:
                    current_next_addr = True
                    next_addr_data = {'next_address': None}

        elif event == 'end':
            if elem.tag == 'param':
                key = elem.get('key', '')
                value = elem.get('value', '')

                if current_addr and not current_station and not current_next_addr:
                    if key == 'draw-x':
                        try: addr_data['x'] = float(value)
                        except: pass
                    elif key == 'draw-y':
                        try: addr_data['y'] = float(value)
                        except: pass

                elif current_station:
                    if key == 'port-id':
                        station_data['port_id'] = value
                    elif key == 'type':
                        try: station_data['type'] = int(value)
                        except: pass
                    elif key == 'no':
                        try: station_data['no'] = int(value)
                        except: pass

                elif current_next_addr:
                    if key == 'next-address':
                        try: next_addr_data['next_address'] = int(value)
                        except: pass

            elif elem.tag == 'group':
                name = elem.get('name', '')

                if current_next_addr and 'NextAddr' in name:
                    if next_addr_data.get('next_address') is not None:
                        addr_data['next_addrs'].append(next_addr_data['next_address'])
                    current_next_addr = False

                elif current_station and 'Station' in name:
                    if current_station_no:
                        stations[current_station_no] = station_data.copy()
                        if current_addr_no:
                            addr_data['stations'].append(current_station_no)
                    current_station = False
                    current_station_no = None

                elif current_addr and 'Addr' in name and 'NextAddr' not in name:
                    if current_addr_no is not None:
                        addresses[current_addr_no] = addr_data.copy()
                        for next_addr in addr_data['next_addrs']:
                            connections.append((current_addr_no, next_addr))
                        count += 1
                        if count % 1000 == 0:
                            print(f"  Processed {count} addresses...")
                    current_addr = False
                    current_addr_no = None

            elem.clear()

    print(f"Parsing complete: {len(addresses)} addresses, {len(stations)} stations, {len(connections)} connections")
    return addresses, stations, connections


def generate_html(addresses, stations, connections, output_file):
    """Generate interactive HTML visualization"""

    min_x = min(a['x'] for a in addresses.values() if a['x'] != 0)
    max_x = max(a['x'] for a in addresses.values())
    min_y = min(a['y'] for a in addresses.values() if a['y'] != 0)
    max_y = max(a['y'] for a in addresses.values())

    print(f"Layout bounds: X({min_x:.1f} - {max_x:.1f}), Y({min_y:.1f} - {max_y:.1f})")

    addr_list = [{'no': no, 'x': d['x'], 'y': d['y'], 'stations': d['stations']} 
                 for no, d in addresses.items()]
    station_list = [{'no': no, 'port_id': d.get('port_id',''), 'type': d.get('type',0)} 
                    for no, d in stations.items()]
    valid_connections = [(f, t) for f, t in connections if f in addresses and t in addresses]

    # HTML 템플릿 (JavaScript 포함)
    html = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>OHT Layout</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:sans-serif;background:#1a1a2e;color:#eee;overflow:hidden}}
#header{{position:fixed;top:0;left:0;right:0;background:#16213e;padding:10px 20px;z-index:1000;display:flex;justify-content:space-between}}
#header h1{{font-size:18px;color:#00d4ff}}
#controls{{position:fixed;top:50px;left:10px;background:rgba(22,33,62,0.95);padding:15px;border-radius:8px;z-index:1000;font-size:12px}}
#controls label{{display:block;margin:5px 0;color:#00d4ff}}
#controls button{{margin:5px 2px;padding:5px 10px;background:#0077b6;border:none;color:white;border-radius:4px;cursor:pointer}}
#info{{position:fixed;bottom:10px;left:10px;background:rgba(22,33,62,0.95);padding:10px;border-radius:8px;z-index:1000;font-size:11px;max-width:280px}}
canvas{{display:block}}
</style></head><body>
<div id="header"><h1>SK Hynix M14 OHT Layout</h1><span>Addr:{len(addresses):,} | Stn:{len(stations):,}</span></div>
<div id="controls">
<label>Zoom:<span id="zv">100%</span></label><input type="range" id="zoom" min="10" max="500" value="100" style="width:120px">
<div><input type="checkbox" id="sr" checked>Rails <input type="checkbox" id="ss" checked>Stations</div>
<button onclick="fit()">Fit</button><button onclick="reset()">Reset</button>
<label>Search:</label><input id="q" style="width:100px;background:#1a1a2e;border:1px solid #0077b6;color:#fff" placeholder="ST-10001">
<button onclick="search()">Find</button>
</div>
<div id="info"><b>Info</b><div id="ic">Drag to pan, scroll to zoom</div></div>
<canvas id="c"></canvas>
<script>
const A={json.dumps(addr_list)};
const S={json.dumps(station_list)};
const C={json.dumps(valid_connections)};
const AM=new Map();A.forEach(a=>AM.set(a.no,a));
const SM=new Map();S.forEach(s=>SM.set(s.no,s));
const B={{minX:{min_x},maxX:{max_x},minY:{min_y},maxY:{max_y}}};
const c=document.getElementById('c'),x=c.getContext('2d');
let w,h,sc=0.5,ox=0,oy=0,dr=false,lx,ly;
function rs(){{w=innerWidth;h=innerHeight-50;c.width=w;c.height=h;rn()}}
function w2s(px,py){{return{{x:(px-B.minX)*sc+ox,y:h-((py-B.minY)*sc+oy)}}}}
function s2w(sx,sy){{return{{x:(sx-ox)/sc+B.minX,y:(h-sy-oy)/sc+B.minY}}}}
function rn(){{
x.fillStyle='#1a1a2e';x.fillRect(0,0,w,h);
if(document.getElementById('sr').checked){{
x.strokeStyle='#00ff8855';x.lineWidth=Math.max(1,sc*0.5);x.beginPath();
for(const[f,t]of C){{const a=AM.get(f),b=AM.get(t);if(a&&b){{const p1=w2s(a.x,a.y),p2=w2s(b.x,b.y);x.moveTo(p1.x,p1.y);x.lineTo(p2.x,p2.y)}}}}
x.stroke()}}
if(document.getElementById('ss').checked){{
for(const a of A){{if(a.stations&&a.stations.length){{const p=w2s(a.x,a.y);
if(p.x>-20&&p.x<w+20&&p.y>-20&&p.y<h+20){{
for(const sn of a.stations){{const st=SM.get(sn);if(st){{
x.fillStyle=(st.type>=5&&st.type<=9)?'#ff6b6b':'#ffd93d';
x.beginPath();x.arc(p.x,p.y,Math.max(2,sc*0.8),0,Math.PI*2);x.fill()}}}}}}}}}}}}}}
function fit(){{const lw=B.maxX-B.minX,lh=B.maxY-B.minY;sc=Math.min((w-100)/lw,(h-100)/lh);ox=(w-lw*sc)/2;oy=(h-lh*sc)/2;document.getElementById('zoom').value=sc*100;document.getElementById('zv').textContent=Math.round(sc*100)+'%';rn()}}
function reset(){{sc=0.5;ox=0;oy=0;document.getElementById('zoom').value=50;document.getElementById('zv').textContent='50%';rn()}}
function search(){{const q=document.getElementById('q').value.trim().toUpperCase();if(!q)return;
for(const s of S){{if(s.port_id&&s.port_id.toUpperCase().includes(q)){{
for(const a of A){{if(a.stations&&a.stations.includes(s.no)){{
const p=w2s(a.x,a.y);ox+=w/2-p.x;oy+=h/2-(h-p.y);
document.getElementById('ic').innerHTML='<b>'+s.port_id+'</b><br>No:'+s.no+'<br>Type:'+s.type;rn();return}}}}}}}}
document.getElementById('ic').innerHTML='Not found'}}
c.onmousedown=e=>{{dr=true;lx=e.clientX;ly=e.clientY}};
c.onmousemove=e=>{{if(dr){{ox+=e.clientX-lx;oy-=e.clientY-ly;lx=e.clientX;ly=e.clientY;rn()}}}};
c.onmouseup=()=>{{dr=false}};
c.onwheel=e=>{{e.preventDefault();const z=e.deltaY>0?0.9:1.1,mx=e.clientX,my=e.clientY-50;
const wb=s2w(mx,my);sc*=z;sc=Math.max(0.05,Math.min(10,sc));const sa=w2s(wb.x,wb.y);
ox+=mx-sa.x;oy-=my-(h-sa.y);document.getElementById('zoom').value=sc*100;document.getElementById('zv').textContent=Math.round(sc*100)+'%';rn()}};
document.getElementById('zoom').oninput=e=>{{sc=e.target.value/100;document.getElementById('zv').textContent=Math.round(sc*100)+'%';rn()}};
document.getElementById('sr').onchange=rn;document.getElementById('ss').onchange=rn;
document.getElementById('q').onkeypress=e=>{{if(e.key==='Enter')search()}};
onresize=rs;rs();fit();
</script></body></html>'''

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML generated: {output_file}")


def main():
    # 경로 수정해서 사용하세요
    xml_file = 'layout.xml'  # 입력 XML 파일
    output_file = 'oht_layout_viewer.html'  # 출력 HTML 파일

    print("=" * 50)
    print("OHT Layout XML to HTML Visualizer")
    print("=" * 50)

    addresses, stations, connections = parse_layout_xml(xml_file)
    generate_html(addresses, stations, connections, output_file)
    print("\nDone! Open HTML in browser.")


if __name__ == '__main__':
    main()