#!/usr/bin/env python3
"""
OHT Layout XML → HTML 변환기
- HID ID → Zone ID / MCP ID 매핑 뷰어
- CSV 매핑 테이블 저장
"""

import xml.etree.ElementTree as ET
import json
import re
import os
import csv
from tkinter import Tk, filedialog, messagebox

# HID_Zone_Master.csv 생성 - 이 파일 내에서 직접 생성 (hid_zone_csv_cre.py와 동일한 포맷)

def select_file():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="OHT Layout XML 파일 선택",
        filetypes=[("XML files", "*.xml"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

def parse_xml(xml_file):
    print("XML 파싱 중...")
    addresses = {}
    stations = {}
    mcp_zones = {}
    connections = []
    
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
                        addr_data = {'x': 0, 'y': 0, 'stations': [], 'next_addrs': [], 'mcp_zone_id': '', 'mcp_id': ''}
                        current_addr = True
                        
                elif 'Station' in name and 'Station' in cls:
                    match = re.search(r'Station(\d+)', name)
                    if match:
                        current_station_no = int(match.group(1))
                        station_data = {'port_id': '', 'type': 0}
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
                            print(f"  {count} addresses...")
                    current_addr = False
                    current_addr_no = None
                    
            elem.clear()
    
    print(f"  Address/Station 완료: {len(addresses):,} addr, {len(stations):,} stn")
    
    # McpZone 파싱 (그룹 깊이 추적 방식)
    print("McpZone 파싱 중...")
    with open(xml_file, 'r', encoding='utf-8') as f:
        xml_content = f.read()

    current_mcp_zone = None    # McpZone 파싱 중인지 여부
    current_mcp_id = None      # McpZone123의 123
    current_zone_id = None     # param id의 값
    current_entries = []
    current_exits = []
    in_entry = False
    in_exit = False
    entry_start = None
    entry_end = None
    exit_start = None
    exit_end = None
    zone_depth = 0  # McpZone 내부 그룹 깊이 추적

    lines = xml_content.split('\n')
    total = len(lines)

    for i, line in enumerate(lines):
        if i % 500000 == 0 and i > 0:
            print(f"  {i:,}/{total:,} ({i*100//total}%)")

        line = line.strip()

        # McpZone 그룹 시작
        if '<group name="McpZone' in line and 'mcpzone.McpZone"' in line:
            # 이전 zone 저장 (있다면)
            if current_mcp_zone is not None and current_zone_id is not None:
                mcp_zones[current_zone_id] = {
                    'mcp_id': current_mcp_id,
                    'zone_id': current_zone_id,
                    'entries': current_entries.copy(),
                    'exits': current_exits.copy()
                }

            # McpZone name에서 숫자 추출
            match = re.search(r'McpZone(\d+)', line)
            if match:
                current_mcp_id = int(match.group(1))
            else:
                current_mcp_id = None

            current_mcp_zone = True
            current_zone_id = None
            current_entries = []
            current_exits = []
            in_entry = False
            in_exit = False
            zone_depth = 1  # McpZone 그룹 시작
            continue

        # current_mcp_zone이 없으면 스킵
        if current_mcp_zone is None:
            continue

        # Entry 그룹 시작
        if '<group name="Entry' in line and 'mcpzone.Entry"' in line:
            in_entry = True
            entry_start = None
            entry_end = None
            zone_depth += 1
            continue

        # Exit 그룹 시작
        if '<group name="Exit' in line and 'mcpzone.Exit"' in line:
            in_exit = True
            exit_start = None
            exit_end = None
            zone_depth += 1
            continue

        # CutLane 그룹 시작 (depth만 추적)
        if '<group name="CutLane' in line and 'mcpzone.CutLane"' in line:
            zone_depth += 1
            continue

        # 다른 그룹 시작 (Entry/Exit/CutLane 외)
        if '<group ' in line and '>' in line and '/>' not in line:
            zone_depth += 1
            continue

        # 파라미터 파싱 (McpZone 직속 - zone_depth == 1)
        if '<param ' in line and 'key="id"' in line and zone_depth == 1:
            match = re.search(r'value="(\d+)"', line)
            if match:
                current_zone_id = int(match.group(1))

        if in_entry and '<param ' in line:
            if 'key="start"' in line:
                match = re.search(r'value="(\d+)"', line)
                if match:
                    entry_start = int(match.group(1))
            elif 'key="end"' in line:
                match = re.search(r'value="(\d+)"', line)
                if match:
                    entry_end = int(match.group(1))

        if in_exit and '<param ' in line:
            if 'key="start"' in line:
                match = re.search(r'value="(\d+)"', line)
                if match:
                    exit_start = int(match.group(1))
            elif 'key="end"' in line:
                match = re.search(r'value="(\d+)"', line)
                if match:
                    exit_end = int(match.group(1))

        # 그룹 종료
        if '</group>' in line:
            # Entry/Exit 그룹 종료 (depth 감소 전에 처리)
            if in_entry:
                if entry_start is not None and entry_end is not None:
                    current_entries.append((entry_start, entry_end))
                in_entry = False
                zone_depth -= 1
                continue
            elif in_exit:
                if exit_start is not None and exit_end is not None:
                    current_exits.append((exit_start, exit_end))
                in_exit = False
                zone_depth -= 1
                continue

            zone_depth -= 1

            # McpZone 그룹 종료
            if zone_depth == 0:
                if current_zone_id is not None:
                    mcp_zones[current_zone_id] = {
                        'mcp_id': current_mcp_id,
                        'zone_id': current_zone_id,
                        'entries': current_entries.copy(),
                        'exits': current_exits.copy()
                    }
                current_mcp_zone = None
                current_zone_id = None
                current_entries = []
                current_exits = []

    print(f"  McpZone 완료: {len(mcp_zones):,} zones")
    
    # Address에 McpZone ID 및 MCP ID 매핑
    print("Address ↔ Zone 매핑 중...")
    addr_to_mcp = {}  # addr_no → {zone_id, mcp_id}
    
    for zone_id, zone_data in mcp_zones.items():
        mcp_id = zone_data.get('mcp_id', '')
        for entry_start, entry_end in zone_data['entries']:
            addr_to_mcp[entry_start] = {'zone_id': zone_id, 'mcp_id': mcp_id}
            addr_to_mcp[entry_end] = {'zone_id': zone_id, 'mcp_id': mcp_id}
        for exit_start, exit_end in zone_data['exits']:
            addr_to_mcp[exit_start] = {'zone_id': zone_id, 'mcp_id': mcp_id}
            addr_to_mcp[exit_end] = {'zone_id': zone_id, 'mcp_id': mcp_id}
    
    mapped_count = 0
    for addr_no in addresses:
        if addr_no in addr_to_mcp:
            addresses[addr_no]['mcp_zone_id'] = str(addr_to_mcp[addr_no]['zone_id'])
            addresses[addr_no]['mcp_id'] = str(addr_to_mcp[addr_no]['mcp_id'])
            mapped_count += 1
    
    print(f"  매핑 완료: {mapped_count:,}/{len(addresses):,} addresses")
    print(f"완료: {len(addresses):,} addr, {len(stations):,} stn, {len(mcp_zones):,} zones, {len(connections):,} conn")
    
    return addresses, stations, mcp_zones, connections

def save_mapping_csv(addresses, stations, output_csv):
    """HID → Zone / MCP ID 매핑 CSV 저장 (layout_HID_Zone_MCP_Mapping.csv 형식)"""
    print(f"CSV 저장 중: {output_csv}")

    hid_zone_map = []
    for addr_no, addr_data in addresses.items():
        zone_id = addr_data.get('mcp_zone_id', '')
        mcp_id = addr_data.get('mcp_id', '')
        for stn_no in addr_data.get('stations', []):
            if stn_no in stations:
                hid_id = stations[stn_no].get('port_id', '')
                stn_type = stations[stn_no].get('type', 0)
                if hid_id:
                    hid_zone_map.append({
                        'HID_ID': hid_id,
                        'Zone_ID': zone_id,
                        'MCP_ID': mcp_id,
                        'Addr_No': addr_no,
                        'Station_No': stn_no,
                        'Station_Type': stn_type,
                        'X': addr_data.get('x', 0),
                        'Y': addr_data.get('y', 0)
                    })

    # HID_ID로 정렬
    hid_zone_map.sort(key=lambda x: x['HID_ID'])

    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = ['HID_ID', 'Zone_ID', 'MCP_ID', 'Addr_No', 'Station_No', 'Station_Type', 'X', 'Y']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(hid_zone_map)

    print(f"  {len(hid_zone_map):,}개 매핑 저장 완료")
    return hid_zone_map


def derive_bay_zone(zone_no: int) -> str:
    """Zone 번호에서 Bay_Zone 추정"""
    bay_mapping = {
        1: 'B01', 2: 'B01', 3: 'B02', 4: 'B03', 5: 'B04',
        6: 'B05', 7: 'B05', 8: 'B06', 9: 'B06',
    }
    if zone_no in bay_mapping:
        return bay_mapping[zone_no]
    bay_num = (zone_no // 10) + 1
    return f'B{bay_num:02d}'


def save_hid_zone_master_csv(addresses, stations, mcp_zones, output_csv, project_name="M14 Project Ph-1"):
    """HID_Zone_Master.csv 형식으로 저장 (hid_zone_csv_cre.py와 동일한 포맷)"""
    print(f"HID_Zone_Master.csv 저장 중: {output_csv}")

    # CSV 헤더 (hid_zone_csv_cre.py와 동일)
    headers = [
        'Zone_ID', 'HID_No', 'Bay_Zone', 'Sub_Region', 'Full_Name',
        'Territory', 'Type', 'IN_Count', 'OUT_Count', 'IN_Lanes', 'OUT_Lanes',
        'Vehicle_Max', 'Vehicle_Precaution', 'Project', 'ZCU', 'HID_Type',
        'HID_ID', 'Zone_ID2', 'Addr_No', 'Station_No'
    ]

    # Zone별 HID 매핑 구축
    zone_hid_map = {}  # zone_id -> [{'HID_ID', 'Addr_No', 'Station_No'}, ...]
    for addr_no, addr_data in addresses.items():
        zone_id = addr_data.get('mcp_zone_id', '')
        if not zone_id:
            continue
        try:
            zone_id_int = int(zone_id)
        except:
            continue

        for stn_no in addr_data.get('stations', []):
            if stn_no in stations:
                hid_id = stations[stn_no].get('port_id', '')
                if hid_id:
                    if zone_id_int not in zone_hid_map:
                        zone_hid_map[zone_id_int] = []
                    zone_hid_map[zone_id_int].append({
                        'HID_ID': hid_id,
                        'Addr_No': str(addr_no),
                        'Station_No': str(stn_no)
                    })

    rows = []

    # Zone 정렬
    sorted_zone_ids = sorted(mcp_zones.keys())

    for zone_id in sorted_zone_ids:
        zone_data = mcp_zones[zone_id]
        zone_no = zone_id

        # Entry/Exit lanes
        entries = zone_data.get('entries', [])
        exits = zone_data.get('exits', [])

        in_lanes = '; '.join([f"{e[0]}→{e[1]}" for e in entries])
        out_lanes = '; '.join([f"{e[0]}→{e[1]}" for e in exits])

        # ZCU (없음)
        zcu = ''

        # Bay_Zone 및 Sub_Region 추정
        bay_zone = derive_bay_zone(zone_no)
        sub_region = ((zone_no - 1) % 2) + 1

        # HID 타입 (기본 HID4)
        hid_type = 'HID4'

        # HID 맵핑 정보 가져오기
        hid_list = zone_hid_map.get(zone_id, [])

        if hid_list:
            # 맵핑된 HID가 있으면 각 HID별로 행 생성
            for hid_info in hid_list:
                row = {
                    'Zone_ID': zone_id,
                    'HID_No': f'HID-OHT-{zone_id:03d}',
                    'Bay_Zone': bay_zone,
                    'Sub_Region': sub_region,
                    'Full_Name': f'HID-{bay_zone}-{sub_region}({zone_id:03d})',
                    'Territory': 1,
                    'Type': 'HID',
                    'IN_Count': len(entries),
                    'OUT_Count': len(exits),
                    'IN_Lanes': in_lanes,
                    'OUT_Lanes': out_lanes,
                    'Vehicle_Max': 0,
                    'Vehicle_Precaution': 0,
                    'Project': project_name,
                    'ZCU': zcu,
                    'HID_Type': hid_type,
                    'HID_ID': hid_info['HID_ID'],
                    'Zone_ID2': zone_id,
                    'Addr_No': hid_info['Addr_No'],
                    'Station_No': hid_info['Station_No']
                }
                rows.append(row)
        else:
            # 맵핑된 HID가 없으면 빈 값으로 1행만 생성
            row = {
                'Zone_ID': zone_id,
                'HID_No': f'HID-OHT-{zone_id:03d}',
                'Bay_Zone': bay_zone,
                'Sub_Region': sub_region,
                'Full_Name': f'HID-{bay_zone}-{sub_region}({zone_id:03d})',
                'Territory': 1,
                'Type': 'HID',
                'IN_Count': len(entries),
                'OUT_Count': len(exits),
                'IN_Lanes': in_lanes,
                'OUT_Lanes': out_lanes,
                'Vehicle_Max': 0,
                'Vehicle_Precaution': 0,
                'Project': project_name,
                'ZCU': zcu,
                'HID_Type': hid_type,
                'HID_ID': '',
                'Zone_ID2': '',
                'Addr_No': '',
                'Station_No': ''
            }
            rows.append(row)

    # CSV 파일 작성 (UTF-8 BOM)
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  완료: {len(rows):,}개 행")
    return rows

def generate_html(addresses, stations, mcp_zones, connections, hid_zone_map, output_file):
    min_x = min((a['x'] for a in addresses.values() if a['x'] != 0), default=0)
    max_x = max((a['x'] for a in addresses.values()), default=1000)
    min_y = min((a['y'] for a in addresses.values() if a['y'] != 0), default=0)
    max_y = max((a['y'] for a in addresses.values()), default=1000)

    addr_list = [{'no': no, 'x': d['x'], 'y': d['y'], 'stations': d['stations'], 'zone': d.get('mcp_zone_id',''), 'mcp': d.get('mcp_id','')} for no, d in addresses.items()]
    station_list = [{'no': no, 'port_id': d.get('port_id',''), 'type': d.get('type',0)} for no, d in stations.items()]
    zone_list = [{'id': str(zid), 'mcp': str(zd.get('mcp_id',''))} for zid, zd in mcp_zones.items()]
    valid_connections = [(f, t) for f, t in connections if f in addresses and t in addresses]

    # JSON용 매핑 데이터
    hzm_json = [{'hid': m['HID_ID'], 'zone': m['Zone_ID'], 'mcp': m['MCP_ID'], 'addr': m['Addr_No']} for m in hid_zone_map]

    html = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>HID → Zone/MCP ID 매핑 뷰어</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui;background:#0d1117;color:#c9d1d9;overflow:hidden}}
#ui{{position:fixed;top:10px;left:10px;z-index:100;display:flex;gap:8px;align-items:center;flex-wrap:wrap;max-width:90%}}
#ui button{{padding:6px 12px;background:#21262d;border:1px solid #30363d;color:#c9d1d9;border-radius:6px;cursor:pointer;font-size:12px}}
#ui button:hover{{background:#30363d}}
#ui input[type=text]{{padding:6px 10px;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;border-radius:6px;width:180px;font-size:12px}}
#ui label{{font-size:12px;display:flex;align-items:center;gap:4px;color:#8b949e}}
#stat{{position:fixed;top:10px;right:10px;font-size:11px;color:#8b949e}}
#rot{{position:fixed;bottom:10px;right:10px;font-size:11px;color:#8b949e;background:#21262d;padding:5px 10px;border-radius:6px}}
#info{{position:fixed;bottom:10px;left:10px;font-size:13px;color:#58a6ff;background:#21262d;padding:12px 16px;border-radius:6px;max-width:500px;font-weight:600}}
#mapping{{position:fixed;top:50px;right:10px;font-size:11px;background:#161b22;border:1px solid #30363d;border-radius:6px;max-height:400px;overflow-y:auto;display:none;width:380px}}
#mapping table{{width:100%;border-collapse:collapse}}
#mapping th,#mapping td{{padding:5px 8px;text-align:left;border-bottom:1px solid #30363d}}
#mapping th{{background:#21262d;position:sticky;top:0}}
#mapping tr:hover{{background:#30363d;cursor:pointer}}
canvas{{display:block}}
</style></head><body>
<div id="ui">
<button onclick="fit()">Fit</button>
<button onclick="reset()">Reset</button>
<button onclick="rotate(-90)">↺ 90°</button>
<button onclick="rotate(90)">↻ 90°</button>
<label><input type="checkbox" id="sr" checked>Rail</label>
<label><input type="checkbox" id="ss" checked>Station</label>
<label><input type="checkbox" id="sz" checked>Zone</label>
<label><input type="checkbox" id="sn">HID</label>
<input type="text" id="q" placeholder="HID ID → Zone/MCP 찾기">
<button onclick="search()">Find</button>
<button onclick="toggleMapping()">매핑표</button>
<span id="msg" style="color:#7ee787;font-weight:bold"></span>
</div>
<div id="stat">HID:{len(hid_zone_map):,} | Zone:{len(mcp_zones):,}</div>
<div id="rot">회전: 0°</div>
<div id="info">HID ID 입력 → Zone ID / MCP ID 확인</div>
<div id="mapping">
<table><thead><tr><th>HID ID</th><th>→</th><th>Zone ID</th><th>MCP ID</th></tr></thead>
<tbody id="mapBody"></tbody></table>
</div>
<canvas id="c"></canvas>
<script>
const A={json.dumps(addr_list)};
const S={json.dumps(station_list)};
const Z={json.dumps(zone_list)};
const C={json.dumps(valid_connections)};
const HZM={json.dumps(hzm_json)};

const AM=new Map();A.forEach(a=>AM.set(a.no,a));
const SM=new Map();S.forEach(s=>SM.set(s.no,s));

const HID2Zone=new Map();
const HID2Mcp=new Map();
const HID2Addr=new Map();
HZM.forEach(m=>{{
  HID2Zone.set(m.hid.toUpperCase(),m.zone);
  HID2Mcp.set(m.hid.toUpperCase(),m.mcp);
  const addr=AM.get(m.addr);
  if(addr)HID2Addr.set(m.hid.toUpperCase(),addr);
}});

const Zone2HIDs=new Map();
HZM.forEach(m=>{{
  const z=m.zone;
  if(!Zone2HIDs.has(z))Zone2HIDs.set(z,[]);
  Zone2HIDs.get(z).push(m.hid);
}});

const B={{minX:{min_x},maxX:{max_x},minY:{min_y},maxY:{max_y}}};
const c=document.getElementById('c'),x=c.getContext('2d');
let w,h,sc=0.5,ox=0,oy=0,dr=false,lx,ly,rot=0;

const zoneColors={{}};
let colorIdx=0;
const palette=['#f85149','#d29922','#238636','#58a6ff','#a371f7','#f778ba','#79c0ff','#7ee787','#ffa657','#ff7b72'];
Z.forEach(z=>{{zoneColors[z.id]=palette[colorIdx++%palette.length]}});

function rs(){{w=innerWidth;h=innerHeight;c.width=w;c.height=h;rn()}}

function rotatePoint(px,py,cx,cy,angle){{
  const rad=angle*Math.PI/180;
  const cos=Math.cos(rad),sin=Math.sin(rad);
  return{{x:(px-cx)*cos-(py-cy)*sin+cx,y:(px-cx)*sin+(py-cy)*cos+cy}};
}}

function w2s(px,py){{
  const cx=(B.minX+B.maxX)/2,cy=(B.minY+B.maxY)/2;
  const rp=rotatePoint(px,py,cx,cy,rot);
  return{{x:(rp.x-B.minX)*sc+ox,y:h-((rp.y-B.minY)*sc+oy)}};
}}

function s2w(sx,sy){{
  const wx=(sx-ox)/sc+B.minX;
  const wy=(h-sy-oy)/sc+B.minY;
  const cx=(B.minX+B.maxX)/2,cy=(B.minY+B.maxY)/2;
  return rotatePoint(wx,wy,cx,cy,-rot);
}}

function rn(){{
  x.fillStyle='#0d1117';x.fillRect(0,0,w,h);
  const showRail=document.getElementById('sr').checked;
  const showStn=document.getElementById('ss').checked;
  const showZone=document.getElementById('sz').checked;
  const showHID=document.getElementById('sn').checked;
  
  if(showRail){{
    x.strokeStyle='#238636';x.lineWidth=Math.max(1,sc*0.4);x.beginPath();
    for(const[f,t]of C){{
      const a=AM.get(f),b=AM.get(t);
      if(a&&b){{const p1=w2s(a.x,a.y),p2=w2s(b.x,b.y);x.moveTo(p1.x,p1.y);x.lineTo(p2.x,p2.y);}}
    }}
    x.stroke();
  }}
  
  if(showZone && sc>0.1){{
    const zoneCenters={{}};
    for(const a of A){{
      if(a.zone){{
        if(!zoneCenters[a.zone])zoneCenters[a.zone]={{sumX:0,sumY:0,cnt:0}};
        zoneCenters[a.zone].sumX+=a.x;
        zoneCenters[a.zone].sumY+=a.y;
        zoneCenters[a.zone].cnt++;
      }}
    }}
    x.font='bold '+Math.max(10,sc*3)+'px system-ui';
    x.textAlign='center';
    for(const[zid,data]of Object.entries(zoneCenters)){{
      if(data.cnt>0){{
        const cx=data.sumX/data.cnt,cy=data.sumY/data.cnt;
        const p=w2s(cx,cy);
        x.fillStyle=zoneColors[zid]||'#58a6ff';
        x.globalAlpha=0.8;
        x.fillText(zid,p.x,p.y);
        x.globalAlpha=1;
      }}
    }}
    x.textAlign='left';
  }}
  
  if(showStn||showHID){{
    const r=Math.max(2,sc*0.6);
    x.font=Math.max(8,sc*1.5)+'px system-ui';
    for(const a of A){{
      if(a.stations&&a.stations.length){{
        const p=w2s(a.x,a.y);
        if(p.x>-50&&p.x<w+50&&p.y>-50&&p.y<h+50){{
          for(const sn of a.stations){{
            const st=SM.get(sn);
            if(st){{
              if(showStn){{
                const zColor=a.zone&&zoneColors[a.zone]?zoneColors[a.zone]:'#d29922';
                x.fillStyle=zColor;
                x.beginPath();x.arc(p.x,p.y,r,0,Math.PI*2);x.fill();
              }}
              if(showHID&&sc>0.3){{
                x.fillStyle='#c9d1d9';
                x.fillText(st.port_id||sn,p.x+r+2,p.y+3);
              }}
            }}
          }}
        }}
      }}
    }}
  }}
}}

function rotate(deg){{rot=(rot+deg)%360;document.getElementById('rot').textContent='회전: '+rot+'°';rn()}}
function fit(){{const lw=B.maxX-B.minX,lh=B.maxY-B.minY;sc=Math.min((w-50)/lw,(h-50)/lh);ox=(w-lw*sc)/2;oy=(h-lh*sc)/2;rn()}}
function reset(){{sc=0.5;ox=0;oy=0;rot=0;document.getElementById('rot').textContent='회전: 0°';rn()}}

function toggleMapping(){{
  const m=document.getElementById('mapping');
  if(m.style.display==='none'||!m.style.display){{
    m.style.display='block';
    const body=document.getElementById('mapBody');
    body.innerHTML='';
    HZM.slice(0,200).forEach(item=>{{
      const tr=document.createElement('tr');
      tr.innerHTML='<td>'+item.hid+'</td><td>→</td><td style="color:'+(zoneColors[item.zone]||'#58a6ff')+'">'+item.zone+'</td><td style="color:#8b949e">'+item.mcp+'</td>';
      tr.onclick=()=>{{document.getElementById('q').value=item.hid;search();}};
      body.appendChild(tr);
    }});
    if(HZM.length>200){{
      const tr=document.createElement('tr');
      tr.innerHTML='<td colspan="4" style="text-align:center;color:#8b949e">... 외 '+(HZM.length-200)+'개</td>';
      body.appendChild(tr);
    }}
  }}else{{
    m.style.display='none';
  }}
}}

function search(){{
  const q=document.getElementById('q').value.trim().toUpperCase();
  if(!q)return;
  
  const zoneId=HID2Zone.get(q);
  const mcpId=HID2Mcp.get(q);
  if(zoneId!==undefined){{
    const addr=HID2Addr.get(q);
    if(addr){{
      const p=w2s(addr.x,addr.y);
      ox+=w/2-p.x;oy+=h/2-(h-p.y);
    }}
    document.getElementById('msg').textContent='✓ '+q+' → Zone:'+zoneId+' MCP:'+mcpId;
    document.getElementById('info').textContent=q+' → Zone:'+zoneId+' / MCP:'+mcpId;
    rn();return;
  }}
  
  for(const[hid,zone]of HID2Zone){{
    if(hid.includes(q)){{
      const mcp=HID2Mcp.get(hid);
      const addr=HID2Addr.get(hid);
      if(addr){{
        const p=w2s(addr.x,addr.y);
        ox+=w/2-p.x;oy+=h/2-(h-p.y);
      }}
      document.getElementById('msg').textContent='✓ '+hid+' → Zone:'+zone+' MCP:'+mcp;
      document.getElementById('info').textContent=hid+' → Zone:'+zone+' / MCP:'+mcp;
      rn();return;
    }}
  }}
  
  if(Zone2HIDs.has(q)){{
    const hids=Zone2HIDs.get(q);
    for(const a of A){{
      if(a.zone===q){{
        const p=w2s(a.x,a.y);
        ox+=w/2-p.x;oy+=h/2-(h-p.y);
        break;
      }}
    }}
    document.getElementById('msg').textContent=q+': '+hids.length+'개 HID';
    document.getElementById('info').textContent='Zone '+q+' → '+hids.slice(0,5).join(', ')+(hids.length>5?' ...':'');
    rn();return;
  }}
  
  document.getElementById('msg').textContent='Not found: '+q;
}}

c.onmousemove=e=>{{
  if(dr){{
    ox+=e.clientX-lx;oy-=e.clientY-ly;lx=e.clientX;ly=e.clientY;rn();
  }}else{{
    const wp=s2w(e.clientX,e.clientY);
    let found=null;
    const threshold=50/sc;
    for(const a of A){{
      const dx=a.x-wp.x,dy=a.y-wp.y;
      if(Math.sqrt(dx*dx+dy*dy)<threshold){{found=a;break;}}
    }}
    if(found){{
      let info='';
      if(found.stations&&found.stations.length){{
        const hids=found.stations.map(sn=>{{const st=SM.get(sn);return st?st.port_id:'';}}).filter(h=>h);
        if(hids.length)info=hids.join(', ')+' → ';
      }}
      info+='Zone:'+(found.zone||'N/A')+' / MCP:'+(found.mcp||'N/A');
      document.getElementById('info').textContent=info;
    }}else{{
      document.getElementById('info').textContent='HID ID 입력 → Zone ID / MCP ID 확인';
    }}
  }}
}};

c.onmousedown=e=>{{dr=true;lx=e.clientX;ly=e.clientY}};
c.onmouseup=()=>{{dr=false}};
c.onmouseleave=()=>{{dr=false}};
c.onwheel=e=>{{
  e.preventDefault();
  const z=e.deltaY>0?0.9:1.1,mx=e.clientX,my=e.clientY;
  const wb=s2w(mx,my);
  sc*=z;sc=Math.max(0.01,Math.min(20,sc));
  const sa=w2s(wb.x,wb.y);
  ox+=mx-sa.x;oy-=my-(h-sa.y);
  rn();
}};

document.getElementById('sr').onchange=rn;
document.getElementById('ss').onchange=rn;
document.getElementById('sz').onchange=rn;
document.getElementById('sn').onchange=rn;
document.getElementById('q').onkeypress=e=>{{if(e.key==='Enter')search()}};
onresize=rs;rs();fit();
</script></body></html>'''

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML 저장: {output_file}")

def main():
    print("=" * 50)
    print("HID ID → Zone ID / MCP ID 매핑 뷰어")
    print("=" * 50)
    
    xml_file = select_file()
    if not xml_file:
        print("취소됨")
        return
    
    print(f"파일: {xml_file}")
    
    base = os.path.splitext(xml_file)[0]
    output_html = f"{base}_hid_zone_viewer.html"
    output_csv = f"{base}_HID_Zone_MCP_Mapping.csv"

    # 스크립트 디렉토리 (HID_Zone_Master.csv 위치)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_hid_zone_csv = os.path.join(script_dir, 'HID_Zone_Master.csv')

    addresses, stations, mcp_zones, connections = parse_xml(xml_file)
    if not addresses:
        print("데이터 없음")
        return

    # 1. layout_HID_Zone_MCP_Mapping.csv 저장 (xml 위치 + 스크립트 디렉토리)
    hid_zone_map = save_mapping_csv(addresses, stations, output_csv)
    # 스크립트 디렉토리에도 저장
    mapping_csv_for_hid = os.path.join(script_dir, 'layout_HID_Zone_MCP_Mapping.csv')
    save_mapping_csv(addresses, stations, mapping_csv_for_hid)

    # 2. HID_Zone_Master.csv 생성 (hid_zone_csv_cre.py와 동일한 포맷)
    print()
    save_hid_zone_master_csv(addresses, stations, mcp_zones, output_hid_zone_csv)

    # HTML 생성
    generate_html(addresses, stations, mcp_zones, connections, hid_zone_map, output_html)

    root = Tk()
    root.withdraw()
    messagebox.showinfo("완료", f"저장 완료!\n\nHTML: {output_html}\nCSV: {output_csv}\nHID Zone: {output_hid_zone_csv}")
    root.destroy()
    
    import webbrowser
    webbrowser.open(output_html)

if __name__ == '__main__':
    main()