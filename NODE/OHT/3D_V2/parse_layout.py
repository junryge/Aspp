"""
OHT Layout XML Parser (Enhanced)
Extracts address coordinates, connections, stations, HID info, and MCP Zone data
from layout.xml and generates:
  1) JSON data for 3D visualization
  2) {FAB}_HID_Zone_Master.csv (통합 마스터 CSV)
Supports loading different layout XML files (master data).
"""
import xml.etree.ElementTree as ET
import json
import csv
import sys
import os

def parse_layout_xml(xml_path, output_dir, fab_name='M14-Pro'):
    """
    Parse layout XML and generate JSON + CSV master data files.

    Args:
        xml_path: Path to layout.xml
        output_dir: Directory to save output files
        fab_name: FAB name for identification (e.g., 'M14-Pro', 'M14-Q', 'M16')
    """
    print(f"{'='*60}")
    print(f"  OHT Layout Parser - FAB: {fab_name}")
    print(f"{'='*60}")
    print(f"Parsing {xml_path}...")
    print("This may take a while for large files...")

    os.makedirs(output_dir, exist_ok=True)

    nodes = {}
    edges = []
    mcp_zones = []
    hid_zones = []  # HID Zone Label 데이터
    hid_controls = []  # HidControl → HID-MCP Zone 매핑

    # Use iterparse for memory efficiency on large files
    context = ET.iterparse(xml_path, events=('start', 'end'))

    current_addr = None
    current_addr_name = None
    current_next_addr = None
    current_station = None
    current_list_key = None
    in_addr_group = False
    in_next_addr = False
    in_station = False

    # MCP Zone parsing state
    in_mcp_zone_control = False
    in_mcp_zone = False
    in_mcp_sub = False
    mcp_zone_data = {}
    mcp_sub_data = {}
    mcp_sub_type = ''

    # HID Zone Label parsing state
    in_hid_label = False
    hid_label_data = {}

    # HidControl parsing state
    in_hid_control = False
    in_hid_entry = False
    hid_entry_data = {}

    depth = 0
    addr_depth = 0

    # Temp storage
    addr_data = {}
    next_addr_data = {}
    station_data = {}

    count = 0

    for event, elem in context:
        if event == 'start':
            depth += 1

            if elem.tag == 'group':
                name = elem.get('name', '')
                cls = elem.get('class', '')

                if 'hid.HidControl' in cls:
                    in_hid_control = True

                elif in_hid_control and 'hid.Hid' in cls and 'HidControl' not in cls:
                    in_hid_entry = True
                    hid_entry_data = {'hid_id': '', 'mcpzone_no': 0, 'group_name': name}

                elif 'McpZoneControl' in cls:
                    in_mcp_zone_control = True

                elif in_mcp_zone_control and 'McpZone' in cls and 'CutLane' not in cls and 'Entry' not in cls and 'Exit' not in cls:
                    in_mcp_zone = True
                    mcp_zone_data = {
                        'id': 0, 'no': 0, 'name': name,
                        'vehicle_max': 0, 'vehicle_precaution': 0,
                        'type': 0,
                        'cut_lanes': [], 'entries': [], 'exits': []
                    }

                elif in_mcp_zone and ('CutLane' in cls or 'Entry' in cls or 'Exit' in cls):
                    in_mcp_sub = True
                    mcp_sub_type = 'cut_lane' if 'CutLane' in cls else ('entry' if 'Entry' in cls else 'exit')
                    mcp_sub_data = {'start': 0, 'end': 0, 'stop_no': 0, 'stop_zcu': '', 'count_type': True}

                elif name.startswith('LabelHID') and 'label.Label' in cls:
                    in_hid_label = True
                    hid_label_data = {
                        'label_name': name.replace('Label', ''),
                        'machine_id': '',
                        'address': 0,
                        'draw_x': 0,
                        'draw_y': 0,
                        'point': 0
                    }

                elif name.startswith('Addr') and 'address.Addr' in cls:
                    in_addr_group = True
                    addr_depth = depth
                    current_addr_name = name
                    addr_data = {
                        'draw_x': 0, 'draw_y': 0,
                        'cad_x': 0, 'cad_y': 0,
                        'address': 0,
                        'symbol_name': '',
                        'is_station': 0,
                        'branch': False,
                        'junction': False,
                        'hid_included': -1,
                        'stopzone': 0,
                        'next_addrs': [],
                        'stations': []
                    }

                elif in_addr_group and name.startswith('NextAddr') and 'address.NextAddr' in cls:
                    in_next_addr = True
                    next_addr_data = {
                        'next_address': 0,
                        'distance_puls': 0,
                        'speed': 0,
                        'direction': 0,
                        'branch_direction': 0,
                        'basic_direction': True,
                        'nextposition': 0.0
                    }

                elif in_addr_group and name.startswith('Station') and 'address.Station' in cls:
                    in_station = True
                    station_data = {
                        'no': 0,
                        'port_id': '',
                        'category': 0,
                        'type': 0,
                        'position': 0
                    }

            elif elem.tag == 'param':
                key = elem.get('key', '')
                value = elem.get('value', '')

                if in_hid_entry and in_hid_control:
                    if key == 'id':
                        hid_entry_data['hid_id'] = value
                    elif key == 'mcpzone-no':
                        try:
                            hid_entry_data['mcpzone_no'] = int(value)
                        except:
                            hid_entry_data['mcpzone_no'] = 0

                elif in_mcp_sub and in_mcp_zone:
                    if key == 'start':
                        mcp_sub_data['start'] = int(value)
                    elif key == 'end':
                        mcp_sub_data['end'] = int(value)
                    elif key == 'stop-no':
                        mcp_sub_data['stop_no'] = int(value)
                    elif key == 'stop-zcu':
                        mcp_sub_data['stop_zcu'] = value
                    elif key == 'count-type':
                        mcp_sub_data['count_type'] = value == 'true'

                elif in_mcp_zone and not in_mcp_sub:
                    if key == 'id':
                        mcp_zone_data['id'] = int(value)
                    elif key == 'no':
                        mcp_zone_data['no'] = int(value)
                    elif key == 'vehicle-max':
                        mcp_zone_data['vehicle_max'] = int(value)
                    elif key == 'vehicle-precaution':
                        mcp_zone_data['vehicle_precaution'] = int(value)
                    elif key == 'type':
                        mcp_zone_data['type'] = int(value)

                elif in_hid_label:
                    if key == 'machine-id':
                        hid_label_data['machine_id'] = value
                    elif key == 'address':
                        try:
                            hid_label_data['address'] = int(value)
                        except:
                            hid_label_data['address'] = 0
                    elif key == 'draw-x':
                        hid_label_data['draw_x'] = float(value)
                    elif key == 'draw-y':
                        hid_label_data['draw_y'] = float(value)
                    elif key == 'point':
                        try:
                            hid_label_data['point'] = int(value)
                        except:
                            hid_label_data['point'] = 0

                elif in_addr_group:
                    if in_next_addr:
                        if key == 'next-address':
                            next_addr_data['next_address'] = int(value)
                        elif key == 'distance-puls':
                            next_addr_data['distance_puls'] = int(value)
                        elif key == 'speed':
                            next_addr_data['speed'] = int(value)
                        elif key == 'direction':
                            next_addr_data['direction'] = int(value)
                        elif key == 'branch-direction':
                            next_addr_data['branch_direction'] = int(value)
                        elif key == 'basic-direction':
                            next_addr_data['basic_direction'] = value == 'true'
                        elif key == 'nextposition':
                            try:
                                next_addr_data['nextposition'] = float(value)
                            except:
                                next_addr_data['nextposition'] = 0.0

                    elif in_station:
                        if key == 'no':
                            station_data['no'] = int(value)
                        elif key == 'port-id':
                            station_data['port_id'] = value
                        elif key == 'category':
                            station_data['category'] = int(value)
                        elif key == 'type':
                            station_data['type'] = int(value)
                        elif key == 'position':
                            station_data['position'] = int(value)

                    else:
                        if key == 'draw-x':
                            addr_data['draw_x'] = float(value)
                        elif key == 'draw-y':
                            addr_data['draw_y'] = float(value)
                        elif key == 'cad-x':
                            try:
                                addr_data['cad_x'] = float(value)
                            except:
                                addr_data['cad_x'] = 0.0
                        elif key == 'cad-y':
                            try:
                                addr_data['cad_y'] = float(value)
                            except:
                                addr_data['cad_y'] = 0.0
                        elif key == 'address':
                            addr_data['address'] = int(value)
                        elif key == 'symbol-name':
                            addr_data['symbol_name'] = value
                        elif key == 'isstation':
                            addr_data['is_station'] = int(value)
                        elif key == 'branch':
                            addr_data['branch'] = value == 'true'
                        elif key == 'junction':
                            addr_data['junction'] = value == 'true'
                        elif key == 'hid-included':
                            try:
                                addr_data['hid_included'] = int(value)
                            except:
                                addr_data['hid_included'] = 0
                        elif key == 'stopzone':
                            try:
                                addr_data['stopzone'] = int(value)
                            except:
                                addr_data['stopzone'] = 0

        elif event == 'end':
            if elem.tag == 'group':
                name = elem.get('name', '')
                cls = elem.get('class', '')

                if in_mcp_sub and ('CutLane' in cls or 'Entry' in cls or 'Exit' in cls):
                    in_mcp_sub = False
                    if mcp_sub_type == 'cut_lane':
                        mcp_zone_data['cut_lanes'].append(dict(mcp_sub_data))
                    elif mcp_sub_type == 'entry':
                        mcp_zone_data['entries'].append(dict(mcp_sub_data))
                    elif mcp_sub_type == 'exit':
                        mcp_zone_data['exits'].append(dict(mcp_sub_data))

                elif in_mcp_zone and 'McpZone' in cls and 'CutLane' not in cls and 'Entry' not in cls and 'Exit' not in cls:
                    in_mcp_zone = False
                    in_mcp_sub = False
                    mcp_zones.append(dict(mcp_zone_data))

                elif 'McpZoneControl' in cls:
                    in_mcp_zone_control = False

                elif in_hid_entry and 'hid.Hid' in cls and 'HidControl' not in cls:
                    in_hid_entry = False
                    if hid_entry_data['hid_id']:
                        hid_controls.append(dict(hid_entry_data))

                elif in_hid_control and 'hid.HidControl' in cls:
                    in_hid_control = False

                elif in_hid_label and name.startswith('LabelHID') and 'label.Label' in cls:
                    in_hid_label = False
                    if hid_label_data['machine_id']:
                        hid_zones.append(dict(hid_label_data))

                elif in_next_addr and name.startswith('NextAddr') and 'address.NextAddr' in cls:
                    in_next_addr = False
                    if next_addr_data['next_address'] > 0:
                        addr_data['next_addrs'].append(dict(next_addr_data))

                elif in_station and name.startswith('Station') and 'address.Station' in cls:
                    in_station = False
                    if station_data['port_id']:
                        addr_data['stations'].append(dict(station_data))

                elif in_addr_group and name.startswith('Addr') and 'address.Addr' in cls:
                    in_addr_group = False
                    in_next_addr = False
                    in_station = False

                    addr_id = addr_data['address']
                    if addr_id > 0:
                        nodes[addr_id] = {
                            'id': addr_id,
                            'x': addr_data['draw_x'],
                            'y': addr_data['draw_y'],
                            'cad_x': addr_data['cad_x'],
                            'cad_y': addr_data['cad_y'],
                            'symbol': addr_data['symbol_name'],
                            'is_station': addr_data['is_station'],
                            'branch': addr_data['branch'],
                            'junction': addr_data['junction'],
                            'hid_included': addr_data['hid_included'],
                            'stopzone': addr_data['stopzone'],
                            'stations': addr_data['stations']
                        }

                        for na in addr_data['next_addrs']:
                            edges.append({
                                'from': addr_id,
                                'to': na['next_address'],
                                'distance': na['distance_puls'],
                                'speed': na['speed'],
                                'direction': na['direction'],
                                'branch_dir': na['branch_direction']
                            })

                        count += 1
                        if count % 500 == 0:
                            print(f"  Processed {count} addresses...")

            depth -= 1
            elem.clear()

    print(f"\nTotal addresses parsed: {count}")
    print(f"Total connections: {len(edges)}")
    print(f"Total MCP zones: {len(mcp_zones)}")
    print(f"Total HID zones: {len(hid_zones)}")
    print(f"Total HID controls (HID→MCP mapping): {len(hid_controls)}")

    # Calculate bounds
    if nodes:
        xs = [n['x'] for n in nodes.values()]
        ys = [n['y'] for n in nodes.values()]
        bounds = {
            'min_x': min(xs), 'max_x': max(xs),
            'min_y': min(ys), 'max_y': max(ys)
        }
    else:
        bounds = {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}

    # Station summary
    station_nodes = [n for n in nodes.values() if n['stations']]
    all_stations = []
    for n in station_nodes:
        for s in n['stations']:
            all_stations.append({
                'port_id': s['port_id'],
                'category': s['category'],
                'type': s['type'],
                'no': s['no'],
                'position': s['position'],
                'node_id': n['id'],
                'x': n['x'],
                'y': n['y']
            })

    # Zone → address mapping
    zone_addr_map = {}
    for z in mcp_zones:
        zid = z['id']
        addrs = set()
        for e in z['entries']:
            addrs.add(e['start']); addrs.add(e['end'])
        for e in z['exits']:
            addrs.add(e['start']); addrs.add(e['end'])
        for c in z['cut_lanes']:
            addrs.add(c['start']); addrs.add(c['end'])
        zone_addr_map[zid] = list(addrs)

    # HID → MCP Zone 매핑 딕셔너리 구축
    mcp_zone_map = {z['no']: z for z in mcp_zones}  # mcpzone-no → zone data
    hid_label_map = {}  # hid_id(B01-1) → label data
    for h in hid_zones:
        # machine_id: HID-B01-1(001) → hid_id: B01-1
        mid = h['machine_id']
        if mid.startswith('HID-'):
            hid_id = mid[4:]  # Remove 'HID-' prefix
            paren = hid_id.find('(')
            if paren > 0:
                hid_id = hid_id[:paren]
            hid_label_map[hid_id] = h

    # hid_master: HID Label + HidControl + MCP Zone 통합 데이터
    hid_master = []
    for hc in sorted(hid_controls, key=lambda x: x['mcpzone_no']):
        hid_id = hc['hid_id']  # e.g. B01-1
        mcpzone_no = hc['mcpzone_no']
        zone = mcp_zone_map.get(mcpzone_no, {})
        label = hid_label_map.get(hid_id, {})
        full_name = label.get('machine_id', f'HID-{hid_id}')

        # IN/OUT lanes
        entries = zone.get('entries', [])
        exits = zone.get('exits', [])
        in_lanes = '; '.join([f"{e['start']}→{e['end']}" for e in entries])
        out_lanes = '; '.join([f"{e['start']}→{e['end']}" for e in exits])

        # ZCU (first entry's stop_zcu)
        zcu = ''
        for e in entries:
            if e.get('stop_zcu'):
                zcu = e['stop_zcu']
                break

        hid_master.append({
            'zone_id': mcpzone_no,
            'hid_id': hid_id,
            'full_name': full_name,
            'address': label.get('address', 0),
            'vehicle_max': zone.get('vehicle_max', 0),
            'vehicle_precaution': zone.get('vehicle_precaution', 0),
            'zone_type': zone.get('type', 0),
            'in_count': len(entries),
            'out_count': len(exits),
            'in_lanes': in_lanes,
            'out_lanes': out_lanes,
            'zcu': zcu,
        })

    # ============================================
    # 1) JSON output (for 3D visualization)
    # ============================================
    result = {
        'project': fab_name,
        'line': 'OHT',
        'client': 'HYNIX',
        'fab_name': fab_name,
        'total_nodes': len(nodes),
        'total_edges': len(edges),
        'total_stations': len(all_stations),
        'total_mcp_zones': len(mcp_zones),
        'total_hid_zones': len(hid_zones),
        'bounds': bounds,
        'nodes': list(nodes.values()),
        'edges': edges,
        'stations': all_stations,
        'mcp_zones': mcp_zones,
        'hid_zones': hid_zones,
        'hid_master': hid_master,
        'zone_addr_map': zone_addr_map
    }

    json_path = os.path.join(output_dir, 'layout_data.json')
    print(f"\nWriting JSON to {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f)
    json_size = os.path.getsize(json_path)
    print(f"  JSON size: {json_size / 1024 / 1024:.1f} MB")

    # ============================================
    # 2) CSV Master Data: HID_Zone_Master.csv만 생성
    # ============================================
    csv_dir = os.path.join(output_dir, 'master_csv')
    os.makedirs(csv_dir, exist_ok=True)

    # --- {FAB}_HID_Zone_Master.csv (통합 마스터) ---
    hid_csv = os.path.join(csv_dir, f'{fab_name}_HID_Zone_Master.csv')
    print(f"Writing {hid_csv}...")
    with open(hid_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Zone_ID', 'HID_ID', 'Full_Name', 'Address',
                     'Type', 'IN_Count', 'OUT_Count', 'IN_Lanes', 'OUT_Lanes',
                     'Vehicle_Max', 'Vehicle_Precaution', 'ZCU', 'FAB'])
        for h in hid_master:
            w.writerow([h['zone_id'], h['hid_id'], h['full_name'], h['address'],
                         h['zone_type'], h['in_count'], h['out_count'],
                         h['in_lanes'], h['out_lanes'],
                         h['vehicle_max'], h['vehicle_precaution'],
                         h['zcu'], fab_name])
    print(f"  {len(hid_master)} rows")

    print(f"\n{'='*60}")
    print(f"  파싱 완료!")
    print(f"  FAB: {fab_name}")
    print(f"  노드: {len(nodes):,}  |  엣지: {len(edges):,}")
    print(f"  스테이션: {len(all_stations):,}  |  MCP Zone: {len(mcp_zones):,}")
    print(f"  HID Zone 라벨: {len(hid_zones):,}")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_dir}/")
    print(f"{'='*60}")

    return result


if __name__ == '__main__':
    xml_path = r'F:\M14_Q\oht_xml\oht_layout\layout\layout\layout.xml'
    output_dir = r'F:\M14_Q\oht_xml\oht_layout'
    fab_name = 'M14-Pro'

    # Command line args: python parse_layout.py [xml_path] [output_dir] [fab_name]
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    if len(sys.argv) > 3:
        fab_name = sys.argv[3]

    parse_layout_xml(xml_path, output_dir, fab_name)
