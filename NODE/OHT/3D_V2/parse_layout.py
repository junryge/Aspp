"""
OHT Layout XML Parser (Enhanced)

python parse_layout.py --scan /path/to/MAP
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
import zipfile
import tempfile
from datetime import datetime

def parse_layout_xml(xml_path, output_dir, fab_name='M14-Pro', json_filename=None):
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

    json_path = os.path.join(output_dir, json_filename or 'layout_data.json')
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


def parse_from_zip(zip_path, output_dir, fab_name=None):
    """
    Extract layout.xml from a zip file and parse it.

    Args:
        zip_path: Path to *.layout.zip or *.zip file
        output_dir: Directory to save output files
        fab_name: FAB name (auto-detected from filename if None)
    """
    if fab_name is None:
        # Remove all extensions: A.layout.zip -> A, layout.zip -> layout
        base = os.path.basename(zip_path)
        fab_name = base.split('.')[0]

    print(f"\n  ZIP: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as z:
        # layout.xml 또는 LAYOUT.XML 등 찾기 (하위 폴더 포함)
        xml_files = [f for f in z.namelist() if f.lower().endswith('layout.xml')]
        if not xml_files:
            print(f"  [ERROR] No layout.xml found in {zip_path}")
            print(f"  Files in zip: {z.namelist()[:20]}")
            return None

        xml_name = xml_files[0]
        print(f"  Found: {xml_name}")

        with tempfile.TemporaryDirectory() as tmp:
            z.extract(xml_name, tmp)
            xml_full_path = os.path.join(tmp, xml_name)
            return parse_layout_xml(
                xml_full_path, output_dir, fab_name,
                json_filename=f'{fab_name}.json'
            )


def _find_layout_zips(dir_path):
    """
    디렉토리에서 layout.xml을 포함한 ZIP 파일 찾기.
    1순위: *.layout.zip
    2순위: 모든 *.zip 중 layout.xml 포함된 것
    """
    # 1) *.layout.zip 패턴
    layout_zips = sorted([
        f for f in os.listdir(dir_path)
        if f.lower().endswith('.layout.zip')
    ])
    if layout_zips:
        return layout_zips

    # 2) 일반 *.zip 중 layout.xml 포함된 것
    found = []
    for f in sorted(os.listdir(dir_path)):
        if not f.lower().endswith('.zip'):
            continue
        zip_path = os.path.join(dir_path, f)
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                xml_files = [n for n in z.namelist() if n.lower().endswith('layout.xml')]
                if xml_files:
                    found.append(f)
        except (zipfile.BadZipFile, Exception):
            continue
    return found


def scan_and_parse_map(map_dir, output_base_dir):
    """
    Scan MAP directory structure and parse all FABs.

    Expected structure:
        MAP/
            M14A/  ->  *.layout.zip or *.zip (with layout.xml inside)
            M14B/  ->  *.zip
            M16A/  ->  A.layout.zip, BR.layout.zip, E.layout.zip
            M16B/  ->  B.layout.zip

    ZIP 안의 구조:
        *.zip
            └── LAYOUT/LAYOUT.XML  (또는 layout.xml)

    Args:
        map_dir: Path to MAP directory
        output_base_dir: Base output directory (fab_data/ will be created inside)
    """
    fab_data_dir = os.path.join(output_base_dir, 'fab_data')
    os.makedirs(fab_data_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MAP Directory Scanner")
    print(f"  Scanning: {map_dir}")
    print(f"  Output:   {fab_data_dir}")
    print(f"{'='*60}")

    results = []

    for entry in sorted(os.listdir(map_dir)):
        fab_path = os.path.join(map_dir, entry)
        if not os.path.isdir(fab_path):
            continue

        zip_files = _find_layout_zips(fab_path)

        if not zip_files:
            # 디렉토리 내용물 보여주기 (디버그용)
            contents = os.listdir(fab_path)
            print(f"\n  [SKIP] {entry}: layout.xml 포함 ZIP 없음")
            print(f"         폴더 내용: {contents[:10]}")
            continue

        for zf in zip_files:
            prefix = zf.split('.')[0]  # A.layout.zip -> A, layout.zip -> layout
            # If multiple zip files in same dir, use prefix as subsystem
            if len(zip_files) > 1:
                fab_name = f"{entry}-{prefix}"
            else:
                fab_name = entry

            zip_path = os.path.join(fab_path, zf)
            print(f"\n{'='*60}")
            print(f"  Parsing FAB: {fab_name}")
            print(f"{'='*60}")

            # FAB별 독립 디렉토리: fab_data/{fab_name}/
            fab_output_dir = os.path.join(fab_data_dir, fab_name)
            os.makedirs(fab_output_dir, exist_ok=True)

            result = parse_from_zip(zip_path, fab_output_dir, fab_name)
            if result:
                results.append({
                    'fab_name': fab_name,
                    'json_path': os.path.join(fab_output_dir, f'{fab_name}.json'),
                    'csv_dir': os.path.join(fab_output_dir, 'master_csv'),
                    'nodes': result.get('total_nodes', 0),
                    'edges': result.get('total_edges', 0),
                    'stations': result.get('total_stations', 0),
                    'mcp_zones': result.get('total_mcp_zones', 0),
                })

    # Save FAB registry
    registry_path = os.path.join(fab_data_dir, '_fab_registry.json')
    registry = {
        'fabs': results,
        'parsed_at': datetime.now().isoformat(),
        'map_dir': map_dir,
    }
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Scan Complete: {len(results)} FABs parsed")
    for r in results:
        print(f"    {r['fab_name']}: {r['nodes']:,} nodes, {r['edges']:,} edges")
    print(f"  Registry: {registry_path}")
    print(f"{'='*60}\n")

    return results


def auto_detect_and_parse(base_dir=None):
    """
    자동 감지 모드 - 그냥 실행하면 알아서 찾아서 파싱.

    탐색 순서:
      1) 하위 폴더에 layout.xml 포함 ZIP 있으면 → MAP 구조로 스캔
      2) 현재 폴더에 layout.xml 포함 ZIP 있으면 → 각각 파싱
      3) 현재 폴더에 layout.xml 있으면 → 직접 파싱
      4) 형제 폴더(상위 폴더의 하위들)에서 MAP 구조 탐색
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    fab_data_dir = os.path.join(base_dir, 'fab_data')
    found = False

    print(f"\n{'='*60}")
    print(f"  OHT Layout Auto Parser")
    print(f"  탐색 경로: {base_dir}")
    print(f"{'='*60}")

    # 1) 하위 폴더에 layout.xml 포함 ZIP 있는지 확인 (MAP 구조)
    sub_dirs_with_zip = []
    for entry in sorted(os.listdir(base_dir)):
        sub_path = os.path.join(base_dir, entry)
        if not os.path.isdir(sub_path) or entry.startswith('_') or entry.startswith('.'):
            continue
        zips = _find_layout_zips(sub_path)
        if zips:
            sub_dirs_with_zip.append(entry)

    if sub_dirs_with_zip:
        print(f"\n  [감지] MAP 구조 발견: {sub_dirs_with_zip}")
        scan_and_parse_map(base_dir, base_dir)
        found = True

    # 2) 현재 폴더에 layout.xml 포함 ZIP 있으면
    if not found:
        cur_zips = _find_layout_zips(base_dir)
        if cur_zips:
            os.makedirs(fab_data_dir, exist_ok=True)
            results = []
            for zf in cur_zips:
                zip_path = os.path.join(base_dir, zf)
                prefix = zf.split('.')[0]
                fab_name = prefix
                # FAB별 독립 디렉토리
                fab_output_dir = os.path.join(fab_data_dir, fab_name)
                os.makedirs(fab_output_dir, exist_ok=True)
                print(f"\n  [감지] ZIP 파일: {zf} → FAB: {fab_name}")
                result = parse_from_zip(zip_path, fab_output_dir, fab_name)
                if result:
                    results.append({
                        'fab_name': fab_name,
                        'json_path': os.path.join(fab_output_dir, f'{fab_name}.json'),
                        'csv_dir': os.path.join(fab_output_dir, 'master_csv'),
                        'nodes': result.get('total_nodes', 0),
                        'edges': result.get('total_edges', 0),
                        'stations': result.get('total_stations', 0),
                        'mcp_zones': result.get('total_mcp_zones', 0),
                    })
            # Save registry
            if results:
                reg_path = os.path.join(fab_data_dir, '_fab_registry.json')
                with open(reg_path, 'w', encoding='utf-8') as f:
                    json.dump({'fabs': results, 'parsed_at': datetime.now().isoformat()}, f, indent=2, ensure_ascii=False)
                found = True

    # 3) 현재 폴더에 layout.xml 있으면
    if not found:
        xml_path = os.path.join(base_dir, 'layout.xml')
        if os.path.exists(xml_path):
            print(f"\n  [감지] layout.xml 발견")
            folder_name = os.path.basename(base_dir)
            parse_layout_xml(xml_path, base_dir, folder_name)
            found = True

    # 4) 형제 폴더에서 MAP 구조 탐색 (상위 폴더의 하위 디렉토리)
    if not found:
        parent_dir = os.path.dirname(base_dir)
        print(f"\n  [탐색] 형제 폴더 탐색 중: {parent_dir}")
        for entry in sorted(os.listdir(parent_dir)):
            sibling_path = os.path.join(parent_dir, entry)
            if sibling_path == base_dir or not os.path.isdir(sibling_path):
                continue
            if entry.startswith('.') or entry.startswith('_'):
                continue
            # 형제 폴더 안에 MAP 구조 있는지 확인
            sibling_sub_dirs = []
            for sub in sorted(os.listdir(sibling_path)):
                sub_path = os.path.join(sibling_path, sub)
                if os.path.isdir(sub_path):
                    zips = _find_layout_zips(sub_path)
                    if zips:
                        sibling_sub_dirs.append(sub)
            if sibling_sub_dirs:
                print(f"\n  [감지] 형제 폴더 '{entry}' 에서 MAP 구조 발견: {sibling_sub_dirs}")
                scan_and_parse_map(sibling_path, base_dir)
                found = True
                break
            # 형제 폴더에 직접 ZIP 있는지
            zips = _find_layout_zips(sibling_path)
            if zips:
                print(f"\n  [감지] 형제 폴더 '{entry}' 에서 layout ZIP 발견: {zips}")
                os.makedirs(fab_data_dir, exist_ok=True)
                for zf in zips:
                    zip_path = os.path.join(sibling_path, zf)
                    fab_name = entry
                    fab_output_dir = os.path.join(fab_data_dir, fab_name)
                    os.makedirs(fab_output_dir, exist_ok=True)
                    parse_from_zip(zip_path, fab_output_dir, fab_name)
                found = True

    if not found:
        print(f"\n  [오류] 파싱할 파일을 찾지 못했습니다.")
        print(f"         현재 폴더 또는 하위/형제 폴더에 다음 중 하나가 필요합니다:")
        print(f"           - *.zip 파일 (ZIP 안에 layout.xml 또는 LAYOUT/LAYOUT.XML)")
        print(f"           - *.layout.zip 파일")
        print(f"           - layout.xml 파일")
        print(f"           - MAP 디렉토리 구조 (M14A/, M14B/ 등에 ZIP 포함)")
        print(f"\n  사용법:")
        print(f"    python parse_layout.py                        # 자동 감지")
        print(f"    python parse_layout.py --scan /path/to/MAP    # MAP 폴더 지정")
        print(f"    python parse_layout.py file.zip               # 단일 ZIP")
    else:
        print(f"\n{'='*60}")
        print(f"  파싱 완료! python server.py 로 서버를 실행하세요.")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # 인자 없이 실행 → 자동 감지 모드
        auto_detect_and_parse()

    elif sys.argv[1] == '--scan':
        # MAP 디렉토리 스캔 모드
        # Usage: python parse_layout.py --scan [MAP_DIR] [OUTPUT_DIR]
        map_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
        output_dir = sys.argv[3] if len(sys.argv) > 3 else '.'
        scan_and_parse_map(map_dir, output_dir)

    elif sys.argv[1].lower().endswith('.zip'):
        # 단일 ZIP 파일 모드
        # Usage: python parse_layout.py file.layout.zip [OUTPUT_DIR] [FAB_NAME]
        zip_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
        fab_name = sys.argv[3] if len(sys.argv) > 3 else None
        parse_from_zip(zip_path, output_dir, fab_name)

    elif sys.argv[1].lower().endswith('.xml'):
        # 단일 XML 파일 모드
        # Usage: python parse_layout.py layout.xml [OUTPUT_DIR] [FAB_NAME]
        xml_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
        fab_name = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(os.path.dirname(os.path.abspath(xml_path)))
        parse_layout_xml(xml_path, output_dir, fab_name)

    else:
        # 기존 호환: python parse_layout.py [xml_path] [output_dir] [fab_name]
        xml_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
        fab_name = sys.argv[3] if len(sys.argv) > 3 else 'M14-Pro'
        parse_layout_xml(xml_path, output_dir, fab_name)