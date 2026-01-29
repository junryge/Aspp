#!/usr/bin/env python3
"""
OHT Layout XML → HTML 변환기
- 파일 선택 → HTML 저장
- 화면 회전 기능
- ZONE ID 표시
- HID ID 표시 (메인 식별자)
"""

import xml.etree.ElementTree as ET
import json
import re
import os
from tkinter import Tk, filedialog, messagebox

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
    zones = {}
    connections = []
    
    context = ET.iterparse(xml_file, events=('start', 'end'))
    current_addr = None
    current_addr_no = None
    current_station = None
    current_station_no = None
    current_next_addr = None
    current_zone = None
    current_zone_id = None
    addr_data = {}
    station_data = {}
    next_addr_data = {}
    zone_data = {}
    count = 0

    for event, elem in context:
        if event == 'start':
            if elem.tag == 'group':
                name = elem.get('name', '')
                cls = elem.get('class', '')
                
                # Zone 파싱
                if 'Zone' in name and 'zone.Zone' in cls:
                    match = re.search(r'Zone(\d+)', name)
                    if match:
                        current_zone_id = match.group(1)
                        zone_data = {'id': current_zone_id, 'name': '', 'addresses': []}
                        current_zone = True
                
                # Address 파싱
                elif 'Addr' in name and 'address.Addr' in cls and 'NextAddr' not in name:
                    match = re.search(r'Addr(\d+)', name)
                    if match:
                        current_addr_no = int(match.group(1))
                        addr_data = {'x': 0, 'y': 0, 'stations': [], 'next_addrs': [], 'zone_id': ''}
                        current_addr = True
                        
                # Station 파싱
                elif 'Station' in name and 'Station' in cls:
                    match = re.search(r'Station(\d+)', name)
                    if match:
                        current_station_no = int(match.group(1))
                        station_data = {'port_id': '', 'type': 0}
                        current_station = True
                        
                # NextAddr 파싱
                elif 'NextAddr' in name and 'NextAddr' in cls:
                    current_next_addr = True
                    next_addr_data = {'next_address': None}
                    
        elif event == 'end':
            if elem.tag == 'param':
                key = elem.get('key', '')
                value = elem.get('value', '')
                
                if current_zone and not current_addr:
                    if key == 'zone-id':
                        zone_data['id'] = value
                    elif key == 'zone-name' or key == 'name':
                        zone_data['name'] = value
                
                elif current_addr and not current_station and not current_next_addr:
                    if key == 'draw-x':
                        try: addr_data['x'] = float(value)
                        except: pass
                    elif key == 'draw-y':
                        try: addr_data['y'] = float(value)
                        except: pass
                    elif key == 'zone-id':
                        addr_data['zone_id'] = value
                        
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
                    
                elif current_zone and 'Zone' in name and 'zone.Zone' in elem.get('class', ''):
                    if current_zone_id:
                        zones[current_zone_id] = zone_data.copy()
                    current_zone = False
                    current_zone_id = None
                    
            elem.clear()
    
    print(f"완료: {len(addresses):,} addr, {len(stations):,} stn, {len(zones):,} zones, {len(connections):,} conn")
    return addresses, stations, zones, connections

def generate_html(addresses, stations, zones, connections, output_file):
    min_x = min((a['x'] for a in addresses.values() if a['x'] != 0), default=0)
    max_x = max((a['x'] for a in addresses.values()), default=1000)
    min_y = min((a['y'] for a in addresses.values() if a['y'] != 0), default=0)
    max_y = max((a['y'] for a in addresses.values()), default=1000)

    addr_list = [{'no': no, 'x': d['x'], 'y': d['y'], 'stations': d['stations'], 'zone': d.get('zone_id','')} for no, d in addresses.items()]
    station_list = [{'no': no, 'port_id': d.get('port_id',''), 'type': d.get('type',0)} for no, d in stations.items()]
    zone_list = [{'id': zid, 'name': d.get('name','')} for zid, d in zones.items()]
    valid_connections = [(f, t) for f, t in connections if f in addresses and t in addresses]

    html = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>OHT Layout - HID View</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui;background:#0d1117;color:#c9d1d9;overflow:hidden}}
#ui{{position:fixed;top:10px;left:10px;z-index:100;display:flex;gap:8px;align-items:center;flex-wrap:wrap;max-width:90%}}
#ui button{{padding:6px 12px;background:#21262d;border:1px solid #30363d;color:#c9d1d9;border-radius:6px;cursor:pointer;font-size:12px}}
#ui button:hover{{background:#30363d}}
#ui button.active{{background:#238636;border-color:#238636}}
#ui input[type=text]{{padding:6px 10px;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;border-radius:6px;width:140px;font-size:12px}}
#ui label{{font-size:12px;display:flex;align-items:center;gap:4px;color:#8b949e}}
#stat{{position:fixed;top:10px;right:10px;font-size:11px;color:#8b949e}}
#rot{{position:fixed;bottom:10px;right:10px;font-size:11px;color:#8b949e;background:#21262d;padding:5px 10px;border-radius:6px}}
#info{{position:fixed;bottom:10px;left:10px;font-size:12px;color:#58a6ff;background:#21262d;padding:10px 14px;border-radius:6px;max-width:400px;font-weight:500}}
canvas{{display:block}}
</style></head><body>
<div id="ui">
<button onclick="fit()">Fit</button>
<button onclick="reset()">Reset</button>
<button onclick="rotate(-90)">↺ 90°</button>
<button onclick="rotate(90)">↻ 90°</button>
<label><input type="checkbox" id="sr" checked>Rail</label>
<label><input type="checkbox" id="ss" checked>Station</label>
<label><input type="checkbox" id="sz" checked>Zone ID</label>
<label><input type="checkbox" id="sn">HID ID</label>
<input type="text" id="q" placeholder="HID ID / Zone 검색">
<button onclick="search()">Find</button>
<span id="msg"></span>
</div>
<div id="stat">HID:{len(stations):,} | Addr:{len(addresses):,} | Zone:{len(zones):,}</div>
<div id="rot">회전: 0°</div>
<div id="info">마우스 오버: HID 정보 표시</div>
<canvas id="c"></canvas>
<script>
const A={json.dumps(addr_list)};
const S={json.dumps(station_list)};
const Z={json.dumps(zone_list)};
const C={json.dumps(valid_connections)};
const AM=new Map();A.forEach(a=>AM.set(a.no,a));
const SM=new Map();S.forEach(s=>SM.set(s.no,s));
const ZM=new Map();Z.forEach(z=>ZM.set(z.id,z));

// HID port_id → Addr 역매핑
const HID2Addr=new Map();
A.forEach(a=>{{
  if(a.stations)a.stations.forEach(sn=>{{
    const st=SM.get(sn);
    if(st&&st.port_id)HID2Addr.set(st.port_id.toUpperCase(),a);
  }});
}});

const B={{minX:{min_x},maxX:{max_x},minY:{min_y},maxY:{max_y}}};
const c=document.getElementById('c'),x=c.getContext('2d');
let w,h,sc=0.5,ox=0,oy=0,dr=false,lx,ly,rot=0;

// Zone별 색상 생성
const zoneColors={{}};
let colorIdx=0;
const palette=['#f85149','#d29922','#238636','#58a6ff','#a371f7','#f778ba','#79c0ff','#7ee787','#ffa657','#ff7b72'];
Z.forEach(z=>{{zoneColors[z.id]=palette[colorIdx++%palette.length]}});

function rs(){{w=innerWidth;h=innerHeight;c.width=w;c.height=h;rn()}}

function rotatePoint(px,py,cx,cy,angle){{
  const rad=angle*Math.PI/180;
  const cos=Math.cos(rad),sin=Math.sin(rad);
  const nx=(px-cx)*cos-(py-cy)*sin+cx;
  const ny=(px-cx)*sin+(py-cy)*cos+cy;
  return{{x:nx,y:ny}};
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
  
  // Rail 그리기
  if(showRail){{
    x.strokeStyle='#238636';x.lineWidth=Math.max(1,sc*0.4);x.beginPath();
    for(const[f,t]of C){{
      const a=AM.get(f),b=AM.get(t);
      if(a&&b){{
        const p1=w2s(a.x,a.y),p2=w2s(b.x,b.y);
        x.moveTo(p1.x,p1.y);x.lineTo(p2.x,p2.y);
      }}
    }}
    x.stroke();
  }}
  
  // Zone ID 라벨 (Zone별 중심에 표시)
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
        const zInfo=ZM.get(zid);
        const label=zInfo&&zInfo.name?zid+' ('+zInfo.name+')':zid;
        x.fillStyle=zoneColors[zid]||'#58a6ff';
        x.globalAlpha=0.8;
        x.fillText(label,p.x,p.y);
        x.globalAlpha=1;
      }}
    }}
    x.textAlign='left';
  }}
  
  // Station 및 HID ID 그리기
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
                // Zone 색상 또는 기본색
                const zColor=a.zone&&zoneColors[a.zone]?zoneColors[a.zone]:((st.type>=5&&st.type<=9)?'#f85149':'#d29922');
                x.fillStyle=zColor;
                x.beginPath();x.arc(p.x,p.y,r,0,Math.PI*2);x.fill();
              }}
              if(showHID&&sc>0.3){{
                x.fillStyle='#c9d1d9';
                // HID ID (port_id) 표시
                x.fillText(st.port_id||'HID-'+sn,p.x+r+2,p.y+3);
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

function search(){{
  const q=document.getElementById('q').value.trim().toUpperCase();
  if(!q)return;
  
  // 1. HID ID (port_id)로 직접 검색
  const addrByHID=HID2Addr.get(q);
  if(addrByHID){{
    const p=w2s(addrByHID.x,addrByHID.y);
    ox+=w/2-p.x;oy+=h/2-(h-p.y);
    const stNames=addrByHID.stations.map(sn=>{{const st=SM.get(sn);return st?st.port_id||sn:sn}}).join(', ');
    document.getElementById('msg').textContent='Found: '+stNames+' (Zone:'+addrByHID.zone+')';
    rn();return;
  }}
  
  // 2. HID ID 부분 매칭
  for(const s of S){{
    if(s.port_id&&s.port_id.toUpperCase().includes(q)){{
      for(const a of A){{
        if(a.stations&&a.stations.includes(s.no)){{
          const p=w2s(a.x,a.y);
          ox+=w/2-p.x;oy+=h/2-(h-p.y);
          document.getElementById('msg').textContent='Found: '+s.port_id+' (Zone:'+a.zone+')';
          rn();return;
        }}
      }}
    }}
  }}
  
  // 3. Zone ID로 검색
  for(const a of A){{
    if(a.zone&&a.zone.toUpperCase().includes(q)){{
      const p=w2s(a.x,a.y);
      ox+=w/2-p.x;oy+=h/2-(h-p.y);
      document.getElementById('msg').textContent='Zone: '+a.zone;
      rn();return;
    }}
  }}
  
  // 4. Addr 번호로 검색 (숫자만 입력시)
  const addrNo=parseInt(q);
  if(!isNaN(addrNo)&&AM.has(addrNo)){{
    const a=AM.get(addrNo);
    const p=w2s(a.x,a.y);
    ox+=w/2-p.x;oy+=h/2-(h-p.y);
    const stNames=a.stations.map(sn=>{{const st=SM.get(sn);return st?st.port_id||sn:sn}}).join(', ');
    document.getElementById('msg').textContent='Addr:'+addrNo+(stNames?' → HID:'+stNames:'');
    rn();return;
  }}
  
  document.getElementById('msg').textContent='Not found: '+q;
}}

// 마우스 오버 정보 - HID ID 우선 표시
c.onmousemove=e=>{{
  if(dr){{
    ox+=e.clientX-lx;oy-=e.clientY-ly;lx=e.clientX;ly=e.clientY;rn();
  }}else{{
    const wp=s2w(e.clientX,e.clientY);
    let found=null;
    const threshold=50/sc;
    for(const a of A){{
      const dx=a.x-wp.x,dy=a.y-wp.y;
      if(Math.sqrt(dx*dx+dy*dy)<threshold){{
        found=a;break;
      }}
    }}
    if(found){{
      let info='';
      // HID ID를 먼저 표시
      if(found.stations&&found.stations.length){{
        const stInfo=found.stations.map(sn=>{{
          const st=SM.get(sn);
          return st?(st.port_id||'HID-'+sn)+' (type:'+st.type+')':'HID-'+sn;
        }}).join(', ');
        info='HID: '+stInfo+' | ';
      }}
      info+='Addr: '+found.no+' | Zone: '+(found.zone||'N/A');
      document.getElementById('info').textContent=info;
    }}else{{
      document.getElementById('info').textContent='마우스 오버: HID 정보 표시';
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
    print("=" * 40)
    print("OHT Layout XML → HTML 변환기 (HID View)")
    print("=" * 40)
    
    xml_file = select_file()
    if not xml_file:
        print("취소됨")
        return
    
    print(f"파일: {xml_file}")
    
    base = os.path.splitext(xml_file)[0]
    output_file = f"{base}_hid_viewer.html"
    
    addresses, stations, zones, connections = parse_xml(xml_file)
    if not addresses:
        print("데이터 없음")
        return
    
    generate_html(addresses, stations, zones, connections, output_file)
    
    root = Tk()
    root.withdraw()
    messagebox.showinfo("완료", f"HTML 저장 완료!\n{output_file}")
    root.destroy()
    
    import webbrowser
    webbrowser.open(output_file)

if __name__ == '__main__':
    main()