#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_map.py - layout.xml -> 실제로 그려지는 2D 지도(HTML, canvas)

OHT2/layout_map_cre.py 는 데이터만 넣고 렌더링 코드가 없어 빈 화면이 나온다.
이 스크립트는 canvas 로 노드/연결을 실제로 그리고, pan/zoom 과
리프터(*ABL*) 포트 강조 표시를 제공한다. 외부 라이브러리 불필요.

사용법:
    python3 make_map.py <layout.xml|layout.zip> <출력.html> [station.dat]

예:
    python3 make_map.py MAP/M16A/BR.layout.xml MAP/M16A/BR.map.html MAP/M16A/BR.station.dat
"""
import sys, os, re, json, zipfile, csv


def load_xml(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            name = next((n for n in zf.namelist() if n.lower().endswith("layout.xml")), None)
            if not name:
                raise FileNotFoundError("zip 안에 layout.xml 없음")
            return zf.read(name).decode("utf-8", "replace")
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_layout(xml_content):
    """nodes: {addr:(x,y)}, connections: [[from,to],...]"""
    nodes = {}
    conns = []
    cur = None          # 현재 Addr params
    nx = None           # 현재 NextAddr params
    in_nx = False
    key_re = re.compile(r'key="([^"]+)"')
    val_re = re.compile(r'value="([^"]*)"')

    def commit_addr(c):
        if c and "address" in c:
            try:
                a = int(c["address"])
                if a > 0:
                    nodes[a] = (round(float(c.get("draw-x", 0)), 2),
                                round(float(c.get("draw-y", 0)), 2))
            except ValueError:
                pass

    for line in xml_content.split("\n"):
        line = line.strip()
        if '<group name="Addr' in line and 'address.Addr"' in line:
            commit_addr(cur)
            cur = {}
            in_nx = False
            continue
        if '<group name="NextAddr' in line and 'NextAddr"' in line:
            in_nx = True
            nx = {}
            continue
        if in_nx and '</group>' in line:
            if cur and "address" in cur and nx and "next-address" in nx:
                try:
                    fa, ta = int(cur["address"]), int(nx["next-address"])
                    if fa > 0 and ta > 0:
                        conns.append([fa, ta])
                except ValueError:
                    pass
            in_nx = False
            continue
        if '<param ' in line and 'key="' in line and 'value="' in line:
            k = key_re.search(line)
            v = val_re.search(line)
            if k and v:
                if in_nx:
                    nx[k.group(1)] = v.group(1)
                elif cur is not None:
                    cur[k.group(1)] = v.group(1)
    commit_addr(cur)
    return nodes, conns


def parse_lifters(station_path):
    """station.dat -> {addr: port_name} (모든 <숫자>ABL 리프터 포트. 4=M14, 6=M16)"""
    if not station_path or not os.path.exists(station_path):
        return {}
    out = {}
    for line in open(station_path, encoding="utf-8", errors="replace"):
        if "ABL" not in line:
            continue
        m = re.search(r'STATION\s*=\s*(.+)', line)
        if not m:
            continue
        parts = [p.strip().strip('"') for p in m.group(1).split(",")]
        try:
            port, addr = parts[3], int(parts[6])
        except (IndexError, ValueError):
            continue
        if re.match(r'\dABL', port) and ("_AI" in port or "_AO" in port):
            out[addr] = port
    return out


def _convex_hull(pts):
    """단조 체인 convex hull. 점 3개 미만이면 그대로 반환."""
    pts = sorted(set(pts))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo = []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    up = []
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def compute_lifter_hids(station_path, layout_input=None, nodes=None, lift=None):
    """각 리프터의 '근처 HID4 구간'을 구함 (count_lifter_inout 과 동일 기준).
    리프터에 경계(lane)가 가장 가까운 HID4(1~37) 구역을 매핑.
    HID_Zone_Master CSV 가 없으면 hid_zone_csv_cre 로 자동 생성.
    반환: {lifter_id: [HID4번호]}"""
    import csv as _csv
    import math
    from collections import defaultdict
    if not station_path or not nodes or not lift:
        return {}
    sd = os.path.dirname(os.path.abspath(station_path))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prefix = os.path.basename(station_path).split(".")[0]
    fab = os.path.basename(sd)

    # 이미 만들어둔 리프터_근처HID4.csv 가 있으면 그걸 사용 (카운트와 동일 기준)
    for cand in (os.path.join(script_dir, "리프터_근처HID4.csv"),
                 os.path.join(sd, "리프터_근처HID4.csv")):
        if os.path.exists(cand):
            out = {}
            with open(cand, encoding="utf-8-sig") as f:
                for r in _csv.DictReader(f):
                    z = (r.get("근처HID4") or "").strip()
                    if z:
                        out[r["Lifter"]] = [z]
            print(f"  리프터별 근처 HID4: {len(out)}기 (출처 {os.path.basename(cand)})")
            return out

    name = f"HID_Zone_Master_{fab}_{prefix}.csv"
    csv_path = None
    for cand in (os.path.join(script_dir, name), os.path.join(sd, name)):
        if os.path.exists(cand):
            csv_path = cand; break
    if csv_path is None and layout_input:
        csv_path = os.path.join(script_dir, name)
        try:
            import hid_zone_csv_cre as H
            print(f"  {name} 없음 -> make_map.py 폴더에 생성 중...")
            H.create_hid_zone_csv(layout_input, csv_path, project_name=f"{fab} Project")
        except Exception as e:
            print(f"  (HID CSV 자동생성 실패: {e})")
    if not csv_path or not os.path.exists(csv_path):
        print("  (HID CSV 없음 -> 리프터 HID 생략)")
        return {}

    # HID4(1~37) lane 좌표
    hid4 = defaultdict(list)
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in _csv.DictReader(f):
            zid = (r.get("Zone_ID") or "").strip()
            if not (zid.isdigit() and 1 <= int(zid) <= 37):
                continue
            for fld in ("IN_Lanes", "OUT_Lanes"):
                for seg in (r.get(fld) or "").split(";"):
                    m = re.match(r'\s*(\d+)\s*→\s*(\d+)', seg)
                    if m:
                        for a in (int(m.group(1)), int(m.group(2))):
                            if a in nodes:
                                hid4[zid].append(nodes[a])
    # 리프터 포트 좌표
    lpts = defaultdict(list)
    for a, p in lift.items():
        if a in nodes:
            lpts[p.split("_")[0]].append(nodes[a])
    # 리프터 -> 최근접 HID4
    out = {}
    for lf in sorted(lpts):
        best, bd = None, 1e18
        for z, pts in hid4.items():
            for px, py in pts:
                for lx, ly in lpts[lf]:
                    d = (lx - px) ** 2 + (ly - py) ** 2
                    if d < bd:
                        bd, best = d, z
        if best:
            out[lf] = [best]
    print(f"  리프터별 근처 HID4: {len(out)}기 (출처 {os.path.basename(csv_path)})")
    return out


HTML = """<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>{title}</title>
<style>
  html,body{{margin:0;height:100%;background:#0d1117;color:#ddd;font-family:monospace;overflow:hidden}}
  #info{{position:fixed;top:8px;left:8px;z-index:10;font-size:13px;background:#161b22cc;padding:6px 10px;border-radius:6px}}
  #legend{{position:fixed;top:8px;right:8px;z-index:10;font-size:12px;background:#161b22cc;padding:6px 10px;border-radius:6px}}
  canvas{{display:block}}
</style></head><body>
<div id="info">{title} · 노드 {nn} · 연결 {nc} · 리프터포트 {nl} · HID매핑 {nz}기<br>휠=확대/축소, 드래그=이동, H=좌우반전, F=상하반전, R=리셋</div>
<div id="legend"><span style="color:#6e7681">●</span>노드 &nbsp; <span style="color:#ffd54f">●</span>IN &nbsp; <span style="color:#3fb950">●</span>OUT &nbsp; <span style="color:#22d3ee">HID n</span>=근방HID &nbsp; <span style="color:#ff8c42">▭</span>리프터M16 &nbsp; <span style="color:#58a6ff">▭</span>리프터M14</div>
<canvas id="cv"></canvas>
<script>
const NODES={nodes_json};      // {{addr:[x,y]}}
const CONNS={conns_json};      // [[from,to]]
const LIFT={lift_json};        // {{addr:port}}
const LHID={lhid_json};        // {{lifter: [HID번호,...]}} 근방 HID
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
// 리프터별 포트 좌표 그룹 -> 범위 박스 계산용
const LGROUP={{}};
for(const a in LIFT){{const p=NODES[a];if(!p)continue;const lf=LIFT[a].split('_')[0];
  (LGROUP[lf]=LGROUP[lf]||[]).push(p);}}
function resize(){{cv.width=innerWidth;cv.height=innerHeight;draw();}}
window.addEventListener('resize',resize);

// 좌표 범위
let xs=[],ys=[];for(const a in NODES){{xs.push(NODES[a][0]);ys.push(NODES[a][1]);}}
const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
const W=maxX-minX||1,H=maxY-minY||1;

let scale=1,ox=0,oy=0,init=false;
function fit(){{
  const m=40;const sx=(cv.width-2*m)/W,sy=(cv.height-2*m)/H;
  scale=Math.min(sx,sy);
  ox=m-minX*scale+(cv.width-2*m-W*scale)/2;
  oy=m-minY*scale+(cv.height-2*m-H*scale)/2;
}}
// 화면좌표 — H/F 키로 좌우·상하 반전
let flipX=false, flipY=false;
function SX(x){{const v=x*scale+ox;return flipX ? cv.width-v : v;}}
function SY(y){{const v=y*scale+oy;return flipY ? cv.height-v : v;}}

function draw(){{
  if(!init){{fit();init=true;}}
  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,cv.width,cv.height);
  // 연결
  ctx.strokeStyle='#58a6ff55';ctx.lineWidth=1;ctx.beginPath();
  for(const c of CONNS){{const f=NODES[c[0]],t=NODES[c[1]];if(!f||!t)continue;
    ctx.moveTo(SX(f[0]),SY(f[1]));ctx.lineTo(SX(t[0]),SY(t[1]));}}
  ctx.stroke();
  // 노드
  const r=Math.max(1.2,1.6*Math.min(scale,1.5));
  ctx.fillStyle='#6e7681';
  for(const a in NODES){{if(LIFT[a])continue;const p=NODES[a];
    ctx.beginPath();ctx.arc(SX(p[0]),SY(p[1]),r,0,7);ctx.fill();}}
  // 리프터 범위 사각형 + 이름 + 근방 HID번호 (4=M14 파랑, 6=M16 주황)
  const pad=14;
  ctx.lineWidth=1.5;ctx.font='bold 13px monospace';
  for(const lf in LGROUP){{const ps=LGROUP[lf];
    const fab=lf[0]==='6'?'M16':(lf[0]==='4'?'M14':'?');
    const col=lf[0]==='6'?'#ff8c42':'#58a6ff';
    let aX=ps.map(p=>SX(p[0])),aY=ps.map(p=>SY(p[1]));
    const x1=Math.min(...aX)-pad,x2=Math.max(...aX)+pad,y1=Math.min(...aY)-pad,y2=Math.max(...aY)+pad;
    ctx.strokeStyle=col;ctx.setLineDash([5,3]);
    ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.setLineDash([]);
    ctx.fillStyle=col;ctx.fillText(fab+'-'+lf,x1,y1-4);
    // 근방 HID 번호 (박스 아래)
    const hids=LHID[lf];
    if(hids&&hids.length){{ctx.fillStyle='#22d3ee';ctx.font='bold 11px monospace';
      ctx.fillText('HID '+hids.join(','),x1,y2+13);ctx.font='bold 13px monospace';}}
  }}
  // 리프터 포트 강조: IN=노랑, OUT=초록
  for(const a in LIFT){{const p=NODES[a];if(!p)continue;const port=LIFT[a];
    ctx.fillStyle=port.includes('_AI')?'#ffd54f':'#3fb950';
    ctx.beginPath();ctx.arc(SX(p[0]),SY(p[1]),Math.max(3,r+2),0,7);ctx.fill();
    if(scale>0.6){{ctx.fillStyle='#ffffff';ctx.font='10px monospace';
      ctx.fillText(port.split('_')[1],SX(p[0])+5,SY(p[1])-3);}}
  }}
}}
// pan
let drag=false,lx,ly;
cv.addEventListener('mousedown',e=>{{drag=true;lx=e.clientX;ly=e.clientY;}});
addEventListener('mouseup',()=>drag=false);
addEventListener('mousemove',e=>{{if(!drag)return;
  ox+=(flipX?-(e.clientX-lx):(e.clientX-lx));
  oy+=(flipY?-(e.clientY-ly):(e.clientY-ly));
  lx=e.clientX;ly=e.clientY;draw();}});
// zoom (마우스 위치 기준)
cv.addEventListener('wheel',e=>{{e.preventDefault();const f=e.deltaY<0?1.15:1/1.15;
  const mx=flipX?cv.width-e.clientX:e.clientX, my=flipY?cv.height-e.clientY:e.clientY;
  ox=mx-(mx-ox)*f;oy=my-(my-oy)*f;scale*=f;draw();}},{{passive:false}});
// 키: H=좌우반전, F=상하반전, R=리셋
addEventListener('keydown',e=>{{const k=e.key.toLowerCase();
  if(k==='h'){{flipX=!flipX;draw();}}
  if(k==='f'){{flipY=!flipY;draw();}}
  if(k==='r'){{init=false;draw();}}}});
resize();
</script></body></html>"""


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    inp = sys.argv[1]
    out = os.path.basename(sys.argv[2])   # 항상 '실행한 폴더(현재 위치)'에 생성
    station = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"입력: {inp}")

    # 입력 파일 존재 확인 (없으면 친절 안내)
    if not os.path.exists(inp):
        print(f"\n[오류] 파일을 찾을 수 없습니다: {inp}")
        d = os.path.dirname(inp) or "."
        if os.path.isdir(d):
            cand = [f for f in os.listdir(d) if f.lower().endswith((".xml", ".zip"))]
            print(f"  '{d}' 폴더의 layout 후보: {cand if cand else '없음'}")
            # 같은 폴더에 .zip 있으면 그것으로 자동 대체 제안
            base = os.path.basename(inp)
            zipname = base.replace(".layout.xml", ".layout.zip").replace(".xml", ".zip")
            zpath = os.path.join(d, zipname)
            if os.path.exists(zpath):
                print(f"  -> .zip 발견! 이걸로 다시 실행하세요:\n     python make_map.py \"{zpath}\" \"{out}\""
                      + (f" \"{station}\"" if station else ""))
        else:
            print(f"  '{d}' 폴더 자체가 없습니다. 경로를 확인하세요.")
        sys.exit(1)

    xml = load_xml(inp)
    print(f"  XML 크기: {len(xml):,} bytes")
    nodes, conns = parse_layout(xml)
    print(f"  노드 {len(nodes)} · 연결 {len(conns)}")
    lift = parse_lifters(station)
    print(f"  리프터 포트 {len(lift)}")
    lhid = compute_lifter_hids(station, inp, nodes, lift) if station else {}

    title = os.path.splitext(os.path.basename(out))[0]
    html = HTML.format(
        title=title, nn=len(nodes), nc=len(conns), nl=len(lift), nz=len(lhid),
        nodes_json=json.dumps({str(k): v for k, v in nodes.items()}),
        conns_json=json.dumps(conns),
        lift_json=json.dumps({str(k): v for k, v in lift.items()}),
        lhid_json=json.dumps(lhid),
    )
    outdir = os.path.dirname(os.path.abspath(out))
    os.makedirs(outdir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"생성 완료: {os.path.abspath(out)} ({os.path.getsize(out):,} bytes)")
    print(f"  -> 이 파일을 브라우저로 여세요")

    # 리프터-근방HID 매핑 CSV 도 실행 폴더에 생성
    if lhid:
        csv_out = "리프터_HID.csv"
        with open(csv_out, "w", newline="", encoding="utf-8-sig") as cf:
            w = csv.writer(cf)
            w.writerow(["Lifter", "FAB", "근방HID_개수", "근방HID_Zone번호"])
            for lf in sorted(lhid):
                fab = "M16" if lf[0] == "6" else ("M14" if lf[0] == "4" else "?")
                zids = lhid[lf]
                w.writerow([lf, fab, len(zids), "; ".join(zids)])
        print(f"리프터-HID CSV: {os.path.abspath(csv_out)} ({len(lhid)}기)")


if __name__ == "__main__":
    main()
