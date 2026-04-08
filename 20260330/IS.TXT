#!/usr/bin/env python3
"""5대 OHT 차량의 5분(600틱) 이동 데이터를 TXT로 생성"""

import json
import math
import random
import urllib.request
import sys

SERVER_URL = "http://localhost:10003"
INTERVAL = 0.5
DURATION = 300  # 5분
TOTAL_TICKS = int(DURATION / INTERVAL)  # 600

VEHICLES = [
    {"id": "V00795", "cur": 12340, "next": 12341, "dist": 14, "state": 1, "isFull": 1,
     "equipId": "4PDMV608", "src": "4ABL3301A_OUT04", "dst": "4ANZ19-701", "vel": 90},
    {"id": "V00564", "cur": 2115,  "next": 2116,  "dist": 9,  "state": 1, "isFull": 1,
     "equipId": "4PDD0270", "src": "4ANZ25-205",      "dst": "4KCW3301_3", "vel": 50},
    {"id": "V00975", "cur": 6033,  "next": 3294,  "dist": 0,  "state": 1, "isFull": 1,
     "equipId": "6PDB1402", "src": "4EPR5301_3",      "dst": "4ANZ03-302", "vel": 50},
    {"id": "V00313", "cur": 1071,  "next": 1072,  "dist": 0,  "state": 1, "isFull": 1,
     "equipId": "6PDN4064", "src": "4CSC1603_3",      "dst": "4AFZ47-328", "vel": 50},
    {"id": "V00649", "cur": 1448,  "next": 1449,  "dist": 0,  "state": 1, "isFull": 1,
     "equipId": "4NDNA076", "src": "4ALFE001_AO31",   "dst": "4AFZ15-160", "vel": 50},
]


def make_msg(v):
    return (
        f"2,OHT,{v['id']},{v['state']},{v['isFull']},0000,1,"
        f"{v['cur']},{v['dist']},{v['next']},"
        f"4,4,{v['equipId']},{v['dst']},00000000,0000,"
        f"{v['src']},{v['dst']},{v['vel']},0,0"
    )


def fetch_layout(url):
    print(f"레이아웃 로딩: {url}/api/layout")
    with urllib.request.urlopen(f"{url}/api/layout", timeout=10) as resp:
        data = json.loads(resp.read())

    node_xy = {}
    for n in data.get("nodes", []):
        node_xy[n["no"]] = (n["x"], n["y"])

    adj = {}
    edge_dist = {}
    for e in data.get("edges", []):
        f, t = e["from"], e["to"]
        adj.setdefault(f, []).append(t)
        if f in node_xy and t in node_xy:
            dx = node_xy[t][0] - node_xy[f][0]
            dy = node_xy[t][1] - node_xy[f][1]
            edge_dist[(f, t)] = int(math.sqrt(dx*dx + dy*dy) * 10)

    print(f"  노드: {len(node_xy)}개  엣지: {len(edge_dist)}개")
    return adj, edge_dist


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else SERVER_URL
    adj, edge_dist = fetch_layout(url)

    vehicles = []
    for v in VEHICLES:
        vv = dict(v)
        vv["edge_max"] = edge_dist.get((vv["cur"], vv["next"]), 100)
        vehicles.append(vv)

    output = "oht_5v_data.txt"
    lines = []

    for tick in range(TOTAL_TICKS):
        batch = []
        for v in vehicles:
            v["dist"] += 5
            if v["dist"] >= v["edge_max"]:
                v["dist"] = 0
                v["cur"] = v["next"]
                neighbors = adj.get(v["cur"], [])
                if neighbors:
                    v["next"] = random.choice(neighbors)
                    v["edge_max"] = edge_dist.get((v["cur"], v["next"]), 100)
            batch.append(make_msg(v))
        lines.append("\n".join(batch))

        if tick % 100 == 0:
            print(f"  tick {tick}/{TOTAL_TICKS}")

    with open(output, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines) + "\n")

    print(f"\n생성 완료: {output}")
    print(f"  {len(vehicles)}대 × {TOTAL_TICKS}틱 = {len(vehicles)*TOTAL_TICKS}건")
    print(f"  재생 시간: {DURATION}초 ({DURATION//60}분)")


if __name__ == "__main__":
    main()
