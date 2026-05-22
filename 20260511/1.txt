"""
독립 API 수신 테스트 서버

목적: 외부 시스템 (Demos / Hermes / 회사 vLLM / 그 외 어떤 거든)
       이 보내는 API 결과를 받아서 저장하고 화면에 보여주기.
       demos_v1 / hermes-web 과 완전 분리. 단독 실행.

엔드포인트:
  POST /api/receive       - 임의 JSON 받음. 메모리+파일 저장. 200 OK 응답.
  POST /api/receive/<tag> - tag 별로 따로 저장 (예: /api/receive/llm, /api/receive/demos)
  GET  /api/last          - 최근 수신 N개 (?limit=20&tag=llm)
  GET  /api/clear         - 메모리 비우기 (파일 보존)
  GET  /                  - 간단 대시보드 (자동 새로고침)
  GET  /api/health        - 살아있나

실행:
  python api_receiver.py            # 포트 9100 (기본)
  python api_receiver.py 9200       # 포트 지정

테스트:
  curl -X POST http://localhost:9100/api/receive \
       -H "Content-Type: application/json" \
       -d '{"hello":"world","value":42}'

  curl -X POST http://localhost:9100/api/receive/llm \
       -H "Content-Type: application/json" \
       -d '{"model":"qwen","output":"안녕하세요"}'

  curl http://localhost:9100/api/last?limit=5
"""
import os
import sys
import json
import time
from collections import deque
from datetime import datetime
from flask import Flask, request, jsonify, Response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "received")
os.makedirs(LOG_DIR, exist_ok=True)

MAX_MEM = 200
_received = deque(maxlen=MAX_MEM)

app = Flask(__name__)


def _save_to_file(tag, payload):
    ts = datetime.now().strftime("%Y%m%d")
    path = os.path.join(LOG_DIR, f"{tag}_{ts}.jsonl")
    line = json.dumps({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "payload": payload,
    }, ensure_ascii=False)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[file-save 실패] {e}")


def _receive(tag):
    payload = request.get_json(silent=True)
    if payload is None:
        raw = request.get_data(as_text=True) or ""
        payload = {"_raw": raw[:5000]}

    rec = {
        "id": int(time.time() * 1000),
        "ts": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "remote": request.remote_addr,
        "headers": {k: v for k, v in request.headers.items()
                    if k.lower() not in ("cookie", "authorization")},
        "payload": payload,
    }
    _received.append(rec)
    _save_to_file(tag, payload)

    print(f"[{rec['ts']}] [{tag}] from={rec['remote']} "
          f"keys={list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__}")

    return jsonify({"ok": True, "id": rec["id"], "tag": tag, "ts": rec["ts"]})


@app.route("/api/receive", methods=["POST"])
def receive_default():
    return _receive("default")


@app.route("/api/receive/<tag>", methods=["POST"])
def receive_tagged(tag):
    safe_tag = "".join(c for c in tag if c.isalnum() or c in "-_")[:32] or "default"
    return _receive(safe_tag)


@app.route("/api/last")
def api_last():
    try:
        limit = max(1, min(MAX_MEM, int(request.args.get("limit", 20))))
    except ValueError:
        limit = 20
    tag_filter = request.args.get("tag", "").strip()

    items = list(_received)
    if tag_filter:
        items = [r for r in items if r["tag"] == tag_filter]
    items = items[-limit:][::-1]

    return jsonify({
        "count": len(items),
        "total_in_memory": len(_received),
        "items": items,
    })


@app.route("/api/clear")
def api_clear():
    n = len(_received)
    _received.clear()
    return jsonify({"cleared": n})


@app.route("/api/health")
def api_health():
    return jsonify({
        "ok": True,
        "in_memory": len(_received),
        "log_dir": LOG_DIR,
        "files": sorted(os.listdir(LOG_DIR)) if os.path.isdir(LOG_DIR) else [],
    })


_DASHBOARD_HTML = """<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8"><title>API Receiver</title>
<style>
  body{font-family:ui-monospace,Menlo,Consolas,monospace;margin:16px;background:#0f1115;color:#e7e9ec}
  h1{margin:0 0 8px;font-size:18px}
  .bar{display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
  .bar input,.bar select{background:#1a1d24;color:#e7e9ec;border:1px solid #2a2f3a;border-radius:6px;padding:6px 8px}
  .bar button{background:#2a6df4;color:#fff;border:0;border-radius:6px;padding:6px 12px;cursor:pointer}
  .bar button.gray{background:#374151}
  .item{background:#1a1d24;border:1px solid #2a2f3a;border-radius:8px;padding:10px;margin:8px 0}
  .head{display:flex;gap:10px;font-size:12px;color:#9ca3af;margin-bottom:6px}
  .tag{background:#2a6df4;color:#fff;padding:1px 6px;border-radius:4px;font-size:11px}
  pre{white-space:pre-wrap;word-break:break-all;margin:0;font-size:12px;color:#cbd5e1}
  .stat{font-size:12px;color:#9ca3af}
</style></head>
<body>
<h1>API Receiver Dashboard</h1>
<div class="bar">
  <label>tag <input id="tag" placeholder="(all)" style="width:120px"></label>
  <label>limit <input id="limit" type="number" value="20" min="1" max="200" style="width:60px"></label>
  <label><input id="auto" type="checkbox" checked> auto refresh 2s</label>
  <button onclick="load()">refresh</button>
  <button class="gray" onclick="if(confirm('메모리 비우기?'))fetch('/api/clear').then(load)">clear</button>
  <span id="stat" class="stat"></span>
</div>
<div id="list"></div>
<script>
async function load(){
  const tag = document.getElementById('tag').value.trim();
  const lim = document.getElementById('limit').value || 20;
  const u = '/api/last?limit=' + lim + (tag ? '&tag=' + encodeURIComponent(tag) : '');
  const r = await fetch(u);
  const j = await r.json();
  document.getElementById('stat').textContent =
    '표시 ' + j.count + ' / 메모리 ' + j.total_in_memory;
  const html = (j.items||[]).map(it =>
    '<div class="item">' +
      '<div class="head">' +
        '<span class="tag">' + escapeHtml(it.tag) + '</span>' +
        '<span>' + escapeHtml(it.ts) + '</span>' +
        '<span>from ' + escapeHtml(it.remote||'') + '</span>' +
        '<span>id=' + it.id + '</span>' +
      '</div>' +
      '<pre>' + escapeHtml(JSON.stringify(it.payload, null, 2)) + '</pre>' +
    '</div>'
  ).join('');
  document.getElementById('list').innerHTML = html || '<div class="stat">(수신 없음)</div>';
}
function escapeHtml(s){
  return String(s==null?'':s).replace(/[&<>"']/g, c=>(
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
  ));
}
load();
setInterval(()=>{ if(document.getElementById('auto').checked) load(); }, 2000);
</script>
</body></html>"""


@app.route("/")
def dashboard():
    return Response(_DASHBOARD_HTML, mimetype="text/html; charset=utf-8")


if __name__ == "__main__":
    port = 9100
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    print("=" * 60)
    print("  API Receiver (독립 단일 파일)")
    print("=" * 60)
    print(f"  포트: {port}")
    print(f"  로그 디렉토리: {LOG_DIR}")
    print(f"  대시보드:  http://localhost:{port}/")
    print(f"  엔드포인트: POST http://localhost:{port}/api/receive")
    print(f"             POST http://localhost:{port}/api/receive/<tag>")
    print("=" * 60)

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
