# -*- coding: utf-8 -*-
"""
AMHS Daily Report System - 서버 (파이썬 표준 라이브러리만 사용 — 별도 설치 불필요)
실행: python server.py  →  http://localhost:8000

폴더 구조:
  server.py
  config.json                      ← 관리자 계정/암호/포트 설정 (여기서 수정)
  reports_db.json                  ← 리포트 데이터 (자동 생성)
  users_db.json                    ← 운영담당자 계정 (관리자가 웹 화면에서 생성)
  INDEX/Daily_Report_System.html   ← 웹 화면

계정 체계:
  - 최고 관리자(레포트 등록자): config.json 의 adminId / adminPassword 로 로그인
    → 리포트 등록·수정·삭제 + 운영담당자 계정 생성/삭제/비밀번호 변경
  - 운영담당자: 관리자가 웹 화면의 "👥 계정 관리" 에서 생성
    → 등록된 리포트 열람 + 리포트 안 내용 입력·저장만 가능
  - 관리자 비밀번호를 잊으면 config.json 의 adminPassword 를 수정 후 서버 재시작

config.json:
  adminId       : 최고 관리자 로그인 아이디
  adminPassword : 최고 관리자 로그인 비밀번호 (완료 리포트 삭제 암호 겸용)
  editPassword  : 완료 리포트 "수정 잠금 해제" 암호
  host / port   : 서버 주소·포트
※ config.json 수정 후 서버 재시작하면 적용
"""
import json
import hashlib
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

BASE = Path(__file__).resolve().parent
HTML = BASE / "INDEX" / "Daily_Report_System.html"
DB = BASE / "reports_db.json"
USERS = BASE / "users_db.json"
CONFIG = BASE / "config.json"
LOCK = threading.Lock()

SESSIONS = {}  # token -> {id, role, name, exp}
SESSION_TTL = 12 * 3600  # 12시간 (사용 중이면 자동 연장)
REPORT_DATE = re.compile(r"^/api/reports/(\d{4}-\d{2}-\d{2})$")
USER_PW = re.compile(r"^/api/users/([^/]+)/password$")
USER_ONE = re.compile(r"^/api/users/([^/]+)$")

DEFAULT_CONFIG = {
    "adminId": "admin",
    "adminPassword": "AMHS1234",
    "editPassword": "AMHS1234",
    "host": "0.0.0.0",
    "port": 8000,
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG.exists():
        try:
            user = json.loads(CONFIG.read_text(encoding="utf-8"))
            if isinstance(user, dict):
                cfg.update(user)
        except Exception as e:
            print(f"[경고] config.json 읽기 실패, 기본값 사용: {e}")
    else:
        CONFIG.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[안내] config.json 기본 파일을 생성했습니다.")
    return cfg


def load_json(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def hash_pw(pw: str, salt: str = None) -> str:
    salt = salt or secrets.token_hex(8)
    return salt + "$" + hashlib.sha256((salt + pw).encode("utf-8")).hexdigest()


def check_pw(pw: str, stored: str) -> bool:
    try:
        salt, h = str(stored).split("$", 1)
        return hashlib.sha256((salt + pw).encode("utf-8")).hexdigest() == h
    except Exception:
        return False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    # ---------- 공통 ----------
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self):
        # 반드시 응답 전에 호출해서 요청 본문을 소진할 것 (keep-alive 오염 방지)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return None
            if n > 64 * 1024 * 1024:
                self.close_connection = True
                return None
            raw = self.rfile.read(n)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return None
        except Exception:
            self.close_connection = True
            return None

    def _session(self):
        token = self.headers.get("X-Auth-Token") or ""
        s = SESSIONS.get(token)
        if not s:
            return None
        if s["exp"] < time.time():
            SESSIONS.pop(token, None)
            return None
        s["exp"] = time.time() + SESSION_TTL  # 사용 중이면 연장
        return s

    def _auth(self, admin_only=False):
        s = self._session()
        if not s:
            self._send(401, {"ok": False, "error": "로그인이 필요합니다"})
            return None
        if admin_only and s["role"] != "admin":
            self._send(403, {"ok": False, "error": "관리자만 가능합니다"})
            return None
        return s

    # ---------- GET ----------
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            if not HTML.exists():
                self._send(404, {"error": "INDEX/Daily_Report_System.html not found"})
                return
            self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        elif p == "/health":
            self._send(200, {"ok": True, "reports": len(load_json(DB))})
        elif p == "/api/me":
            s = self._session()
            if not s:
                self._send(401, {"ok": False})
                return
            self._send(200, {"ok": True, "id": s["id"], "role": s["role"], "name": s["name"]})
        elif p == "/api/config":
            if not self._auth(admin_only=True):
                return
            cfg = load_config()
            self._send(200, {"editPassword": cfg["editPassword"], "adminPassword": cfg["adminPassword"]})
        elif p == "/api/reports":
            if not self._auth():
                return
            with LOCK:
                self._send(200, load_json(DB))
        elif p == "/api/users":
            if not self._auth(admin_only=True):
                return
            cfg = load_config()
            users = load_json(USERS)
            out = [{"id": cfg["adminId"], "name": "최고 관리자", "role": "admin"}]
            for uid in sorted(users):
                out.append({"id": uid, "name": users[uid].get("name") or "", "role": "operator"})
            self._send(200, out)
        else:
            self._send(404, {"ok": False, "error": "not found"})

    # ---------- POST ----------
    def do_POST(self):
        p = self.path.split("?")[0]
        if p == "/api/login":
            body = self._json_body() or {}
            uid = str(body.get("id") or "").strip()
            pw = str(body.get("pw") or "")
            cfg = load_config()
            role = name = None
            if uid and uid == str(cfg["adminId"]) and pw == str(cfg["adminPassword"]):
                role, name = "admin", "최고 관리자"
            else:
                u = load_json(USERS).get(uid)
                if u and check_pw(pw, u.get("pw") or ""):
                    role, name = "operator", (u.get("name") or uid)
            if not role:
                self._send(401, {"ok": False, "error": "아이디 또는 비밀번호가 올바르지 않습니다"})
                return
            token = secrets.token_hex(16)
            SESSIONS[token] = {"id": uid, "role": role, "name": name, "exp": time.time() + SESSION_TTL}
            self._send(200, {"ok": True, "token": token, "id": uid, "role": role, "name": name})
        elif p == "/api/logout":
            SESSIONS.pop(self.headers.get("X-Auth-Token") or "", None)
            self._send(200, {"ok": True})
        elif p == "/api/users":
            body = self._json_body() or {}
            if not self._auth(admin_only=True):
                return
            uid = str(body.get("id") or "").strip()
            pw = str(body.get("pw") or "")
            name = str(body.get("name") or "").strip()
            cfg = load_config()
            if not uid or not pw:
                self._send(400, {"ok": False, "error": "아이디와 비밀번호가 필요합니다"})
                return
            if len(pw) < 4:
                self._send(400, {"ok": False, "error": "비밀번호는 4자 이상이어야 합니다"})
                return
            if uid == str(cfg["adminId"]):
                self._send(400, {"ok": False, "error": "관리자 아이디와 같은 이름은 사용할 수 없습니다"})
                return
            with LOCK:
                users = load_json(USERS)
                if uid in users:
                    self._send(400, {"ok": False, "error": "이미 존재하는 아이디입니다"})
                    return
                users[uid] = {"pw": hash_pw(pw), "name": name}
                save_json(USERS, users)
            self._send(200, {"ok": True})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    # ---------- PUT ----------
    def do_PUT(self):
        p = self.path.split("?")[0]
        m = REPORT_DATE.match(p)
        if m:  # 날짜별 리포트 저장 (운영담당자는 이미 등록된 날짜만 가능)
            rec = self._json_body()  # 본문 먼저 소진 (응답을 먼저 보내면 keep-alive 오염)
            s = self._auth()
            if not s:
                return
            if not isinstance(rec, dict):
                self._send(400, {"ok": False, "error": "invalid json"})
                return
            ds = m.group(1)
            with LOCK:
                db = load_json(DB)
                if s["role"] != "admin" and ds not in db:
                    self._send(403, {"ok": False, "error": "리포트 등록은 관리자만 할 수 있습니다"})
                    return
                db[ds] = rec
                save_json(DB, db)
            self._send(200, {"ok": True})
            return
        if p == "/api/reports":  # 전체 교체 (JSON 복원) — 관리자 전용
            data = self._json_body()
            if not self._auth(admin_only=True):
                return
            if not isinstance(data, dict):
                self._send(400, {"ok": False, "error": "object expected"})
                return
            with LOCK:
                save_json(DB, data)
            self._send(200, {"ok": True, "count": len(data)})
            return
        m = USER_PW.match(p)
        if m:  # 운영담당자 비밀번호 변경 — 관리자 전용
            body = self._json_body() or {}
            if not self._auth(admin_only=True):
                return
            uid = unquote(m.group(1))
            pw = str(body.get("pw") or "")
            if len(pw) < 4:
                self._send(400, {"ok": False, "error": "비밀번호는 4자 이상이어야 합니다"})
                return
            with LOCK:
                users = load_json(USERS)
                if uid not in users:
                    self._send(404, {"ok": False, "error": "없는 계정입니다"})
                    return
                users[uid]["pw"] = hash_pw(pw)
                save_json(USERS, users)
            self._send(200, {"ok": True})
            return
        self._send(404, {"ok": False, "error": "not found"})

    # ---------- DELETE ----------
    def do_DELETE(self):
        p = self.path.split("?")[0]
        m = REPORT_DATE.match(p)
        if m:  # 리포트 삭제 — 관리자 전용
            if not self._auth(admin_only=True):
                return
            ds = m.group(1)
            with LOCK:
                db = load_json(DB)
                if ds in db:
                    del db[ds]
                    save_json(DB, db)
            self._send(200, {"ok": True})
            return
        m = USER_ONE.match(p)
        if m:  # 운영담당자 계정 삭제 — 관리자 전용
            if not self._auth(admin_only=True):
                return
            uid = unquote(m.group(1))
            with LOCK:
                users = load_json(USERS)
                if uid in users:
                    del users[uid]
                    save_json(USERS, users)
            # 해당 계정의 로그인 세션 즉시 종료
            for t in [t for t, ss in list(SESSIONS.items()) if ss["id"] == uid]:
                SESSIONS.pop(t, None)
            self._send(200, {"ok": True})
            return
        self._send(404, {"ok": False, "error": "not found"})


if __name__ == "__main__":
    cfg = load_config()
    print("=" * 52)
    print("  AMHS Daily Report System")
    print(f"  브라우저에서 열기 →  http://localhost:{cfg['port']}")
    print(f"  최고 관리자 로그인: {cfg['adminId']} / config.json 의 adminPassword")
    print("  운영담당자 계정: 관리자 로그인 후 '👥 계정 관리' 에서 생성")
    print("=" * 52)
    ThreadingHTTPServer((cfg["host"], int(cfg["port"])), Handler).serve_forever()
