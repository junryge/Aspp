# -*- coding: utf-8 -*-
"""
ML_LO.py — ML 예측 결과 Logpresso 적재기

Rule_LO.py 와 동일 패턴 (json "{...}" | import <table>) 이지만 별도 테이블 사용:
  - 룰베이스: test_table3 (Rule_LO)
  - ML:       test_table4 (본 모듈)

사용 (ml_predict_runner_v41.py 에서 import):
    import ML_LO
    ML_LO.start()
    ML_LO.upload(fields, row_values)   # 매 예측마다
    ML_LO.stop()

설정: 같은 config.json 사용. ml_table_name, ml_file_label 키 추가.
"""
import json
import logging
import os
import queue
import threading
import time
import urllib.parse
import urllib3

import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = logging.getLogger("ML_LO")
log.setLevel(logging.INFO)
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [ML_LO] %(message)s"))
    log.addHandler(h)


# ============================================================
# 설정 로드 (config.json + api_key.txt)
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_config():
    path = os.path.join(_HERE, "config.json")
    if not os.path.exists(path):
        log.warning(f"config.json 없음 ({path}) — ML_LO 비활성")
        return {"enabled": False}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error(f"config.json 파싱 실패: {e}")
        return {"enabled": False}


def _load_api_key(key_file_name):
    for p in [os.path.join(_HERE, key_file_name),
              os.path.join(os.getcwd(), key_file_name)]:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return f.read().strip().splitlines()[0].strip()
    log.warning(f"{key_file_name} 없음 — 인증 실패 가능")
    return ""


CFG = _load_config()
# ML 전용 enabled 키 (없으면 전체 enabled 따라감)
ENABLED = bool(CFG.get("ml_enabled", CFG.get("enabled", False)))
API_KEY = _load_api_key(CFG.get("api_key_file", "api_key.txt")) if ENABLED else ""

LOGPRESSO_BASE = CFG.get("logpresso_base", "http://localhost:8888/logpresso")
TABLE_NAME     = CFG.get("ml_table_name", "test_table4")   # ★ ML 전용 테이블
FILE_LABEL     = CFG.get("ml_file_label", "ML_system")      # ★ ML 식별자
ASYNC_UPLOAD   = bool(CFG.get("async_upload", True))
QUEUE_MAX      = int(CFG.get("queue_max_size", 10000))
RETRY_ON_FAIL  = int(CFG.get("retry_on_fail", 3))
RETRY_BACKOFF  = float(CFG.get("retry_backoff", 1.0))
LOG_EVERY_N    = int(CFG.get("log_every_n", 60))
FAIL_SILENT    = bool(CFG.get("fail_silent", True))
HTTP_TIMEOUT   = int(CFG.get("http_timeout", 30))

INSERT_PATH    = CFG.get("_endpoints", {}).get("insert", "/httpexport/query.csv")
INSERT_URL     = LOGPRESSO_BASE.rstrip("/") + INSERT_PATH


# ============================================================
# 내부 상태
# ============================================================
_queue: "queue.Queue" = None
_worker_thread = None
_stop_flag = threading.Event()
_count = 0
_fail_count = 0


# ============================================================
# Logpresso 쿼리 (json "{...}" | import <table>) — Rule_LO 와 동일 패턴
# ============================================================
def _to_maru_literal(row_dict):
    parts = []
    for k, v in row_dict.items():
        if v is None or v == '':
            parts.append(f"{k} = null")
        else:
            s = str(v).replace("'", "\\'")
            parts.append(f"{k} = '{s}'")
    return "{" + ", ".join(parts) + "}"


def _build_query(row_dict):
    literal = _to_maru_literal(row_dict)
    escaped = literal.replace('"', '\\"')
    return f'json "{escaped}" | import {TABLE_NAME}'


def _post_query(q):
    qs = " ".join(q.split())
    url = f"{INSERT_URL}?_apikey={API_KEY}&_q={urllib.parse.quote(qs, safe='')}"
    r = requests.get(url, verify=False, timeout=HTTP_TIMEOUT)
    if r.status_code != 200 or r.text.strip().startswith("<"):
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r


def _send_one(row_dict):
    q = _build_query(row_dict)
    last_err = None
    for attempt in range(RETRY_ON_FAIL):
        try:
            _post_query(q)
            return True
        except Exception as e:
            last_err = e
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
    log.warning(f"ML 적재 실패 (재시도 {RETRY_ON_FAIL}회): {type(last_err).__name__}: {last_err}")
    return False


# ============================================================
# 비동기 워커
# ============================================================
def _worker():
    global _count, _fail_count
    while not _stop_flag.is_set() or (_queue and not _queue.empty()):
        try:
            row_dict = _queue.get(timeout=1.0)
        except queue.Empty:
            continue
        ok = _send_one(row_dict)
        if ok:
            _count += 1
            if LOG_EVERY_N and _count % LOG_EVERY_N == 0:
                log.info(f"ML 적재 누적 {_count}행")
        else:
            _fail_count += 1


# ============================================================
# 외부 API
# ============================================================
def start():
    global _queue, _worker_thread
    if not ENABLED:
        log.info("ML_LO 비활성 (config.json: enabled/ml_enabled=false)")
        return
    if not API_KEY:
        log.warning("API key 없음 — 적재 시도하나 실패 예상")
    log.info(f"ML 적재 활성 — {LOGPRESSO_BASE} / 테이블={TABLE_NAME} / file={FILE_LABEL}")
    if ASYNC_UPLOAD:
        _queue = queue.Queue(maxsize=QUEUE_MAX)
        _stop_flag.clear()
        _worker_thread = threading.Thread(target=_worker, daemon=True, name="ML_LO-worker")
        _worker_thread.start()
        log.info(f"비동기 워커 시작 (queue_max={QUEUE_MAX})")


def upload(fields, row):
    """ML 예측 행 적재. fields/row 는 ml_predict_runner_v41 의 OUT_HEADER / 행 값."""
    if not ENABLED:
        return
    try:
        row_dict = dict(zip(fields, row))
        row_dict['file'] = FILE_LABEL   # 무조건 'ML_system' 하드코딩
        if ASYNC_UPLOAD and _queue is not None:
            try:
                _queue.put_nowait(row_dict)
            except queue.Full:
                try:
                    _queue.get_nowait()
                    _queue.put_nowait(row_dict)
                except queue.Empty:
                    pass
        else:
            _send_one(row_dict)
    except Exception as e:
        if not FAIL_SILENT:
            raise
        log.debug(f"upload 예외 무시: {e}")


def stop():
    if not ENABLED:
        return
    _stop_flag.set()
    if _worker_thread and _worker_thread.is_alive():
        _worker_thread.join(timeout=10.0)
    log.info(f"ML 종료 — 적재 성공 {_count}, 실패 {_fail_count}")


def stats():
    return {
        "enabled": ENABLED,
        "uploaded": _count,
        "failed": _fail_count,
        "queued": _queue.qsize() if _queue else 0,
        "url": INSERT_URL,
        "table": TABLE_NAME,
        "file_label": FILE_LABEL,
    }
