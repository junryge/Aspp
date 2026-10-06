# -*- coding: utf-8 -*-
"""
run_oht.py — HID_VHL_OHT.py + Rule_hid.py (+ OHT_MAP_INDEX.py) 같이 돌리기

    python run_oht.py                     실시간 (계속 돈다, Ctrl+C 로 멈춤)
    python run_oht.py --no-problem-map    실시간인데 문제맵은 안 만듦
    (HID_VHL_OHT.py 의 옵션을 그대로 넘긴다)

  50초마다  HID_VHL_OHT 가 로그프레소에서 차량 보고를 받아 판정
  1분 끝나면 그 분 한 줄 → CSV  (HID_BOTTLENECK/{FAB}/HID_BOTTLENECK_{FAB}_YYYYMMDD.csv)
                          → 같은 줄 + FAB → 로그프레소 AMHS_VHL_OHT  (Rule_hid, 매분 51초에 모아서)
                          → 경계 이상이면 문제맵 (HID_BOTTLENECK/PROBLEM_MAP/…)
                          → 문제맵 경로 → ../m16a_hubroom_event_prediction/oht_map/OHT_MAP_YYYYMMDD.csv
                                          (OHT_MAP_INDEX — 다운로드 화면용 목록)
  돌릴 FAB 은 HID_CONFIG.json 의 RUN_FABS. 적재 · 목록 설정은 config.json (Rule_LO 와 같은 파일).
  다시 켜도 이미 CSV 에 쓴 분은 다시 안 넣는다 (CSV · AMHS_VHL_OHT · 목록 중복 없음).
  영문 등급은 정상 NONE · 경계 WARNING · 위험 CRITICAL · 초위험 EMERGENCY (ALARM_EN_NEW)
"""
import json
import sys
import threading
import time
from pathlib import Path

import HID_VHL_OHT as HID
import Rule_hid

# ★ 로그프레소 저장 시각 — 매분 이 초에 그동안 CSV 에 새로 쓴 줄을 한 번에 AMHS_VHL_OHT 로
#   config.json 의 "hid_upload_at_sec" 로 바꿀 수 있다 (0~59, 기본 51)
try:
    _cfg = json.loads((Path(__file__).resolve().parent / "config.json").read_text(encoding="utf-8"))
except Exception:
    _cfg = {}
UPLOAD_AT_SEC = int(_cfg.get("hid_upload_at_sec", 51)) % 60

# ★ 영문 등급 (2026-10 변경) — HID_VHL_OHT.py 는 그대로 두고 여기서 바꿔 끼운다
#   CSV ALARM_EN · 문제맵 파일 이름 · 로그프레소 alarm_en 이 모두 이 이름으로 나간다
ALARM_EN_NEW = {"정상": "NONE", "경계": "WARNING", "위험": "CRITICAL", "초위험": "EMERGENCY"}


def _rename_alarm_en():
    """HID 안의 한글→영문 사전(ALARM_EN_OF 등)과 영문→한글 사전 값을 새 이름으로 바꾼다."""
    kr = set(ALARM_EN_NEW)
    en = {"NORMAL", "WARNING", "DANGER", "CRITICAL", "NONE", "EMERGENCY"}   # 예전 · 새 영문 이름
    done = []
    for name, d in list(vars(HID).items()):
        if not isinstance(d, dict) or not d:
            continue
        # 키 · 값이 전부 문자열인 사전만 본다 (DEFAULT_LEVELS 처럼 값이 사전인 것 · 숫자 · None 키는 건너뜀)
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in d.items()):
            continue
        if set(d) <= kr and set(d.values()) <= en:          # 한글 → 영문 (색 등 다른 사전은 안 건드림)
            for k in d:
                d[k] = ALARM_EN_NEW[k]
            done.append(name)
        elif set(d) <= en and set(d.values()) <= kr:        # 영문 → 한글
            new = {ALARM_EN_NEW[v]: v for v in d.values()}
            d.clear()
            d.update(new)
            done.append(name)
    HID.log.info(f"  영문 등급: " + " · ".join(f"{k} {v}" for k, v in ALARM_EN_NEW.items())
                 + (f"  ({', '.join(done)})" if done else "  (⚠ 바꿀 사전을 못 찾음 — 예전 이름 그대로)"))

try:
    import OHT_MAP_INDEX as MAP_INDEX
except Exception as _e:                               # 없거나 깨져도 본 기능은 그대로 돈다
    MAP_INDEX = None
    print(f"[run_oht] OHT_MAP_INDEX 불러오기 실패 — 문제맵 목록 CSV 없이 진행: {_e}")


def _hook_map_index():
    """문제맵을 만들 때마다 바로 목록 CSV 에 넣는다 (HID_VHL_OHT.py 는 그대로)."""
    if MAP_INDEX is None or not MAP_INDEX.ENABLED:
        return
    try:
        n = MAP_INDEX.sync()                          # 켤 때 — 그동안 만들어진 맵 전부 (빠진 것만)
        HID.log.info(f"  문제맵 목록 → {MAP_INDEX.INDEX_DIR}" + (f"  ({n}개 추가)" if n else ""))
    except Exception as e:
        HID.log.warning(f"  문제맵 목록 정리 실패 — 계속 진행: {e}")
    orig = HID.problem_map

    def problem_map(*a, **kw):
        res = orig(*a, **kw)
        try:
            MAP_INDEX.sync_recent()
        except Exception as e:
            HID.log.warning(f"  문제맵 목록 추가 실패 — 다음 맵 때 같이 넣는다: {e}")
        return res

    HID.problem_map = problem_map


# ---------- 로그프레소 저장: 매분 UPLOAD_AT_SEC 초 ----------
_upload_now = Rule_hid.upload_rows                    # 원래 함수 (바로 보냄)
_queue, _sent = [], set()                             # 보낼 줄 · 최근 보낸 줄 (같은 줄 두 번 안 보냄)
_lock = threading.Lock()
_stop = threading.Event()


def _queue_rows(fab, header, rows):
    """CSV 에 1분 줄을 쓸 때 — 바로 보내지 않고 모아 둔다."""
    with _lock:
        for r in rows:
            k = (fab, tuple(r))
            if k in _sent or any(k == (f, tuple(x)) for f, _, x in _queue):
                continue
            _queue.append((fab, header, list(r)))


def _flush():
    with _lock:
        todo = list(_queue)
        _queue.clear()
    if not todo:
        return
    groups = {}                                       # (FAB, 헤더) 별로 묶어 원래 함수로
    for fab, header, r in todo:
        groups.setdefault((fab, tuple(header)), []).append(r)
    for (fab, header), rows in groups.items():
        try:
            _upload_now(fab, list(header), rows)
        except Exception as e:
            HID.log.warning(f"  AMHS_VHL_OHT 저장 실패 ({fab} {len(rows)}줄) — CSV 는 그대로: {e}")
        with _lock:
            _sent.update((fab, tuple(r)) for r in rows)
            if len(_sent) > 5000:                     # 오래된 것은 비운다 (CSV 쪽에서도 중복을 막는다)
                _sent.clear()


def _uploader():
    while not _stop.is_set():
        wait = (UPLOAD_AT_SEC - time.time() % 60) % 60
        if _stop.wait(wait if wait > 0.05 else 60):
            break
        _flush()


def _hook_upload():
    """Rule_hid.upload_rows 를 '모아 두기' 로 바꿔 끼운다.
       HID_VHL_OHT 가 스스로 등록하는 것도 같은 함수가 되어 바로 보내는 일이 없다."""
    Rule_hid.upload_rows = _queue_rows
    HID.SAVE_HOOKS[:] = [h for h in HID.SAVE_HOOKS if h is not _upload_now]
    if _queue_rows not in HID.SAVE_HOOKS:
        HID.SAVE_HOOKS.append(_queue_rows)            # CSV 에 1분 줄을 쓸 때마다 → 모아 둠
    threading.Thread(target=_uploader, name="AMHS_VHL_OHT", daemon=True).start()
    HID.log.info(f"  로그프레소 AMHS_VHL_OHT 저장: 매분 {UPLOAD_AT_SEC}초")


def main():
    try:
        _rename_alarm_en()
    except Exception as e:                            # 이름 바꾸기 때문에 멈추는 일은 없게
        HID.log.warning(f"  영문 등급 바꾸기 실패 — 예전 이름으로 진행: {e}")
    Rule_hid.start()
    _hook_upload()
    _hook_map_index()
    try:
        HID.main()                                    # 50초마다 판정 (옵션은 HID_VHL_OHT.py 와 같다)
    except KeyboardInterrupt:
        HID.log.info("멈춤 (Ctrl+C)")
    finally:
        _stop.set()
        _flush()                                      # 모아 둔 줄 마저 보내고
        Rule_hid.stop()                               # 끝


if __name__ == "__main__":
    main()
