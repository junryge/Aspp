"""
demos_v1/report_graphs.py — 리포트 자동 그래프 (재사용 레지스트리)

설계
  - 렌더러 하나당 (detect, render) 한 쌍.  detect(headers)=이 CSV에 이 그래프가 맞나,
    render(rows)=인라인 SVG 문자열.  리포트 종류가 늘면 RENDERERS 에 추가만 하면 된다.
  - build_report_graph(headers, rows) → 본문에 그대로 붙일 마크다운 블록(<div>…SVG…</div>)
    또는 None.  LLM 이 못 그리는 그래프를 백엔드가 만들어 응답 끝에 자동 삽입한다.
  - 기본은 인라인 SVG(의존성 0, 브라우저·다운로드 HTML 에서 렌더).  서버에 headless
    크로미움이 있고 DEMOS_GRAPH_PNG=1 이면 PNG(base64)로 자동 전환(svg_to_png_data_uri).

발동이벤트 24h 그래프(hub_evt_24h)
  ① 종합 위험점수 24h + 등급 배경 + 최고점
  ② 최고점 reason 이 '실제로 발동시킨' 컬럼만 골라 지표별 패널 — 실제 raw 컬럼명 표기
     (raw 컬럼 매핑은 hubroom_predictor_V2.py 원본 그대로)
"""
from __future__ import annotations
import os
import re
from datetime import datetime, timedelta

# ── 실제 raw 컬럼 매핑 (hubroom_predictor_V2.py 원본) ──────────────────────
_RA_RAW = {
    'M16HUB': 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN',
    'M14':    'M14.QUE.LOAD.AVGLOADTIME1MIN',
    'M14B':   'M14B.QUE.TIME.AVGTOTALTIME1MIN',
    'M16A':   'M16A.QUE.LOAD.AVGLOADTIME1MIN',
    'M16B':   'M16B.QUE.LOAD.AVGLOADTIME1MIN',
}
_OHT_RAW = {a: f'{a}.QUE.OHT.OHTUTIL' for a in ('M16HUB', 'M14', 'M14B', 'M16A', 'M16B')}
_SLA_RAW = {a: f'{a}.QUE.ALL.TRANSPORT4MINOVERRATIO' for a in ('M16HUB', 'M14', 'M16A', 'M16B')}
_SORTER_RAW = {a: f'{a}.SORTER.ABN.SORTERWAITCOUNTOVER'
               for a in ('M14', 'M14B', 'M16A', 'M16B', 'M16HUB')}
_FAB_RAW = 'M16HUB.STRATE.ALL.FABSTORAGERATIO'
_STB_RAW = 'M16HUB.STRATE.STB.3F_STORAGE_UTIL'
_REV_RAW = 'M16HUB.QUE.LFT.3F_LFT_REVERSALCNT'

# 등급 배경 밴드 (1~100 척도 — 경계54/위험75/초위험90, 54 미만은 알람 없음(회색))
_BANDS = [
    (0, 54, "#f1f5f9"),      # 알람 없음 (등급 미표기)
    (54, 75, "#ffedd5"),     # 🟠 경계
    (75, 90, "#fee2e2"),     # 🔴 위험
    (90, 100, "#fecaca"),    # ⛔ 초위험
]
_SCORE_COLOR = "#2563eb"
_KIND_COLOR = {
    "반송시간": "#dc2626", "FAB저장율": "#ea580c", "STB저장율": "#d97706",
    "OHT가동률": "#7c3aed", "4분초과율": "#0891b2", "소터대기": "#16a34a",
    "리프터막힘": "#be185d",
}
_PALETTE = ["#dc2626", "#ea580c", "#d97706", "#16a34a", "#0891b2",
            "#2563eb", "#7c3aed", "#be185d", "#0d9488", "#b45309"]


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _metric_color(label, idx):
    for kind, col in _KIND_COLOR.items():
        if kind in label:
            return col
    return _PALETTE[idx % len(_PALETTE)]


def _parse_dt(s):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime((s or "")[:19], fmt)
        except ValueError:
            pass
    return None


def parse_reason_metrics(reason):
    """peak reason → [{col, raw, label, unit}] (영역별, reason 등장 순서, 중복 제거).
    M16_PKT·M16_WT 는 제외(사용자 요청 — 발동이벤트 분석/그래프에서 불필요)."""
    out, seen = [], set()
    _EXCLUDE = ("M16_PKT", "M16_WT")

    def add(col, raw, label, unit):
        if col and col not in seen and not any(x in col or x in raw or x in label for x in _EXCLUDE):
            seen.add(col)
            out.append({"col": col, "raw": raw, "label": label, "unit": unit})

    body = (reason or "").split("발동:", 1)[-1]
    body = re.split(r"흐름:|운영자조치:", body)[0]
    for m in re.finditer(r"(M16HUB|M14B|M16A|M16B|M14)\s*\[(.*?)\]", body):
        area, inner = m.group(1), m.group(2)
        if "AVGTOTALTIME1MIN" in inner or "AVGLOADTIME1MIN" in inner:
            add(f"{area}_ra", _RA_RAW.get(area, f"{area}.QUE.TIME.AVGTOTALTIME1MIN"),
                f"{area} 반송시간", "분")
        if "FAB저장" in inner:
            add("M16HUB_rd_fab", _FAB_RAW, "M16HUB FAB저장율", "%")
        if re.search(r"\bSTB", inner):
            add("M16HUB_stb_util", _STB_RAW, "M16HUB STB저장율", "%")
        if "OHT=" in inner or "OHT가동" in inner:
            add(f"{area}_rd_oht", _OHT_RAW.get(area, f"{area}.QUE.OHT.OHTUTIL"),
                f"{area} OHT가동률", "%")
        if "R-C" in inner:
            add("M16HUB_rev_count", _REV_RAW, "M16HUB 리프터막힘", "회")
        if "SLA(" in inner or "4분초과" in inner:
            add(f"sla_{area}", _SLA_RAW.get(area, f"{area}.QUE.ALL.TRANSPORT4MINOVERRATIO"),
                f"{area} 4분초과율", "%")
        if "SORT(" in inner or "소터" in inner:
            add(f"sorter_{area}", _SORTER_RAW.get(area, f"{area}.SORTER.ABN.SORTERWAITCOUNTOVER"),
                f"{area} 소터대기", "건")
    return out


def _downsample(pts, step_min=2):
    """좌표 수 절감(localStorage·바이트 절약): step_min 간격 + 마지막점 보존."""
    if len(pts) <= 2:
        return pts
    out, last_key = [], None
    for t, v in pts:
        key = int(t.timestamp() // (step_min * 60))
        if key != last_key:
            out.append((t, v))
            last_key = key
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


_LEVEL_HL = {"경계": "#ea580c", "위험": "#dc2626", "초위험": "#7c3aed", "정상": "#3b82f6"}


def render_hub_evt_24h(rows, title=None, width=900, incidents=None):
    """발동이벤트 rows → (점수 패널 + 발동지표 스택 패널) 인라인 SVG.
    incidents: [{start,end,idx,peak_score,level,hot}] 주면 점수 패널에 사건 구간을 음영+라벨로 표시."""
    rows = [{(k or "").lstrip("﻿"): v for k, v in r.items()} for r in rows]
    data = []
    for r in rows:
        t = _parse_dt(r.get("datetime") or "")
        if t is None:
            continue
        try:
            sc = float(r.get("unified_risk_score") or 0)
        except (TypeError, ValueError):
            sc = 0.0
        data.append((t, sc, r))
    if len(data) < 2:
        return None
    data.sort(key=lambda x: x[0])
    t0, t1 = data[0][0], data[-1][0]
    span = (t1 - t0).total_seconds() or 1.0

    peak_t, peak_v, peak_r = max(data, key=lambda x: x[1])
    if peak_v <= 0:
        return None
    metrics_def = parse_reason_metrics(peak_r.get("reason") or "")

    series = []
    for md in metrics_def:
        pts = []
        for t, _sc, r in data:
            v = (r.get(md["col"]) or "").strip()
            if not v:
                continue
            try:
                pts.append((t, float(v)))
            except ValueError:
                pass
        if pts and any(v != 0 for _, v in pts):
            series.append({**md, "pts": pts})

    if title is None:
        day = (rows[0].get("date") or (rows[0].get("datetime") or "")[:10])
        title = f"📅 {day} M16 BR 발동이벤트 24시간 종합"

    L, R = 200, 18                 # 좌측 거터 넓힘 — raw 컬럼명 잘 보이게
    pw = width - L - R
    TITLE_H = 30
    H_SCORE = 150                  # ① 종합 위험점수 패널 높이
    PH, PGAP = 72, 10              # 지표 패널(라벨 크게)
    top = TITLE_H + 8              # 점수 패널 top
    y = top + H_SCORE + 26         # 지표 패널 시작
    height = int(y + (len(series) * (PH + PGAP) if series else 0) + 22)

    def X(t):
        return round(L + pw * ((t - t0).total_seconds() / span))

    px = X(peak_t)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'font-family="-apple-system,\'Malgun Gothic\',sans-serif" '
           f'style="max-width:100%;height:auto;background:#fff;border:1px solid #e2e8f0;border-radius:10px;">']
    out.append(f'<text x="14" y="20" font-size="15" font-weight="800" fill="#16213e">{_esc(title)}</text>')

    _span_h = span / 3600.0      # 윈도우 길이에 맞춰 눈금 간격 적응(짧은 사건줌에서도 보이게)
    _tick_min = 15 if _span_h <= 2 else (30 if _span_h <= 3.5 else (60 if _span_h <= 7 else 120))

    def xticks(ptop, h, labels=False):
        base = t0.replace(minute=0, second=0, microsecond=0)
        k = 0
        while True:
            cur = base + timedelta(minutes=k * _tick_min)
            if cur > t1:
                break
            if cur >= t0:
                x = X(cur)
                out.append(f'<line x1="{x}" y1="{ptop}" x2="{x}" y2="{ptop+h}" stroke="#eef2f7" stroke-width=".7"/>')
                if labels:
                    _lbl = cur.strftime("%H:%M") if _tick_min < 60 else cur.strftime("%H시")
                    out.append(f'<text x="{x}" y="{ptop+h+14}" font-size="9" fill="#64748b" text-anchor="middle">{_lbl}</text>')
            k += 1

    # ① 종합 위험점수 패널 (1~100 척도, 등급 배경)
    ymax = 100.0

    def Y1(v):
        return round(top + H_SCORE * (1 - min(v, ymax) / ymax))

    for lo, hi, color in _BANDS:
        if lo >= ymax:
            continue
        yh, yl = Y1(min(hi, ymax)), Y1(lo)
        out.append(f'<rect x="{L}" y="{yh}" width="{pw}" height="{yl-yh}" fill="{color}"/>')
    for v in (54, 75, 90, 100):
        yy = Y1(v)
        out.append(f'<line x1="{L}" y1="{yy}" x2="{L+pw}" y2="{yy}" stroke="#e2e8f0" stroke-width=".5" stroke-dasharray="2 3"/>')
        out.append(f'<text x="{L-6}" y="{yy+3}" font-size="9" fill="#94a3b8" text-anchor="end">{v}</text>')
    xticks(top, H_SCORE)
    # 사건 구간 음영 + 라벨(여러 사건을 한 그래프에) — 점수선보다 먼저 그려 배경이 되게
    for _inc in (incidents or []):
        try:
            _ix0, _ix1 = X(_inc["start"]), X(_inc["end"])
        except Exception:
            continue
        _ix0 = max(L, min(_ix0, L + pw)); _ix1 = max(L, min(_ix1, L + pw))
        if _ix1 - _ix0 < 2:
            _ix1 = _ix0 + 2
        _hc = _LEVEL_HL.get(_inc.get("level"), "#3b82f6")
        out.append(f'<rect x="{_ix0}" y="{top}" width="{_ix1-_ix0}" height="{H_SCORE}" fill="{_hc}" opacity="0.10"/>')
        out.append(f'<line x1="{_ix0}" y1="{top}" x2="{_ix0}" y2="{top+H_SCORE}" stroke="{_hc}" stroke-width="1" stroke-dasharray="3 2" opacity="0.65"/>')
        out.append(f'<line x1="{_ix1}" y1="{top}" x2="{_ix1}" y2="{top+H_SCORE}" stroke="{_hc}" stroke-width="1" stroke-dasharray="3 2" opacity="0.65"/>')
        _lbl = f'사건{_inc.get("idx","")} · {int(_inc.get("peak_score") or 0)}점'
        out.append(f'<text x="{(_ix0+_ix1)/2:.0f}" y="{top+12}" font-size="10" font-weight="800" fill="{_hc}" '
                   f'text-anchor="middle" style="paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round">{_esc(_lbl)}</text>')
    sd = _downsample([(t, s) for t, s, _ in data])
    d = "M" + " L".join(f"{X(t)},{Y1(s)}" for t, s in sd)
    out.append(f'<path d="{d}" fill="none" stroke="{_SCORE_COLOR}" stroke-width="1.7"/>')
    out.append(f'<line x1="{px}" y1="{top}" x2="{px}" y2="{top+H_SCORE}" stroke="#dc2626" stroke-width="1" stroke-dasharray="4 3"/>')
    out.append(f'<circle cx="{px}" cy="{Y1(peak_v)}" r="4" fill="#dc2626"/>')
    _la = "end" if px > L + pw * 0.7 else "start"
    _lx = px - 7 if _la == "end" else px + 7
    out.append(f'<text x="{_lx}" y="{Y1(peak_v)-7}" font-size="11" font-weight="800" fill="#dc2626" '
               f'text-anchor="{_la}" style="paint-order:stroke;stroke:#fff;stroke-width:3.2px;stroke-linejoin:round">'
               f'▲ 최고 {int(peak_v)}점 · {peak_t.strftime("%H:%M")} · {_esc(peak_r.get("hot_area") or "")}</text>')

    # ② 지표 스택 패널
    if series:
        out.append(f'<text x="14" y="{y-6}" font-size="11.5" font-weight="700" fill="#334155">'
                   f'최고점({peak_t.strftime("%H:%M")} · {int(peak_v)}점 · {_esc(peak_r.get("hot_area") or "")}) 발동 지표 — 실제 raw 컬럼 24시간 추이</text>')

    # ② 지표 스택 패널
    for i, s in enumerate(series):
        ptop = round(y + i * (PH + PGAP))
        pc = _metric_color(s["label"], i)
        pts = s["pts"]
        vmax = max(v for _, v in pts)
        vmin = min(v for _, v in pts)
        rng = (vmax - vmin) or 1.0
        last = i == len(series) - 1

        def Yp(v, _ptop=ptop):
            return round(_ptop + (PH - 12) * (1 - (v - vmin) / rng) + 4)

        out.append(f'<rect x="{L}" y="{ptop}" width="{pw}" height="{PH}" fill="#fcfdfe" stroke="#eef2f7"/>')
        out.append(f'<rect x="0" y="{ptop}" width="{L}" height="{PH}" fill="#fff"/>')
        out.append(f'<rect x="{L-4}" y="{ptop}" width="3.5" height="{PH}" fill="{pc}"/>')
        xticks(ptop, PH, labels=last)
        out.append(f'<text x="8" y="{ptop+18}" font-size="12" font-weight="800" fill="{pc}">{_esc(s["label"])} ({_esc(s["unit"])})</text>')
        out.append(f'<text x="8" y="{ptop+35}" font-size="9.5" font-weight="600" fill="#1e293b" font-family="ui-monospace,Consolas,monospace">{_esc(s["raw"])}</text>')
        out.append(f'<text x="8" y="{ptop+50}" font-size="9" fill="#94a3b8">범위 {vmin:g}~{vmax:g}{_esc(s["unit"])}</text>')
        dpts = _downsample(pts)
        dd = "M" + " L".join(f"{X(t)},{Yp(v)}" for t, v in dpts)
        out.append(f'<path d="{dd}" fill="none" stroke="{pc}" stroke-width="1.6"/>')
        area_d = dd + f" L{X(dpts[-1][0])},{ptop+PH-2} L{X(dpts[0][0])},{ptop+PH-2} Z"
        out.append(f'<path d="{area_d}" fill="{pc}" opacity="0.07"/>')
        out.append(f'<line x1="{px}" y1="{ptop}" x2="{px}" y2="{ptop+PH}" stroke="#dc2626" stroke-width="1" stroke-dasharray="4 3"/>')
        pv = next((v for t, v in pts if t == peak_t), None)
        if pv is not None:
            out.append(f'<circle cx="{px}" cy="{Yp(pv)}" r="3" fill="#dc2626"/>')
            pa = "end" if px > L + pw * 0.7 else "start"
            pxx = px - 6 if pa == "end" else px + 6
            out.append(f'<text x="{pxx}" y="{Yp(pv)-5}" font-size="9.5" font-weight="800" fill="#dc2626" '
                       f'text-anchor="{pa}" style="paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round">{pv:g}{_esc(s["unit"])}</text>')

    out.append('</svg>')
    return "".join(out)


# ══ 사건단위 타임라인 그래프 ════════════════════════════════════════════
_LEVEL_COLOR = {"관심": "#16a34a", "주의": "#ca8a04", "경계": "#ea580c",
                "위험": "#dc2626", "발동": "#991b1b"}
_LEVEL_EMOJI = {"관심": "🟢", "주의": "🟡", "경계": "🟠", "위험": "🔴", "발동": "⛔"}
_LEVEL_ORDER = {"정상": 0, "관심": 1, "주의": 2, "경계": 3, "위험": 4, "발동": 5}


def _hhmm_to_min(s):
    try:
        h, m = str(s).split(":")[:2]
        return int(h) * 60 + int(m)
    except Exception:
        return None


def _incident_cause(relation, risk_factors=""):
    txt = (relation or risk_factors or "").strip()
    parts = [p.strip() for p in txt.split("|") if p.strip()]
    return parts[0] if parts else ""


def render_incident_timeline(rows, title=None, width=900):
    """사건단위 rows → 하루 위험사건 타임라인 SVG (사건당 가로 막대)."""
    rows = [{(k or "").lstrip("﻿"): v for k, v in r.items()} for r in rows]
    incs = []
    day = ""
    for r in rows:
        s = _hhmm_to_min(r.get("start_time"))
        e = _hhmm_to_min(r.get("end_time"))
        if s is None or e is None:
            continue
        if e < s:
            e += 1440   # 자정 넘김
        p = _hhmm_to_min(r.get("predict_time"))
        if p is not None and p > s:
            p -= 1440
        try:
            score = int(float(r.get("max_risk_score") or 0))
        except (TypeError, ValueError):
            score = 0
        day = day or (r.get("date") or "")
        incs.append({
            "s": s, "e": e, "p": p, "score": score,
            "level": (r.get("max_risk_level") or "").strip(),
            "hot": (r.get("hot_area") or "").strip(),
            "dur": r.get("duration_min") or "",
            "cause": _incident_cause(r.get("relation"), r.get("risk_factors")),
        })
    if not incs:
        return None
    incs.sort(key=lambda x: (-_LEVEL_ORDER.get(x["level"], 0), x["s"]))

    if title is None:
        title = f"📅 {day} M16 HUBROOM 위험사건 타임라인 ({len(incs)}건)"

    t_min = min((i["p"] if i["p"] is not None else i["s"]) for i in incs)
    t_max = max(i["e"] for i in incs)
    t_min = (t_min // 60) * 60
    t_max = ((t_max + 59) // 60) * 60
    span = max(1, t_max - t_min)

    L, R = 96, 18
    pw = width - L - R
    TITLE_H = 34
    ROW_H, ROW_GAP = 30, 8
    top = TITLE_H + 6
    height = int(top + len(incs) * (ROW_H + ROW_GAP) + 30)

    def X(mn):
        return round(L + pw * ((mn - t_min) / span))

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'font-family="-apple-system,\'Malgun Gothic\',sans-serif" '
           f'style="max-width:100%;height:auto;background:#fff;border:1px solid #e2e8f0;border-radius:10px;">']
    out.append(f'<text x="{L}" y="22" font-size="15" font-weight="800" fill="#16213e">{_esc(title)}</text>')

    # 시간축 (2시간 간격 세로선 + HH시)
    hr = t_min
    while hr <= t_max:
        x = X(hr)
        out.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{height-22}" stroke="#eef2f7" stroke-width=".8"/>')
        out.append(f'<text x="{x}" y="{height-8}" font-size="9" fill="#64748b" text-anchor="middle">{(hr//60)%24:02d}시</text>')
        hr += 120

    for idx, it in enumerate(incs):
        ry = top + idx * (ROW_H + ROW_GAP)
        color = _LEVEL_COLOR.get(it["level"], "#64748b")
        x0, x1 = X(it["s"]), X(it["e"])
        bw = max(3, x1 - x0)
        # 좌측 라벨: 등급 이모지 + 진원지
        emo = _LEVEL_EMOJI.get(it["level"], "·")
        out.append(f'<text x="6" y="{ry+ROW_H*0.62:.0f}" font-size="11" font-weight="700" fill="{color}">{emo} {_esc(it["hot"] or "-")}</text>')
        # 예측→시작 선행구간 (점선)
        if it["p"] is not None and it["p"] < it["s"]:
            xp = X(it["p"])
            out.append(f'<line x1="{xp}" y1="{ry+ROW_H/2:.0f}" x2="{x0}" y2="{ry+ROW_H/2:.0f}" stroke="{color}" stroke-width="1" stroke-dasharray="3 3" opacity=".6"/>')
            out.append(f'<circle cx="{xp}" cy="{ry+ROW_H/2:.0f}" r="2.5" fill="{color}" opacity=".6"/>')
        # 사건 막대
        out.append(f'<rect x="{x0}" y="{ry+5}" width="{bw}" height="{ROW_H-10}" rx="4" fill="{color}" opacity="0.88"/>')
        # 막대 안/옆 라벨: 점수 + 지속(분) + 원인 (막대가 좁으면 오른쪽에)
        sh, sm = divmod(it["s"], 60)
        eh, em = divmod(it["e"] % 1440, 60)
        try:
            dmin = int(float(it["dur"]))
        except (TypeError, ValueError):
            dmin = it["e"] - it["s"]
        info = f'{it["level"]} {it["score"]}점 · {sh%24:02d}:{sm:02d}~{eh:02d}:{em:02d} · {dmin}분'
        if bw > 230:
            out.append(f'<text x="{x0+8}" y="{ry+ROW_H*0.62:.0f}" font-size="9.5" font-weight="700" fill="#fff" style="paint-order:stroke;stroke:{color};stroke-width:2px;">{_esc(info)}</text>')
        else:
            out.append(f'<text x="{x1+6}" y="{ry+ROW_H*0.62:.0f}" font-size="9.5" font-weight="700" fill="{color}">{_esc(info)}</text>')
        # 원인 한 줄 (막대 아래 회색 작게)
        if it["cause"]:
            cz = it["cause"]
            cz = cz[:78] + "…" if len(cz) > 80 else cz
            out.append(f'<text x="{L}" y="{ry+ROW_H-1:.0f}" font-size="7.5" fill="#94a3b8">{_esc(cz)}</text>')

    out.append('</svg>')
    return "".join(out)


# ── 렌더러 레지스트리 ───────────────────────────────────────────────────
def _detect_hub_evt(headers):
    h = set(headers or [])
    return ("unified_risk_score" in h) and ("reason" in h) and ("datetime" in h)


def _detect_incident(headers):
    h = set(headers or [])
    return ("start_time" in h) and ("end_time" in h) and ("max_risk_level" in h)


RENDERERS = [
    {"name": "hub_evt_24h", "detect": _detect_hub_evt, "render": render_hub_evt_24h},
    {"name": "incident_timeline", "detect": _detect_incident, "render": render_incident_timeline},
]


def svg_to_png_data_uri(svg, width=900):
    """headless 크로미움이 있으면 SVG→PNG(base64 data URI). 없으면 None."""
    import base64
    import shutil
    import subprocess
    import tempfile
    cand = [os.environ.get("DEMOS_GRAPH_CHROMIUM"),
            "/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell",
            shutil.which("chromium"), shutil.which("chromium-browser"),
            shutil.which("google-chrome"), shutil.which("headless_shell")]
    binp = next((c for c in cand if c and os.path.exists(c)), None)
    if not binp:
        return None
    m = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    w, h = (int(m.group(1)), int(m.group(2))) if m else (width, 1000)
    try:
        with tempfile.TemporaryDirectory() as td:
            hp = os.path.join(td, "g.html")
            pp = os.path.join(td, "g.png")
            with open(hp, "w", encoding="utf-8") as f:
                f.write(f'<!doctype html><meta charset="utf-8"><body style="margin:0;background:#fff">{svg}</body>')
            subprocess.run([binp, "--headless", "--no-sandbox", "--disable-gpu",
                            "--force-device-scale-factor=2", "--hide-scrollbars",
                            f"--window-size={w},{h}", f"--screenshot={pp}", f"file://{hp}"],
                           capture_output=True, timeout=30)
            if os.path.exists(pp):
                b = base64.b64encode(open(pp, "rb").read()).decode("ascii")
                return f"data:image/png;base64,{b}"
    except Exception:
        return None
    return None


def _level_from_score(s):
    try:
        s = float(s)
    except (TypeError, ValueError):
        return "정상"
    return "초위험" if s >= 90 else ("위험" if s >= 75 else ("경계" if s >= 54 else "정상"))


def derive_incidents_from_evt(rows, gap_min=10, min_score=54):
    """발동이벤트 rows → 사건 목록 (예측기 로직 그대로: stage=3 확정 시작,
    gap_min 분 동안 무s3면 종료, 사건 내 최고점수 ≥ min_score 만 기록)."""
    data = []
    for r in rows:
        r = {(k or "").lstrip("﻿"): v for k, v in r.items()}
        t = _parse_dt(r.get("datetime") or "")
        if t is None:
            continue
        try:
            sc = float(r.get("unified_risk_score") or 0)
        except (TypeError, ValueError):
            sc = 0.0
        data.append((t, sc, str(r.get("stage") or "").strip(), r))
    data.sort(key=lambda x: x[0])
    incs, cur = [], None
    for t, sc, st, r in data:
        s3 = (st == "3")
        if cur is None:
            if s3:
                cur = {"start": t, "end": t, "last_s3": t, "peak_score": sc, "peak_t": t, "peak_row": r}
        elif s3:
            cur["last_s3"] = t
            cur["end"] = t
            if sc > cur["peak_score"]:
                cur["peak_score"], cur["peak_t"], cur["peak_row"] = sc, t, r
        elif (t - cur["last_s3"]).total_seconds() / 60.0 >= gap_min:
            incs.append(cur)
            cur = None
    if cur:
        incs.append(cur)
    return [i for i in incs if i["peak_score"] >= min_score]


def render_incident_zoom(all_data, inc, idx, total, width=900):
    """한 사건의 ±60분 윈도우 그래프 (점수 + 발동지표). all_data = 정렬된 [(t,row),...]."""
    w0 = inc["start"] - timedelta(minutes=60)
    w1 = inc["end"] + timedelta(minutes=60)
    win = [r for (t, r) in all_data if w0 <= t <= w1]
    if len(win) < 2:
        return None
    lvl = _level_from_score(inc["peak_score"])
    emo = _LEVEL_EMOJI.get(lvl, "")
    hot = (inc["peak_row"].get("hot_area") or "")
    title = (f"{emo} 사건 {idx}/{total} · {inc['start'].strftime('%H:%M')}~{inc['end'].strftime('%H:%M')}"
             f" · 최고 {int(inc['peak_score'])}점 · {_esc(hot)}")
    return render_hub_evt_24h(win, title=title, width=width)


def _is_incident_query(query):
    """질문이 '사건(사건발생/사건단위) 모드'인가. 발동이벤트는 제외(그래프 그대로 줘야 함)."""
    q = str(query or "")
    if "발동이벤트" in q:
        return False
    return any(k in q for k in (
        "사건단위", "사건 단위", "위험사건", "사건별", "사건 분석", "사건분석",
        "사건발생", "사건 발생", "사건",
    ))


def build_report_graph(headers, rows, query=None, prefer_png=None):
    """헤더+질문으로 렌더러 선택 → 본문에 붙일 마크다운 블록(<div>…</div>) 또는 None.
    질문에 '사건단위/위험사건' 키워드 + 발동이벤트 CSV → 사건별 ±60분 줌(여러 개).
    그 외 → 스키마 자동 감지(발동이벤트=24h 종합, 사건단위=타임라인)."""
    if not headers or not rows:
        return None
    q = str(query or "")
    want_incident = _is_incident_query(q)
    svgs = []
    if want_incident and _detect_hub_evt(headers):
        rows2 = [{(k or "").lstrip("﻿"): v for k, v in r.items()} for r in rows]
        all_data = sorted(((_parse_dt(r.get("datetime") or ""), r) for r in rows2
                           if _parse_dt(r.get("datetime") or "")), key=lambda x: x[0])
        incs = derive_incidents_from_evt(rows)
        incs = sorted(incs, key=lambda i: -i["peak_score"])[:8]   # 점수 상위 8건만
        incs = sorted(incs, key=lambda i: i["start"])            # 표시는 시간순
        if incs:
            # ★ 사건들을 한 그래프에 — 모든 사건을 감싸는 윈도우 1개 + 사건 구간 음영/라벨
            w0 = min(i["start"] for i in incs) - timedelta(minutes=60)
            w1 = max(i["end"] for i in incs) + timedelta(minutes=60)
            win = [r for (t, r) in all_data if w0 <= t <= w1]
            inc_meta = [{"start": i["start"], "end": i["end"], "idx": n,
                         "peak_score": int(i["peak_score"]),
                         "level": _level_from_score(i["peak_score"]),
                         "hot": (i["peak_row"].get("hot_area") or "")}
                        for n, i in enumerate(incs, 1)]
            day0 = (win[0].get("date") if win else "") or ""
            ttl = (f"📅 {day0} M16 BR 사건 통합 ({len(incs)}건) · "
                   f"{incs[0]['start'].strftime('%H:%M')}~{incs[-1]['end'].strftime('%H:%M')}")
            s = render_hub_evt_24h(win, title=ttl, incidents=inc_meta)
            if s:
                svgs.append(s)
        if not svgs:
            return None
    else:
        renderer = next((r for r in RENDERERS if r["detect"](headers)), None)
        if not renderer:
            return None
        try:
            s = renderer["render"](rows)
        except Exception:
            return None
        if not s:
            return None
        svgs = [s]
    if prefer_png is None:
        prefer_png = os.environ.get("DEMOS_GRAPH_PNG", "") in ("1", "true", "yes")
    svg = "".join(f'<div style="margin:12px 0;">{s}</div>' for s in svgs) if len(svgs) > 1 else svgs[0]
    inner = svg
    if prefer_png:
        uri = svg_to_png_data_uri(svg)
        if uri:
            inner = f'<img src="{uri}" alt="리포트 그래프" style="max-width:100%;border-radius:10px;"/>'
    # 블록 레벨 <div> 로 감싸 python-markdown/marked 통과 보장
    return f'\n\n<div class="hub-report-graph" style="margin:14px 0;">{inner}</div>\n\n'


class GraphStreamInjector:
    """SSE 스트림에 리포트 그래프를 '제목(# …) 바로 밑, 본문 위'로 1회 삽입.

    routes_chat 의 gen_api 침투를 최소화하기 위한 어댑터.  코어 파일은
        inj = GraphStreamInjector(_data_for_script)
        ... for p in inj.feed(vis): yield _tok(p)
        ... for p in inj.flush():   yield _tok(p)
    처럼 feed()/flush() 만 호출하면 된다(버퍼링·정규식·그래프 생성은 전부 여기).

    feed/flush 는 'SSE로 감싸기 전의 텍스트 조각'을 yield 한다(코어가 _tok 으로 감쌈).
    """

    def __init__(self, data_for_script, prefer_png=None, head_limit=4000, query=None):
        self.src = data_for_script if (data_for_script and data_for_script.get("rows")) else None
        self.prefer_png = prefer_png
        self.head_limit = head_limit
        self.query = query or ""   # 질문 키워드로 그래프 종류 라우팅(사건단위 vs 발동이벤트)
        self.buf = ""
        self._block = None
        self._built = False
        # ★ 강한 스코프: '등록 렌더러가 인식하는 스키마(발동이벤트 등)'의 CSV 일 때만 활성.
        #   그 외(다른 개인에이전트·일반 데모스 채팅·다른 CSV)는 done=True → 토큰을 그대로
        #   통과만 시킨다(버퍼링조차 안 함). 즉 발동이벤트 분석이 아니면 완전 무영향.
        self.done = True
        # 우선순위 1: 등록 스크립트가 직접 출력한 SVG(사용자가 100% 통제).
        self.script_svg = (self.src or {}).get("_script_svg") if self.src else None
        if self.script_svg:
            self.done = False
        # 우선순위 2: 내장 렌더러가 인식하는 스키마(발동이벤트 24h 자동생성).
        elif self.src:
            try:
                hdr = [str(h).lstrip("﻿") for h in (self.src.get("headers") or [])]
                if any(r["detect"](hdr) for r in RENDERERS):
                    self.done = False
            except Exception:
                self.done = True

        # ★ 사건(사건발생) 모드인데 사건이 0건(정상)이면 그래프를 아예 주지 않는다 — 텍스트만.
        #   스크립트 SVG·내장 그래프 둘 다 차단. (발동이벤트 모드는 _is_incident_query=False 라 무영향.)
        if not self.done and self.src and _is_incident_query(self.query):
            try:
                hdr = [str(h).lstrip("﻿") for h in (self.src.get("headers") or [])]
                if _detect_hub_evt(hdr):
                    dict_rows = [dict(zip(hdr, r)) for r in (self.src.get("rows") or [])]
                    if not derive_incidents_from_evt(dict_rows):
                        self.done = True          # 토큰 그대로 통과 — 그래프 미삽입
                        self.script_svg = None     # 스크립트가 준 SVG 도 버림
                        print("  📈 [GRAPH] 사건 0건(정상) → 그래프 생략, 텍스트만")
            except Exception:
                pass

    def _block_md(self):
        if self._built:
            return self._block
        self._built = True
        # 스크립트가 준 SVG 가 있으면 그대로 끼운다(내장 렌더러보다 우선).
        if self.script_svg:
            self._block = (f'\n\n<div class="hub-report-graph" style="margin:14px 0;">'
                           f'{self.script_svg}</div>\n\n')
            print(f"  📈 [GRAPH] 스크립트 SVG 삽입 ({len(self._block)}자)")
            return self._block
        try:
            hdr = [str(h).lstrip("﻿") for h in (self.src.get("headers") or [])]
            rows = self.src.get("rows") or []
            dict_rows = [dict(zip(hdr, r)) for r in rows]
            self._block = build_report_graph(hdr, dict_rows, query=self.query, prefer_png=self.prefer_png)
        except Exception as e:
            print(f"  📈 [GRAPH] 생성 예외: {e}")
            self._block = None
        if self._block:
            print(f"  📈 [GRAPH] 리포트 그래프 삽입 ({len(self._block)}자)")
        return self._block

    def feed(self, vis):
        """가시 토큰 1조각 → 내보낼 텍스트 조각들(제목 직후 그래프 포함)을 yield."""
        if self.done:
            if vis:
                yield vis
            return
        self.buf += vis
        m = re.search(r'(?m)^#\s+.+?\n', self.buf)
        if m:
            head, rest = self.buf[:m.end()], self.buf[m.end():]
            self.done = True
            self.buf = ""
            if head:
                yield head
            blk = self._block_md()
            if blk:
                yield blk
            if rest:
                yield rest
        elif len(self.buf) > self.head_limit:
            # 제목(# …)을 못 만남 → 그래프 먼저, 그다음 본문(폴백)
            self.done = True
            blk = self._block_md()
            if blk:
                yield blk
            yield self.buf
            self.buf = ""

    def flush(self):
        """스트림 종료 시: 제목을 못 만났으면 그래프+남은 버퍼를 마저 흘린다."""
        if self.done:
            return
        self.done = True
        blk = self._block_md()
        if blk:
            yield blk
        if self.buf:
            yield self.buf
            self.buf = ""

