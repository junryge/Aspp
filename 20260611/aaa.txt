#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
운영로그 파서 v2 — Episode 단위 장애 라벨링

v1 (운영로그_파서.py, 메시지 단위) 의 한계를 보완:
  - 메시지 4개(요청→조치→복구→원복) 가 v1 에선 4건 장애로 카운트
  - v2 는 같은 장비 + 30분 window 로 Episode 1건으로 묶음

핵심 차이:
  v1: 1메시지 = 1장애 (other 제외 약 150건/월 과대 집계)
  v2: 1Episode = 1장애 (메시지 N개 묶음, 약 19건/월 실제)

알고리즘 (6/12 검토결과 + 실데이터 검증 기반):
  1. 헤더 기준 메시지 분리      (v1 재사용)
  2. 전처리                       — 멘션/URL/이미지 제거
  3. 카테고리 분류 6종           — 복구 키워드 우선 (Case 4)
  4. 장애유형 7종 분류           — fault 메시지만
  5. 장비명/라인 추출
  6. Episode 클러스터링         — find_open_episode (score-based)
     · 같은 장비 + 30분 window → 같은 Episode
     · recovery/rollback → Episode 종료
     · "발생" 키워드 시 fault 우선 (Case 5)
  7. Episode 라벨링              — start/end/장비/유형/지속/severity/상태
  8. Output: 메시지 CSV + Episode CSV

사용법:
    python3 운영로그_파서_v2.py <운영로그.txt>
    python3 운영로그_파서_v2.py <운영로그.txt> --out ./output

출력:
    YYYYMMDD_HHMMSS_message.csv  — 메시지 단위 (v1 호환)
    YYYYMMDD_HHMMSS_episode.csv  — Episode 단위 (v2 신규)
    YYYYMMDD_HHMMSS_summary.json — 통계 요약
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, List, Set


# ============================================================
# 설정 (6/12 검토결과 기반)
# ============================================================

SESSION_WINDOW_MIN = 30      # Episode 묶기 시간 윈도우
REOPEN_WINDOW_MIN = 15       # 복구 후 재발생 분리
SCORE_THRESHOLD = 6          # find_open_episode 매칭 임계 (장비 일치 = 최소)


# ── 카테고리 6종 (복구 키워드 우선 — Case 4) ──────────────────
CATEGORY_RULES = [
    # (카테고리, 키워드들) — 위에서부터 검사
    ('cancel',   ['오탐', '취소', '무시', '장애 아님', '이슈 없음', '정상 확인']),
    ('recovery', ['조치되어', '조치 완료', '복구 완료', '정상화', '해소', 'clear',
                  'Open 하겠', 'Open 했', 'OPEN 했', 'open 했', '복구되', '처리 완료',
                  '정상 동작']),
    ('rollback', ['원복']),
    ('action',   ['Close 하겠', 'Close했', 'Close 했', 'CLOSE 하겠', '조치중',
                  '확인중', '점검중', '재기동', 'reset', '확인 예정']),
    ('fault',    ['장애', '이상', '에러', 'Error', 'ERROR', 'Err', 'error', 'fail',
                  '발생', '다운', '멈춤', '정체', '몰림', '인터락', 'alarm', 'Alarm',
                  '지연', 'Q-OVER', 'T-OVER', 'timeout', 'Timeout', 'TIMEOUT',
                  'Down', '미동작', '통신불량']),
]
# normal = 위 어디에도 매칭 안 됨


# ── Case 5: "발생" 포함 시 fault 우선 (action 보다 먼저) ─────
FAULT_PRIORITY_KEYWORDS = ['발생', 'Error 발생', 'ERROR 발생', '이상 발생', '장애 발생', 'Down 발생']


# ── 장애 유형 7종 (fault 메시지 대상) ──────────────────────
FAULT_TYPE_RULES = [
    ('정체/병목', ['정체', '몰림', '밀림', '밀려', 'Q-OVER', 'T-OVER', '지연',
                   'Storage 증가', '증가하고', 'Queue']),
    ('브릿지',    ['Bridge', 'bridge', 'BRIDGE', 'ZT', '브릿지']),
    ('리프터',    ['4ABLD', '6ABL', 'Lifter', 'lifter', 'LIFTER', '리프터', 'LIFT', 'LFT']),
    ('CNV',       ['4AFC', 'CNV', '컨베이어', 'Conveyor']),
    ('MLUD',      ['MLUD', 'mlud', 'Mlud', 'M16HUBTOM14MANUAL']),
    ('통신/에러', ['통신', 'timeout', 'Timeout', 'TIMEOUT', '인터락', 'alarm',
                   'Alarm', 'ALARM', '알람']),
]
# fault 인데 위 유형 미해당 → '기타'


# ── 정규식 ──────────────────────────────────────────────
HEADER_RE = re.compile(
    r'^(?P<author>[^,\n]+),(?P<org>[^,\n]+),(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*$'
)
# 장비번호 — 좁은 매칭 (구체적 호기명)
#   4ABLD131, 6ABL6032, 4AFC3201, OHT, ZT4-2, ZT#6, ZT 3호기, 132호기, M16E OHT
EQUIP_RE = re.compile(
    r'\b('
    r'\d[A-Z]{2,5}\d{2,4}(?:-\d+)?'      # 4ABLD131, 6ABL6032 등
    r'|OHT(?:\d+)?'                       # OHT, OHT01
    r'|ZT\s*#?\s*\d+(?:-\d+)?(?:호기)?'   # ZT4, ZT#6, ZT4-2, ZT 3호기
    r'|M16E\s*OHT'                        # M16E OHT
    r'|\d{2,3}호기'                       # 132호기, 3호기
    r')\b'
)
# 라인/구역 (장비 없을 때 fallback 키)
LINE_RE = re.compile(r'\b(M14A|M14B|M14|M16HUB|HUBROOM|Hubroom|hubroom|M16A|M16B|M16E|M16|BRIDGE|Bridge)\b')
FLOOR_RE = re.compile(r'\b(\d+F)\b')      # 3F, 6F, 7F, 10F
# CAPA 메시지 식별
CAPA_RE = re.compile(r'(CAPA|Capa|capa|MAXCAPA|MAX\s*CAPA|MaxCapa)')


# ============================================================
# 데이터 구조
# ============================================================

@dataclass
class Message:
    msg_id: int
    sender: str
    org: str
    ts: datetime
    text: str
    text_raw: str = ''             # 전처리 전 원본
    category: str = 'normal'        # fault/action/recovery/rollback/cancel/normal
    fault_type: str = ''            # 정체/병목, 리프터 등 (fault 일 때만)
    equipment: Set[str] = field(default_factory=set)
    line: Set[str] = field(default_factory=set)
    floor: Set[str] = field(default_factory=set)    # 3F, 6F, 7F, 10F
    is_capa: bool = False           # CAPA/MAXCAPA 변경 메시지

    def group_keys(self) -> List[str]:
        """Episode 묶기 key 우선순위 리스트.

        ★ 핵심 원칙: 장비번호가 있는 메시지는 EQ 키만 사용 (라인 키 흡수 방지).
                    라인 키는 장비번호가 없을 때만 fallback.

        우선순위:
          [장비 있을 때]
            1) EQ:장비 (예: EQ:4ABLD131)
          [장비 없을 때 — fallback]
            2) CAPA 메시지: CAPA:라인+층 또는 CAPA:라인
            3) 라인+층 (LN:M14B@7F)
            4) 라인만 (LN:M14B)
        """
        keys = []
        if self.equipment:
            # 장비번호가 있으면 EQ 키만 — 라인 키로 인한 흡수 방지
            for eq in sorted(self.equipment):
                keys.append(f'EQ:{eq}')
            return keys

        # 장비 없을 때만 라인 기반 fallback
        if self.is_capa:
            # CAPA 는 라인+층 우선, 그 다음 라인만
            for ln in sorted(self.line):
                for fl in sorted(self.floor):
                    keys.append(f'CAPA:{ln}@{fl}')
                keys.append(f'CAPA:{ln}')
        else:
            for ln in sorted(self.line):
                for fl in sorted(self.floor):
                    keys.append(f'LN:{ln}@{fl}')
                keys.append(f'LN:{ln}')
        return keys


@dataclass
class Episode:
    episode_id: int
    start_time: datetime
    last_time: datetime
    end_time: Optional[datetime] = None
    equipment: str = ''
    line: str = ''
    fault_type: str = ''
    severity: str = 'LOW'           # LOW(<15분) / MED(15~60) / HIGH(>60)
    status: str = 'open'             # open / closed / orphan
    message_ids: List[int] = field(default_factory=list)
    message_count: int = 0
    is_orphan: bool = False         # 선행 fault 없이 recovery 만 — Case 1

    def duration_min(self) -> float:
        end = self.end_time or self.last_time
        return (end - self.start_time).total_seconds() / 60.0

    def update_severity(self):
        d = self.duration_min()
        if d > 60: self.severity = 'HIGH'
        elif d >= 15: self.severity = 'MED'
        else: self.severity = 'LOW'


# ============================================================
# 1. 헤더 기준 메시지 파싱
# ============================================================

def strip_attachments(text: str) -> str:
    """전처리: 멘션/URL/이미지 토큰 제거."""
    text = re.sub(r'\{\{@[^}]+\}\}', '', text)         # {{@이름::ID}}
    text = re.sub(r'<<이미지>>[^\n]*', '', text)
    text = re.sub(r'<<파일>>[^\n]*', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_messages(path: str) -> List[Message]:
    """로그 파일 → Message 객체 리스트."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    messages = []
    i = 0
    msg_id = 0
    while i < len(lines):
        m = HEADER_RE.match(lines[i].rstrip())
        if not m:
            i += 1
            continue
        try:
            ts = datetime.strptime(m['ts'], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            i += 1
            continue
        author = m['author'].strip()
        org = m['org'].strip()

        body = []
        i += 1
        while i < len(lines):
            if HEADER_RE.match(lines[i].rstrip()):
                break
            body.append(lines[i].rstrip())
            i += 1
        raw = '\n'.join(body).strip()
        cleaned = strip_attachments(raw)
        if not cleaned:
            continue

        msg_id += 1
        messages.append(Message(
            msg_id=msg_id, sender=author, org=org, ts=ts,
            text=cleaned, text_raw=raw,
        ))

    messages.sort(key=lambda x: x.ts)
    return messages


# ============================================================
# 2~4. 카테고리/장애유형/장비 분류
# ============================================================

def classify_category(text: str) -> str:
    """6종 카테고리 분류. 복구 우선 + Case 5(발생 키워드 시 fault 우선)."""
    # Case 5: "발생" 포함 + fault 의도 → fault 우선
    if any(k in text for k in FAULT_PRIORITY_KEYWORDS):
        return 'fault'
    # ★ CAPA 변경 액션 — 별도 우선 처리 (normal 로 빠지지 않게)
    if CAPA_RE.search(text):
        # 원복은 기존 룰의 rollback 으로 가게 (변경 키워드보다 원복 우선 검사)
        if '원복' not in text:
            if any(k in text for k in ['변경 부탁', '변경 했', '변경했', '변경 합', '변경합',
                                        '변경 완료', '변경완료', '변경 진행', '바꾸겠', '바꿔',
                                        '1로', '50으로', '3으로', '으로 변경', '로 변경']):
                return 'action'
    for cat, keywords in CATEGORY_RULES:
        if any(k in text for k in keywords):
            return cat
    return 'normal'


def classify_fault_type(text: str) -> str:
    """fault 카테고리의 장애 유형 7종 분류."""
    for name, keywords in FAULT_TYPE_RULES:
        if any(k in text for k in keywords):
            return name
    return '기타'


def extract_equipment(text: str) -> Set[str]:
    return set(EQUIP_RE.findall(text))


def extract_line(text: str) -> Set[str]:
    return set(LINE_RE.findall(text))


def enrich_messages(messages: List[Message]) -> None:
    """카테고리/유형/장비/라인/층/CAPA 플래그 채우기."""
    for m in messages:
        m.category = classify_category(m.text)
        if m.category == 'fault':
            m.fault_type = classify_fault_type(m.text)
        m.equipment = extract_equipment(m.text)
        m.line = extract_line(m.text)
        m.floor = set(FLOOR_RE.findall(m.text))
        m.is_capa = bool(CAPA_RE.search(m.text))
        # CAPA 변경 메시지에 fault_type 부여 (rollback/action 도 분류)
        if m.is_capa and not m.fault_type:
            m.fault_type = 'CAPA변경'


# ============================================================
# 5~6. Episode 클러스터링 (find_open_episode + 상태 머신)
# ============================================================

# 자연스러운 역할 흐름 (find_open_episode score +3)
VALID_TRANSITIONS = {
    'fault':    ['action', 'recovery', 'rollback'],
    'action':   ['recovery', 'rollback', 'action'],
    'recovery': ['rollback'],
    'rollback': ['rollback'],
}


def find_open_episode(msg: Message, open_episodes: dict) -> Optional[Episode]:
    """매칭되는 open episode 찾기. score-based.
    조건: 장비 일치(+6) / 라인 일치(+3) / 역할 흐름 자연스러움(+3).
    threshold = 6 (장비 일치만 해도 매칭).
    """
    candidates = []
    for ep in open_episodes.values():
        if ep.status != 'open':
            continue
        # 시간 window 검사
        if (msg.ts - ep.last_time).total_seconds() / 60.0 > SESSION_WINDOW_MIN:
            continue
        score = 0
        if ep.equipment in msg.equipment:
            score += 6
        if ep.line and msg.line and (ep.line in msg.line):
            score += 3
        # 마지막 메시지의 역할 흐름
        # (간단화: open 상태에서 fault/action 이어지면 +3)
        if msg.category in VALID_TRANSITIONS.get('fault', []):
            score += 3
        if score >= SCORE_THRESHOLD:
            candidates.append((score, ep))

    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def cluster_episodes(messages: List[Message]) -> List[Episode]:
    """메시지 → Episode 묶기 (group_key 우선순위 매칭).

    매칭 우선순위:
      1) 장비번호 (EQ:4ABLD131)
      2) CAPA 라인 (CAPA:M14B)
      3) 라인+층 (LN:M14B@7F)
      4) 라인만 (LN:M14B)

    그 어느 키로도 open episode 매칭 안 되면 → 새 Episode (fault/action) 또는 orphan (recovery/rollback).
    """
    episodes: List[Episode] = []
    open_episodes: dict = {}     # group_key -> Episode
    ep_id = 0

    for m in messages:
        # normal/cancel 은 Episode 제외
        if m.category in ('normal', 'cancel'):
            continue

        keys = m.group_keys()
        # 어떤 키도 못 만들면 (장비/라인 다 없는 fault) → 마지막 fallback: 카테고리만
        # 이건 너무 광범위해서 묶기 위험 — 그냥 단독 Episode 처리
        if not keys:
            # 장비/라인 둘 다 없는 경우 — fault/action 이면 단독 Episode
            if m.category in ('fault', 'action'):
                ep_id += 1
                ep = Episode(
                    episode_id=ep_id, start_time=m.ts, last_time=m.ts,
                    equipment='', line='', fault_type=m.fault_type,
                    message_ids=[m.msg_id], message_count=1,
                )
                episodes.append(ep)   # 즉시 종료 (매칭 키 없으므로)
            # recovery/rollback 이면 orphan
            elif m.category in ('recovery', 'rollback'):
                ep_id += 1
                ep = Episode(
                    episode_id=ep_id, start_time=m.ts, last_time=m.ts, end_time=m.ts,
                    equipment='', line='', fault_type=m.fault_type,
                    status='orphan', is_orphan=True,
                    message_ids=[m.msg_id], message_count=1,
                )
                episodes.append(ep)
            continue

        # 우선순위 키 순서대로 open episode 매칭 시도
        matched_ep = None
        matched_key = None
        for k in keys:
            if k in open_episodes:
                cand = open_episodes[k]
                if (m.ts - cand.last_time).total_seconds() / 60.0 <= SESSION_WINDOW_MIN:
                    matched_ep = cand
                    matched_key = k
                    break

        if matched_ep is not None:
            matched_ep.message_ids.append(m.msg_id)
            matched_ep.message_count = len(matched_ep.message_ids)
            matched_ep.last_time = m.ts
            if m.fault_type and not matched_ep.fault_type:
                matched_ep.fault_type = m.fault_type
            # recovery/rollback → 종료
            if m.category in ('recovery', 'rollback'):
                matched_ep.end_time = m.ts
                matched_ep.status = 'closed'
                matched_ep.update_severity()
                episodes.append(matched_ep)
                del open_episodes[matched_key]
            continue

        # 매칭 실패 — 새 Episode 또는 orphan
        primary_key = keys[0]       # 가장 구체적 키
        # 장비명 / 라인 추출
        eq_val = next(iter(m.equipment), '')
        ln_val = next(iter(m.line), '') if m.line else ''

        if m.category in ('fault', 'action'):
            # ★ 같은 키로 이미 open 인 Episode 가 있으면 (window 초과로 매칭 실패한 경우)
            #   덮어쓰기 전에 episodes 리스트에 마감해서 추가 (재발생 분리)
            if primary_key in open_episodes:
                stale = open_episodes[primary_key]
                stale.update_severity()
                episodes.append(stale)
            ep_id += 1
            ep = Episode(
                episode_id=ep_id, start_time=m.ts, last_time=m.ts,
                equipment=eq_val, line=ln_val, fault_type=m.fault_type,
                message_ids=[m.msg_id], message_count=1,
            )
            open_episodes[primary_key] = ep
        elif m.category in ('recovery', 'rollback'):
            # Case 1: 선행 fault 없는 복구 → orphan
            ep_id += 1
            ep = Episode(
                episode_id=ep_id, start_time=m.ts, last_time=m.ts, end_time=m.ts,
                equipment=eq_val, line=ln_val, fault_type=m.fault_type,
                status='orphan', is_orphan=True,
                message_ids=[m.msg_id], message_count=1,
            )
            episodes.append(ep)

    # 끝까지 열린 채 남은 Episode → open 상태로 마감
    for ep in open_episodes.values():
        ep.update_severity()
        episodes.append(ep)

    episodes.sort(key=lambda x: x.start_time)
    # episode_id 재부여 (시간순)
    for i, ep in enumerate(episodes, 1):
        ep.episode_id = i
    return episodes


# ============================================================
# 7. CSV 출력
# ============================================================

def write_message_csv(messages: List[Message], out_path: Path) -> None:
    """메시지 단위 CSV (v1 호환)."""
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['msg_id', 'datetime', 'sender', 'org', 'category',
                    'fault_type', 'equipment', 'line', 'text'])
        for m in messages:
            w.writerow([
                m.msg_id,
                m.ts.strftime('%Y-%m-%d %H:%M:%S'),
                m.sender, m.org, m.category, m.fault_type,
                '|'.join(sorted(m.equipment)),
                '|'.join(sorted(m.line)),
                m.text[:200],
            ])


def write_episode_csv(episodes: List[Episode], out_path: Path) -> None:
    """Episode 단위 CSV — 핵심 출력."""
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['episode_id', 'start_time', 'end_time', 'duration_min',
                    'equipment', 'line', 'fault_type', 'severity', 'status',
                    'is_orphan', 'message_count', 'message_ids'])
        for ep in episodes:
            end = ep.end_time or ep.last_time
            w.writerow([
                ep.episode_id,
                ep.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                end.strftime('%Y-%m-%d %H:%M:%S'),
                f'{ep.duration_min():.1f}',
                ep.equipment, ep.line, ep.fault_type,
                ep.severity, ep.status,
                'Y' if ep.is_orphan else 'N',
                ep.message_count,
                '|'.join(str(i) for i in ep.message_ids),
            ])


def write_summary_json(messages: List[Message], episodes: List[Episode],
                       out_path: Path) -> None:
    """통계 요약 JSON."""
    cat_cnt = Counter(m.category for m in messages)
    type_cnt = Counter(m.fault_type for m in messages if m.fault_type)
    ep_status = Counter(e.status for e in episodes)
    ep_severity = Counter(e.severity for e in episodes if not e.is_orphan)
    ep_type = Counter(e.fault_type for e in episodes if e.fault_type)

    real_episodes = [e for e in episodes if not e.is_orphan]
    asis_fault_count = sum(1 for m in messages if m.category != 'normal')

    summary = {
        'input_messages': len(messages),
        'asis_fault_count': asis_fault_count,
        'asis_note': '메시지 단위 (normal 제외 전부 장애로 카운트 — v1 방식)',
        'v2_episode_count': len(real_episodes),
        'v2_orphan_count': sum(1 for e in episodes if e.is_orphan),
        'reduction_pct': round(
            (1 - len(real_episodes) / max(asis_fault_count, 1)) * 100, 1
        ),
        'category_distribution': dict(cat_cnt),
        'fault_type_distribution': dict(type_cnt),
        'episode_status': dict(ep_status),
        'episode_severity': dict(ep_severity),
        'episode_fault_type': dict(ep_type),
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ============================================================
# 메인
# ============================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', help='운영로그 .txt 파일 경로')
    p.add_argument('--out', default='./output', help='출력 폴더 (기본: ./output)')
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f'❌ 입력 파일 없음: {inp}')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f'📂 입력: {inp}')
    print(f'📂 출력: {out_dir}')
    print()

    # 1. 파싱
    messages = parse_messages(str(inp))
    print(f'[1/4] 메시지 파싱: {len(messages)}건')

    # 2~4. 분류/유형/장비
    enrich_messages(messages)
    cat = Counter(m.category for m in messages)
    print(f'[2/4] 카테고리 분류: {dict(cat)}')

    # 5~6. Episode 클러스터링
    episodes = cluster_episodes(messages)
    real_ep = [e for e in episodes if not e.is_orphan]
    print(f'[3/4] Episode 묶기: {len(real_ep)}건 (orphan {len(episodes)-len(real_ep)}건 별도)')

    # 7. CSV 출력
    msg_csv = out_dir / f'{stamp}_message.csv'
    ep_csv = out_dir / f'{stamp}_episode.csv'
    sum_json = out_dir / f'{stamp}_summary.json'
    write_message_csv(messages, msg_csv)
    write_episode_csv(episodes, ep_csv)
    write_summary_json(messages, episodes, sum_json)
    print(f'[4/4] 출력 완료:')
    print(f'      {msg_csv.name}  ({len(messages)}행)')
    print(f'      {ep_csv.name}   ({len(episodes)}행)')
    print(f'      {sum_json.name}')
    print()

    # 비교 출력
    asis = sum(1 for m in messages if m.category != 'normal')
    reduction = (1 - len(real_ep) / max(asis, 1)) * 100
    print(f'📊 v1 (메시지 단위) 장애 카운트 : {asis}건')
    print(f'📊 v2 (Episode 단위) 장애 카운트: {len(real_ep)}건')
    print(f'📉 거품 제거: {reduction:.1f}%')


if __name__ == '__main__':
    main()
