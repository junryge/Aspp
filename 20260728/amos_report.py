# -*- coding: utf-8 -*-
"""demos_v1/amos_report.py — 사건발생 보고서 HTML에 AMOS 인터랙티브 블록 주입.

_maybe_generate_md_html 이 만든 body_html 에 'AMOS 이상 감지 내역' 섹션이 있으면:
  ① AMOS 표 → 인터랙티브 표(id=amos-detection-table)로 변환
     · 실제 발생여부 = O/X 라디오
     · 작업 여부 = O/X 라디오 (신규 열)
  ② '실제 이상 발생내역' 섹션 아래 → 인터랙티브 표 + '+ 수동 기입' 버튼 주입
     (실제 발생여부 O → 행 자동 생성 / X·미선택 → 제거 / 수동기입은 항상 가능)
     · 발생·완료 시간, 조치 시간, 원복 시간 모두 시/분 분리 셀렉트
     (저장 형식은 'HH:MM' 유지 → 예전 저장본도 그대로 복원됨)
  ③ 리포트 그래프(hub-report-graph)를 '위험 이벤트 상세' 헤딩 아래로 이동
  ④ 상단 저장 툴바 + 저장/복원 JS (같은 브라우저 자동 복원, 파일로 저장 지원)

코어(routes_chat) 침투 최소화: amosify(body_html) 한 번만 부르면 된다.
실패하면 (원본, False) 반환 — 보고서 생성은 절대 안 깨진다.
"""
import re

# ── 인터랙티브 CSS (샘플5 이식) ──────────────────────────────────────────
AMOS_CSS = """
.table-wrap{width:100%;overflow-x:auto;margin:1em 0}
.amos-table{min-width:1040px}
.actual-incident-table{min-width:1300px}
.center-cell{text-align:center;vertical-align:middle}
.section-note{margin:.5em 0 1em;color:#475569;font-size:.9em}
/* 체크박스·라디오는 어떤 배치(flex/grid) 안에서도 같은 크기로 고정 —
   inline-flex 라벨 안에서 글자 길이에 따라 눌려 작아지던 문제 방지 */
input[type="checkbox"],input[type="radio"]{width:16px;height:16px;min-width:16px;min-height:16px;
  flex:0 0 16px;margin:0;padding:0;vertical-align:middle;accent-color:#2563eb}
select,textarea{box-sizing:border-box;border:1px solid #cbd5e1;border-radius:6px;background:#fff;color:#1a1a2e;font:inherit}
select{min-width:92px;padding:.25em .35em}
textarea{width:100%;min-width:170px;min-height:62px;padding:.45em .55em;resize:vertical;line-height:1.45}
.time-selects{display:flex;flex-direction:column;gap:.35rem}
.time-selects label{display:inline-flex;align-items:center;gap:.25rem;white-space:nowrap;font-size:.92em}
/* 시/분 분리 셀렉트 */
.time-field{display:inline-flex;align-items:center;gap:.22rem;white-space:nowrap;font-size:.92em}
.time-field .time-field-label{color:#334155;font-weight:650;margin-right:.15rem}
.time-field .time-unit{color:#475569}
.time-field select{min-width:62px;padding:.22em .25em}
/* O/X 라디오 그룹 */
.ox-group{display:inline-flex;align-items:center;justify-content:center;gap:.7rem}
.ox-group label{display:inline-flex;align-items:center;gap:.25rem;white-space:nowrap;font-weight:700;color:#334155}
.action-area-options{display:grid;grid-template-columns:repeat(2,minmax(72px,1fr));gap:.25rem .5rem}
.action-area-options label{display:inline-flex;align-items:center;gap:.25rem;white-space:nowrap}
.empty-row td{text-align:center;color:#94a3b8;padding:1.1em;background:#f8fafc}
.score-criteria{display:inline;color:#475569;font-size:.72em;font-weight:650}
.actual-incident-table th,.actual-incident-table td{padding:.55em .65em}
.action-editor{min-width:310px}
.action-toolbar{display:flex;align-items:center;justify-content:space-between;gap:.5rem;margin-bottom:.45rem}
.action-toolbar-title{color:#334155;font-weight:700;white-space:nowrap}
.action-list{display:grid;gap:.55rem}
.action-item{padding:.5rem;border:1px solid #e2e8f0;border-radius:8px;background:#f8fafc}
.action-item-controls{display:flex;align-items:center;justify-content:space-between;gap:.5rem;margin-bottom:.4rem}
.action-time-label{display:inline-flex;align-items:center;gap:.3rem;color:#334155;font-size:.92em;font-weight:650;white-space:nowrap}
.action-times{display:flex;flex-wrap:wrap;align-items:center;gap:.3rem .7rem}
.add-action-button,.remove-action-button{border:1px solid #94a3b8;border-radius:6px;background:#fff;color:#334155;font:inherit;font-weight:700;cursor:pointer;white-space:nowrap}
.add-action-button{padding:.28em .65em;border-color:#2563eb;color:#1d4ed8}
.remove-action-button{padding:.22em .52em;color:#b91c1c;border-color:#fca5a5}
.add-action-button:hover{background:#eff6ff}
.remove-action-button:hover{background:#fef2f2}
.add-action-button:focus-visible,.remove-action-button:focus-visible,select:focus-visible,textarea:focus-visible,input:focus-visible{outline:2px solid #2563eb;outline-offset:2px}
.action-item textarea{min-width:280px;min-height:54px}
.comment-textarea{min-width:220px}
textarea::placeholder{color:#94a3b8;opacity:1}
.top-toolbar{position:sticky;top:0;z-index:50;display:flex;align-items:center;flex-wrap:wrap;gap:.6rem;padding:.6rem .2rem;margin:-1rem 0 1rem;background:rgba(255,255,255,.96);border-bottom:1px solid #e2e8f0;backdrop-filter:blur(4px)}
.save-button{border:none;border-radius:8px;background:#2563eb;color:#fff;font:inherit;font-weight:700;padding:.5em 1.3em;cursor:pointer}
.save-button:hover{background:#1d4ed8}
.save-file-button{border:1px solid #94a3b8;border-radius:8px;background:#fff;color:#334155;font:inherit;font-weight:700;padding:.45em 1em;cursor:pointer}
.save-file-button:hover{background:#f1f5f9}
.file-sync-warn{color:#b45309;font-size:.85em;font-weight:700}
.save-status{color:#16a34a;font-size:.88em;font-weight:650}
.toolbar-note{color:#64748b;font-size:.82em}
.manual-add-bar{margin:.25em 0 1.2em;display:flex;align-items:center;gap:.55em;flex-wrap:wrap}
.manual-help{color:#64748b;font-size:.85em}
.manual-incident-row td{background:#fffbeb}
.manual-detection-row td{background:#fffbeb}
.manual-input{box-sizing:border-box;width:100%;border:1px solid #cbd5e1;border-radius:6px;padding:.3em .45em;font:inherit;background:#fff;color:#1a1a2e}
.manual-badge{display:inline-block;margin-left:.3em;font-size:.72em;color:#b45309;font-weight:700;white-space:nowrap}
.remove-manual-button{display:block;margin-top:.35em;border:1px solid #fca5a5;border-radius:6px;background:#fff;color:#b91c1c;font:inherit;font-size:.78em;font-weight:700;padding:.15em .45em;cursor:pointer;white-space:nowrap}
.remove-manual-button:hover{background:#fef2f2}
"""

TOOLBAR_HTML = """
<div class="top-toolbar">
<button type="button" id="save-report-button" class="save-button">\U0001F4BE 저장</button>
<button type="button" id="save-file-button" class="save-file-button">\U0001F4C4 파일로 저장</button>
<span id="save-status" class="save-status" role="status" aria-live="polite"></span>
<span id="file-sync-status" class="file-sync-warn" role="status" aria-live="polite"></span>
<span class="toolbar-note">저장: 클릭 즉시 저장, 선택창 없음. 같은 PC·같은 브라우저에서 이 파일을 다시 열면 자동 복원됩니다. 파일 자체에 내용을 담아 공유·보관할 때만 '파일로 저장'을 쓰세요. (Ctrl+S = 저장)</span>
</div>
"""

SECTION3_HTML = """
<p class="section-note">2. AMOS 이상 감지 내역의 <strong>실제 발생여부</strong>를 <strong>O</strong>로 선택하면 아래 표에 해당 번호의 행이 자동으로 생성됩니다. <strong>X</strong>로 바꾸거나 선택을 비우면 해당 행은 표에서 제외됩니다.</p>
<div class="table-wrap">
<table id="actual-incident-table" class="actual-incident-table">
<colgroup>
<col style="width:55px"><col style="width:120px"><col style="width:250px">
<col style="width:160px"><col style="width:405px"><col style="width:225px">
</colgroup>
<thead>
<tr><th>번호</th><th>AMOS 이상감지 시간</th><th>실제 발생시간/조치완료시간</th><th>조치 구간</th><th>조치내용</th><th>Comment(or 발생원인)</th></tr>
</thead>
<tbody id="actual-incident-tbody">
<tr class="empty-row"><td colspan="6">실제 발생여부가 O 로 선택된 AMOS 이상 감지 항목이 없습니다.</td></tr>
</tbody>
</table>
</div>
<div class="manual-add-bar">
<button type="button" id="add-manual-incident" class="add-action-button">+ 수동 기입</button>
<span class="manual-help">AMOS 가 감지하지 못한 이상이 발생 한 경우 수동 기입</span>
</div>
"""

# ── 인터랙티브 JS (샘플5 이식 — 체크→행 생성 / 수동기입 / 저장·복원) ──────
AMOS_JS = r"""
<script type="application/json" id="saved-data">null</script>
<script>
(function () {
  'use strict';
  window.__AMHS_NATIVE_SAVE = true;
  const amosTbody = document.querySelector('#amos-detection-table tbody');
  const actualTbody = document.getElementById('actual-incident-tbody');
  const saveButton = document.getElementById('save-report-button');
  const saveFileButton = document.getElementById('save-file-button');
  const fileSyncStatus = document.getElementById('file-sync-status');
  const saveStatus = document.getElementById('save-status');
  const addManualButton = document.getElementById('add-manual-incident');
  const savedState = new Map();
  let manualIncidents = [];
  const actionAreas = ['CNV', 'M14BLFT', 'M16LFT', 'FIO'];

  function pad2(v) { return String(v).padStart(2, '0'); }
  // '실제 발생여부' O 라디오 = 판정 기준 (X/미선택이면 3번 표에 행을 만들지 않는다)
  function getOccurrenceChecks() { return Array.from(document.querySelectorAll('.actual-occurrence-check')); }

  // ── O/X 라디오 (실제 발생여부 · 작업 여부) ─────────────────────────
  function oxNodes(cls, no) {
    const key = String(no);
    const pick = (sel) => Array.from(document.querySelectorAll(sel))
      .find((node) => node.dataset.incidentNo === key) || null;
    return { o: pick('.' + cls), x: pick('.' + cls + '-no') };
  }

  function readOx(cls, no) {
    const { o, x } = oxNodes(cls, no);
    if (o && o.checked) return 'O';
    if (x && x.checked) return 'X';
    return '';
  }

  function applyOx(cls, no, value) {
    const { o, x } = oxNodes(cls, no);
    if (o) o.checked = value === 'O';
    if (x) x.checked = value === 'X';
  }

  // ── 시/분 분리 셀렉트 ──────────────────────────────────────────────
  // 저장 형식은 기존과 같은 'HH:MM' 문자열 유지 (예전에 저장한 보고서도 그대로 복원됨)
  function splitHM(value) {
    const m = /^(\d{1,2}):(\d{1,2})$/.exec(String(value || '').trim());
    return m ? [pad2(m[1]), pad2(m[2])] : ['', ''];
  }

  function buildUnitOptions(count, selectedValue) {
    const fragment = document.createDocumentFragment();
    const placeholder = document.createElement('option');
    placeholder.value = ''; placeholder.textContent = '--';
    fragment.appendChild(placeholder);
    for (let i = 0; i < count; i += 1) {
      const value = pad2(i);
      const option = document.createElement('option');
      option.value = value; option.textContent = value;
      if (value === selectedValue) option.selected = true;
      fragment.appendChild(option);
    }
    return fragment;
  }

  function createTimeSplit(no, field, labelText, selectedValue) {
    const [hh, mm] = splitHM(selectedValue);
    const wrap = document.createElement('span');
    wrap.className = 'time-field';
    if (labelText) {
      const tag = document.createElement('span');
      tag.className = 'time-field-label'; tag.textContent = labelText;
      wrap.appendChild(tag);
    }
    const hour = document.createElement('select');
    hour.dataset.field = `${field}-h`;
    hour.setAttribute('aria-label', `${no}번 ${labelText} 시`);
    hour.appendChild(buildUnitOptions(24, hh));
    const hUnit = document.createElement('span');
    hUnit.className = 'time-unit'; hUnit.textContent = '시';
    const minute = document.createElement('select');
    minute.dataset.field = `${field}-m`;
    minute.setAttribute('aria-label', `${no}번 ${labelText} 분`);
    minute.appendChild(buildUnitOptions(60, mm));
    const mUnit = document.createElement('span');
    mUnit.className = 'time-unit'; mUnit.textContent = '분';
    wrap.appendChild(hour); wrap.appendChild(hUnit);
    wrap.appendChild(minute); wrap.appendChild(mUnit);
    return wrap;
  }

  function readTime(scope, field) {
    if (!scope) return '';
    const hh = scope.querySelector(`[data-field="${field}-h"]`)?.value || '';
    const mm = scope.querySelector(`[data-field="${field}-m"]`)?.value || '';
    if (!hh && !mm) return '';
    return `${hh || '00'}:${mm || '00'}`;
  }

  function defaultAction() { return { time: '', restoreTime: '', text: '' }; }

  function defaultState() {
    return { actualTime: '', completeTime: '', areas: [], actions: [defaultAction()], commentText: '' };
  }

  function getActionsFromRow(row) {
    const items = Array.from(row.querySelectorAll('[data-action-item]'));
    const actions = items.map((item) => ({
      time: readTime(item, 'action-time'),
      restoreTime: readTime(item, 'restore-time'),
      text: item.querySelector('[data-field="action-text"]')?.value || ''
    }));
    return actions.length ? actions : [defaultAction()];
  }

  function saveCurrentRows() {
    if (!actualTbody) return;
    actualTbody.querySelectorAll('tr[data-incident-no]').forEach((row) => {
      savedState.set(row.dataset.incidentNo, {
        actualTime: readTime(row, 'actual-time'),
        completeTime: readTime(row, 'complete-time'),
        areas: Array.from(row.querySelectorAll('[data-field="action-area"]:checked')).map((i) => i.value),
        actions: getActionsFromRow(row),
        commentText: row.querySelector('[data-field="comment-text"]')?.value || ''
      });
    });
  }

  function createActionAreaOptions(no, selectedAreas) {
    const wrapper = document.createElement('div');
    wrapper.className = 'action-area-options';
    actionAreas.forEach((area) => {
      const optionId = `action-area-${no}-${area}`;
      const label = document.createElement('label');
      label.setAttribute('for', optionId);
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox'; checkbox.id = optionId; checkbox.value = area;
      checkbox.dataset.field = 'action-area';
      checkbox.checked = selectedAreas.includes(area);
      label.appendChild(checkbox);
      label.appendChild(document.createTextNode(area));
      wrapper.appendChild(label);
    });
    return wrapper;
  }

  function createTextarea(no, field, ariaLabel, value, placeholderText, className) {
    const textarea = document.createElement('textarea');
    textarea.dataset.field = field;
    textarea.setAttribute('aria-label', `${no}번 ${ariaLabel}`);
    textarea.value = value || '';
    if (placeholderText) textarea.placeholder = placeholderText;
    if (className) textarea.className = className;
    return textarea;
  }

  function createActionItem(no, action, index) {
    const item = document.createElement('div');
    item.className = 'action-item'; item.dataset.actionItem = '';
    const controls = document.createElement('div');
    controls.className = 'action-item-controls';
    const times = document.createElement('div');
    times.className = 'action-times';
    times.appendChild(createTimeSplit(no, 'action-time', '조치 시간', action?.time || ''));
    times.appendChild(createTimeSplit(no, 'restore-time', '원복 시간', action?.restoreTime || ''));
    const removeButton = document.createElement('button');
    removeButton.type = 'button'; removeButton.className = 'remove-action-button';
    removeButton.dataset.action = 'remove-action'; removeButton.textContent = '삭제';
    removeButton.setAttribute('aria-label', `${no}번 ${index + 1}번째 조치내용 삭제`);
    controls.appendChild(times); controls.appendChild(removeButton);
    const textarea = createTextarea(no, 'action-text', `${index + 1}번째 조치내용`, action?.text || '',
      'ex) M16 6F Maxcapa 3조정,Maxcapa원복');
    item.appendChild(controls); item.appendChild(textarea);
    return item;
  }

  function refreshActionItemControls(row) {
    const items = Array.from(row.querySelectorAll('[data-action-item]'));
    const no = row.dataset.incidentNo;
    items.forEach((item, index) => {
      const textarea = item.querySelector('[data-field="action-text"]');
      const removeButton = item.querySelector('[data-action="remove-action"]');
      [['action-time', '조치 시간'], ['restore-time', '원복 시간']].forEach(([field, label]) => {
        item.querySelector(`[data-field="${field}-h"]`)
          ?.setAttribute('aria-label', `${no}번 ${index + 1}번째 ${label} 시`);
        item.querySelector(`[data-field="${field}-m"]`)
          ?.setAttribute('aria-label', `${no}번 ${index + 1}번째 ${label} 분`);
      });
      if (textarea) textarea.setAttribute('aria-label', `${no}번 ${index + 1}번째 조치내용`);
      if (removeButton) {
        removeButton.hidden = items.length === 1;
        removeButton.setAttribute('aria-label', `${no}번 ${index + 1}번째 조치내용 삭제`);
      }
    });
  }

  function createActionEditor(no, actions) {
    const editor = document.createElement('div');
    editor.className = 'action-editor';
    const toolbar = document.createElement('div');
    toolbar.className = 'action-toolbar';
    const title = document.createElement('span');
    title.className = 'action-toolbar-title'; title.textContent = '조치내용';
    const addButton = document.createElement('button');
    addButton.type = 'button'; addButton.className = 'add-action-button';
    addButton.dataset.action = 'add-action'; addButton.textContent = '+ 추가';
    addButton.setAttribute('aria-label', `${no}번 조치내용 추가`);
    toolbar.appendChild(title); toolbar.appendChild(addButton);
    const list = document.createElement('div');
    list.className = 'action-list';
    const normalizedActions = Array.isArray(actions) && actions.length ? actions : [defaultAction()];
    normalizedActions.forEach((action, index) => { list.appendChild(createActionItem(no, action, index)); });
    editor.appendChild(toolbar); editor.appendChild(list);
    return editor;
  }

  function createIncidentRow(no, detectionTime, isManual) {
    const state = savedState.get(no) || defaultState();
    const row = document.createElement('tr');
    row.dataset.incidentNo = no;
    const noCell = document.createElement('td');
    noCell.appendChild(document.createTextNode(no));
    if (isManual) {
      row.classList.add('manual-incident-row');
      row.dataset.manualIncident = '1';
      const badge = document.createElement('span');
      badge.className = 'manual-badge'; badge.textContent = '수동';
      noCell.appendChild(badge);
      const removeButton = document.createElement('button');
      removeButton.type = 'button'; removeButton.className = 'remove-manual-button';
      removeButton.dataset.action = 'remove-manual'; removeButton.textContent = '삭제';
      removeButton.setAttribute('aria-label', `${no}번 수동 기입 행 삭제`);
      noCell.appendChild(removeButton);
    }
    const detectionTimeCell = document.createElement('td');
    detectionTimeCell.textContent = detectionTime || '-';
    const timeCell = document.createElement('td');
    const timeWrapper = document.createElement('div');
    timeWrapper.className = 'time-selects';
    timeWrapper.appendChild(createTimeSplit(no, 'actual-time', '발생', state.actualTime));
    timeWrapper.appendChild(createTimeSplit(no, 'complete-time', '완료', state.completeTime));
    timeCell.appendChild(timeWrapper);
    const areaCell = document.createElement('td');
    areaCell.appendChild(createActionAreaOptions(no, state.areas || []));
    const actionCell = document.createElement('td');
    actionCell.appendChild(createActionEditor(no, state.actions));
    const commentCell = document.createElement('td');
    commentCell.appendChild(createTextarea(no, 'comment-text', 'Comment 또는 발생원인',
      state.commentText, 'ex) 00작업, 물량증가, OHT Stop...', 'comment-textarea'));
    row.appendChild(noCell); row.appendChild(detectionTimeCell); row.appendChild(timeCell);
    row.appendChild(areaCell); row.appendChild(actionCell); row.appendChild(commentCell);
    refreshActionItemControls(row);
    return row;
  }

  function renderActualIncidentRows() {
    if (!actualTbody) return;
    saveCurrentRows();
    actualTbody.innerHTML = '';
    const checkedItems = getOccurrenceChecks().filter((check) => check.checked);
    if (checkedItems.length === 0 && manualIncidents.length === 0) {
      const emptyRow = document.createElement('tr');
      emptyRow.className = 'empty-row';
      const emptyCell = document.createElement('td');
      emptyCell.colSpan = 6;
      emptyCell.textContent = '실제 발생여부가 O 로 선택된 AMOS 이상 감지 항목이 없습니다.';
      emptyRow.appendChild(emptyCell);
      actualTbody.appendChild(emptyRow);
      return;
    }
    checkedItems.forEach((check) => {
      actualTbody.appendChild(createIncidentRow(check.dataset.incidentNo, check.dataset.detectionTime, false));
    });
    manualIncidents.forEach((no) => { actualTbody.appendChild(createIncidentRow(no, '-', true)); });
  }

  function nextIncidentNo() {
    const nos = getOccurrenceChecks()
      .map((check) => parseInt(check.dataset.incidentNo, 10))
      .concat(manualIncidents.map((no) => parseInt(no, 10)))
      .filter((v) => !Number.isNaN(v));
    return nos.length ? Math.max(...nos) + 1 : 1;
  }

  function collectFullState() {
    saveCurrentRows();
    const staticChecks = {};
    const occurrenceStatus = {};
    const workStatus = {};
    getOccurrenceChecks().forEach((check) => {
      const no = check.dataset.incidentNo;
      staticChecks[no] = check.checked;   // 하위호환 (예전 저장본 형식)
      occurrenceStatus[no] = readOx('actual-occurrence-check', no);
      workStatus[no] = readOx('work-status-check', no);
    });
    const incidents = {};
    savedState.forEach((value, key) => { incidents[key] = value; });
    return { version: 2, savedAt: new Date().toISOString(), staticChecks, occurrenceStatus, workStatus,
             manualIncidents: manualIncidents.slice(), incidents };
  }

  let fileHandle = null;
  const STORAGE_KEY = 'amos-report:' + (window.__AMHS_DATE || window.location.pathname);
  let fileDirty = false;

  function setFileDirty(value) {
    fileDirty = value;
    if (fileSyncStatus) {
      fileSyncStatus.textContent = fileDirty
        ? '⚠ 파일 미반영 변경 있음 — 공유·보관 전 ‘파일로 저장’ 필요' : '';
    }
  }

  function currentFileName() {
    try {
      const name = decodeURIComponent(window.location.pathname.split('/').pop() || '');
      const lower = name.toLowerCase();
      if (lower.endsWith('.html') || lower.endsWith('.htm')) return name;
    } catch (error) { /* ignore */ }
    return '반송이벤트_보고서.html';
  }

  function buildHtmlText(state) {
    const docClone = document.documentElement.cloneNode(true);
    const cloneActualTbody = docClone.querySelector('#actual-incident-tbody');
    if (cloneActualTbody) {
      cloneActualTbody.innerHTML = '<tr class="empty-row"><td colspan="6">실제 발생여부가 O 로 선택된 AMOS 이상 감지 항목이 없습니다.</td></tr>';
    }
    docClone.querySelectorAll('.manual-detection-row').forEach((row) => row.remove());
    docClone.querySelectorAll('#__amhs_bridge').forEach((node) => node.remove());
    const dataScript = docClone.querySelector('#saved-data');
    if (dataScript) dataScript.textContent = JSON.stringify(state).replace(/</g, '\\u003c');
    const statusSpan = docClone.querySelector('#save-status');
    if (statusSpan) statusSpan.textContent = '';
    const syncSpan = docClone.querySelector('#file-sync-status');
    if (syncSpan) syncSpan.textContent = '';
    return '<!DOCTYPE html>\n' + docClone.outerHTML;
  }

  function markSaved(message) {
    if (!saveStatus) return;
    const d = new Date();
    saveStatus.textContent = `${message} · ${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
  }

  function syncLocal(state) {
    try { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); return true; }
    catch (error) { return false; }
  }

  function postToParent(state) {
    try {
      if (!window.parent || window.parent === window) return false;
      window.parent.postMessage({ type: 'amhs-report-save', date: window.__AMHS_DATE || null, html: buildHtmlText(state) }, '*');
      return true;
    } catch (error) { return false; }
  }

  function saveLocal() {
    const state = collectFullState();
    const localOk = syncLocal(state);
    if (postToParent(state)) { markSaved('저장됨 · 리포트 시스템 반영'); setFileDirty(false); return; }
    if (localOk) { markSaved('저장됨'); }
    else if (saveStatus) { saveStatus.textContent = '브라우저 저장 실패 — ‘파일로 저장’ 버튼을 사용하세요.'; }
  }

  function downloadFallback(htmlText) {
    const blob = new Blob([htmlText], { type: 'text/html;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = currentFileName();
    document.body.appendChild(link);
    link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 2000);
    markSaved('다운로드 저장됨 (브라우저 미지원 대체)');
    setFileDirty(false);
  }

  async function saveReport() {
    const state = collectFullState();
    syncLocal(state); postToParent(state);
    const htmlText = buildHtmlText(state);
    if (window.showSaveFilePicker) {
      try {
        if (!fileHandle) {
          fileHandle = await window.showSaveFilePicker({
            suggestedName: currentFileName(),
            types: [{ description: 'HTML 파일', accept: { 'text/html': ['.html', '.htm'] } }]
          });
        }
        const writable = await fileHandle.createWritable();
        await writable.write(htmlText); await writable.close();
        markSaved('파일 저장됨'); setFileDirty(false);
        return;
      } catch (error) {
        if (error && error.name === 'AbortError') {
          if (saveStatus) saveStatus.textContent = '파일 저장 취소됨';
          return;
        }
        fileHandle = null;
      }
    }
    downloadFallback(htmlText);
  }

  function parseState(text) {
    try { const state = JSON.parse(text); return state && typeof state === 'object' ? state : null; }
    catch (error) { return null; }
  }

  function loadSavedData() {
    const embedded = parseState(document.getElementById('saved-data')?.textContent || 'null');
    let local = null;
    try { local = parseState(window.localStorage.getItem(STORAGE_KEY) || 'null'); }
    catch (error) { local = null; }
    let state = null;
    if (embedded && local) {
      const embeddedAt = Date.parse(embedded.savedAt || '') || 0;
      const localAt = Date.parse(local.savedAt || '') || 0;
      state = localAt >= embeddedAt ? local : embedded;
      if (localAt > embeddedAt) setFileDirty(true);
    } else {
      state = local || embedded;
      if (local && !embedded) setFileDirty(true);
    }
    if (!state) return;
    Object.entries(state.incidents || {}).forEach(([no, value]) => { savedState.set(String(no), value); });
    const occStatus = state.occurrenceStatus || {};
    const wrkStatus = state.workStatus || {};
    const legacy = state.staticChecks || {};
    getOccurrenceChecks().forEach((check) => {
      const no = check.dataset.incidentNo;
      // version 2 = 'O'/'X'/'' · version 1(예전 저장본) = boolean → O 로 환산
      const occ = occStatus[no] !== undefined ? occStatus[no] : (legacy[no] ? 'O' : '');
      applyOx('actual-occurrence-check', no, occ);
      applyOx('work-status-check', no, wrkStatus[no] || '');
    });
    manualIncidents = (state.manualIncidents || []).map(String);
  }

  amosTbody?.addEventListener('change', (event) => {
    setFileDirty(true);
    const cl = event.target.classList;
    // O 든 X 든 바뀌면 3번 표를 다시 그린다 (O → 행 생성, X/미선택 → 행 제외)
    if (cl.contains('actual-occurrence-check') || cl.contains('actual-occurrence-check-no')) {
      renderActualIncidentRows();
    }
  });

  addManualButton?.addEventListener('click', () => {
    if (!actualTbody) return;
    const no = String(nextIncidentNo());
    manualIncidents.push(no);
    savedState.set(no, defaultState());
    setFileDirty(true);
    renderActualIncidentRows();
    const newRow = actualTbody.querySelector(`tr[data-incident-no="${no}"]`);
    if (newRow && typeof newRow.scrollIntoView === 'function') newRow.scrollIntoView({ block: 'nearest' });
    newRow?.querySelector('[data-field="actual-time-h"]')?.focus();
  });

  saveButton?.addEventListener('click', saveLocal);
  saveFileButton?.addEventListener('click', saveReport);
  document.addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && (event.key === 's' || event.key === 'S')) {
      event.preventDefault(); saveLocal();
    }
  });

  actualTbody?.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-action]');
    if (!button) return;
    const row = button.closest('tr[data-incident-no]');
    if (!row) return;
    if (button.dataset.action === 'remove-manual') {
      const no = row.dataset.incidentNo;
      if (!window.confirm(`${no}번 수동 기입 행을 삭제할까요? 입력한 내용도 함께 삭제됩니다.`)) return;
      manualIncidents = manualIncidents.filter((v) => v !== no);
      row.remove(); savedState.delete(no); setFileDirty(true);
      renderActualIncidentRows();
      return;
    }
    const actionList = row.querySelector('.action-list');
    if (!actionList) return;
    if (button.dataset.action === 'add-action') {
      const index = actionList.querySelectorAll('[data-action-item]').length;
      actionList.appendChild(createActionItem(row.dataset.incidentNo, defaultAction(), index));
      refreshActionItemControls(row); saveCurrentRows();
      actionList.lastElementChild?.querySelector('[data-field="action-time-h"]')?.focus();
      return;
    }
    if (button.dataset.action === 'remove-action') {
      const item = button.closest('[data-action-item]');
      const items = actionList.querySelectorAll('[data-action-item]');
      if (item && items.length > 1) item.remove();
      refreshActionItemControls(row); saveCurrentRows();
    }
  });

  actualTbody?.addEventListener('input', () => { saveCurrentRows(); setFileDirty(true); });
  actualTbody?.addEventListener('change', () => { saveCurrentRows(); setFileDirty(true); });

  loadSavedData();
  renderActualIncidentRows();
})();
</script>
"""

_TAG_RE = re.compile(r"<[^>]+>")


def _cell_text(html_cell):
    return _TAG_RE.sub("", html_cell).replace("&nbsp;", " ").strip()


def _ox_cell(no, group, cls, label, extra=""):
    """O/X 라디오 한 쌍이 든 <td>. O 쪽에만 판정용 class 를 준다."""
    return (
        f'<td class="center-cell"><span class="ox-group" role="radiogroup" '
        f'aria-label="{no}번 {label}">'
        f'<label><input type="radio" name="{group}-{no}" class="{cls}" value="O" '
        f'data-incident-no="{no}"{extra} aria-label="{no}번 {label} O">O</label>'
        f'<label><input type="radio" name="{group}-{no}" class="{cls}-no" value="X" '
        f'data-incident-no="{no}" aria-label="{no}번 {label} X">X</label>'
        f"</span></td>"
    )


def _transform_amos_table(table_html):
    """AMOS 표 → id/class + 마지막 열(실제 발생여부)을 O/X 라디오로,
    그 뒤에 '작업 여부' O/X 열을 신규 추가."""
    table_html = re.sub(r"<table\b[^>]*>",
                        '<table id="amos-detection-table" class="amos-table">',
                        table_html, count=1)

    # 1) 머리행에 '작업 여부' th 추가 (마지막 th 뒤)
    hm = re.search(r"<thead>([\s\S]*?)</thead>", table_html)
    if hm:
        head = hm.group(1)
        ths = re.findall(r"<th\b[^>]*>[\s\S]*?</th>", head)
        if ths and "작업 여부" not in head:
            head2 = head.replace(ths[-1], ths[-1] + "<th>작업 여부</th>", 1)
            table_html = table_html.replace(head, head2, 1)
    # colgroup 이 있으면 열 하나 늘려준다 (없으면 그대로)
    cm = re.search(r"<colgroup>[\s\S]*?</colgroup>", table_html)
    if cm and cm.group(0).count("<col") >= 1:
        table_html = table_html.replace(
            cm.group(0), cm.group(0).replace("</colgroup>", '<col style="width:92px"></colgroup>'), 1)

    m = re.search(r"<tbody>([\s\S]*?)</tbody>", table_html)
    if not m:
        return table_html
    body = m.group(1)

    def _fix_row(rm):
        row = rm.group(0)
        tds = re.findall(r"<td\b[^>]*>[\s\S]*?</td>", row)
        if len(tds) < 2:
            return row
        no = _cell_text(tds[0]) or "?"
        tmm = re.search(r"\d{1,2}:\d{2}", _cell_text(tds[1]))
        tm = tmm.group(0) if tmm else _cell_text(tds[1])
        # 마지막 td(실제 발생여부 자리) → O/X 라디오, 그 뒤에 '작업 여부' O/X 열 추가
        occ = _ox_cell(no, "occurrence", "actual-occurrence-check", "실제 발생여부",
                       extra=f' data-detection-time="{tm}"')
        work = _ox_cell(no, "work", "work-status-check", "작업 여부")
        return row.replace(tds[-1], occ + work, 1)

    body2 = re.sub(r"<tr\b[^>]*>[\s\S]*?</tr>", _fix_row, body)
    return table_html.replace(body, body2, 1)


def amosify(body_html):
    """body_html 에 AMOS 인터랙티브 블록 주입. 반환 (body_html, has_amos).
    실패 시 원본 그대로 (보고서 생성 절대 안 깨짐)."""
    try:
        if "AMOS" not in body_html:
            return body_html, False
        # 1) AMOS 헤딩 + 바로 다음 <table> 변환
        hm = re.search(r"<h[123]\b[^>]*>[^<]*AMOS[^<]*감지[^<]*</h[123]>", body_html)
        if not hm:
            return body_html, False

        # 0) 총평 섹션의 '점수 등급 기준' 표 제거 (등급 기준은 헤딩 인라인으로 충분 — 고객 요청)
        def _strip_grade_table(html):
            for tm0 in re.finditer(r"<table\b[^>]*>[\s\S]*?</table>", html):
                seg = tm0.group(0)
                if ("경계" in seg) and ("초위험" in seg) and re.search(r"50\s*~\s*70", seg):
                    return html[:tm0.start()] + html[tm0.end():]
            return html
        head_part = body_html[:hm.start()]
        head_part2 = _strip_grade_table(head_part)
        if head_part2 != head_part:
            body_html = head_part2 + body_html[hm.start():]
            hm = re.search(r"<h[123]\b[^>]*>[^<]*AMOS[^<]*감지[^<]*</h[123]>", body_html)
        after = body_html[hm.end():]
        tm = re.search(r"<table\b[^>]*>[\s\S]*?</table>", after)
        if tm:
            new_table = ('<div class="table-wrap">'
                         + _transform_amos_table(tm.group(0))
                         + "</div>")
            body_html = body_html[:hm.end()] + after[:tm.start()] + new_table + after[tm.end():]

        # 2) '실제 이상 발생내역' 헤딩 아래 내용을 인터랙티브 스켈레톤으로 교체(없으면 삽입)
        rm = re.search(r"<h[123]\b[^>]*>[^<]*실제[^<]*발생[^<]*내역[^<]*</h[123]>", body_html)
        if rm:
            tail = body_html[rm.end():]
            nx = re.search(r"<h[12]\b", tail)
            cut = nx.start() if nx else len(tail)
            body_html = body_html[:rm.end()] + SECTION3_HTML + tail[cut:]
        else:
            # 헤딩 자체가 없으면 AMOS 표 뒤에 섹션째 삽입
            hm2 = re.search(r"</table>\s*</div>", body_html)
            ins = "<h2>3. 실제 이상 발생내역</h2>" + SECTION3_HTML
            if hm2:
                body_html = body_html[:hm2.end()] + ins + body_html[hm2.end():]
            else:
                body_html += ins

        # 3) 리포트 그래프를 '위험 이벤트 상세' 헤딩 아래로 이동 (샘플5 배치)
        gm = re.search(r'<div class="hub-report-graph"[\s\S]*?</div>\s*', body_html)
        dm = re.search(r"<h[123]\b[^>]*>[^<]*위험[^<]*이벤트[^<]*상세[^<]*</h[123]>", body_html)
        if gm and dm and gm.start() < dm.start():
            graph = gm.group(0)
            body_html = body_html[:gm.start()] + body_html[gm.end():]
            dm = re.search(r"<h[123]\b[^>]*>[^<]*위험[^<]*이벤트[^<]*상세[^<]*</h[123]>", body_html)
            body_html = body_html[:dm.end()] + graph + body_html[dm.end():]

        # 4) 금지어 스크럽 — LLM이 raw 컬럼(LFT_REVERSALCNT 등)을 "리프터 역방향 카운트"로
        #    풀어써도 무조건 "리프터 정체"로 치환. 한글 '역방향'·'카운트'는 서술문에만 존재
        #    (raw 컬럼명은 전부 ASCII) → HTML 태그·컬럼명 훼손 없이 안전.
        body_html = _scrub_forbidden_words(body_html)

        return body_html, True
    except Exception:
        return body_html, False


def _scrub_forbidden_words(html):
    """'역방향'·'카운트' 금지어를 '리프터 정체' 계열로 결정적 치환 (AMOS 보고서 전용)."""
    try:
        pairs = [
            ("리프터 역방향 카운트", "리프터 정체"),
            ("리프터 역방향", "리프터 정체"),
            ("역방향 카운트", "정체"),
            ("역방향", "정체"),
        ]
        for a, b in pairs:
            html = html.replace(a, b)
        # 남은 단독 '카운트' 제거 (예: "카운트 증가" → "증가")
        html = re.sub(r"\s*카운트\s*", " ", html)
        html = re.sub(r"[ \t]{2,}", " ", html)
        return html
    except Exception:
        return html
