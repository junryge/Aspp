---
name: weekly-report
tags: 주간보고, weekly, report, PPT, 금주실적, 차주계획
category: document
description: "주간보고 PPT 생성. 금주 실적과 차주 계획을 python-pptx로 PPT 변환"
date: 2026-04-13
---

# 주간보고 PPT 생성

## 중요 규칙
1. 아래 코드를 **한 글자도 수정하지 말고 그대로 복사**하여 실행할 것
2. 함수 코드를 **재작성하거나 변형 절대 금지**
3. **projects 데이터만 변경**할 것
4. 코드를 요약하거나 줄이지 말 것
5. 상태 기호: ▶ 진행중 (고정)

## 복사 전용 코드 (수정 금지)

아래 코드 블록을 통째로 복사해서 .py 파일로 저장 후 실행한다.
projects 리스트의 데이터와 마지막 줄의 파일명만 수정한다.

```python
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR


def _set_cell(table, row, col, text, bold=False, bg=None, align=PP_ALIGN.LEFT, size=10):
    cell = table.cell(row, col)
    cell.text = ""
    p = cell.text_frame.paragraphs[0]
    p.text = str(text)
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.name = "맑은 고딕"
    p.alignment = align
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    if bg is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg


def _add_text(slide, left, top, width, height, text, size=10, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = str(text)
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.name = "맑은 고딕"
    p.alignment = align


GRAY = RGBColor(0xD9, 0xD9, 0xD9)
LGRAY = RGBColor(0xF2, 0xF2, 0xF2)
RED = RGBColor(0xFF, 0x00, 0x00)
YELLOW = RGBColor(0xFF, 0xD7, 0x00)
BLUE = RGBColor(0x44, 0x72, 0xC4)


# ============================================================
# 여기부터 projects 데이터만 수정할 것
# ============================================================
projects = [
    {
        "name": "smartATLAS",
        "current": [
            {"content": "▶ ATLAS 소스 개발/운영 동일화 계획(분석)\n   1) DATA 소스 영역 분석\n     ▶문제 영역 찾아서 수정 계획 제공", "date": "4/22", "progress": "97%"},
            {"content": "▶ ICPKT AGV MAP 구현\n   1)테이블 확인후 구현 완료\n     ▶문서 및 HTML 문서 작성 및 전달", "date": "4/08", "progress": "100%"},
            {"content": "▶ ICPKT CNV MAP 구현\n   1)테이블 확인후 구현 완료\n     ▶문서 및 HTML 문서 작성 및 전달", "date": "4/08", "progress": "100%"},
            {"content": "▶ HID IN OUT MAS 테이블 변경\n   1) VHL COUNT Limit,VHL Precaution 2개 영역 수집 필요함\n     ▶ 기존HID IN OUT MAS 테이블 2개 데이터 추가 분석 중(실패)", "date": "4/08", "progress": "100%"},
        ],
        "next": [
            {"content": "▶ ATLAS 소스 개발/운영 동일화 계획(분석)\n   1) DATA 소스 영역 분석\n     ▶파이썬,일반영역 분리작업 준비", "date": "4/22", "progress": "98%"},
            {"content": "▶ OHT,XML 데이터 통한 ADD 위치 데이터 파싱 처리(분석,개발)\n   1) 기존 메시지 UDP 파싱 필요함 시나리오 재분석\n     ▶Adrance area 영역 분석/개발 진행중", "date": "4/08", "progress": "100%"},
        ],
        "issues": ""
    },
]
# ============================================================
# 여기까지 데이터 수정 영역
# ============================================================


# PPT 생성 (아래 코드 수정 금지)
OUTPUT_FILE = "smartATLAS_주간보고_20260408.pptx"

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)

for idx, proj in enumerate(projects):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    _add_text(slide, Cm(1), Cm(0.5), Cm(20), Cm(1.2),
              f"{idx+1}. {proj['name']}", size=24, bold=True)

    for i, c in enumerate([RED, YELLOW, BLUE]):
        bar = slide.shapes.add_shape(1, Cm(1 + 10.5 * i), Cm(2.0), Cm(10.5), Cm(0.25))
        bar.fill.solid()
        bar.fill.fore_color.rgb = c
        bar.line.fill.background()

    n_cur = len(proj.get("current", []))
    n_nxt = len(proj.get("next", []))
    data_rows = max(n_cur, n_nxt, 1)
    total_rows = 2 + data_rows + 1

    ts = slide.shapes.add_table(total_rows, 6, Cm(1), Cm(2.5), Cm(31.5), Cm(1.2 * total_rows))
    tbl = ts.table

    tbl.columns[0].width = Cm(12)
    tbl.columns[1].width = Cm(2.5)
    tbl.columns[2].width = Cm(2.5)
    tbl.columns[3].width = Cm(12)
    tbl.columns[4].width = Cm(2.5)
    tbl.columns[5].width = Cm(2.5)

    tbl.cell(0, 0).merge(tbl.cell(0, 2))
    tbl.cell(0, 3).merge(tbl.cell(0, 5))
    _set_cell(tbl, 0, 0, "금주 실적", bold=True, bg=GRAY, align=PP_ALIGN.CENTER, size=12)
    _set_cell(tbl, 0, 3, "차주 계획", bold=True, bg=GRAY, align=PP_ALIGN.CENTER, size=12)

    for ci, h in enumerate(["추진 내용", "납기", "진척율", "추진 내용", "납기", "진척율"]):
        _set_cell(tbl, 1, ci, h, bold=True, bg=LGRAY, align=PP_ALIGN.CENTER)

    for ri in range(data_rows):
        r = ri + 2
        if ri < n_cur:
            it = proj["current"][ri]
            _set_cell(tbl, r, 0, it.get("content", ""), size=9)
            _set_cell(tbl, r, 1, it.get("date", ""), align=PP_ALIGN.CENTER, size=9)
            _set_cell(tbl, r, 2, it.get("progress", ""), align=PP_ALIGN.CENTER, size=9)
        if ri < n_nxt:
            it = proj["next"][ri]
            _set_cell(tbl, r, 3, it.get("content", ""), size=9)
            _set_cell(tbl, r, 4, it.get("date", ""), align=PP_ALIGN.CENTER, size=9)
            _set_cell(tbl, r, 5, it.get("progress", ""), align=PP_ALIGN.CENTER, size=9)

    ir = 2 + data_rows
    tbl.cell(ir, 0).merge(tbl.cell(ir, 5))
    itxt = "Issue 및 협의사항"
    if proj.get("issues"):
        itxt = f"Issue 및 협의사항: {proj['issues']}"
    _set_cell(tbl, ir, 0, itxt, bold=True, bg=GRAY)

    _add_text(slide, Cm(1), Cm(17), Cm(20), Cm(0.8),
              "● : 완료  ○ : 계획  ▶ : 진행중  ※ : Issue/특이사항", size=9)
    _add_text(slide, Cm(15), Cm(17.5), Cm(3), Cm(0.6),
              str(idx + 1), size=10, align=PP_ALIGN.CENTER)

prs.save(OUTPUT_FILE)
print(f"생성 완료: {OUTPUT_FILE}")
```
