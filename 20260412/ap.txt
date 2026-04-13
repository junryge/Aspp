---
name: weekly-report
tags: 주간보고, weekly, report, PPT, 금주실적, 차주계획
category: document
description: "주간보고 PPT 생성. 금주 실적과 차주 계획을 PPT로 변환. 이 스킬이 로드되면 pptx 스킬의 코드 방식 대신 이 스킬의 코드를 사용할 것."
date: 2026-04-13
---

# 주간보고 PPT 생성

## 절대 금지 사항
1. **slide_layouts[0], [1], [5] 사용 금지** — 반드시 slide_layouts[6] (blank) 사용
2. **placeholders 사용 금지** — placeholders[0], placeholders[1] 절대 사용 금지
3. **pptx 스킬의 코드 패턴 사용 금지** — 이 스킬의 코드만 사용
4. **Chart 추가 금지** — 차트 절대 넣지 말 것
5. **코드를 재작성하지 말 것** — 아래 코드를 그대로 복사하고 DATA 영역만 수정

## PPT 생성 코드

주간보고 PPT 요청 시, 아래 코드를 **그대로** 출력하고 DATA 영역의 projects만 수정할 것.

```python
from pptx import Presentation
from pptx.util import Inches, Pt, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

def _sc(tbl, r, c, txt, b=False, bg=None, al=PP_ALIGN.LEFT, sz=10):
    cl = tbl.cell(r, c)
    cl.text = ""
    p = cl.text_frame.paragraphs[0]
    p.text = str(txt)
    p.font.size = Pt(sz)
    p.font.bold = b
    p.font.name = "맑은 고딕"
    p.alignment = al
    cl.vertical_anchor = MSO_ANCHOR.MIDDLE
    if bg:
        cl.fill.solid()
        cl.fill.fore_color.rgb = bg

def _tx(sl, l, t, w, h, txt, sz=10, b=False, al=PP_ALIGN.LEFT):
    bx = sl.shapes.add_textbox(l, t, w, h)
    tf = bx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = str(txt)
    p.font.size = Pt(sz)
    p.font.bold = b
    p.font.name = "맑은 고딕"
    p.alignment = al

GY = RGBColor(0xD9, 0xD9, 0xD9)
LG = RGBColor(0xF2, 0xF2, 0xF2)

# ===== DATA 시작 (여기만 수정) =====
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
OUTPUT = "smartATLAS_주간보고_20260408.pptx"
# ===== DATA 끝 =====

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)
for idx, pj in enumerate(projects):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    _tx(sl, Cm(1), Cm(0.5), Cm(20), Cm(1.2), f"{idx+1}. {pj['name']}", sz=24, b=True)
    for i, c in enumerate([RGBColor(0xFF,0,0), RGBColor(0xFF,0xD7,0), RGBColor(0x44,0x72,0xC4)]):
        br = sl.shapes.add_shape(1, Cm(1+10.5*i), Cm(2.0), Cm(10.5), Cm(0.25))
        br.fill.solid()
        br.fill.fore_color.rgb = c
        br.line.fill.background()
    nc = len(pj.get("current", []))
    nn = len(pj.get("next", []))
    dr = max(nc, nn, 1)
    tr = 2 + dr + 1
    ts = sl.shapes.add_table(tr, 6, Cm(1), Cm(2.5), Cm(31.5), Cm(1.2*tr))
    tb = ts.table
    tb.columns[0].width = Cm(12)
    tb.columns[1].width = Cm(2.5)
    tb.columns[2].width = Cm(2.5)
    tb.columns[3].width = Cm(12)
    tb.columns[4].width = Cm(2.5)
    tb.columns[5].width = Cm(2.5)
    tb.cell(0,0).merge(tb.cell(0,2))
    tb.cell(0,3).merge(tb.cell(0,5))
    _sc(tb, 0, 0, "금주 실적", b=True, bg=GY, al=PP_ALIGN.CENTER, sz=12)
    _sc(tb, 0, 3, "차주 계획", b=True, bg=GY, al=PP_ALIGN.CENTER, sz=12)
    for ci, h in enumerate(["추진 내용","납기","진척율","추진 내용","납기","진척율"]):
        _sc(tb, 1, ci, h, b=True, bg=LG, al=PP_ALIGN.CENTER)
    for ri in range(dr):
        r = ri + 2
        if ri < nc:
            it = pj["current"][ri]
            _sc(tb, r, 0, it.get("content",""), sz=9)
            _sc(tb, r, 1, it.get("date",""), al=PP_ALIGN.CENTER, sz=9)
            _sc(tb, r, 2, it.get("progress",""), al=PP_ALIGN.CENTER, sz=9)
        if ri < nn:
            it = pj["next"][ri]
            _sc(tb, r, 3, it.get("content",""), sz=9)
            _sc(tb, r, 4, it.get("date",""), al=PP_ALIGN.CENTER, sz=9)
            _sc(tb, r, 5, it.get("progress",""), al=PP_ALIGN.CENTER, sz=9)
    ir = 2 + dr
    tb.cell(ir,0).merge(tb.cell(ir,5))
    _sc(tb, ir, 0, "Issue 및 협의사항" + (": "+pj["issues"] if pj.get("issues") else ""), b=True, bg=GY)
    _tx(sl, Cm(1), Cm(17), Cm(20), Cm(0.8), "● : 완료  ○ : 계획  ▶ : 진행중  ※ : Issue/특이사항", sz=9)
    _tx(sl, Cm(15), Cm(17.5), Cm(3), Cm(0.6), str(idx+1), sz=10, al=PP_ALIGN.CENTER)
prs.save(OUTPUT)
```
