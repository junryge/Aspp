---
name: weekly-report
tags: 주간보고, weekly, report, PPT, 금주실적, 차주계획
category: document
description: "주간보고 PPT 생성. 금주 실적과 차주 계획을 python-pptx로 PPT 변환"
date: 2026-04-13
---

# 주간보고 PPT 생성

## 사용법
"주간보고 PPT 만들어줘" 요청 시, 아래 코드를 **그대로 복사**하여 실행한다.
개인 지식에 등록된 주간보고 데이터를 읽어서 projects 리스트에 채운다.

## 완성 코드 (그대로 사용할 것)

```python
from pptx import Presentation
from pptx.util import Inches, Pt, Cm, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

def set_cell_text(table, row, col, text, bold=False, bg_color=None, align=PP_ALIGN.LEFT, font_size=10):
    """테이블 셀에 텍스트 설정"""
    cell = table.cell(row, col)
    cell.text = ""
    para = cell.text_frame.paragraphs[0]
    para.text = str(text)
    para.font.size = Pt(font_size)
    para.font.bold = bold
    para.font.name = "맑은 고딕"
    para.alignment = align
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    if bg_color is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg_color

def add_textbox(slide, left, top, width, height, text, font_size=10, bold=False, font_name="맑은 고딕", align=PP_ALIGN.LEFT):
    """슬라이드에 텍스트 박스 추가"""
    shape = slide.shapes.add_textbox(left, top, width, height)
    frame = shape.text_frame
    frame.word_wrap = True
    para = frame.paragraphs[0]
    para.text = str(text)
    para.font.size = Pt(font_size)
    para.font.bold = bold
    para.font.name = font_name
    para.alignment = align
    return shape

def create_weekly_report(filename, projects):
    """
    주간보고 PPT 생성
    filename: 저장할 파일명 (예: "smartATLAS_주간보고_20260408.pptx")
    projects: 프로젝트 데이터 리스트
    """
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    GRAY = RGBColor(0xD9, 0xD9, 0xD9)
    LIGHT_GRAY = RGBColor(0xF2, 0xF2, 0xF2)
    RED = RGBColor(0xFF, 0x00, 0x00)
    YELLOW = RGBColor(0xFF, 0xD7, 0x00)
    BLUE = RGBColor(0x44, 0x72, 0xC4)

    for idx, proj in enumerate(projects):
        slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout

        # ── 제목 ──
        add_textbox(slide, Cm(1), Cm(0.5), Cm(20), Cm(1.2),
                    f"{idx+1}. {proj['name']}", font_size=24, bold=True)

        # ── 3색 구분선 ──
        bar_y = Cm(2.0)
        bar_h = Cm(0.25)
        bar_w = Cm(10.5)
        for i, color in enumerate([RED, YELLOW, BLUE]):
            bar = slide.shapes.add_shape(1, Cm(1) + Cm(10.5) * i, bar_y, bar_w, bar_h)
            bar.fill.solid()
            bar.fill.fore_color.rgb = color
            bar.line.fill.background()

        # ── 테이블 ──
        max_rows = max(len(proj.get("current", [])), len(proj.get("next", [])), 1)
        total_rows = 2 + max_rows + 1  # 헤더2줄 + 데이터행 + Issue행
        total_cols = 6

        tbl_shape = slide.shapes.add_table(total_rows, total_cols, Cm(1), Cm(2.5), Cm(31.5), Cm(1.2 * total_rows))
        tbl = tbl_shape.table

        # 컬럼 너비
        tbl.columns[0].width = Cm(12)
        tbl.columns[1].width = Cm(2.5)
        tbl.columns[2].width = Cm(2.5)
        tbl.columns[3].width = Cm(12)
        tbl.columns[4].width = Cm(2.5)
        tbl.columns[5].width = Cm(2.5)

        # Row 0: 금주 실적 | 차주 계획
        tbl.cell(0, 0).merge(tbl.cell(0, 2))
        tbl.cell(0, 3).merge(tbl.cell(0, 5))
        set_cell_text(tbl, 0, 0, "금주 실적", bold=True, bg_color=GRAY, align=PP_ALIGN.CENTER, font_size=12)
        set_cell_text(tbl, 0, 3, "차주 계획", bold=True, bg_color=GRAY, align=PP_ALIGN.CENTER, font_size=12)

        # Row 1: 추진 내용 | 납기 | 진척율 x 2
        for ci, header in enumerate(["추진 내용", "납기", "진척율", "추진 내용", "납기", "진척율"]):
            set_cell_text(tbl, 1, ci, header, bold=True, bg_color=LIGHT_GRAY, align=PP_ALIGN.CENTER, font_size=10)

        # 데이터 행
        for ri in range(max_rows):
            row = ri + 2
            if ri < len(proj.get("current", [])):
                item = proj["current"][ri]
                set_cell_text(tbl, row, 0, item.get("content", ""), font_size=9)
                set_cell_text(tbl, row, 1, item.get("date", ""), align=PP_ALIGN.CENTER, font_size=9)
                set_cell_text(tbl, row, 2, item.get("progress", ""), align=PP_ALIGN.CENTER, font_size=9)
            if ri < len(proj.get("next", [])):
                item = proj["next"][ri]
                set_cell_text(tbl, row, 3, item.get("content", ""), font_size=9)
                set_cell_text(tbl, row, 4, item.get("date", ""), align=PP_ALIGN.CENTER, font_size=9)
                set_cell_text(tbl, row, 5, item.get("progress", ""), align=PP_ALIGN.CENTER, font_size=9)

        # Issue 행
        issue_row = 2 + max_rows
        tbl.cell(issue_row, 0).merge(tbl.cell(issue_row, 5))
        issue_text = f"Issue 및 협의사항: {proj.get('issues', '')}" if proj.get("issues") else "Issue 및 협의사항"
        set_cell_text(tbl, issue_row, 0, issue_text, bold=True, bg_color=GRAY, font_size=10)

        # ── 범례 ──
        add_textbox(slide, Cm(1), Cm(17), Cm(20), Cm(0.8),
                    "● : 완료  ○ : 계획  ▶ : 진행중  ※ : Issue/특이사항", font_size=9)

        # ── 페이지 번호 ──
        add_textbox(slide, Cm(15), Cm(17.5), Cm(3), Cm(0.6),
                    str(idx + 1), font_size=10, align=PP_ALIGN.CENTER)

    prs.save(filename)
    print(f"주간보고 PPT 생성 완료: {filename}")
    return filename
```

## 데이터 구조

```python
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
    }
]

create_weekly_report("smartATLAS_주간보고_20260408.pptx", projects)
```

## 규칙
- 상태 기호: ▶ 진행중 (고정)
- 파일명: {프로젝트명}_주간보고_{YYYYMMDD}.pptx
- 프로젝트 여러 개면 슬라이드 추가
- 코드를 변형하지 말고 위 함수를 그대로 사용할 것
