---
tags: 주간보고 weekly report PPT
category: document
description: "주간보고 PPT 생성 템플릿 - python-pptx 코드 포함"
date: 2026-04-13
---

# 주간보고 PPT 생성 템플릿

## 사용법
"주간보고 PPT 만들어줘" 요청 시, 아래 python-pptx 코드를 기반으로 PPT를 생성한다.
개인 지식에 등록된 주간보고 데이터를 읽어서 `projects` 리스트에 채운다.

## python-pptx 생성 코드

```python
from pptx import Presentation
from pptx.util import Inches, Pt, Cm, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

def create_weekly_report(filename, projects):
    """
    주간보고 PPT 생성

    projects = [
        {
            "name": "smartATLAS",
            "current": [  # 금주 실적
                {"content": "▶ ATLAS 소스 개발/운영 동일화 계획(분석)\n   1) DATA 소스 영역 분석\n     ▶문제 영역 찾아서 수정 계획 제공", "date": "4/22", "progress": "97%"},
                {"content": "▶ ICPKT AGV MAP 구현\n   1)테이블 확인후 구현 완료\n     ▶문서 및 HTML 문서 작성 및 전달", "date": "4/08", "progress": "100%"},
            ],
            "next": [  # 차주 계획
                {"content": "▶ ATLAS 소스 개발/운영 동일화 계획(분석)\n   1) DATA 소스 영역 분석\n     ▶파이썬,일반영역 분리작업 준비", "date": "4/22", "progress": "98%"},
            ],
            "issues": ""  # Issue 및 협의사항
        }
    ]
    """
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    for idx, proj in enumerate(projects):
        slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

        # === 제목 ===
        title_box = slide.shapes.add_textbox(Cm(1), Cm(0.5), Cm(20), Cm(1.2))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = f"{idx+1}. {proj['name']}"
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.name = "맑은 고딕"

        # === 3색 구분선 (빨강/노랑/파랑) ===
        bar_y = Cm(2.0)
        bar_h = Cm(0.25)
        total_w = Cm(31.5)
        colors = [RGBColor(0xFF, 0x00, 0x00), RGBColor(0xFF, 0xD7, 0x00), RGBColor(0x44, 0x72, 0xC4)]
        for i, color in enumerate(colors):
            bar = slide.shapes.add_shape(1, Cm(1) + Emu(int(total_w * i / 3)), bar_y, Emu(int(total_w / 3)), bar_h)
            bar.fill.solid()
            bar.fill.fore_color.rgb = color
            bar.line.fill.background()

        # === 테이블 ===
        max_rows = max(len(proj["current"]), len(proj["next"]))
        content_rows = max(max_rows, 1)
        total_rows = 2 + content_rows + 1  # 헤더2줄 + 데이터 + Issue
        total_cols = 6  # 추진내용,납기,진척율 x 2

        table_shape = slide.shapes.add_table(total_rows, total_cols, Cm(1), Cm(2.5), Cm(31.5), Cm(1.5 * total_rows))
        table = table_shape.table

        # 컬럼 너비: 추진내용(넓게), 납기(좁게), 진척율(좁게)
        table.columns[0].width = Cm(12)   # 금주 추진내용
        table.columns[1].width = Cm(2.5)  # 금주 납기
        table.columns[2].width = Cm(2.5)  # 금주 진척율
        table.columns[3].width = Cm(12)   # 차주 추진내용
        table.columns[4].width = Cm(2.5)  # 차주 납기
        table.columns[5].width = Cm(2.5)  # 차주 진척율

        def set_cell(row, col, text, bold=False, bg=None, align=PP_ALIGN.LEFT, size=10):
            cell = table.cell(row, col)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.text = str(text)
            p.font.size = Pt(size)
            p.font.bold = bold
            p.font.name = "맑은 고딕"
            p.alignment = align
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            if bg:
                cell.fill.solid()
                cell.fill.fore_color.rgb = bg

        header_bg = RGBColor(0xD9, 0xD9, 0xD9)
        sub_bg = RGBColor(0xF2, 0xF2, 0xF2)

        # Row 0: 금주 실적 | 차주 계획 (merge)
        table.cell(0, 0).merge(table.cell(0, 2))
        table.cell(0, 3).merge(table.cell(0, 5))
        set_cell(0, 0, "금주 실적", bold=True, bg=header_bg, align=PP_ALIGN.CENTER, size=12)
        set_cell(0, 3, "차주 계획", bold=True, bg=header_bg, align=PP_ALIGN.CENTER, size=12)

        # Row 1: 추진 내용 | 납기 | 진척율 x 2
        headers = ["추진 내용", "납기", "진척율", "추진 내용", "납기", "진척율"]
        for ci, h in enumerate(headers):
            align = PP_ALIGN.CENTER if ci in [1, 2, 4, 5] else PP_ALIGN.CENTER
            set_cell(1, ci, h, bold=True, bg=sub_bg, align=align, size=10)

        # Data rows
        for ri in range(content_rows):
            row = ri + 2
            # 금주 실적
            if ri < len(proj["current"]):
                item = proj["current"][ri]
                set_cell(row, 0, item["content"], size=9)
                set_cell(row, 1, item["date"], align=PP_ALIGN.CENTER, size=9)
                set_cell(row, 2, item["progress"], align=PP_ALIGN.CENTER, size=9)
            # 차주 계획
            if ri < len(proj["next"]):
                item = proj["next"][ri]
                set_cell(row, 0 + 3, item["content"], size=9)
                set_cell(row, 1 + 3, item["date"], align=PP_ALIGN.CENTER, size=9)
                set_cell(row, 2 + 3, item["progress"], align=PP_ALIGN.CENTER, size=9)

        # Issue row
        issue_row = 2 + content_rows
        table.cell(issue_row, 0).merge(table.cell(issue_row, 5))
        issue_text = f"Issue 및 협의사항: {proj.get('issues', '')}" if proj.get('issues') else "Issue 및 협의사항"
        set_cell(issue_row, 0, issue_text, bold=True, bg=header_bg, size=10)

        # === 범례 ===
        legend_box = slide.shapes.add_textbox(Cm(1), Cm(17), Cm(20), Cm(0.8))
        lf = legend_box.text_frame
        lp = lf.paragraphs[0]
        lp.text = "● : 완료  ○ : 계획  ▶ : 진행중  ※ : Issue/특이사항"
        lp.font.size = Pt(9)
        lp.font.name = "맑은 고딕"

        # === 페이지 번호 ===
        page_box = slide.shapes.add_textbox(Cm(15), Cm(17.5), Cm(3), Cm(0.6))
        pf = page_box.text_frame
        pp = pf.paragraphs[0]
        pp.text = str(idx + 1)
        pp.alignment = PP_ALIGN.CENTER
        pp.font.size = Pt(10)
        pp.font.name = "맑은 고딕"

    prs.save(filename)
    print(f"주간보고 PPT 생성 완료: {filename}")


# === 사용 예시 ===
# projects 데이터를 개인 지식에서 읽어서 채운 후 호출
# create_weekly_report("smartATLAS_주간보고_20260408.pptx", projects)
```

## 데이터 구조 예시

```python
projects = [
    {
        "name": "smartATLAS",
        "current": [
            {
                "content": "▶ ATLAS 소스 개발/운영 동일화 계획(분석)\n   1) DATA 소스 영역 분석\n     ▶문제 영역 찾아서 수정 계획 제공",
                "date": "4/22",
                "progress": "97%"
            },
            {
                "content": "▶ ICPKT AGV MAP 구현\n   1)테이블 확인후 구현 완료\n     ▶문서 및 HTML 문서 작성 및 전달",
                "date": "4/08",
                "progress": "100%"
            },
            {
                "content": "▶ ICPKT CNV MAP 구현\n   1)테이블 확인후 구현 완료\n     ▶문서 및 HTML 문서 작성 및 전달",
                "date": "4/08",
                "progress": "100%"
            },
            {
                "content": "▶ HID IN OUT MAS 테이블 변경\n   1) VHL COUNT Limit,VHL Precaution 2개 영역 수집 필요함\n     ▶ 기존HID IN OUT MAS 테이블 2개 데이터 추가 분석 중(실패)",
                "date": "4/08",
                "progress": "100%"
            },
        ],
        "next": [
            {
                "content": "▶ ATLAS 소스 개발/운영 동일화 계획(분석)\n   1) DATA 소스 영역 분석\n     ▶파이썬,일반영역 분리작업 준비",
                "date": "4/22",
                "progress": "98%"
            },
            {
                "content": "▶ OHT,XML 데이터 통한 ADD 위치 데이터 파싱 처리(분석,개발)\n   1) 기존 메시지 UDP 파싱 필요함 시나리오 재분석\n     ▶Adrance area 영역 분석/개발 진행중",
                "date": "4/08",
                "progress": "100%"
            },
        ],
        "issues": ""
    }
]
```
