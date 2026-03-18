"""
Graph Generator - app.py의 Chart.js 기반 그래프 생성을 독립 Python 모듈로 분리
=============================================================================
다른 프로젝트에서 import하여 사용 가능.

지원 차트 유형: bar, line, pie, doughnut, radar, scatter, bubble, polarArea

사용법:
    from graph_generator import GraphGenerator

    # 1) 간단 API
    gen = GraphGenerator()
    gen.bar(labels=["1월","2월","3월"], datasets=[{"label":"매출","data":[120,190,300]}])
    gen.save("output.png")

    # 2) Chart.js JSON config 호환
    gen.from_chartjs_config({
        "type": "bar",
        "data": {
            "labels": ["1월","2월","3월"],
            "datasets": [{"label":"매출","data":[120,190,300]}]
        }
    })
    gen.save("output.png")

    # 3) base64 반환 (웹 임베딩용)
    b64 = gen.to_base64()
"""

import io
import os
import json
import base64
import math
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


# ─── 한글 폰트 설정 ───
def _setup_korean_font():
    """시스템에서 한글 폰트를 찾아 matplotlib에 설정"""
    korean_fonts = [
        "NanumGothic", "NanumBarunGothic", "Malgun Gothic",
        "AppleGothic", "Noto Sans KR", "Noto Sans CJK KR",
        "UnDotum", "D2Coding",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in korean_fonts:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_korean_font()

# ─── 기본 색상 팔레트 (app.py 추천 색상과 동일) ───
DEFAULT_COLORS = [
    "#6366f1",  # 인디고
    "#8b5cf6",  # 보라
    "#ec4899",  # 핑크
    "#f59e0b",  # 앰버
    "#10b981",  # 에메랄드
    "#3b82f6",  # 블루
    "#ef4444",  # 레드
    "#84cc16",  # 라임
    "#14b8a6",  # 틸
    "#f97316",  # 오렌지
]


def _hex_to_rgb(hex_color: str):
    """#RRGGBB → (r, g, b) 0~1 float tuple"""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def _hex_to_rgba(hex_color: str, alpha: float = 1.0):
    r, g, b = _hex_to_rgb(hex_color)
    return (r, g, b, alpha)


def _get_colors(n: int, provided: list = None):
    """n개의 색상 반환. provided가 있으면 우선 사용, 부족하면 기본 팔레트에서 보충"""
    colors = []
    for i in range(n):
        if provided and i < len(provided):
            colors.append(provided[i])
        else:
            colors.append(DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
    return colors


class GraphGenerator:
    """matplotlib 기반 그래프 생성기. Chart.js JSON config 호환."""

    def __init__(self, figsize=(10, 6), dpi=100, style="default"):
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
        self.ax = None
        if style and style in plt.style.available:
            plt.style.use(style)

    def _init_figure(self):
        """새 figure/axes 생성"""
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

    def _apply_title(self, title: str = None):
        if title and self.ax:
            self.ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # ════════════════════════════════════════
    #  개별 차트 메서드
    # ════════════════════════════════════════

    def bar(self, labels: list, datasets: list, title: str = None,
            horizontal: bool = False, stacked: bool = False):
        """막대 그래프 (세로/가로)"""
        self._init_figure()
        n_datasets = len(datasets)
        x = np.arange(len(labels))
        width = 0.8 / n_datasets if not stacked else 0.8

        for i, ds in enumerate(datasets):
            color = ds.get("backgroundColor", DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
            if isinstance(color, list):
                color = color[:len(labels)]
            label = ds.get("label", f"시리즈 {i+1}")
            data = ds["data"]

            if stacked:
                bottom = None
                if i > 0:
                    bottom = np.zeros(len(labels))
                    for j in range(i):
                        bottom += np.array(datasets[j]["data"])
                if horizontal:
                    self.ax.barh(x, data, height=width, left=bottom, label=label, color=color)
                else:
                    self.ax.bar(x, data, width=width, bottom=bottom, label=label, color=color)
            else:
                offset = x + (i - n_datasets / 2 + 0.5) * width
                if horizontal:
                    self.ax.barh(offset, data, height=width, label=label, color=color)
                else:
                    self.ax.bar(offset, data, width=width, label=label, color=color)

        if horizontal:
            self.ax.set_yticks(x)
            self.ax.set_yticklabels(labels)
        else:
            self.ax.set_xticks(x)
            self.ax.set_xticklabels(labels, rotation=45 if len(labels) > 6 else 0, ha="right")

        if n_datasets > 1:
            self.ax.legend()
        self._apply_title(title)
        self.fig.tight_layout()
        return self

    def line(self, labels: list, datasets: list, title: str = None,
             fill: bool = False, markers: bool = True):
        """꺾은선 그래프"""
        self._init_figure()
        x = np.arange(len(labels))

        for i, ds in enumerate(datasets):
            color = ds.get("borderColor", ds.get("backgroundColor", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]))
            if isinstance(color, list):
                color = color[0] if color else DEFAULT_COLORS[i]
            label = ds.get("label", f"시리즈 {i+1}")
            data = ds["data"]
            marker = "o" if markers else None
            line_width = ds.get("borderWidth", 2)

            self.ax.plot(x, data, label=label, color=color, marker=marker,
                         linewidth=line_width, markersize=5)
            if fill or ds.get("fill"):
                self.ax.fill_between(x, data, alpha=0.15, color=color)

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels, rotation=45 if len(labels) > 6 else 0, ha="right")
        self.ax.grid(True, alpha=0.3)
        if len(datasets) > 1:
            self.ax.legend()
        self._apply_title(title)
        self.fig.tight_layout()
        return self

    def pie(self, labels: list, datasets: list, title: str = None):
        """원형 그래프"""
        self._init_figure()
        ds = datasets[0]
        data = ds["data"]
        colors = _get_colors(len(data), ds.get("backgroundColor"))

        wedges, texts, autotexts = self.ax.pie(
            data, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.85
        )
        for t in autotexts:
            t.set_fontsize(9)
        self.ax.set_aspect("equal")
        self._apply_title(title)
        self.fig.tight_layout()
        return self

    def doughnut(self, labels: list, datasets: list, title: str = None):
        """도넛 차트"""
        self._init_figure()
        ds = datasets[0]
        data = ds["data"]
        colors = _get_colors(len(data), ds.get("backgroundColor"))

        wedges, texts, autotexts = self.ax.pie(
            data, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.85,
            wedgeprops=dict(width=0.4)
        )
        for t in autotexts:
            t.set_fontsize(9)
        self.ax.set_aspect("equal")
        self._apply_title(title)
        self.fig.tight_layout()
        return self

    def radar(self, labels: list, datasets: list, title: str = None):
        """방사형(레이더) 차트"""
        if self.fig:
            plt.close(self.fig)
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = self.fig.add_subplot(111, polar=True)

        n = len(labels)
        angles = [i / n * 2 * math.pi for i in range(n)]
        angles += angles[:1]  # 닫기

        for i, ds in enumerate(datasets):
            color = ds.get("borderColor", ds.get("backgroundColor", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]))
            if isinstance(color, list):
                color = color[0] if color else DEFAULT_COLORS[i]
            label = ds.get("label", f"시리즈 {i+1}")
            data = list(ds["data"]) + [ds["data"][0]]

            self.ax.plot(angles, data, linewidth=2, label=label, color=color)
            self.ax.fill(angles, data, alpha=0.15, color=color)

        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(labels, fontsize=10)
        if len(datasets) > 1:
            self.ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        self._apply_title(title)
        self.fig.tight_layout()
        return self

    def scatter(self, labels: list, datasets: list, title: str = None):
        """산점도"""
        self._init_figure()

        for i, ds in enumerate(datasets):
            color = ds.get("backgroundColor", DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
            if isinstance(color, list):
                color = color[0] if color else DEFAULT_COLORS[i]
            label = ds.get("label", f"시리즈 {i+1}")
            data = ds["data"]

            if data and isinstance(data[0], dict):
                xs = [p["x"] for p in data]
                ys = [p["y"] for p in data]
            else:
                xs = list(range(len(data)))
                ys = data

            self.ax.scatter(xs, ys, label=label, color=color, s=50, alpha=0.7)

        self.ax.grid(True, alpha=0.3)
        if len(datasets) > 1:
            self.ax.legend()
        self._apply_title(title)
        self.fig.tight_layout()
        return self

    def bubble(self, labels: list, datasets: list, title: str = None):
        """버블 차트"""
        self._init_figure()

        for i, ds in enumerate(datasets):
            color = ds.get("backgroundColor", DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
            if isinstance(color, list):
                color = color[0] if color else DEFAULT_COLORS[i]
            label = ds.get("label", f"시리즈 {i+1}")
            data = ds["data"]

            if data and isinstance(data[0], dict):
                xs = [p["x"] for p in data]
                ys = [p["y"] for p in data]
                rs = [p.get("r", 10) * 20 for p in data]
            else:
                xs = list(range(len(data)))
                ys = data
                rs = [100] * len(data)

            self.ax.scatter(xs, ys, s=rs, label=label, color=color, alpha=0.5, edgecolors="white")

        self.ax.grid(True, alpha=0.3)
        if len(datasets) > 1:
            self.ax.legend()
        self._apply_title(title)
        self.fig.tight_layout()
        return self

    def polar_area(self, labels: list, datasets: list, title: str = None):
        """극좌표 영역 차트 (polarArea)"""
        if self.fig:
            plt.close(self.fig)
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = self.fig.add_subplot(111, polar=True)

        ds = datasets[0]
        data = ds["data"]
        n = len(data)
        colors = _get_colors(n, ds.get("backgroundColor"))

        angles = [i / n * 2 * math.pi for i in range(n)]
        width = 2 * math.pi / n

        bars = self.ax.bar(angles, data, width=width, color=colors, alpha=0.7, edgecolor="white")
        self.ax.set_xticks(angles)
        self.ax.set_xticklabels(labels, fontsize=10)
        self._apply_title(title)
        self.fig.tight_layout()
        return self

    # ════════════════════════════════════════
    #  Chart.js JSON Config 호환 입력
    # ════════════════════════════════════════

    def from_chartjs_config(self, config: dict):
        """
        Chart.js JSON config를 받아서 matplotlib 그래프 생성.
        app.py의 ```chart 코드블록에서 사용하는 동일한 JSON 형식 지원.

        Args:
            config: Chart.js 호환 JSON config dict 또는 JSON 문자열
        """
        if isinstance(config, str):
            raw = config
            raw = raw.replace("//", "")  # 한 줄 주석 간이 제거
            try:
                config = json.loads(raw)
            except json.JSONDecodeError:
                raw = raw.replace(",\n}", "\n}").replace(",\n]", "\n]")
                raw = raw.replace("'", '"')
                config = json.loads(raw)

        chart_type = config.get("type", "bar")
        data = config.get("data", {})
        options = config.get("options", {})

        labels = data.get("labels", [])
        datasets = data.get("datasets", [])

        # title 추출
        title = None
        plugins = options.get("plugins", {})
        title_cfg = plugins.get("title", {})
        if title_cfg.get("display") and title_cfg.get("text"):
            title = title_cfg["text"]

        # 차트 유형 매핑
        type_map = {
            "bar": self.bar,
            "horizontalBar": lambda **kw: self.bar(horizontal=True, **kw),
            "line": self.line,
            "pie": self.pie,
            "doughnut": self.doughnut,
            "radar": self.radar,
            "scatter": self.scatter,
            "bubble": self.bubble,
            "polarArea": self.polar_area,
        }

        # bar + horizontal 옵션 체크
        if chart_type == "bar":
            indexAxis = options.get("indexAxis", "x")
            if indexAxis == "y":
                chart_type = "horizontalBar"

        chart_fn = type_map.get(chart_type, self.bar)

        # stacked 체크
        extra_kwargs = {}
        if chart_type in ("bar", "horizontalBar"):
            scales = options.get("scales", {})
            x_stacked = scales.get("x", {}).get("stacked", False)
            y_stacked = scales.get("y", {}).get("stacked", False)
            if x_stacked or y_stacked:
                extra_kwargs["stacked"] = True

        chart_fn(labels=labels, datasets=datasets, title=title, **extra_kwargs)
        return self

    def from_json_string(self, json_str: str):
        """JSON 문자열로부터 차트 생성 (app.py의 ```chart 블록 내용과 동일)"""
        return self.from_chartjs_config(json_str)

    # ════════════════════════════════════════
    #  출력 메서드
    # ════════════════════════════════════════

    def save(self, filepath: str, transparent: bool = False):
        """그래프를 파일로 저장 (PNG, SVG, PDF, JPG 등)"""
        if not self.fig:
            raise RuntimeError("그래프가 생성되지 않았습니다. 차트 메서드를 먼저 호출하세요.")
        self.fig.savefig(filepath, transparent=transparent, bbox_inches="tight")
        plt.close(self.fig)
        self.fig = None
        self.ax = None
        return filepath

    def to_base64(self, fmt: str = "png") -> str:
        """그래프를 base64 문자열로 반환 (웹 임베딩용)"""
        if not self.fig:
            raise RuntimeError("그래프가 생성되지 않았습니다. 차트 메서드를 먼저 호출하세요.")
        buf = io.BytesIO()
        self.fig.savefig(buf, format=fmt, bbox_inches="tight")
        plt.close(self.fig)
        self.fig = None
        self.ax = None
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def to_bytes(self, fmt: str = "png") -> bytes:
        """그래프를 bytes로 반환"""
        if not self.fig:
            raise RuntimeError("그래프가 생성되지 않았습니다. 차트 메서드를 먼저 호출하세요.")
        buf = io.BytesIO()
        self.fig.savefig(buf, format=fmt, bbox_inches="tight")
        plt.close(self.fig)
        self.fig = None
        self.ax = None
        buf.seek(0)
        return buf.read()

    def show(self):
        """그래프를 화면에 표시 (GUI 환경)"""
        if not self.fig:
            raise RuntimeError("그래프가 생성되지 않았습니다.")
        plt.show()
        self.fig = None
        self.ax = None

    def close(self):
        """현재 figure 닫기"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# ════════════════════════════════════════
#  편의 함수 (빠른 사용)
# ════════════════════════════════════════

def generate_chart(config: dict, output_path: str = None,
                   figsize=(10, 6), dpi=100) -> Optional[str]:
    """
    Chart.js JSON config로 그래프 생성.

    Args:
        config: Chart.js 호환 JSON config (dict 또는 str)
        output_path: 저장 경로. None이면 base64 반환
        figsize: 그래프 크기
        dpi: 해상도

    Returns:
        output_path가 있으면 파일 경로, 없으면 base64 문자열

    Example:
        # 파일 저장
        generate_chart({
            "type": "bar",
            "data": {
                "labels": ["A", "B", "C"],
                "datasets": [{"label": "값", "data": [10, 20, 30]}]
            }
        }, "chart.png")

        # base64 반환
        b64 = generate_chart(config)
    """
    gen = GraphGenerator(figsize=figsize, dpi=dpi)
    gen.from_chartjs_config(config)
    if output_path:
        return gen.save(output_path)
    return gen.to_base64()


def quick_bar(labels, data, title=None, output_path=None, **kwargs):
    """빠른 막대 그래프 생성"""
    gen = GraphGenerator(**kwargs) if kwargs else GraphGenerator()
    gen.bar(labels, [{"label": title or "데이터", "data": data}], title=title)
    return gen.save(output_path) if output_path else gen.to_base64()


def quick_line(labels, data, title=None, output_path=None, **kwargs):
    """빠른 꺾은선 그래프 생성"""
    gen = GraphGenerator(**kwargs) if kwargs else GraphGenerator()
    gen.line(labels, [{"label": title or "데이터", "data": data}], title=title)
    return gen.save(output_path) if output_path else gen.to_base64()


def quick_pie(labels, data, title=None, output_path=None, **kwargs):
    """빠른 원형 그래프 생성"""
    gen = GraphGenerator(**kwargs) if kwargs else GraphGenerator()
    gen.pie(labels, [{"data": data}], title=title)
    return gen.save(output_path) if output_path else gen.to_base64()


# ════════════════════════════════════════
#  CLI 테스트
# ════════════════════════════════════════

if __name__ == "__main__":
    print("=== GraphGenerator 테스트 ===")

    # 테스트 1: 막대 그래프
    gen = GraphGenerator()
    gen.bar(
        labels=["1월", "2월", "3월", "4월"],
        datasets=[
            {"label": "매출", "data": [120, 190, 300, 250], "backgroundColor": "#6366f1"},
            {"label": "비용", "data": [80, 100, 150, 130], "backgroundColor": "#ec4899"},
        ],
        title="월별 매출 vs 비용"
    )
    gen.save("test_bar.png")
    print("  test_bar.png 생성 완료")

    # 테스트 2: Chart.js JSON config
    config = {
        "type": "line",
        "data": {
            "labels": ["1분기", "2분기", "3분기", "4분기"],
            "datasets": [{
                "label": "성장률",
                "data": [2.5, 3.1, 4.2, 3.8],
                "borderColor": "#10b981"
            }]
        },
        "options": {
            "plugins": {"title": {"display": True, "text": "분기별 성장률"}}
        }
    }
    generate_chart(config, "test_line.png")
    print("  test_line.png 생성 완료")

    # 테스트 3: 원형 차트
    quick_pie(["Python", "JS", "Go", "Rust"], [45, 30, 15, 10],
              title="언어별 사용 비율", output_path="test_pie.png")
    print("  test_pie.png 생성 완료")

    print("\n모든 테스트 완료!")
