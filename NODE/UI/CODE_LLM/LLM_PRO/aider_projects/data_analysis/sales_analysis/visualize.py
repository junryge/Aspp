#!/usr/bin/env python3
"""매출 데이터 시각화 (matplotlib 사용)"""

import os

# matplotlib 선택적 사용
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.family"] = "Malgun Gothic"  # 한글 폰트
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[경고] matplotlib 미설치 - 텍스트 차트로 대체합니다")

from analyze import load_sales_data, monthly_summary, top_products, category_breakdown


def text_bar_chart(title: str, items: list, max_width: int = 40):
    """텍스트 기반 막대 차트"""
    print(f"\n  {title}")
    print("  " + "-" * (max_width + 20))

    if not items:
        print("  (데이터 없음)")
        return

    max_val = max(v for _, v in items)
    for label, value in items:
        bar_len = int(value / max_val * max_width) if max_val > 0 else 0
        bar = "█" * bar_len
        print(f"  {label:>12s} | {bar} {value:,}")


def plot_monthly_chart(data: list):
    """월별 매출 차트"""
    monthly = monthly_summary(data)
    months = sorted(monthly.keys())
    totals = [monthly[m]["total"] for m in months]

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 5))
        plt.bar(months, totals, color="#6366f1")
        plt.title("월별 매출")
        plt.ylabel("매출 (원)")
        plt.tight_layout()
        plt.savefig("monthly_sales.png", dpi=150)
        print("  [저장] monthly_sales.png")
    else:
        text_bar_chart("월별 매출", list(zip(months, totals)))


def plot_category_pie(data: list):
    """카테고리별 비율 차트"""
    breakdown = category_breakdown(data)

    if HAS_MATPLOTLIB:
        labels = list(breakdown.keys())
        sizes = [breakdown[k]["ratio"] for k in labels]
        colors = ["#6366f1", "#f97316", "#22c55e", "#ef4444"]

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors[:len(labels)])
        plt.title("카테고리별 매출 비율")
        plt.tight_layout()
        plt.savefig("category_pie.png", dpi=150)
        print("  [저장] category_pie.png")
    else:
        items = [(k, v["total"]) for k, v in breakdown.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        text_bar_chart("카테고리별 매출", items)


def plot_top_products(data: list):
    """상위 제품 차트"""
    top = top_products(data, 5)

    if HAS_MATPLOTLIB:
        products = [p for p, _ in top]
        amounts = [a for _, a in top]
        plt.figure(figsize=(10, 5))
        plt.barh(products, amounts, color="#22c55e")
        plt.title("매출 상위 5개 제품")
        plt.xlabel("매출 (원)")
        plt.tight_layout()
        plt.savefig("top_products.png", dpi=150)
        print("  [저장] top_products.png")
    else:
        text_bar_chart("매출 상위 5개 제품", top)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "sales_data.csv")

    if os.path.exists(csv_path):
        sales = load_sales_data(csv_path)
        print("=" * 50)
        print("  매출 데이터 시각화")
        print("=" * 50)
        plot_monthly_chart(sales)
        plot_category_pie(sales)
        plot_top_products(sales)
    else:
        print(f"데이터 파일을 찾을 수 없습니다: {csv_path}")
