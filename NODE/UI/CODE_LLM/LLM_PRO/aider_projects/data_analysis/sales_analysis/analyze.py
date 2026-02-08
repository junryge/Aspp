#!/usr/bin/env python3
"""월별 매출 데이터 분석"""

import csv
import os
from collections import defaultdict


def load_sales_data(filepath: str) -> list:
    """CSV 매출 데이터 로드"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["amount"] = int(row["amount"])
            row["quantity"] = int(row["quantity"])
            data.append(row)
    return data


def monthly_summary(data: list) -> dict:
    """월별 매출 집계"""
    summary = defaultdict(lambda: {"total": 0, "count": 0, "items": 0})
    for row in data:
        month = row["date"][:7]  # YYYY-MM
        summary[month]["total"] += row["amount"]
        summary[month]["count"] += 1
        summary[month]["items"] += row["quantity"]
    return dict(summary)


def top_products(data: list, n: int = 5) -> list:
    """매출 상위 제품"""
    product_sales = defaultdict(int)
    for row in data:
        product_sales[row["product"]] += row["amount"]

    sorted_products = sorted(product_sales.items(), key=lambda x: x[1], reverse=True)
    return sorted_products[:n]


def category_breakdown(data: list) -> dict:
    """카테고리별 매출 비율"""
    cat_total = defaultdict(int)
    grand_total = 0
    for row in data:
        cat_total[row["category"]] += row["amount"]
        grand_total += row["amount"]

    result = {}
    for cat, total in cat_total.items():
        result[cat] = {
            "total": total,
            "ratio": round(total / grand_total * 100, 1) if grand_total > 0 else 0,
        }
    return result


def print_report(data: list):
    """분석 리포트 출력"""
    print("=" * 50)
    print("  월별 매출 분석 리포트")
    print("=" * 50)

    # 월별 요약
    monthly = monthly_summary(data)
    print("\n[월별 매출]")
    for month in sorted(monthly.keys()):
        info = monthly[month]
        print(f"  {month}: {info['total']:>12,}원  ({info['count']}건, {info['items']}개)")

    # 상위 제품
    print("\n[매출 상위 5개 제품]")
    for i, (product, amount) in enumerate(top_products(data), 1):
        print(f"  {i}. {product}: {amount:>12,}원")

    # 카테고리
    print("\n[카테고리별 비율]")
    for cat, info in sorted(category_breakdown(data).items(), key=lambda x: x[1]["total"], reverse=True):
        print(f"  {cat}: {info['total']:>12,}원 ({info['ratio']}%)")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "sales_data.csv")

    if os.path.exists(csv_path):
        sales = load_sales_data(csv_path)
        print_report(sales)
    else:
        print(f"데이터 파일을 찾을 수 없습니다: {csv_path}")
