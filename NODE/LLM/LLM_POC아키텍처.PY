# 데이터 플로우 다이어그램
fig2 = plt.figure(figsize=(18, 12), facecolor='white')
ax2 = fig2.add_subplot(111)
ax2.set_xlim(0, 18)
ax2.set_ylim(0, 12)
ax2.axis('off')

# 섹션 구분 배경
predict_bg = Rectangle((0.5, 5.5), 17, 5.5, 
                       facecolor='#E8F5E9', edgecolor='#4CAF50', 
                       linewidth=2, alpha=0.3, zorder=0)
ax2.add_patch(predict_bg)

rag_bg = Rectangle((0.5, 0.5), 17, 4, 
                   facecolor='#E3F2FD', edgecolor='#2196F3', 
                   linewidth=2, alpha=0.3, zorder=0)
ax2.add_patch(rag_bg)

# 섹션 타이틀
ax2.text(9, 10.7, '예측 + 설명 플로우', fontsize=14, weight='bold', 
        ha='center', color='#2E7D32',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                 edgecolor='#4CAF50', linewidth=2))

ax2.text(9, 4.2, 'RAG 검색 플로우', fontsize=14, weight='bold', 
        ha='center', color='#1565C0',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                 edgecolor='#2196F3', linewidth=2))

# ===== 예측 플로우 =====
boxes_predict = [
    (1, 8, 2.5, 1.2, '입력 데이터\n\n30분 시퀀스\n데이터', '#BBDEFB'),
    (4.5, 8, 2.5, 1.2, 'Feature 생성\n\nMomentum\nhub/storage/cmd\nsurge_risk_score', '#C8E6C9'),
    (8, 8, 2.5, 1.2, 'XGBoost V7\n\n예측 수행\nConfidence 계산', '#FFECB3'),
    (8, 6, 2.5, 1.2, 'SHAP 분석\n\nFeature 영향도\n상위 10개 추출', '#F8BBD0'),
    (11.5, 7, 2.5, 1.4, 'Qwen 80B\n\nSHAP → 자연어\n"hub_high로 인한\n병목 발생"', '#E1BEE7'),
    (15, 7, 2.5, 1.4, '최종 응답\n\n예측: 305.2\n신뢰도: 87%\n주요 원인 3개', '#B2DFDB')
]

for x, y, w, h, text, color in boxes_predict:
    draw_modern_box(ax2, x, y, w, h, text, color, is_header=False)

# 화살표
draw_clean_arrow(ax2, 3.5, 8.6, 4.5, 8.6, '', '#666666')
draw_clean_arrow(ax2, 7, 8.6, 8, 8.6, 'Predict', '#666666')
draw_clean_arrow(ax2, 10.5, 8.6, 11.5, 7.8, 'Result', '#666666')
draw_clean_arrow(ax2, 10.5, 6.6, 11.5, 7.2, 'SHAP', '#666666')
draw_clean_arrow(ax2, 14, 7.7, 15, 7.7, 'Generate', '#666666')

# 분석 화살표
ax2.plot([9.25, 9.25], [8, 7.2], color='#666666', linewidth=2, zorder=5)
ax2.text(9.6, 7.6, 'Analyze', fontsize=8, color='#666666')

# ===== RAG 플로우 =====
boxes_rag = [
    (1, 2, 2.5, 1.2, '사용자 질문\n\n"과거 유사\n케이스는?"', '#BBDEFB'),
    (4.5, 2, 2.5, 1.2, '임베딩 변환\n\nko-sroberta\n512차원 벡터', '#C8E6C9'),
    (8, 2, 2.5, 1.2, 'Vector DB\n검색\n\nTop-5\n유사도 매칭', '#FFECB3'),
    (11.5, 2, 2.5, 1.2, 'Qwen 80B\n\nContext +\nQuery로\n답변 생성', '#E1BEE7'),
    (15, 2, 2.5, 1.2, 'RAG 응답\n\n유사 케이스:\n2024-09-15\n14:23:00', '#B2DFDB')
]

for x, y, w, h, text, color in boxes_rag:
    draw_modern_box(ax2, x, y, w, h, text, color, is_header=False)

# 화살표
draw_clean_arrow(ax2, 3.5, 2.6, 4.5, 2.6, '', '#666666')
draw_clean_arrow(ax2, 7, 2.6, 8, 2.6, 'Embed', '#666666')
draw_clean_arrow(ax2, 10.5, 2.6, 11.5, 2.6, 'Docs', '#666666')
draw_clean_arrow(ax2, 14, 2.6, 15, 2.6, 'Answer', '#666666')

# 메인 타이틀
title_bg2 = Rectangle((3, 11.4), 12, 0.5, 
                      facecolor='#1A237E', edgecolor='none', zorder=10)
ax2.add_patch(title_bg2)
ax2.text(9, 11.65, 'Data Flow Diagram', 
        fontsize=16, weight='bold', ha='center', color='white', zorder=11)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/dataflow_professional.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
print("✅ 데이터 플로우 다이어그램 생성 완료!")
plt.close()

print("\n📦 생성된 파일 목록:")
print("  1. architecture_professional.png - 아키텍처 다이어그램")
print("  2. dataflow_professional.png - 데이터 플로우 다이어그램")
print("  3. architecture_documentation.xlsx - 엑셀 문서")