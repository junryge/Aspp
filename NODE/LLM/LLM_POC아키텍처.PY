import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# 고해상도 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# ===== 1. 아키텍처 다이어그램 생성 =====
fig = plt.figure(figsize=(20, 14), facecolor='white')
ax = fig.add_subplot(111)
ax.set_xlim(0, 20)
ax.set_ylim(0, 16)
ax.axis('off')

COLORS = {
    'user': '#2196F3',
    'router': '#FF9800',
    'rag': '#4CAF50',
    'predict': '#4CAF50',
    'explain': '#4CAF50',
    'db': '#FFC107',
    'llm': '#9C27B0',
    'response': '#00BCD4',
}

def draw_modern_box(ax, x, y, width, height, text, color, is_header=False):
    shadow = FancyBboxPatch(
        (x+0.05, y-0.05), width, height,
        boxstyle="round,pad=0.15",
        facecolor='#CCCCCC',
        edgecolor='none',
        alpha=0.3,
        zorder=1
    )
    ax.add_patch(shadow)
    
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.15",
        facecolor='white',
        edgecolor=color,
        linewidth=3 if is_header else 2.5,
        zorder=2
    )
    ax.add_patch(box)
    
    if is_header:
        header_bar = Rectangle(
            (x, y + height - 0.4), width, 0.4,
            facecolor=color,
            edgecolor='none',
            zorder=3
        )
        ax.add_patch(header_bar)
        
        lines = text.split('\n')
        ax.text(
            x + width/2, y + height - 0.2, lines[0],
            ha='center', va='center',
            fontsize=13, weight='bold',
            color='white',
            zorder=4
        )
        
        body_text = '\n'.join(lines[1:])
        ax.text(
            x + width/2, y + height/2 - 0.3, body_text,
            ha='center', va='center',
            fontsize=10,
            color='#333333',
            zorder=4,
            linespacing=1.6
        )
    else:
        ax.text(
            x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=10,
            color='#333333',
            zorder=4,
            linespacing=1.6
        )

def draw_clean_arrow(ax, x1, y1, x2, y2, label='', color='#666666'):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>',
        color=color,
        linewidth=2.5,
        mutation_scale=25,
        zorder=5,
        connectionstyle="arc3,rad=0"
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mid_x, mid_y + 0.25, label,
            fontsize=9,
            color=color,
            weight='bold',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor=color, linewidth=1.5),
            zorder=6
        )

# 배경 레이어
layer_y = [14.5, 12.3, 9.8, 7.3, 4.8, 2.3]
layer_names = ['User Layer', 'Routing Layer', 'Processing Layer', 'Data Layer', 'LLM Layer', 'Response Layer']
layer_colors = ['#E3F2FD', '#FFF3E0', '#E8F5E9', '#FFF9C4', '#F3E5F5', '#E0F7FA']

for i, (y, name, col) in enumerate(zip(layer_y, layer_names, layer_colors)):
    height = 1.5 if i == 2 else 1.3
    bg = Rectangle((0.5, y - height), 19, height,
                   facecolor=col, edgecolor='none', alpha=0.15, zorder=0)
    ax.add_patch(bg)
    ax.text(0.7, y - height + 0.2, name,
           fontsize=10, style='italic', weight='bold', color='#555555')

# 박스들
draw_modern_box(ax, 7, 13.5, 6, 1.2,
    '사용자 질의\n\n"V7 모델로 예측해줘"  •  "왜 급증 예측했어?"  •  "과거 유사 케이스 찾아줘"',
    COLORS['user'], is_header=True)

draw_modern_box(ax, 7, 11.3, 6, 1.2,
    'FastAPI Router - 질의 분류\n\n질의 유형 자동 판별: SEARCH  |  PREDICT  |  EXPLAIN',
    COLORS['router'], is_header=True)

draw_modern_box(ax, 1.5, 8.8, 5, 1.6,
    'RAG Engine\n\nLangChain + ChromaDB\n과거 이력 검색 • 문서 임베딩 • 유사도 매칭',
    COLORS['rag'], is_header=True)

draw_modern_box(ax, 7.5, 8.8, 5, 1.6,
    'Prediction Engine\n\n모델 레지스트리 관리\nXGBoost V7 • V6 • 기타 모델',
    COLORS['predict'], is_header=True)

draw_modern_box(ax, 13.5, 8.8, 5, 1.6,
    'Explainability Engine\n\nSHAP 분석\nFeature 중요도 • 영향도 계산 • 상위 N개 추출',
    COLORS['explain'], is_header=True)

draw_modern_box(ax, 2.5, 6.3, 4, 1.2,
    'Vector Database\n\nChromaDB\n과거 데이터 임베딩 저장',
    COLORS['db'], is_header=True)

draw_modern_box(ax, 13.5, 6.3, 4, 1.2,
    'Model Storage\n\n.pkl 파일 저장소\nV7.pkl • V6.pkl • 기타 모델',
    COLORS['db'], is_header=True)

draw_modern_box(ax, 5, 3.8, 10, 1.6,
    'Text Generation WebUI - Qwen 80B (H200 GPU)\n\nRAG 답변 생성  •  예측 결과 해석  •  SHAP → 자연어 변환',
    COLORS['llm'], is_header=True)

draw_modern_box(ax, 7, 1.3, 6, 1.2,
    '통합 응답\n\n예측값 + 신뢰도 + 주요 원인 설명 + 과거 유사 케이스',
    COLORS['response'], is_header=True)

# 화살표들
draw_clean_arrow(ax, 10, 13.5, 10, 12.5, 'HTTP Request', COLORS['user'])
draw_clean_arrow(ax, 8.5, 11.3, 4, 10.4, 'SEARCH', COLORS['rag'])
draw_clean_arrow(ax, 10, 11.3, 10, 10.4, 'PREDICT', COLORS['predict'])
draw_clean_arrow(ax, 11.5, 11.3, 16, 10.4, 'EXPLAIN', COLORS['explain'])

ax.plot([4, 4.5], [8.8, 7.5], '--', color=COLORS['db'], linewidth=2, zorder=5)
ax.text(4.7, 8.1, 'Query', fontsize=8, color=COLORS['db'])
ax.plot([16, 15.5], [8.8, 7.5], '--', color=COLORS['db'], linewidth=2, zorder=5)
ax.text(15.2, 8.1, 'Load', fontsize=8, color=COLORS['db'])

draw_clean_arrow(ax, 4, 8.8, 7, 5.4, '검색 결과', COLORS['rag'])
draw_clean_arrow(ax, 10, 8.8, 10, 5.4, '예측값', COLORS['predict'])
draw_clean_arrow(ax, 16, 8.8, 13, 5.4, 'SHAP 값', COLORS['explain'])
draw_clean_arrow(ax, 10, 3.8, 10, 2.5, '자연어 생성', COLORS['llm'])

ax.annotate('', xy=(12.8, 13.7), xytext=(13, 2.5),
            arrowprops=dict(arrowstyle='-|>', color=COLORS['response'],
                          lw=2.5, connectionstyle="arc3,rad=0.3"))
ax.text(14, 8, 'HTTP\nResponse', fontsize=9, color=COLORS['response'],
       weight='bold', ha='center')

# 타이틀
title_bg = Rectangle((3, 15.2), 14, 0.6,
                     facecolor='#1A237E', edgecolor='none', zorder=10)
ax.add_patch(title_bg)
ax.text(10, 15.5, 'Text Generation WebUI RAG Architecture',
        fontsize=18, weight='bold', ha='center', color='white', zorder=11)

legend_x = 0.8
legend_y = 0.5
ax.text(legend_x, legend_y + 0.3, '● Architecture Overview',
       fontsize=10, weight='bold', color='#333333')
ax.text(legend_x, legend_y, '  └ 6-Layer 구조로 설계된 통합 예측/검색 시스템',
       fontsize=8, color='#666666')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/architecture_professional.png',
           dpi=300, bbox_inches='tight', facecolor='white')
print("✅ 아키텍처 다이어그램 생성 완료!")
plt.close()

# ===== 2. 데이터 플로우 다이어그램 생성 =====
fig2 = plt.figure(figsize=(18, 12), facecolor='white')
ax2 = fig2.add_subplot(111)
ax2.set_xlim(0, 18)
ax2.set_ylim(0, 12)
ax2.axis('off')

predict_bg = Rectangle((0.5, 5.5), 17, 5.5,
                       facecolor='#E8F5E9', edgecolor='#4CAF50',
                       linewidth=2, alpha=0.3, zorder=0)
ax2.add_patch(predict_bg)

rag_bg = Rectangle((0.5, 0.5), 17, 4,
                   facecolor='#E3F2FD', edgecolor='#2196F3',
                   linewidth=2, alpha=0.3, zorder=0)
ax2.add_patch(rag_bg)

ax2.text(9, 10.7, '예측 + 설명 플로우', fontsize=14, weight='bold',
        ha='center', color='#2E7D32',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                 edgecolor='#4CAF50', linewidth=2))

ax2.text(9, 4.2, 'RAG 검색 플로우', fontsize=14, weight='bold',
        ha='center', color='#1565C0',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                 edgecolor='#2196F3', linewidth=2))

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

draw_clean_arrow(ax2, 3.5, 8.6, 4.5, 8.6, '', '#666666')
draw_clean_arrow(ax2, 7, 8.6, 8, 8.6, 'Predict', '#666666')
draw_clean_arrow(ax2, 10.5, 8.6, 11.5, 7.8, 'Result', '#666666')
draw_clean_arrow(ax2, 10.5, 6.6, 11.5, 7.2, 'SHAP', '#666666')
draw_clean_arrow(ax2, 14, 7.7, 15, 7.7, 'Generate', '#666666')

ax2.plot([9.25, 9.25], [8, 7.2], color='#666666', linewidth=2, zorder=5)
ax2.text(9.6, 7.6, 'Analyze', fontsize=8, color='#666666')

boxes_rag = [
    (1, 2, 2.5, 1.2, '사용자 질문\n\n"과거 유사\n케이스는?"', '#BBDEFB'),
    (4.5, 2, 2.5, 1.2, '임베딩 변환\n\nko-sroberta\n512차원 벡터', '#C8E6C9'),
    (8, 2, 2.5, 1.2, 'Vector DB\n검색\n\nTop-5\n유사도 매칭', '#FFECB3'),
    (11.5, 2, 2.5, 1.2, 'Qwen 80B\n\nContext +\nQuery로\n답변 생성', '#E1BEE7'),
    (15, 2, 2.5, 1.2, 'RAG 응답\n\n유사 케이스:\n2024-09-15\n14:23:00', '#B2DFDB')
]

for x, y, w, h, text, color in boxes_rag:
    draw_modern_box(ax2, x, y, w, h, text, color, is_header=False)

draw_clean_arrow(ax2, 3.5, 2.6, 4.5, 2.6, '', '#666666')
draw_clean_arrow(ax2, 7, 2.6, 8, 2.6, 'Embed', '#666666')
draw_clean_arrow(ax2, 10.5, 2.6, 11.5, 2.6, 'Docs', '#666666')
draw_clean_arrow(ax2, 14, 2.6, 15, 2.6, 'Answer', '#666666')

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