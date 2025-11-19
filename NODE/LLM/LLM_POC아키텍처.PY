import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# 박스 그리기 함수
def draw_box(ax, x, y, width, height, text, color, fontsize=10, bold=False):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    txt = ax.text(
        x + width/2, y + height/2, text,
        ha='center', va='center',
        fontsize=fontsize,
        weight=weight,
        wrap=True
    )

# 화살표 그리기 함수
def draw_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color='black',
        linewidth=2,
        mutation_scale=20
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ===== 1. 사용자 레이어 =====
draw_box(ax, 3, 12, 4, 1.2, 
         '사용자 질의\n"V7 모델로 예측해줘"\n"왜 급증 예측했어?"\n"과거 유사 케이스 찾아줘"',
         '#E3F2FD', fontsize=10, bold=True)

# ===== 2. 라우팅 레이어 =====
draw_box(ax, 3, 10, 4, 1.2,
         'FastAPI Router\n질의 분류\nsearch | predict | explain',
         '#FFF3E0', fontsize=10, bold=True)

# ===== 3. 처리 엔진 레이어 =====
# RAG
draw_box(ax, 0.5, 7.5, 2.5, 1.5,
         'RAG Engine\n\nLangChain\n+ ChromaDB\n\n과거 이력 검색\n유사도 매칭',
         '#E8F5E9', fontsize=9)

# Prediction
draw_box(ax, 3.5, 7.5, 2.5, 1.5,
         'Prediction\nEngine\n\nXGBoost V7\nXGBoost V6\n기타 모델들',
         '#E8F5E9', fontsize=9)

# Explainability
draw_box(ax, 6.5, 7.5, 2.5, 1.5,
         'Explainability\nEngine\n\nSHAP 분석\nFeature 중요도\n영향도 계산',
         '#E8F5E9', fontsize=9)

# ===== 4. 데이터 레이어 =====
# Vector DB
draw_box(ax, 0.5, 5.5, 2.5, 1.2,
         'Vector DB\n(ChromaDB)\n\n과거 데이터\n문서 임베딩',
         '#FFF9C4', fontsize=9)

# Model Storage
draw_box(ax, 6.5, 5.5, 2.5, 1.2,
         'Model Storage\n\nV7.pkl\nV6.pkl\n기타 모델들',
         '#FFF9C4', fontsize=9)

# ===== 5. LLM 레이어 =====
draw_box(ax, 2, 3, 6, 1.5,
         'Text Generation WebUI\nQwen 80B (H200 GPU)\n\nRAG 답변 생성 | 예측 결과 해석 | SHAP → 자연어 변환',
         '#F3E5F5', fontsize=11, bold=True)

# ===== 6. 응답 레이어 =====
draw_box(ax, 3, 0.8, 4, 1.2,
         '통합 응답\n\n예측값 + 신뢰도 + 주요 원인 설명\n+ 과거 유사 케이스',
         '#E0F7FA', fontsize=10, bold=True)

# ===== 화살표 연결 =====
# 사용자 → Router
draw_arrow(ax, 5, 12, 5, 11.2, 'HTTP Request')

# Router → 엔진들
draw_arrow(ax, 4, 10, 1.75, 9, 'search')
draw_arrow(ax, 5, 10, 4.75, 9, 'predict')
draw_arrow(ax, 6, 10, 7.75, 9, 'explain')

# 엔진 → DB (점선)
draw_arrow(ax, 1.75, 7.5, 1.75, 6.7, 'query', style='-')
draw_arrow(ax, 7.75, 7.5, 7.75, 6.7, 'load', style='-')

# 엔진 → LLM
draw_arrow(ax, 1.75, 7.5, 3.5, 4.5, '검색결과')
draw_arrow(ax, 4.75, 7.5, 5, 4.5, '예측값')
draw_arrow(ax, 7.75, 7.5, 6.5, 4.5, 'SHAP값')

# LLM → 응답
draw_arrow(ax, 5, 3, 5, 2, '자연어\n생성')

# 응답 → 사용자 (리턴)
draw_arrow(ax, 6.5, 1.4, 8, 11.5, 'Response', style='->')

# 제목
ax.text(5, 13.5, 'Text Generation WebUI RAG Architecture', 
        fontsize=16, weight='bold', ha='center')

# 레이어 라벨
ax.text(0.2, 12.5, '① 사용자 레이어', fontsize=9, style='italic', color='blue')
ax.text(0.2, 10.5, '② 라우팅 레이어', fontsize=9, style='italic', color='orange')
ax.text(0.2, 8.5, '③ 처리 엔진 레이어', fontsize=9, style='italic', color='green')
ax.text(0.2, 6.5, '④ 데이터 레이어', fontsize=9, style='italic', color='goldenrod')
ax.text(0.2, 4, '⑤ LLM 레이어', fontsize=9, style='italic', color='purple')
ax.text(0.2, 1.5, '⑥ 응답 레이어', fontsize=9, style='italic', color='teal')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/architecture_diagram.png', dpi=300, bbox_inches='tight')
print("✅ 아키텍처 다이어그램 생성 완료!")