import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# 고해상도 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

fig = plt.figure(figsize=(20, 14), facecolor='white')
ax = fig.add_subplot(111)
ax.set_xlim(0, 20)
ax.set_ylim(0, 16)
ax.axis('off')

# 전문적인 색상 팔레트
COLORS = {
    'user': '#2196F3',      # Blue
    'router': '#FF9800',    # Orange  
    'rag': '#4CAF50',       # Green
    'predict': '#4CAF50',   
    'explain': '#4CAF50',
    'db': '#FFC107',        # Amber
    'llm': '#9C27B0',       # Purple
    'response': '#00BCD4',  # Cyan
    'bg_light': '#F5F5F5'
}

def draw_modern_box(ax, x, y, width, height, text, color, is_header=False):
    """모던한 박스 디자인"""
    # 그림자 효과
    shadow = FancyBboxPatch(
        (x+0.05, y-0.05), width, height,
        boxstyle="round,pad=0.15",
        facecolor='#CCCCCC',
        edgecolor='none',
        alpha=0.3,
        zorder=1
    )
    ax.add_patch(shadow)
    
    # 메인 박스
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.15",
        facecolor='white',
        edgecolor=color,
        linewidth=3 if is_header else 2.5,
        zorder=2
    )
    ax.add_patch(box)
    
    # 헤더 색상 바
    if is_header:
        header_bar = Rectangle(
            (x, y + height - 0.4), width, 0.4,
            facecolor=color,
            edgecolor='none',
            zorder=3
        )
        ax.add_patch(header_bar)
        
        # 헤더 텍스트 (흰색)
        lines = text.split('\n')
        ax.text(
            x + width/2, y + height - 0.2, lines[0],
            ha='center', va='center',
            fontsize=13, weight='bold',
            color='white',
            zorder=4
        )
        
        # 본문 텍스트
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
        # 일반 박스 텍스트
        ax.text(
            x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=10,
            color='#333333',
            zorder=4,
            linespacing=1.6
        )

def draw_clean_arrow(ax, x1, y1, x2, y2, label='', color='#666666'):
    """깔끔한 화살표"""
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

# ===== 배경 레이어 구분 =====
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

# ===== 1층: 사용자 =====
draw_modern_box(ax, 7, 13.5, 6, 1.2,
    '사용자 질의\n\n"V7 모델로 예측해줘"  •  "왜 급증 예측했어?"  •  "과거 유사 케이스 찾아줘"',
    COLORS['user'], is_header=True)

# ===== 2층: 라우터 =====
draw_modern_box(ax, 7, 11.3, 6, 1.2,
    'FastAPI Router - 질의 분류\n\n질의 유형 자동 판별: SEARCH  |  PREDICT  |  EXPLAIN',
    COLORS['router'], is_header=True)

# ===== 3층: 처리 엔진 (3개) =====
draw_modern_box(ax, 1.5, 8.8, 5, 1.6,
    'RAG Engine\n\nLangChain + ChromaDB\n과거 이력 검색 • 문서 임베딩 • 유사도 매칭',
    COLORS['rag'], is_header=True)

draw_modern_box(ax, 7.5, 8.8, 5, 1.6,
    'Prediction Engine\n\n모델 레지스트리 관리\nXGBoost V7 • V6 • 기타 모델',
    COLORS['predict'], is_header=True)

draw_modern_box(ax, 13.5, 8.8, 5, 1.6,
    'Explainability Engine\n\nSHAP 분석\nFeature 중요도 • 영향도 계산 • 상위 N개 추출',
    COLORS['explain'], is_header=True)

# ===== 4층: 데이터베이스 (2개) =====
draw_modern_box(ax, 2.5, 6.3, 4, 1.2,
    'Vector Database\n\nChromaDB\n과거 데이터 임베딩 저장',
    COLORS['db'], is_header=True)

draw_modern_box(ax, 13.5, 6.3, 4, 1.2,
    'Model Storage\n\n.pkl 파일 저장소\nV7.pkl • V6.pkl • 기타 모델',
    COLORS['db'], is_header=True)

# ===== 5층: LLM (중앙 대형) =====
draw_modern_box(ax, 5, 3.8, 10, 1.6,
    'Text Generation WebUI - Qwen 80B (H200 GPU)\n\nRAG 답변 생성  •  예측 결과 해석  •  SHAP → 자연어 변환',
    COLORS['llm'], is_header=True)

# ===== 6층: 최종 응답 =====
draw_modern_box(ax, 7, 1.3, 6, 1.2,
    '통합 응답\n\n예측값 + 신뢰도 + 주요 원인 설명 + 과거 유사 케이스',
    COLORS['response'], is_header=True)

# ===== 화살표 연결 =====
# User → Router
draw_clean_arrow(ax, 10, 13.5, 10, 12.5, 'HTTP Request', COLORS['user'])

# Router → 3 Engines
draw_clean_arrow(ax, 8.5, 11.3, 4, 10.4, 'SEARCH', COLORS['rag'])
draw_clean_arrow(ax, 10, 11.3, 10, 10.4, 'PREDICT', COLORS['predict'])
draw_clean_arrow(ax, 11.5, 11.3, 16, 10.4, 'EXPLAIN', COLORS['explain'])

# Engines → Databases (점선)
ax.plot([4, 4.5], [8.8, 7.5], '--', color=COLORS['db'], linewidth=2, zorder=5)
ax.text(4.7, 8.1, 'Query', fontsize=8, color=COLORS['db'])

ax.plot([16, 15.5], [8.8, 7.5], '--', color=COLORS['db'], linewidth=2, zorder=5)
ax.text(15.2, 8.1, 'Load', fontsize=8, color=COLORS['db'])

# Engines → LLM
draw_clean_arrow(ax, 4, 8.8, 7, 5.4, '검색 결과', COLORS['rag'])
draw_clean_arrow(ax, 10, 8.8, 10, 5.4, '예측값', COLORS['predict'])
draw_clean_arrow(ax, 16, 8.8, 13, 5.4, 'SHAP 값', COLORS['explain'])

# LLM → Response
draw_clean_arrow(ax, 10, 3.8, 10, 2.5, '자연어 생성', COLORS['llm'])

# Response → User (순환)
ax.annotate('', xy=(12.8, 13.7), xytext=(13, 2.5),
            arrowprops=dict(arrowstyle='-|>', color=COLORS['response'], 
                          lw=2.5, connectionstyle="arc3,rad=0.3"))
ax.text(14, 8, 'HTTP\nResponse', fontsize=9, color=COLORS['response'], 
       weight='bold', ha='center')

# ===== 타이틀 =====
title_bg = Rectangle((3, 15.2), 14, 0.6, 
                     facecolor='#1A237E', edgecolor='none', zorder=10)
ax.add_patch(title_bg)

ax.text(10, 15.5, 'Text Generation WebUI RAG Architecture', 
        fontsize=18, weight='bold', ha='center', color='white', zorder=11)

# ===== 범례 =====
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