import graphviz
from graphviz import Digraph

# 아키텍처 다이어그램 생성
def create_architecture_diagram():
    dot = Digraph(comment='Text Gen WebUI RAG Architecture', format='png')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.0')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Malgun Gothic')
    
    # 스타일 정의
    user_style = {'fillcolor': '#E3F2FD', 'color': '#1976D2', 'fontcolor': '#000000', 'penwidth': '2'}
    router_style = {'fillcolor': '#FFF3E0', 'color': '#F57C00', 'fontcolor': '#000000', 'penwidth': '2'}
    engine_style = {'fillcolor': '#E8F5E9', 'color': '#388E3C', 'fontcolor': '#000000', 'penwidth': '2'}
    llm_style = {'fillcolor': '#F3E5F5', 'color': '#7B1FA2', 'fontcolor': '#000000', 'penwidth': '3'}
    db_style = {'fillcolor': '#FFF9C4', 'color': '#F57F17', 'fontcolor': '#000000', 'penwidth': '2', 'shape': 'cylinder'}
    result_style = {'fillcolor': '#E0F7FA', 'color': '#00838F', 'fontcolor': '#000000', 'penwidth': '2'}
    
    # 노드 생성
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='사용자 레이어', style='dashed', color='blue')
        c.node('user', '사용자 질의\n\n"V7 모델로 예측해줘"\n"왜 급증 예측했어?"\n"과거 유사 케이스 찾아줘"', **user_style)
    
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='라우팅 레이어', style='dashed', color='orange')
        c.node('router', 'FastAPI Router\n\n질의 분류\n- search (검색)\n- predict (예측)\n- explain (설명)', **router_style)
    
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='처리 엔진 레이어', style='dashed', color='green')
        c.node('rag', 'RAG Engine\n\nLangChain + ChromaDB\n- 과거 이력 검색\n- 문서 임베딩\n- 유사도 매칭', **engine_style)
        c.node('predict', 'Prediction Engine\n\n모델 레지스트리\n- XGBoost V7\n- XGBoost V6\n- 기타 모델들', **engine_style)
        c.node('explain', 'Explainability Engine\n\nSHAP 분석\n- Feature 중요도\n- 영향도 계산\n- 상위 N개 추출', **engine_style)
    
    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='데이터 레이어', style='dashed', color='#F57F17')
        c.node('vectordb', 'Vector DB\n(ChromaDB)\n\n과거 데이터\n문서 임베딩', **db_style)
        c.node('modeldb', 'Model Storage\n\nV7: .pkl\nV6: .pkl\n기타 모델들', **db_style)
    
    with dot.subgraph(name='cluster_4') as c:
        c.attr(label='LLM 레이어', style='dashed', color='purple')
        c.node('textgen', 'Text Generation WebUI\n\nQwen 80B (H200 GPU)\n\n- RAG 답변 생성\n- 예측 결과 해석\n- SHAP → 자연어 변환', **llm_style)
    
    with dot.subgraph(name='cluster_5') as c:
        c.attr(label='응답 레이어', style='dashed', color='cyan')
        c.node('response', '통합 응답\n\n예측값 + 신뢰도 +\n주요 원인 설명 +\n과거 유사 케이스', **result_style)
    
    # 엣지 연결
    dot.edge('user', 'router', label='HTTP Request')
    
    dot.edge('router', 'rag', label='search')
    dot.edge('router', 'predict', label='predict')
    dot.edge('router', 'explain', label='explain')
    
    dot.edge('rag', 'vectordb', label='query', style='dashed')
    dot.edge('predict', 'modeldb', label='load', style='dashed')
    dot.edge('explain', 'modeldb', label='load', style='dashed')
    
    dot.edge('rag', 'textgen', label='검색결과 → 답변생성')
    dot.edge('predict', 'textgen', label='예측값 → 해석')
    dot.edge('explain', 'textgen', label='SHAP값 → 설명')
    
    dot.edge('textgen', 'response', label='자연어 생성')
    dot.edge('response', 'user', label='HTTP Response')
    
    return dot

# 다이어그램 생성 및 저장
diagram = create_architecture_diagram()
diagram.render('/mnt/user-data/outputs/text_gen_webui_architecture', view=False, cleanup=True)

print("✅ 아키텍처 다이어그램 생성 완료!")