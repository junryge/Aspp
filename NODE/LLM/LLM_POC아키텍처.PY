from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# 워크북 생성
wb = Workbook()
ws = wb.active
ws.title = "Architecture Overview"

# 스타일 정의
header_fill = PatternFill(start_color="1A237E", end_color="1A237E", fill_type="solid")
header_font = Font(bold=True, color="FFFFFF", size=14)

layer_fills = {
    'user': PatternFill(start_color="E3F2FD", end_color="E3F2FD", fill_type="solid"),
    'router': PatternFill(start_color="FFF3E0", end_color="FFF3E0", fill_type="solid"),
    'processing': PatternFill(start_color="E8F5E9", end_color="E8F5E9", fill_type="solid"),
    'data': PatternFill(start_color="FFF9C4", end_color="FFF9C4", fill_type="solid"),
    'llm': PatternFill(start_color="F3E5F5", end_color="F3E5F5", fill_type="solid"),
    'response': PatternFill(start_color="E0F7FA", end_color="E0F7FA", fill_type="solid"),
}

component_font = Font(bold=True, size=11)
border = Border(
    left=Side(style='thin'),
    right=Side(style='thin'),
    top=Side(style='thin'),
    bottom=Side(style='thin')
)

# 열 너비 설정
ws.column_dimensions['A'].width = 20
ws.column_dimensions['B'].width = 35
ws.column_dimensions['C'].width = 50

# 타이틀
ws.merge_cells('A1:C1')
ws['A1'] = "Text Generation WebUI RAG Architecture"
ws['A1'].font = header_font
ws['A1'].fill = header_fill
ws['A1'].alignment = Alignment(horizontal='center', vertical='center')
ws.row_dimensions[1].height = 30

# 헤더
row = 3
ws['A3'] = "Layer"
ws['B3'] = "Component"
ws['C3'] = "Description"
for col in ['A', 'B', 'C']:
    ws[f'{col}3'].font = Font(bold=True, size=12)
    ws[f'{col}3'].fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
    ws[f'{col}3'].border = border
    ws[f'{col}3'].alignment = Alignment(horizontal='center', vertical='center')

# 데이터 입력
data = [
    # Layer 1: User
    ("User Layer", "사용자 질의", "• V7 모델로 예측해줘\n• 왜 급증 예측했어?\n• 과거 유사 케이스 찾아줘"),
    
    # Layer 2: Router
    ("Routing Layer", "FastAPI Router", "질의 유형 자동 판별\n• SEARCH: 데이터 검색\n• PREDICT: 예측 수행\n• EXPLAIN: 예측 설명"),
    
    # Layer 3: Processing
    ("Processing Layer", "RAG Engine", "LangChain + ChromaDB\n• 과거 이력 검색\n• 문서 임베딩\n• 유사도 매칭"),
    ("", "Prediction Engine", "모델 레지스트리 관리\n• XGBoost V7 모델\n• XGBoost V6 모델\n• 기타 예측 모델"),
    ("", "Explainability Engine", "SHAP 분석\n• Feature 중요도 계산\n• 영향도 분석\n• 상위 N개 Feature 추출"),
    
    # Layer 4: Data
    ("Data Layer", "Vector Database", "ChromaDB\n• 과거 데이터 임베딩 저장\n• 유사도 기반 검색\n• 벡터 인덱싱"),
    ("", "Model Storage", ".pkl 파일 저장소\n• V7.pkl (30분→10분 예측)\n• V6.pkl (기존 모델)\n• 기타 모델 파일"),
    
    # Layer 5: LLM
    ("LLM Layer", "Text Generation WebUI", "Qwen 80B (H200 GPU)\n• RAG 답변 생성\n• 예측 결과 해석\n• SHAP → 자연어 변환\n• 통합 설명 생성"),
    
    # Layer 6: Response
    ("Response Layer", "통합 응답", "• 예측값 + 신뢰도\n• 주요 원인 설명\n• 과거 유사 케이스\n• 자연어 설명")
]

row = 4
current_layer = None
for layer, component, description in data:
    # Layer 병합 처리
    if layer and layer != current_layer:
        layer_start_row = row
        current_layer = layer
        ws[f'A{row}'] = layer
        ws[f'A{row}'].font = Font(bold=True, size=11)
        ws[f'A{row}'].alignment = Alignment(horizontal='center', vertical='center')
        
        # Layer별 색상
        if 'User' in layer:
            ws[f'A{row}'].fill = layer_fills['user']
        elif 'Routing' in layer:
            ws[f'A{row}'].fill = layer_fills['router']
        elif 'Processing' in layer:
            ws[f'A{row}'].fill = layer_fills['processing']
        elif 'Data' in layer:
            ws[f'A{row}'].fill = layer_fills['data']
        elif 'LLM' in layer:
            ws[f'A{row}'].fill = layer_fills['llm']
        elif 'Response' in layer:
            ws[f'A{row}'].fill = layer_fills['response']
    
    # Component & Description
    ws[f'B{row}'] = component
    ws[f'C{row}'] = description
    
    ws[f'B{row}'].font = component_font
    ws[f'B{row}'].alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
    ws[f'C{row}'].alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
    
    # 테두리
    for col in ['A', 'B', 'C']:
        ws[f'{col}{row}'].border = border
    
    # 행 높이
    ws.row_dimensions[row].height = 60
    
    row += 1

# Layer 셀 병합
merge_ranges = [
    ('A4:A4', 'user'),      # User Layer
    ('A5:A5', 'router'),    # Routing Layer
    ('A6:A8', 'processing'),# Processing Layer (3행)
    ('A9:A10', 'data'),     # Data Layer (2행)
    ('A11:A11', 'llm'),     # LLM Layer
    ('A12:A12', 'response') # Response Layer
]

for merge_range, layer_type in merge_ranges:
    if ':' in merge_range:
        ws.merge_cells(merge_range)
        first_cell = merge_range.split(':')[0]
        ws[first_cell].fill = layer_fills[layer_type]

# 두 번째 시트: Data Flow
ws2 = wb.create_sheet("Data Flow")
ws2.column_dimensions['A'].width = 15
ws2.column_dimensions['B'].width = 40
ws2.column_dimensions['C'].width = 40

# 타이틀
ws2.merge_cells('A1:C1')
ws2['A1'] = "Data Flow Diagram"
ws2['A1'].font = header_font
ws2['A1'].fill = header_fill
ws2['A1'].alignment = Alignment(horizontal='center', vertical='center')
ws2.row_dimensions[1].height = 30

# Flow 1: 예측 + 설명
row = 3
ws2.merge_cells(f'A{row}:C{row}')
ws2[f'A{row}'] = "Flow 1: 예측 + 설명 플로우"
ws2[f'A{row}'].font = Font(bold=True, size=13, color="2E7D32")
ws2[f'A{row}'].fill = PatternFill(start_color="E8F5E9", end_color="E8F5E9", fill_type="solid")
ws2[f'A{row}'].alignment = Alignment(horizontal='center', vertical='center')
ws2[f'A{row}'].border = border

flow1_data = [
    ("Step 1", "입력 데이터", "30분 시퀀스 데이터"),
    ("Step 2", "Feature 생성", "Momentum Features\nhub/storage/cmd flags\nsurge_risk_score"),
    ("Step 3", "XGBoost V7 예측", "예측 수행 + Confidence 계산"),
    ("Step 4", "SHAP 분석", "Feature 영향도 분석\n상위 10개 추출"),
    ("Step 5", "Qwen 80B 해석", "SHAP → 자연어 변환\n'hub_high로 인한 병목 발생'"),
    ("Step 6", "최종 응답", "예측값: 305.2\n신뢰도: 87%\n주요 원인 3개")
]

row = 4
for step, action, detail in flow1_data:
    ws2[f'A{row}'] = step
    ws2[f'B{row}'] = action
    ws2[f'C{row}'] = detail
    
    ws2[f'A{row}'].font = Font(bold=True)
    ws2[f'A{row}'].alignment = Alignment(horizontal='center', vertical='center')
    ws2[f'B{row}'].font = Font(bold=True)
    ws2[f'B{row}'].alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
    ws2[f'C{row}'].alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
    
    for col in ['A', 'B', 'C']:
        ws2[f'{col}{row}'].border = border
    
    ws2.row_dimensions[row].height = 45
    row += 1

# Flow 2: RAG 검색
row += 1
ws2.merge_cells(f'A{row}:C{row}')
ws2[f'A{row}'] = "Flow 2: RAG 검색 플로우"
ws2[f'A{row}'].font = Font(bold=True, size=13, color="1565C0")
ws2[f'A{row}'].fill = PatternFill(start_color="E3F2FD", end_color="E3F2FD", fill_type="solid")
ws2[f'A{row}'].alignment = Alignment(horizontal='center', vertical='center')
ws2[f'A{row}'].border = border

flow2_data = [
    ("Step 1", "사용자 질문", "과거 유사 케이스는?"),
    ("Step 2", "임베딩 변환", "ko-sroberta\n512차원 벡터"),
    ("Step 3", "Vector DB 검색", "Top-5 유사도 매칭"),
    ("Step 4", "Qwen 80B 답변", "Context + Query로 답변 생성"),
    ("Step 5", "RAG 응답", "유사 케이스: 2024-09-15 14:23:00")
]

row += 1
for step, action, detail in flow2_data:
    ws2[f'A{row}'] = step
    ws2[f'B{row}'] = action
    ws2[f'C{row}'] = detail
    
    ws2[f'A{row}'].font = Font(bold=True)
    ws2[f'A{row}'].alignment = Alignment(horizontal='center', vertical='center')
    ws2[f'B{row}'].font = Font(bold=True)
    ws2[f'B{row}'].alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
    ws2[f'C{row}'].alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
    
    for col in ['A', 'B', 'C']:
        ws2[f'{col}{row}'].border = border
    
    ws2.row_dimensions[row].height = 40
    row += 1

# 세 번째 시트: Technical Specs
ws3 = wb.create_sheet("Technical Specs")
ws3.column_dimensions['A'].width = 25
ws3.column_dimensions['B'].width = 60

# 타이틀
ws3.merge_cells('A1:B1')
ws3['A1'] = "Technical Specifications"
ws3['A1'].font = header_font
ws3['A1'].fill = header_fill
ws3['A1'].alignment = Alignment(horizontal='center', vertical='center')
ws3.row_dimensions[1].height = 30

specs = [
    ("LLM Model", "Qwen 80B (Text Generation WebUI)"),
    ("GPU", "H200 (4개 분산: Dev, Staging, Prod)"),
    ("Vector DB", "ChromaDB"),
    ("RAG Framework", "LangChain"),
    ("Embedding Model", "ko-sroberta-multitask"),
    ("Prediction Models", "XGBoost V7, V6, 기타"),
    ("Explainability", "SHAP (TreeExplainer)"),
    ("API Framework", "FastAPI"),
    ("Feature Engineering", "Momentum, hub/storage/cmd flags, surge_risk_score"),
    ("Model Files", "V7.pkl (30분→10분 예측), V6.pkl (기존 모델)")
]

row = 3
for spec_name, spec_value in specs:
    ws3[f'A{row}'] = spec_name
    ws3[f'B{row}'] = spec_value
    
    ws3[f'A{row}'].font = Font(bold=True, size=11)
    ws3[f'A{row}'].alignment = Alignment(horizontal='left', vertical='center')
    ws3[f'B{row}'].alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
    
    for col in ['A', 'B']:
        ws3[f'{col}{row}'].border = border
    
    ws3.row_dimensions[row].height = 30
    row += 1

# 저장
wb.save('/mnt/user-data/outputs/architecture_documentation.xlsx')
print("✅ 엑셀 문서 생성 완료!")