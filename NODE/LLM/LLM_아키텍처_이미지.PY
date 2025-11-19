# ===== 3. 엑셀 파일 생성 및 이미지 삽입 =====
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.drawing.image import Image as XLImage

wb = Workbook()

# ========== Sheet 1: Architecture Diagram (이미지) ==========
ws1 = wb.active
ws1.title = "Architecture Diagram"

# 타이틀
ws1.merge_cells('A1:J1')
ws1['A1'] = "Text Generation WebUI RAG Architecture"
ws1['A1'].font = Font(bold=True, color="FFFFFF", size=16)
ws1['A1'].fill = PatternFill(start_color="1A237E", end_color="1A237E", fill_type="solid")
ws1['A1'].alignment = Alignment(horizontal='center', vertical='center')
ws1.row_dimensions[1].height = 35

# 이미지 삽입
img1 = XLImage('/mnt/user-data/outputs/architecture_professional.png')
img1.width = 1200  # 너비 조정
img1.height = 840  # 높이 조정
ws1.add_image(img1, 'A3')

# 열 너비 조정
for col in range(1, 11):
    ws1.column_dimensions[chr(64 + col)].width = 15

# ========== Sheet 2: Data Flow Diagram (이미지) ==========
ws2 = wb.create_sheet("Data Flow Diagram")

# 타이틀
ws2.merge_cells('A1:J1')
ws2['A1'] = "Data Flow Diagram"
ws2['A1'].font = Font(bold=True, color="FFFFFF", size=16)
ws2['A1'].fill = PatternFill(start_color="1A237E", end_color="1A237E", fill_type="solid")
ws2['A1'].alignment = Alignment(horizontal='center', vertical='center')
ws2.row_dimensions[1].height = 35

# 이미지 삽입
img2 = XLImage('/mnt/user-data/outputs/dataflow_professional.png')
img2.width = 1080
img2.height = 720
ws2.add_image(img2, 'A3')

# 열 너비 조정
for col in range(1, 11):
    ws2.column_dimensions[chr(64 + col)].width = 15

# ========== Sheet 3: Architecture Overview (표) ==========
ws3 = wb.create_sheet("Architecture Overview")
ws3.column_dimensions['A'].width = 20
ws3.column_dimensions['B'].width = 35
ws3.column_dimensions['C'].width = 50

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

border = Border(
    left=Side(style='thin'),
    right=Side(style='thin'),
    top=Side(style='thin'),
    bottom=Side(style='thin')
)

# 타이틀
ws3.merge_cells('A1:C1')
ws3['A1'] = "Architecture Overview"
ws3['A1'].font = header_font
ws3['A1'].fill = header_fill
ws3['A1'].alignment = Alignment(horizontal='center', vertical='center')
ws3.row_dimensions[1].height = 30

# 헤더
ws3['A3'] = "Layer"
ws3['B3'] = "Component"
ws3['C3'] = "Description"
for col in ['A', 'B', 'C']:
    ws3[f'{col}3'].font = Font(bold=True, size=12)
    ws3[f'{col}3'].fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
    ws3[f'{col}3'].border = border
    ws3[f'{col}3'].alignment = Alignment(horizontal='center', vertical='center')

# 데이터
data = [
    ("User Layer", "사용자 질의", "• V7 모델로 예측해줘\n• 왜 급증 예측했어?\n• 과거 유사 케이스 찾아줘"),
    ("Routing Layer", "FastAPI Router", "질의 유형 자동 판별\n• SEARCH: 데이터 검색\n• PREDICT: 예측 수행\n• EXPLAIN: 예측 설명"),
    ("Processing Layer", "RAG Engine", "LangChain + ChromaDB\n• 과거 이력 검색\n• 문서 임베딩\n• 유사도 매칭"),
    ("", "Prediction Engine", "모델 레지스트리 관리\n• XGBoost V7 모델\n• XGBoost V6 모델\n• 기타 예측 모델"),
    ("", "Explainability Engine", "SHAP 분석\n• Feature 중요도 계산\n• 영향도 분석\n• 상위 N개 Feature 추출"),
    ("Data Layer", "Vector Database", "ChromaDB\n• 과거 데이터 임베딩 저장\n• 유사도 기반 검색\n• 벡터 인덱싱"),
    ("", "Model Storage", ".pkl 파일 저장소\n• V7.pkl (30분→10분 예측)\n• V6.pkl (기존 모델)\n• 기타 모델 파일"),
    ("LLM Layer", "Text Generation WebUI", "Qwen 80B (H200 GPU)\n• RAG 답변 생성\n• 예측 결과 해석\n• SHAP → 자연어 변환\n• 통합 설명 생성"),
    ("Response Layer", "통합 응답", "• 예측값 + 신뢰도\n• 주요 원인 설명\n• 과거 유사 케이스\n• 자연어 설명")
]

row = 4
for layer, component, description in data:
    if layer:
        ws3[f'A{row}'] = layer
        ws3[f'A{row}'].font = Font(bold=True, size=11)
        ws3[f'A{row}'].alignment = Alignment(horizontal='center', vertical='center')
        
        if 'User' in layer:
            ws3[f'A{row}'].fill = layer_fills['user']
        elif 'Routing' in layer:
            ws3[f'A{row}'].fill = layer_fills['router']
        elif 'Processing' in layer:
            ws3[f'A{row}'].fill = layer_fills['processing']
        elif 'Data' in layer:
            ws3[f'A{row}'].fill = layer_fills['data']
        elif 'LLM' in layer:
            ws3[f'A{row}'].fill = layer_fills['llm']
        elif 'Response' in layer:
            ws3[f'A{row}'].fill = layer_fills['response']
    
    ws3[f'B{row}'] = component
    ws3[f'C{row}'] = description
    
    ws3[f'B{row}'].font = Font(bold=True, size=11)
    ws3[f'B{row}'].alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
    ws3[f'C{row}'].alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
    
    for col in ['A', 'B', 'C']:
        ws3[f'{col}{row}'].border = border
    
    ws3.row_dimensions[row].height = 60
    row += 1

# 셀 병합
ws3.merge_cells('A4:A4')
ws3.merge_cells('A5:A5')
ws3.merge_cells('A6:A8')
ws3.merge_cells('A9:A10')
ws3.merge_cells('A11:A11')
ws3.merge_cells('A12:A12')

# ========== Sheet 4: Technical Specs ==========
ws4 = wb.create_sheet("Technical Specs")
ws4.column_dimensions['A'].width = 25
ws4.column_dimensions['B'].width = 60

ws4.merge_cells('A1:B1')
ws4['A1'] = "Technical Specifications"
ws4['A1'].font = header_font
ws4['A1'].fill = header_fill
ws4['A1'].alignment = Alignment(horizontal='center', vertical='center')
ws4.row_dimensions[1].height = 30

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
    ws4[f'A{row}'] = spec_name
    ws4[f'B{row}'] = spec_value
    
    ws4[f'A{row}'].font = Font(bold=True, size=11)
    ws4[f'A{row}'].alignment = Alignment(horizontal='left', vertical='center')
    ws4[f'B{row}'].alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
    
    for col in ['A', 'B']:
        ws4[f'{col}{row}'].border = border
    
    ws4.row_dimensions[row].height = 30
    row += 1

# 저장
wb.save('/mnt/user-data/outputs/architecture_documentation.xlsx')
print("✅ 엑셀 문서 생성 완료! (이미지 포함)")
print("\n📦 생성된 파일:")
print("  • architecture_documentation.xlsx")
print("    - Sheet 1: Architecture Diagram (이미지)")
print("    - Sheet 2: Data Flow Diagram (이미지)")
print("    - Sheet 3: Architecture Overview (표)")
print("    - Sheet 4: Technical Specs (표)")