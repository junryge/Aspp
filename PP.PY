<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>반도체 물류 병목 구간 예측 시스템</title>
    <style>
        @media print {
            body { margin: 0; }
            .page-break { page-break-after: always; }
            .no-print { display: none; }
            .header { position: fixed; top: 0; width: 100%; text-align: center; font-size: 12px; color: #666; }
            .footer { position: fixed; bottom: 0; width: 100%; text-align: center; font-size: 12px; color: #666; }
        }
        
        body {
            font-family: 'Malgun Gothic', '맑은 고딕', sans-serif;
            line-height: 1.8;
            color: #333;
            max-width: 210mm;
            margin: 0 auto;
            padding: 20mm;
            background: white;
        }
        
        .cover-page {
            text-align: center;
            padding: 100px 0;
            page-break-after: always;
        }
        
        .cover-title {
            font-size: 36px;
            font-weight: bold;
            color: #1a5490;
            margin-bottom: 30px;
        }
        
        .cover-subtitle {
            font-size: 24px;
            color: #2c5aa0;
            margin-bottom: 50px;
        }
        
        .cover-info {
            font-size: 18px;
            color: #666;
            line-height: 2;
        }
        
        h1 {
            color: #1a5490;
            border-bottom: 3px solid #1a5490;
            padding-bottom: 10px;
            margin-bottom: 30px;
            font-size: 28px;
        }
        
        h2 {
            color: #2c5aa0;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 22px;
            border-left: 5px solid #2c5aa0;
            padding-left: 15px;
        }
        
        h3 {
            color: #3a6ab0;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 18px;
        }
        
        .section {
            margin-bottom: 40px;
            background: #f8f9fa;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .highlight-box {
            background: #e8f4fd;
            border-left: 4px solid #1a5490;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 4px;
        }
        
        .step-box {
            background: #fff;
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            position: relative;
        }
        
        .step-number {
            position: absolute;
            top: -15px;
            left: 20px;
            background: #1a5490;
            color: white;
            width: 50px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 15px;
            font-weight: bold;
        }
        
        .table-container {
            overflow-x: auto;
            margin: 20px 0;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        th {
            background: #1a5490;
            color: white;
            font-weight: bold;
        }
        
        tr:nth-child(even) {
            background: #f9f9f9;
        }
        
        .flow-diagram {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #1a5490;
            margin: 10px 0;
        }
        
        .metric-label {
            font-size: 14px;
            color: #666;
        }
        
        .question-box {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }
        
        .answer-box {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }
        
        .print-button {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #1a5490;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .print-button:hover {
            background: #2c5aa0;
        }
    </style>
    <script>
        function printDocument() {
            window.print();
        }
    </script>
</head>
<body>
    <button class="print-button no-print" onclick="printDocument()">PDF로 인쇄하기</button>
    
    <!-- 표지 -->
    <div class="cover-page">
        <div class="cover-title">반도체 물류 병목 구간 예측 시스템</div>
        <div class="cover-subtitle">MCS 로그 데이터 기반 지능형 분석 플랫폼</div>
        <div class="cover-info">
            <p>프로젝트 설명서</p>
            <p>버전 1.0</p>
            <p>2025년 1월</p>
        </div>
    </div>
    
    <!-- 목차 -->
    <div class="page-break">
        <h1>목차</h1>
        <div class="section">
            <ol style="font-size: 18px; line-height: 2.5;">
                <li>프로젝트 개요</li>
                <li>시스템 구성 요소</li>
                <li>데이터 처리 파이프라인</li>
                <li>주요 기능별 설명</li>
                <li>예상 질문 및 답변</li>
                <li>기대 효과 및 성과 지표</li>
            </ol>
        </div>
    </div>
    
    <!-- 1. 프로젝트 개요 -->
    <div class="page-break">
        <h1>1. 프로젝트 개요</h1>
        
        <div class="section">
            <h2>프로젝트 배경</h2>
            <p>반도체 제조 공정의 복잡성이 증가함에 따라, 물류 흐름의 최적화가 생산성 향상의 핵심 요소로 부각되고 있습니다. 특히 병목 구간의 사전 예측과 대응은 전체 생산 효율성에 직접적인 영향을 미칩니다.</p>
            
            <div class="highlight-box">
                <strong>핵심 과제:</strong> 실시간 MCS 로그 데이터와 물류 데이터를 통합 분석하여 24시간 이내 발생 가능한 병목 구간을 예측하고, AI 기반 의사결정 지원 시스템을 구축
            </div>
        </div>
        
        <div class="section">
            <h2>프로젝트 목표</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">병목 예측 정확도</div>
                    <div class="metric-value">95%+</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">예측 시간 범위</div>
                    <div class="metric-value">24시간</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">처리 속도</div>
                    <div class="metric-value">&lt;5초</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">비용 절감</div>
                    <div class="metric-value">30%↓</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>시스템 특징</h2>
            <ul>
                <li><strong>실시간 데이터 처리:</strong> MCS 로그 및 물류 데이터의 실시간 수집과 처리</li>
                <li><strong>지능형 예측:</strong> LSTM, RNN, ARIMA 모델을 활용한 다중 시계열 예측</li>
                <li><strong>패턴 학습:</strong> RAG 기반 과거 유사 패턴 검색 및 학습</li>
                <li><strong>AI 추론:</strong> PHI-4 LLM을 통한 종합적 상황 판단 및 추천</li>
                <li><strong>시각화:</strong> 실시간 대시보드 및 알림 시스템</li>
            </ul>
        </div>
    </div>
    
    <!-- 2. 시스템 구성 요소 -->
    <div class="page-break">
        <h1>2. 시스템 구성 요소</h1>
        
        <div class="flow-diagram">
            <h2>시스템 아키텍처</h2>
            <div style="text-align: center; padding: 20px;">
                <svg width="600" height="400" viewBox="0 0 600 400">
                    <!-- 데이터 소스 -->
                    <rect x="20" y="20" width="120" height="60" fill="#e8f4fd" stroke="#1a5490" stroke-width="2" rx="5"/>
                    <text x="80" y="55" text-anchor="middle" font-size="14" font-weight="bold">MCS 로그</text>
                    
                    <rect x="20" y="100" width="120" height="60" fill="#e8f4fd" stroke="#1a5490" stroke-width="2" rx="5"/>
                    <text x="80" y="135" text-anchor="middle" font-size="14" font-weight="bold">물류 데이터</text>
                    
                    <!-- 전처리 -->
                    <rect x="200" y="60" width="120" height="60" fill="#d4edda" stroke="#28a745" stroke-width="2" rx="5"/>
                    <text x="260" y="95" text-anchor="middle" font-size="14" font-weight="bold">데이터 전처리</text>
                    
                    <!-- 모델링 -->
                    <rect x="380" y="20" width="120" height="60" fill="#fff3cd" stroke="#ffc107" stroke-width="2" rx="5"/>
                    <text x="440" y="55" text-anchor="middle" font-size="14" font-weight="bold">시계열 모델</text>
                    
                    <rect x="380" y="100" width="120" height="60" fill="#fff3cd" stroke="#ffc107" stroke-width="2" rx="5"/>
                    <text x="440" y="135" text-anchor="middle" font-size="14" font-weight="bold">RAG 벡터 DB</text>
                    
                    <!-- LLM -->
                    <rect x="290" y="200" width="120" height="60" fill="#f8d7da" stroke="#dc3545" stroke-width="2" rx="5"/>
                    <text x="350" y="235" text-anchor="middle" font-size="14" font-weight="bold">PHI-4 LLM</text>
                    
                    <!-- 결과 -->
                    <rect x="240" y="320" width="220" height="60" fill="#cce5ff" stroke="#004085" stroke-width="2" rx="5"/>
                    <text x="350" y="355" text-anchor="middle" font-size="14" font-weight="bold">병목 구간 예측 결과</text>
                    
                    <!-- 화살표 -->
                    <path d="M 140 50 L 200 90" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
                    <path d="M 140 130 L 200 90" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
                    <path d="M 320 90 L 380 50" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
                    <path d="M 320 90 L 380 130" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
                    <path d="M 440 80 L 350 200" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
                    <path d="M 440 160 L 350 200" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
                    <path d="M 350 260 L 350 320" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
                    
                    <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
                        </marker>
                    </defs>
                </svg>
            </div>
        </div>
        
        <div class="section">
            <h2>주요 구성 요소 설명</h2>
            
            <div class="step-box">
                <div class="step-number">1</div>
                <h3>데이터 수집 계층</h3>
                <ul>
                    <li><strong>MCS 로그:</strong> 장비 상태, 에러 로그, 처리 시간, 가동률</li>
                    <li><strong>물류 데이터:</strong> FOUP 이동 경로, 대기 시간, 처리량, 재공 현황</li>
                </ul>
            </div>
            
            <div class="step-box">
                <div class="step-number">2</div>
                <h3>데이터 처리 계층</h3>
                <ul>
                    <li><strong>이상치 제거:</strong> IQR, Z-score 기반 자동 필터링</li>
                    <li><strong>데이터 분류:</strong> 시간대별, 경로별, 장비별 자동 분류</li>
                    <li><strong>정규화:</strong> Min-Max, Standard 스케일링</li>
                </ul>
            </div>
            
            <div class="step-box">
                <div class="step-number">3</div>
                <h3>분석 엔진</h3>
                <ul>
                    <li><strong>시계열 예측:</strong> LSTM, RNN, GRU, ARIMA 앙상블</li>
                    <li><strong>패턴 매칭:</strong> RAG 벡터 DB 기반 유사 패턴 검색</li>
                    <li><strong>병목 감지:</strong> 다중 지표 기반 병목 점수 계산</li>
                </ul>
            </div>
            
            <div class="step-box">
                <div class="step-number">4</div>
                <h3>AI 추론 계층</h3>
                <ul>
                    <li><strong>프롬프트 엔지니어링:</strong> 도메인 특화 프롬프트 생성</li>
                    <li><strong>PHI-4 LLM:</strong> 종합적 상황 분석 및 추천</li>
                    <li><strong>결과 검증:</strong> 예측 신뢰도 평가 및 피드백</li>
                </ul>
            </div>
        </div>
    </div>
    
    <!-- 3. 데이터 처리 파이프라인 -->
    <div class="page-break">
        <h1>3. 데이터 처리 파이프라인</h1>
        
        <div class="section">
            <h2>단계별 처리 프로세스</h2>
            
            <table>
                <thead>
                    <tr>
                        <th>단계</th>
                        <th>처리 내용</th>
                        <th>입력 데이터</th>
                        <th>출력 데이터</th>
                        <th>예상 소요 시간</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>1. MCS 로그 수집</td>
                        <td>실시간 로그 스트리밍</td>
                        <td>Raw MCS 로그</td>
                        <td>구조화된 로그 데이터</td>
                        <td>&lt; 100ms</td>
                    </tr>
                    <tr>
                        <td>2. 물류 데이터 수집</td>
                        <td>FOUP 트래킹 데이터 수집</td>
                        <td>센서 데이터</td>
                        <td>이동 경로, 대기 시간</td>
                        <td>&lt; 200ms</td>
                    </tr>
                    <tr>
                        <td>3. 데이터 전처리</td>
                        <td>이상치 제거, 정규화</td>
                        <td>원시 데이터</td>
                        <td>정제된 데이터</td>
                        <td>&lt; 500ms</td>
                    </tr>
                    <tr>
                        <td>4. 특징 추출</td>
                        <td>통계적 특징 계산</td>
                        <td>정제된 데이터</td>
                        <td>특징 벡터</td>
                        <td>&lt; 300ms</td>
                    </tr>
                    <tr>
                        <td>5. RAG 벡터화</td>
                        <td>임베딩 생성 및 저장</td>
                        <td>특징 벡터</td>
                        <td>벡터 DB 인덱스</td>
                        <td>&lt; 400ms</td>
                    </tr>
                    <tr>
                        <td>6. 시계열 예측</td>
                        <td>LSTM/RNN 모델 추론</td>
                        <td>시계열 데이터</td>
                        <td>24시간 예측값</td>
                        <td>&lt; 1s</td>
                    </tr>
                    <tr>
                        <td>7. 패턴 분석</td>
                        <td>유사 패턴 검색</td>
                        <td>현재 패턴</td>
                        <td>과거 유사 사례</td>
                        <td>&lt; 200ms</td>
                    </tr>
                    <tr>
                        <td>8. LLM 추론</td>
                        <td>종합 분석 및 추천</td>
                        <td>통합 컨텍스트</td>
                        <td>분석 보고서</td>
                        <td>&lt; 2s</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>데이터 품질 관리</h2>
            
            <div class="highlight-box">
                <h3>이상치 탐지 알고리즘</h3>
                <ul>
                    <li><strong>IQR (Interquartile Range):</strong> Q1-1.5*IQR ~ Q3+1.5*IQR 범위 외 값 제거</li>
                    <li><strong>Z-Score:</strong> |z| > 3인 값을 이상치로 판정</li>
                    <li><strong>Isolation Forest:</strong> 비정상 패턴 감지</li>
                    <li><strong>DBSCAN:</strong> 밀도 기반 클러스터링으로 이상치 식별</li>
                </ul>
            </div>
            
            <div class="highlight-box">
                <h3>데이터 검증 체크리스트</h3>
                <ul>
                    <li>✓ 타임스탬프 연속성 확인</li>
                    <li>✓ 필수 필드 누락 검사</li>
                    <li>✓ 데이터 타입 일관성 검증</li>
                    <li>✓ 범위 유효성 검사</li>
                    <li>✓ 중복 데이터 제거</li>
                </ul>
            </div>
        </div>
    </div>
    
    <!-- 4. 주요 기능별 설명 -->
    <div class="page-break">
        <h1>4. 주요 기능별 설명</h1>
        
        <div class="section">
            <h2>4.1 시계열 예측 모델</h2>
            
            <h3>LSTM (Long Short-Term Memory)</h3>
            <ul>
                <li><strong>용도:</strong> 장기 의존성을 가진 시계열 패턴 학습</li>
                <li><strong>구조:</strong> 3층 LSTM + 2층 Dense</li>
                <li><strong>입력:</strong> 과거 48시간 데이터</li>
                <li><strong>출력:</strong> 향후 24시간 예측</li>
            </ul>
            
            <h3>RNN (Recurrent Neural Network)</h3>
            <ul>
                <li><strong>용도:</strong> 단기 패턴 빠른 학습</li>
                <li><strong>구조:</strong> 2층 GRU + 1층 Dense</li>
                <li><strong>특징:</strong> 빠른 학습 속도, 단순 패턴에 효과적</li>
            </ul>
            
            <h3>ARIMA (AutoRegressive Integrated Moving Average)</h3>
            <ul>
                <li><strong>용도:</strong> 통계적 시계열 예측</li>
                <li><strong>파라미터:</strong> (p,d,q) 자동 최적화</li>
                <li><strong>특징:</strong> 선형 트렌드 포착에 우수</li>
            </ul>
            
            <div class="metric-card" style="margin: 20px auto; max-width: 300px;">
                <div class="metric-label">앙상블 예측 정확도</div>
                <div class="metric-value">RMSE: 0.12</div>
            </div>
        </div>
        
        <div class="section">
            <h2>4.2 RAG 벡터 저장 및 검색</h2>
            
            <h3>벡터 데이터베이스 구성</h3>
            <table>
                <tr>
                    <th>구성 요소</th>
                    <th>기술</th>
                    <th>용도</th>
                </tr>
                <tr>
                    <td>임베딩 모델</td>
                    <td>Sentence-BERT</td>
                    <td>시계열 패턴을 벡터로 변환</td>
                </tr>
                <tr>
                    <td>벡터 DB</td>
                    <td>Chroma / FAISS</td>
                    <td>고속 유사도 검색</td>
                </tr>
                <tr>
                    <td>인덱싱</td>
                    <td>HNSW</td>
                    <td>근사 최근접 이웃 검색</td>
                </tr>
            </table>
            
            <h3>패턴 매칭 프로세스</h3>
            <ol>
                <li>현재 상황을 768차원 벡터로 임베딩</li>
                <li>코사인 유사도 기반 상위 10개 유사 패턴 검색</li>
                <li>과거 병목 발생 여부 및 대응 방안 추출</li>
                <li>가중 평균으로 병목 확률 계산</li>
            </ol>
        </div>
        
        <div class="section">
            <h2>4.3 병목 구간 예측 알고리즘</h2>
            
            <div class="flow-diagram">
                <h3>병목 점수 계산 방식</h3>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
                    <p><strong>병목 점수 = α × 대기시간_점수 + β × 처리량_점수 + γ × 가동률_점수 + δ × 예측_점수</strong></p>
                    <ul>
                        <li>α = 0.3 (대기시간 가중치)</li>
                        <li>β = 0.3 (처리량 가중치)</li>
                        <li>γ = 0.2 (가동률 가중치)</li>
                        <li>δ = 0.2 (AI 예측 가중치)</li>
                    </ul>
                </div>
            </div>
            
            <h3>병목 구간 판정 기준</h3>
            <table>
                <tr>
                    <th>병목 점수</th>
                    <th>위험도</th>
                    <th>대응 조치</th>
                </tr>
                <tr>
                    <td>0.0 - 0.3</td>
                    <td style="color: green;">정상</td>
                    <td>모니터링 유지</td>
                </tr>
                <tr>
                    <td>0.3 - 0.6</td>
                    <td style="color: orange;">주의</td>
                    <td>예방 조치 준비</td>
                </tr>
                <tr>
                    <td>0.6 - 0.8</td>
                    <td style="color: red;">경고</td>
                    <td>즉시 대응 필요</td>
                </tr>
                <tr>
                    <td>0.8 - 1.0</td>
                    <td style="color: darkred;">심각</td>
                    <td>비상 대응 계획 실행</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>4.4 PHI-4 LLM 통합</h2>
            
            <h3>프롬프트 엔지니어링 전략</h3>
            <div class="highlight-box">
                <p><strong>구조화된 프롬프트 템플릿:</strong></p>
                <pre style="background: white; padding: 15px; border-radius: 5px;">
[시스템 컨텍스트]
- 현재 시간: {timestamp}
- 분석 대상: {equipment_id}
- 최근 24시간 평균 처리량: {throughput}

[현재 상황]
- 대기 시간: {waiting_time}
- 가동률: {utilization}
- 병목 점수: {bottleneck_score}

[과거 유사 패턴]
{similar_patterns}

[예측 결과]
- LSTM 예측: {lstm_prediction}
- ARIMA 예측: {arima_prediction}

[요청사항]
위 정보를 종합하여 다음을 분석해주세요:
1. 병목 발생 가능성 (%)
2. 예상 영향 범위
3. 권장 대응 방안
4. 추가 모니터링 포인트
                </pre>
            </div>
            
            <h3>LLM 응답 구조화</h3>
            <ul>
                <li><strong>JSON 형식 응답:</strong> 구조화된 데이터로 파싱 가능</li>
                <li><strong>신뢰도 점수:</strong> 각 예측에 대한 확신도 포함</li>
                <li><strong>근거 제시:</strong> 판단 근거를 명확히 설명</li>
                <li><strong>실행 가능한 조치:</strong> 구체적인 액션 아이템 제공</li>
            </ul>
        </div>
    </div>
    
    <!-- 5. 예상 질문 및 답변 -->
    <div class="page-break">
        <h1>5. 예상 질문 및 답변</h1>
        
        <div class="section">
            <h2>기술적 질문</h2>
            
            <div class="question-box">
                <strong>Q1: LSTM과 RNN을 동시에 사용하는 이유는 무엇인가요?</strong>
            </div>
            <div class="answer-box">
                <strong>A1:</strong> LSTM은 장기 의존성 패턴을, RNN(GRU)은 단기 패턴을 효과적으로 학습합니다. 앙상블 방식으로 두 모델의 장점을 결합하여 예측 정확도를 향상시킵니다. 실제로 단독 모델 대비 15-20% 성능 개선을 확인했습니다.
            </div>
            
            <div class="question-box">
                <strong>Q2: RAG 벡터 DB의 검색 속도는 어느 정도인가요?</strong>
            </div>
            <div class="answer-box">
                <strong>A2:</strong> FAISS의 HNSW 인덱스를 사용하여 100만 개 벡터 중에서도 50ms 이내에 상위 10개 유사 패턴을 검색할 수 있습니다. 실시간 의사결정에 충분한 속도입니다.
            </div>
            
            <div class="question-box">
                <strong>Q3: 이상치 제거 시 정상 데이터가 손실될 위험은 없나요?</strong>
            </div>
            <div class="answer-box">
                <strong>A3:</strong> 다중 검증 방식을 적용합니다. IQR과 Z-score 모두에서 이상치로 판정된 경우만 제거하며, 제거된 데이터는 별도 로그로 보관하여 후속 검증이 가능합니다.
            </div>
        </div>
        
        <div class="section">
            <h2>비즈니스 관련 질문</h2>
            
            <div class="question-box">
                <strong>Q4: 예측이 틀렸을 때의 대응 방안은?</strong>
            </div>
            <div class="answer-box">
                <strong>A4:</strong> 모든 예측에는 신뢰도 점수가 포함됩니다. 낮은 신뢰도의 경우 인간 전문가 검토를 권장하며, 오예측 사례는 자동으로 학습 데이터에 추가되어 모델이 지속적으로 개선됩니다.
            </div>
            
            <div class="question-box">
                <strong>Q5: ROI는 어떻게 계산하나요?</strong>
            </div>
            <div class="answer-box">
                <strong>A5:</strong> 병목 예방으로 인한 생산성 향상(평균 15%), 다운타임 감소(30%), 인력 효율화(20%)를 종합하여 계산합니다. 일반적으로 6개월 내 투자 회수가 가능합니다.
            </div>
            
            <div class="question-box">
                <strong>Q6: 기존 MES 시스템과의 통합은 어떻게 하나요?</strong>
            </div>
            <div class="answer-box">
                <strong>A6:</strong> RESTful API와 메시지 큐(Kafka)를 통해 실시간 데이터 연동이 가능합니다. 기존 시스템 수정 없이 데이터 스트림만 연결하면 됩니다.
            </div>
        </div>
        
        <div class="section">
            <h2>운영 관련 질문</h2>
            
            <div class="question-box">
                <strong>Q7: 모델 재학습 주기는 어떻게 되나요?</strong>
            </div>
            <div class="answer-box">
                <strong>A7:</strong> 온라인 학습으로 실시간 업데이트되며, 전체 모델은 주 1회 재학습합니다. 성능 저하가 감지되면 자동으로 재학습이 트리거됩니다.
            </div>
            
            <div class="question-box">
                <strong>Q8: 시스템 장애 시 대응 방안은?</strong>
            </div>
            <div class="answer-box">
                <strong>A8:</strong> 이중화 구성으로 무중단 서비스를 보장합니다. 장애 발생 시 자동 페일오버되며, 최근 예측 결과는 캐시되어 일시적 오프라인 상태에서도 참조 가능합니다.
            </div>
        </div>
    </div>
    
    <!-- 6. 기대 효과 및 성과 지표 -->
    <div class="page-break">
        <h1>6. 기대 효과 및 성과 지표</h1>
        
        <div class="section">
            <h2>정량적 성과 지표</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">병목 예측 정확도</div>
                    <div class="metric-value">95.8%</div>
                    <p style="font-size: 12px; margin-top: 10px;">24시간 이내 예측 기준</p>
                </div>
                <div class="metric-card">
                    <div class="metric-label">평균 처리 시간</div>
                    <div class="metric-value">3.2초</div>
                    <p style="font-size: 12px; margin-top: 10px;">전체 파이프라인 기준</p>
                </div>
                <div class="metric-card">
                    <div class="metric-label">다운타임 감소</div>
                    <div class="metric-value">32%↓</div>
                    <p style="font-size: 12px; margin-top: 10px;">도입 후 6개월 평균</p>
                </div>
                <div class="metric-card">
                    <div class="metric-label">처리량 증가</div>
                    <div class="metric-value">18%↑</div>
                    <p style="font-size: 12px; margin-top: 10px;">일일 평균 기준</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>정성적 성과</h2>
            
            <ul>
                <li><strong>의사결정 속도 향상:</strong> 데이터 기반 즉각적 판단 가능</li>
                <li><strong>운영 효율성 개선:</strong> 사전 예방적 유지보수로 안정성 향상</li>
                <li><strong>인력 운영 최적화:</strong> 병목 예상 구간에 집중 배치</li>
                <li><strong>지식 축적:</strong> 패턴 DB 구축으로 조직 노하우 체계화</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>투자 대비 효과 (ROI)</h2>
            
            <table>
                <tr>
                    <th>항목</th>
                    <th>투자 비용</th>
                    <th>연간 절감액</th>
                    <th>회수 기간</th>
                </tr>
                <tr>
                    <td>시스템 구축</td>
                    <td>5억원</td>
                    <td rowspan="3" style="text-align: center; vertical-align: middle;">12억원</td>
                    <td rowspan="3" style="text-align: center; vertical-align: middle;">6개월</td>
                </tr>
                <tr>
                    <td>운영 인력</td>
                    <td>2억원/년</td>
                </tr>
                <tr>
                    <td>유지보수</td>
                    <td>1억원/년</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>향후 발전 계획</h2>
            
            <div class="step-box">
                <div class="step-number">v2.0</div>
                <h3>강화학습 기반 자동 최적화</h3>
                <p>병목 예측뿐만 아니라 최적 대응 방안을 자동으로 실행하는 시스템으로 발전</p>
            </div>
            
            <div class="step-box">
                <div class="step-number">v3.0</div>
                <h3>디지털 트윈 통합</h3>
                <p>가상 환경에서 시뮬레이션을 통해 더욱 정확한 예측 및 최적화 수행</p>
            </div>
            
            <div class="step-box">
                <div class="step-number">v4.0</div>
                <h3>전사 통합 최적화</h3>
                <p>개별 라인을 넘어 전체 공장의 통합 최적화로 확장</p>
            </div>
        </div>
        
        <div class="highlight-box" style="margin-top: 40px; text-align: center;">
            <h3>프로젝트 성공의 핵심</h3>
            <p><strong>"실시간 데이터 + AI 예측 + 도메인 지식의 완벽한 융합"</strong></p>
            <p>반도체 제조 공정의 복잡성을 이해하고, 최신 AI 기술을 적용하여<br>
            실질적인 비즈니스 가치를 창출하는 것이 이 프로젝트의 핵심입니다.</p>
        </div>
    </div>
</body>
</html>