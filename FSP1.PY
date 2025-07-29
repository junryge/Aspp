<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>반도체 물류 병목 예측 시스템 - 예상 결과 보고서</title>
    <style>
        @media print {
            body { 
                margin: 0;
                font-size: 11pt;
            }
            .page-break { page-break-after: always; }
            .no-print { display: none !important; }
            .print-section { page-break-inside: avoid; }
            h1, h2 { page-break-after: avoid; }
            table { page-break-inside: avoid; }
            .header-info {
                position: running(header);
                font-size: 10pt;
                text-align: center;
                color: #666;
            }
            @page {
                margin: 15mm 20mm;
                @top-center {
                    content: element(header);
                }
                @bottom-center {
                    content: counter(page) " / " counter(pages);
                    font-size: 10pt;
                    color: #666;
                }
            }
        }
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Malgun Gothic', '맑은 고딕', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            color: #2c3e50;
            max-width: 210mm;
            margin: 0 auto;
            padding: 20mm;
            background: #f5f6fa;
        }
        
        .container {
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        /* 헤더 스타일 */
        .report-header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 30px;
            border-bottom: 3px solid #3498db;
        }
        
        .report-title {
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .report-subtitle {
            font-size: 20px;
            color: #7f8c8d;
            margin-bottom: 20px;
        }
        
        .report-date {
            font-size: 16px;
            color: #95a5a6;
        }
        
        /* 섹션 스타일 */
        h1 {
            color: #2c3e50;
            font-size: 28px;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }
        
        h2 {
            color: #34495e;
            font-size: 22px;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-left: 10px;
            border-left: 4px solid #3498db;
        }
        
        h3 {
            color: #34495e;
            font-size: 18px;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        /* 상태 표시 */
        .status-critical {
            color: #e74c3c;
            font-weight: bold;
        }
        
        .status-warning {
            color: #f39c12;
            font-weight: bold;
        }
        
        .status-normal {
            color: #27ae60;
            font-weight: bold;
        }
        
        /* 테이블 스타일 */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        th {
            background: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        
        tr:nth-child(even) {
            background: #f8f9fa;
        }
        
        tr:hover {
            background: #e8f4f8;
        }
        
        /* 알림 박스 */
        .alert-box {
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            border-left: 5px solid;
        }
        
        .alert-critical {
            background: #fee;
            border-color: #e74c3c;
            color: #c0392b;
        }
        
        .alert-warning {
            background: #fff3cd;
            border-color: #f39c12;
            color: #856404;
        }
        
        .alert-info {
            background: #d1ecf1;
            border-color: #3498db;
            color: #0c5460;
        }
        
        .alert-success {
            background: #d4edda;
            border-color: #27ae60;
            color: #155724;
        }
        
        /* 메트릭 카드 */
        .metrics-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-value {
            font-size: 36px;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .metric-label {
            font-size: 14px;
            opacity: 0.9;
        }
        
        .metric-trend {
            font-size: 16px;
            margin-top: 5px;
        }
        
        /* 프로세스 박스 */
        .process-box {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            position: relative;
        }
        
        .process-number {
            position: absolute;
            top: -15px;
            left: 20px;
            background: #3498db;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        
        /* RAG 검색 결과 */
        .rag-result {
            background: #f0f7ff;
            border: 1px solid #b8daff;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }
        
        .similarity-badge {
            display: inline-block;
            background: #007bff;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            margin-bottom: 10px;
        }
        
        /* 권고사항 */
        .recommendation {
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 15px 20px;
            margin: 15px 0;
        }
        
        .recommendation-urgent {
            background: #ffebee;
            border-left-color: #f44336;
        }
        
        /* 코드/데이터 블록 */
        .code-block {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            overflow-x: auto;
            margin: 20px 0;
        }
        
        /* 인쇄 버튼 */
        .print-button {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #3498db;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s;
            z-index: 1000;
        }
        
        .print-button:hover {
            background: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        }
        
        /* 진행 상황 표시 */
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }
        
        /* 플로우 다이어그램 */
        .flow-diagram {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 30px 0;
            text-align: center;
        }
        
        /* 강조 텍스트 */
        .highlight {
            background: #fff3cd;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
        }
        
        .text-critical { color: #e74c3c; }
        .text-warning { color: #f39c12; }
        .text-success { color: #27ae60; }
        .text-info { color: #3498db; }
    </style>
    <script>
        function printReport() {
            window.print();
        }
    </script>
</head>
<body>
    <button class="print-button no-print" onclick="printReport()">📄 PDF로 인쇄하기</button>
    
    <div class="container">
        <!-- 보고서 헤더 -->
        <div class="report-header">
            <h1 class="report-title">🏭 반도체 물류 병목 구간 예측 시스템</h1>
            <p class="report-subtitle">실시간 예상 결과 보고서</p>
            <p class="report-date">2025년 1월 29일 14:30 기준</p>
        </div>
        
        <!-- 종합 요약 -->
        <div class="alert-box alert-critical">
            <h2 style="margin-top: 0; border: none; padding: 0;">⚠️ 긴급 상황 요약</h2>
            <p><strong>전체 시스템 상태:</strong> <span class="status-critical">경고</span></p>
            <p><strong>주요 위험 구간:</strong> STK_03 (3번 자동창고) - 포화도 95%</p>
            <p><strong>병목 발생 예상:</strong> 2.5시간 이내 (17:00경)</p>
            <p><strong>예상 영향:</strong> 전체 생산량 20-25% 감소</p>
        </div>
        
        <!-- 핵심 메트릭 -->
        <div class="metrics-container">
            <div class="metric-card" style="background: linear-gradient(135deg, #e74c3c, #c0392b);">
                <div class="metric-label">병목 위험도</div>
                <div class="metric-value">87%</div>
                <div class="metric-trend">↑ 15%</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #f39c12, #d68910);">
                <div class="metric-label">STK_03 포화도</div>
                <div class="metric-value">95%</div>
                <div class="metric-trend">↑ 8%/시간</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #3498db, #2980b9);">
                <div class="metric-label">평균 대기시간</div>
                <div class="metric-value">18.5분</div>
                <div class="metric-trend">↑ 42%</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #27ae60, #229954);">
                <div class="metric-label">OHT 가동률</div>
                <div class="metric-value">87%</div>
                <div class="metric-trend">과부하</div>
            </div>
        </div>
        
        <!-- 물류 현황 -->
        <h1>1. 실시간 물류 이동 현황</h1>
        
        <h2>1.1 주요 물류 경로별 상태</h2>
        <table>
            <thead>
                <tr>
                    <th>이동 경로</th>
                    <th>일일 이동량</th>
                    <th>현재 대기</th>
                    <th>평균 이동시간</th>
                    <th>정상 시간</th>
                    <th>병목 위험도</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>포토(Photo) → 식각(Etch)</td>
                    <td>450 FOUP</td>
                    <td>23개</td>
                    <td class="text-warning">12.3분</td>
                    <td>10분</td>
                    <td><span class="status-critical">🔴 높음</span></td>
                </tr>
                <tr>
                    <td>식각(Etch) → 증착(CVD)</td>
                    <td>425 FOUP</td>
                    <td>18개</td>
                    <td>10.5분</td>
                    <td>9분</td>
                    <td><span class="status-warning">🟡 중간</span></td>
                </tr>
                <tr>
                    <td>증착(CVD) → 노광(Litho)</td>
                    <td>410 FOUP</td>
                    <td class="text-critical">31개</td>
                    <td class="text-critical">15.8분</td>
                    <td>11분</td>
                    <td><span class="status-critical">🔴 매우 높음</span></td>
                </tr>
                <tr>
                    <td>노광(Litho) → 검사(Metro)</td>
                    <td>380 FOUP</td>
                    <td>12개</td>
                    <td>8.2분</td>
                    <td>8분</td>
                    <td><span class="status-normal">🟢 정상</span></td>
                </tr>
            </tbody>
        </table>
        
        <h2>1.2 Stocker(자동창고) 상태</h2>
        <table>
            <thead>
                <tr>
                    <th>Stocker ID</th>
                    <th>위치</th>
                    <th>수용량</th>
                    <th>현재 보관</th>
                    <th>점유율</th>
                    <th>입출고 속도</th>
                    <th>상태</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>STK_01</td>
                    <td>포토-식각 사이</td>
                    <td>200</td>
                    <td>156</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 78%;">78%</div>
                        </div>
                    </td>
                    <td>분당 8개</td>
                    <td><span class="status-warning">⚠️ 포화 임박</span></td>
                </tr>
                <tr>
                    <td>STK_02</td>
                    <td>식각-증착 사이</td>
                    <td>250</td>
                    <td>198</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 79%;">79%</div>
                        </div>
                    </td>
                    <td>분당 6개</td>
                    <td><span class="status-warning">⚠️ 포화 임박</span></td>
                </tr>
                <tr>
                    <td><strong>STK_03</strong></td>
                    <td>증착-노광 사이</td>
                    <td>300</td>
                    <td class="text-critical">285</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 95%; background: linear-gradient(90deg, #e74c3c, #c0392b);">95%</div>
                        </div>
                    </td>
                    <td class="text-critical">분당 4개</td>
                    <td><span class="status-critical">🔴 긴급</span></td>
                </tr>
                <tr>
                    <td>STK_04</td>
                    <td>노광-검사 사이</td>
                    <td>200</td>
                    <td>92</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 46%;">46%</div>
                        </div>
                    </td>
                    <td>분당 10개</td>
                    <td><span class="status-normal">🟢 정상</span></td>
                </tr>
            </tbody>
        </table>
        
        <!-- RAG + PHI-4 분석 -->
        <h1>2. RAG + PHI-4 통합 분석 시스템</h1>
        
        <div class="alert-box alert-info">
            <h3 style="margin-top: 0;">🤖 RAG + PHI-4 시스템의 핵심 가치</h3>
            <p><strong>RAG (Retrieval-Augmented Generation)</strong>가 3년간 축적된 15,000개의 실제 물류 패턴을 제공하고,<br>
            <strong>PHI-4 LLM</strong>이 이를 종합 분석하여 데이터 기반의 정확한 예측과 구체적인 해결책을 제시합니다.</p>
        </div>
        
        <h2>2.1 RAG 벡터 검색 결과</h2>
        
        <div class="rag-result">
            <span class="similarity-badge">유사도 94% - 최고 일치</span>
            <h3>🥇 가장 유사한 과거 사례</h3>
            <table style="margin: 10px 0;">
                <tr>
                    <td width="30%"><strong>발생일시:</strong></td>
                    <td>2024년 12월 15일 14:45 (오후 피크)</td>
                </tr>
                <tr>
                    <td><strong>당시 상황:</strong></td>
                    <td>STK_03 93%, C-Block 89%, 증착 88%</td>
                </tr>
                <tr>
                    <td><strong>발생 결과:</strong></td>
                    <td class="text-critical">3시간 후 전면 병목, 생산량 28% 감소</td>
                </tr>
                <tr>
                    <td><strong>적용 해결책:</strong></td>
                    <td>
                        1. STK_04로 30% 물량 긴급 분산<br>
                        2. 비긴급 LOT 4시간 보류<br>
                        3. 증착 공정 속도 10% 감속
                    </td>
                </tr>
                <tr>
                    <td><strong>최종 효과:</strong></td>
                    <td class="text-success">2시간 내 정상화, 손실 5%로 최소화</td>
                </tr>
            </table>
        </div>
        
        <div class="rag-result">
            <span class="similarity-badge">유사도 89%</span>
            <h3>🥈 두 번째 유사 사례</h3>
            <p><strong>발생일:</strong> 2025년 1월 8일 15:20</p>
            <p><strong>결과:</strong> OHT 대체 경로 활성화로 병목 예방 성공</p>
        </div>
        
        <h2>2.2 패턴 기반 통계 분석</h2>
        <div class="process-box">
            <div class="process-number">📊</div>
            <h3>유사 패턴 15건 종합 분석</h3>
            <ul>
                <li><strong>병목 발생:</strong> 12건 (80%)</li>
                <li><strong>병목 미발생:</strong> 3건 (20%)</li>
                <li><strong>평균 발생 시간:</strong> <span class="highlight">2.8시간 후</span></li>
                <li><strong>평균 지속 시간:</strong> 4.5시간</li>
                <li><strong>평균 생산 영향:</strong> 22% 감소</li>
            </ul>
        </div>
        
        <!-- 시계열 예측 -->
        <h1>3. AI 시계열 예측 결과</h1>
        
        <h2>3.1 LSTM 모델 - 24시간 예측</h2>
        <table>
            <thead>
                <tr>
                    <th>시간</th>
                    <th>예측 FOUP 이동량</th>
                    <th>정상 범위</th>
                    <th>STK_03 예상 포화도</th>
                    <th>병목 위험도</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>+1시간 (15:30)</td>
                    <td>445</td>
                    <td>400-450</td>
                    <td>89%</td>
                    <td><span class="status-warning">🟡 중간</span></td>
                </tr>
                <tr>
                    <td>+2시간 (16:30)</td>
                    <td class="text-warning">468</td>
                    <td>400-450</td>
                    <td class="text-warning">96%</td>
                    <td><span class="status-warning">⚠️ 높음</span></td>
                </tr>
                <tr class="text-critical">
                    <td><strong>+3시간 (17:30)</strong></td>
                    <td><strong>485</strong></td>
                    <td>400-450</td>
                    <td><strong>99%</strong></td>
                    <td><span class="status-critical">🔴 매우 높음</span></td>
                </tr>
                <tr>
                    <td>+4시간 (18:30)</td>
                    <td>492</td>
                    <td>400-450</td>
                    <td>100%</td>
                    <td><span class="status-critical">🔴 위험</span></td>
                </tr>
                <tr>
                    <td>+8시간 (22:30)</td>
                    <td>425</td>
                    <td>400-450</td>
                    <td>85%</td>
                    <td><span class="status-normal">🟢 정상</span></td>
                </tr>
            </tbody>
        </table>
        
        <h2>3.2 모델 앙상블 예측 신뢰도</h2>
        <div class="metrics-container">
            <div class="metric-card" style="background: linear-gradient(135deg, #8e44ad, #9b59b6);">
                <div class="metric-label">LSTM 예측</div>
                <div class="metric-value">87%</div>
                <div class="metric-trend">병목 확률</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #16a085, #1abc9c);">
                <div class="metric-label">RNN 예측</div>
                <div class="metric-value">91%</div>
                <div class="metric-trend">병목 확률</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #d35400, #e67e22);">
                <div class="metric-label">ARIMA 예측</div>
                <div class="metric-value">+3.2%</div>
                <div class="metric-trend">시간당 증가율</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #34495e, #2c3e50);">
                <div class="metric-label">종합 신뢰도</div>
                <div class="metric-value">92%</div>
                <div class="metric-trend">예측 정확도</div>
            </div>
        </div>
        
        <!-- PHI-4 종합 분석 -->
        <h1>4. PHI-4 LLM 종합 분석 및 권고사항</h1>
        
        <div class="code-block">
<pre>PHI-4 + RAG 종합 분석 결과:

"검색된 15개 유사 패턴을 분석한 결과, 현재 상황은 
2024년 12월 15일 케이스와 94% 일치합니다.

핵심 위험 요인:
1. 오후 피크 시간대 (14:30)
2. STK_03 포화 임계점 도달 (95%)
3. 증착 공정 고부하 상태 (89%)

예측:
- 병목 발생 확률: 87%
- 예상 시점: 17:00 (2.5시간 후)
- 영향 범위: 전체 라인 25-30%

근거: 과거 동일 조치로 12건 중 10건 성공적 해결"</pre>
        </div>
        
        <h2>4.1 즉시 실행 조치 (10분 이내)</h2>
        
        <div class="recommendation recommendation-urgent">
            <h3>🚨 조치 1: STK_03 긴급 분산</h3>
            <ul>
                <li><strong>내용:</strong> STK_04로 FOUP 30% 즉시 이동</li>
                <li><strong>방법:</strong> OHT 경로 재프로그래밍</li>
                <li><strong>소요 시간:</strong> 15분</li>
                <li><strong>예상 효과:</strong> STK_03 포화 2시간 지연</li>
                <li><strong>담당:</strong> 물류 관제팀</li>
            </ul>
        </div>
        
        <div class="recommendation recommendation-urgent">
            <h3>🚨 조치 2: 우선순위 재조정</h3>
            <ul>
                <li><strong>내용:</strong> Normal 등급 LOT 85개 일시 보류</li>
                <li><strong>영향:</strong> 물류량 20% 즉시 감소</li>
                <li><strong>우선 처리:</strong> Hot/Super Hot LOT만 진행</li>
                <li><strong>담당:</strong> 생산 계획팀</li>
            </ul>
        </div>
        
        <h2>4.2 단기 조치 (1시간 이내)</h2>
        
        <div class="recommendation">
            <h3>📋 조치 3: 대체 경로 활성화</h3>
            <ul>
                <li><strong>C-Block 우회로:</strong> B-라인 50% 활용</li>
                <li><strong>예상 소요 시간:</strong> 30분</li>
                <li><strong>필요 인력:</strong> 엔지니어 2명</li>
                <li><strong>예상 효과:</strong> OHT 혼잡도 15% 감소</li>
            </ul>
        </div>
        
        <div class="recommendation">
            <h3>📋 조치 4: 임시 버퍼 구역 운영</h3>
            <ul>
                <li><strong>위치:</strong> D-Block 유휴 공간</li>
                <li><strong>수용량:</strong> 50 FOUP</li>
                <li><strong>운영 시간:</strong> 4시간</li>
                <li><strong>예상 효과:</strong> 긴급 대피 공간 확보</li>
            </ul>
        </div>
        
        <h2>4.3 실시간 모니터링 지표</h2>
        
        <div class="alert-box alert-warning">
            <h3>📊 핵심 감시 항목</h3>
            <table style="margin: 10px 0;">
                <tr>
                    <td width="40%">STK_03 점유율</td>
                    <td>임계값: 90%</td>
                    <td class="text-critical">현재: 95% ⚠️</td>
                </tr>
                <tr>
                    <td>C-Block OHT 평균 속도</td>
                    <td>임계값: 0.7m/s</td>
                    <td class="text-critical">현재: 0.5m/s ⚠️</td>
                </tr>
                <tr>
                    <td>평균 FOUP 대기 시간</td>
                    <td>임계값: 10분</td>
                    <td class="text-warning">현재: 8.5분</td>
                </tr>
                <tr>
                    <td>시간당 이동량</td>
                    <td>임계값: 450 FOUP</td>
                    <td class="text-warning">현재: 450 FOUP</td>
                </tr>
            </table>
        </div>
        
        <!-- 예상 결과 -->
        <h1>5. 예상 결과 및 성과</h1>
        
        <h2>5.1 조치 시행 vs 미시행 비교</h2>
        <table>
            <thead>
                <tr>
                    <th>구분</th>
                    <th>조치 시행 시</th>
                    <th>미조치 시</th>
                    <th>개선 효과</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>병목 위험</strong></td>
                    <td class="text-success">60% 감소</td>
                    <td class="text-critical">100% 발생</td>
                    <td>40%p ↓</td>
                </tr>
                <tr>
                    <td><strong>생산 영향</strong></td>
                    <td class="text-success">5% 이내</td>
                    <td class="text-critical">25% 감소</td>
                    <td>20%p ↓</td>
                </tr>
                <tr>
                    <td><strong>정상화 시간</strong></td>
                    <td class="text-success">2시간</td>
                    <td class="text-critical">6-8시간</td>
                    <td>4-6시간 ↓</td>
                </tr>
                <tr>
                    <td><strong>영향받는 LOT</strong></td>
                    <td>85개 (Normal급)</td>
                    <td>380개 (전체)</td>
                    <td>295개 ↓</td>
                </tr>
            </tbody>
        </table>
        
        <h2>5.2 시스템 성능 지표</h2>
        <div class="flow-diagram">
            <h3>RAG + PHI-4 통합 시스템 성과</h3>
            <div class="metrics-container" style="margin-top: 20px;">
                <div class="metric-card" style="background: linear-gradient(135deg, #00b894, #00cec9);">
                    <div class="metric-label">예측 정확도</div>
                    <div class="metric-value">92%</div>
                    <div class="metric-trend">vs 룰기반 75%</div>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, #6c5ce7, #a29bfe);">
                    <div class="metric-label">오탐율</div>
                    <div class="metric-value">8%</div>
                    <div class="metric-trend">vs 룰기반 20%</div>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, #fd79a8, #e84393);">
                    <div class="metric-label">응답 시간</div>
                    <div class="metric-value">2.15초</div>
                    <div class="metric-trend">실시간 가능</div>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, #fdcb6e, #f39c12);">
                    <div class="metric-label">일일 학습량</div>
                    <div class="metric-value">45건</div>
                    <div class="metric-trend">지속 개선</div>
                </div>
            </div>
        </div>
        
        <!-- 결론 -->
        <div class="page-break"></div>
        <h1>6. 결론 및 핵심 인사이트</h1>
        
        <div class="alert-box alert-success">
            <h2 style="margin-top: 0;">✅ RAG + PHI-4 시스템의 핵심 가치</h2>
            <p>본 시스템은 <strong>3년간 축적된 15,000개의 실제 물류 패턴</strong>과 <strong>PHI-4의 빠른 추론 능력</strong>을 결합하여:</p>
            <ul>
                <li>✓ <strong>데이터 기반 예측:</strong> 추측이 아닌 과거 사례 기반의 정확한 예측 (92% 정확도)</li>
                <li>✓ <strong>구체적 해결책:</strong> "일반적 대응"이 아닌 "12월 15일 성공 사례" 제시</li>
                <li>✓ <strong>실시간 대응:</strong> 2.15초 내 분석 완료로 즉각적 의사결정 지원</li>
                <li>✓ <strong>지속적 개선:</strong> 매일 45건의 새로운 패턴 자동 학습</li>
            </ul>
        </div>
        
        <h2>6.1 현재 상황 대응 로드맵</h2>
        <div class="process-box">
            <div class="process-number">1</div>
            <h3>즉시 (10분): 긴급 분산</h3>
            <p>STK_04로 30% 물량 이동 → STK_03 포화 2시간 지연</p>
        </div>
        
        <div class="process-box">
            <div class="process-number">2</div>
            <h3>30분 내: 우선순위 조정</h3>
            <p>Normal LOT 보류 → 물류량 20% 감소</p>
        </div>
        
        <div class="process-box">
            <div class="process-number">3</div>
            <h3>1시간 내: 경로 최적화</h3>
            <p>대체 경로 활성화 → OHT 혼잡도 15% 감소</p>
        </div>
        
        <div class="process-box">
            <div class="process-number">4</div>
            <h3>2시간 후: 정상화 확인</h3>
            <p>주요 지표 모니터링 → 추가 조치 여부 결정</p>
        </div>
        
        <h2>6.2 장기적 개선 방향</h2>
        <div class="recommendation">
            <h3>🔮 시스템 발전 로드맵</h3>
            <ul>
                <li><strong>단기 (3개월):</strong> RAG DB 20,000건으로 확장, 예측 정확도 95% 달성</li>
                <li><strong>중기 (6개월):</strong> 실시간 자동 대응 시스템 구축</li>
                <li><strong>장기 (1년):</strong> 전 공장 통합 최적화 플랫폼으로 확장</li>
            </ul>
        </div>
        
        <div class="alert-box alert-info" style="margin-top: 40px; text-align: center;">
            <h2 style="margin-top: 0;">💡 핵심 메시지</h2>
            <p style="font-size: 18px; margin: 20px 0;">
                <strong>"RAG + PHI-4 통합 시스템은 반도체 물류의 복잡성을 이해하고,<br>
                과거의 경험에서 학습하며, 실시간으로 최적의 해결책을 제시하는<br>
                차세대 지능형 물류 관리 플랫폼입니다."</strong>
            </p>
            <p>예측이 아닌 예방, 대응이 아닌 선제적 관리로<br>
            반도체 제조의 생산성을 한 단계 높이겠습니다.</p>
        </div>
        
        <!-- 푸터 -->
        <div style="margin-top: 60px; padding-top: 30px; border-top: 2px solid #ecf0f1; text-align: center; color: #7f8c8d;">
            <p>본 보고서는 RAG + PHI-4 통합 시스템에 의해 자동 생성되었습니다.</p>
            <p>문의: 스마트팩토리 운영팀 | 업데이트: 5분마다 자동 갱신</p>
        </div>
    </div>
</body>
</html>