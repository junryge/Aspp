<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>반도체 물류 예측 시스템 아키텍처</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .diagram-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 1400px;
            width: 100%;
        }

        h1 {
            text-align: center;
            color: #1e3c72;
            margin-bottom: 40px;
            font-size: 2.5em;
        }

        .architecture {
            position: relative;
            min-height: 800px;
        }

        /* 레이어 스타일 */
        .layer {
            position: absolute;
            width: 100%;
            display: flex;
            justify-content: space-around;
            align-items: center;
            padding: 20px;
        }

        .layer-1 { top: 0; }
        .layer-2 { top: 200px; }
        .layer-3 { top: 400px; }
        .layer-4 { top: 600px; }

        /* 컴포넌트 박스 */
        .component {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            text-align: center;
            position: relative;
            transition: all 0.3s ease;
            cursor: pointer;
            min-width: 180px;
            border: 3px solid transparent;
        }

        .component:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
        }

        .component.data {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .component.processing {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }

        .component.ai {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }

        .component.output {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            color: white;
        }

        .component-icon {
            font-size: 3em;
            margin-bottom: 10px;
        }

        .component-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .component-desc {
            font-size: 0.9em;
            opacity: 0.9;
        }

        /* 연결선 */
        .connections {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1;
        }

        .connection {
            stroke: #4facfe;
            stroke-width: 3;
            fill: none;
            opacity: 0.6;
            stroke-dasharray: 5, 5;
            animation: flow 2s linear infinite;
        }

        @keyframes flow {
            from {
                stroke-dashoffset: 0;
            }
            to {
                stroke-dashoffset: -10;
            }
        }

        .connection.highlight {
            stroke: #ff6b6b;
            stroke-width: 4;
            opacity: 1;
        }

        /* 플로우 레이블 */
        .flow-label {
            position: absolute;
            background: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            color: #333;
            font-weight: bold;
            z-index: 10;
        }

        /* 범례 */
        .legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .legend-color {
            width: 30px;
            height: 20px;
            border-radius: 5px;
        }

        /* 상세 정보 팝업 */
        .detail-popup {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            z-index: 1000;
            display: none;
            max-width: 500px;
        }

        .detail-popup.show {
            display: block;
            animation: popIn 0.3s ease;
        }

        @keyframes popIn {
            from {
                opacity: 0;
                transform: translate(-50%, -50%) scale(0.8);
            }
            to {
                opacity: 1;
                transform: translate(-50%, -50%) scale(1);
            }
        }

        .popup-close {
            position: absolute;
            top: 15px;
            right: 15px;
            font-size: 24px;
            cursor: pointer;
            color: #666;
        }

        .popup-close:hover {
            color: #333;
        }

        .overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 999;
            display: none;
        }

        .overlay.show {
            display: block;
        }

        /* 애니메이션 펄스 효과 */
        .pulse {
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% {
                box-shadow: 0 0 0 0 rgba(79, 172, 254, 0.7);
            }
            70% {
                box-shadow: 0 0 0 20px rgba(79, 172, 254, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(79, 172, 254, 0);
            }
        }

        /* 반응형 디자인 */
        @media (max-width: 1200px) {
            .component {
                min-width: 150px;
                padding: 20px;
            }
            .component-icon {
                font-size: 2.5em;
            }
        }

        @media (max-width: 768px) {
            .layer {
                flex-direction: column;
                gap: 20px;
            }
            .layer-1 { position: relative; top: auto; }
            .layer-2 { position: relative; top: auto; }
            .layer-3 { position: relative; top: auto; }
            .layer-4 { position: relative; top: auto; }
            .architecture {
                min-height: auto;
            }
        }
    </style>
</head>
<body>
    <div class="diagram-container">
        <h1>🏭 반도체 물류 시계열+LLM+RAG 시스템 아키텍처</h1>
        
        <div class="architecture">
            <!-- SVG 연결선 -->
            <svg class="connections" viewBox="0 0 1400 800">
                <!-- Layer 1 to Layer 2 -->
                <path class="connection" d="M 200,100 Q 200,150 200,200" />
                <path class="connection" d="M 450,100 Q 450,150 300,200" />
                <path class="connection" d="M 700,100 Q 700,150 400,200" />
                <path class="connection" d="M 950,100 Q 950,150 500,200" />
                
                <!-- Layer 2 to Layer 3 -->
                <path class="connection" d="M 200,300 Q 200,350 300,400" />
                <path class="connection" d="M 300,300 Q 350,350 400,400" />
                <path class="connection" d="M 400,300 Q 450,350 500,400" />
                <path class="connection" d="M 500,300 Q 550,350 600,400" />
                
                <!-- Parallel processing in Layer 3 -->
                <path class="connection highlight" d="M 300,500 Q 350,550 700,500" />
                <path class="connection highlight" d="M 400,500 Q 450,550 700,500" />
                <path class="connection highlight" d="M 500,500 Q 550,550 700,500" />
                <path class="connection highlight" d="M 600,500 Q 650,550 700,500" />
                
                <!-- Layer 3 to Layer 4 -->
                <path class="connection" d="M 700,500 Q 700,600 450,700" />
                <path class="connection" d="M 700,500 Q 700,600 700,700" />
                <path class="connection" d="M 700,500 Q 700,600 950,700" />
            </svg>

            <!-- Layer 1: 데이터 수집 -->
            <div class="layer layer-1">
                <div class="component data" onclick="showDetail('mcs')">
                    <div class="component-icon">📊</div>
                    <div class="component-title">MCS 로그</div>
                    <div class="component-desc">실시간 물류 데이터</div>
                </div>
                <div class="component data" onclick="showDetail('sensor')">
                    <div class="component-icon">🌡️</div>
                    <div class="component-title">센서 데이터</div>
                    <div class="component-desc">환경 모니터링</div>
                </div>
                <div class="component data" onclick="showDetail('erp')">
                    <div class="component-icon">🏭</div>
                    <div class="component-title">ERP/MES</div>
                    <div class="component-desc">생산 정보</div>
                </div>
                <div class="component data" onclick="showDetail('history')">
                    <div class="component-icon">📚</div>
                    <div class="component-title">과거 이력</div>
                    <div class="component-desc">3년치 데이터</div>
                </div>
            </div>

            <!-- Layer 2: 데이터 처리 -->
            <div class="layer layer-2">
                <div class="component processing" onclick="showDetail('etl')">
                    <div class="component-icon">🔄</div>
                    <div class="component-title">ETL 파이프라인</div>
                    <div class="component-desc">데이터 정제/변환</div>
                </div>
                <div class="component processing" onclick="showDetail('feature')">
                    <div class="component-icon">⚡</div>
                    <div class="component-title">특징 추출</div>
                    <div class="component-desc">시계열 특징 생성</div>
                </div>
                <div class="component processing" onclick="showDetail('embedding')">
                    <div class="component-icon">🔍</div>
                    <div class="component-title">임베딩 생성</div>
                    <div class="component-desc">BGE-small 벡터화</div>
                </div>
                <div class="component processing" onclick="showDetail('storage')">
                    <div class="component-icon">💾</div>
                    <div class="component-title">저장소</div>
                    <div class="component-desc">벡터DB + 시계열DB</div>
                </div>
            </div>

            <!-- Layer 3: AI 모델 (병렬 처리) -->
            <div class="layer layer-3">
                <div class="component ai pulse" onclick="showDetail('rag')">
                    <div class="component-icon">📖</div>
                    <div class="component-title">RAG 엔진</div>
                    <div class="component-desc">유사 패턴 검색</div>
                </div>
                <div class="component ai pulse" onclick="showDetail('prophet')">
                    <div class="component-icon">📈</div>
                    <div class="component-title">Prophet</div>
                    <div class="component-desc">시계열 예측</div>
                </div>
                <div class="component ai pulse" onclick="showDetail('arima')">
                    <div class="component-icon">📉</div>
                    <div class="component-title">ARIMA</div>
                    <div class="component-desc">단기 예측</div>
                </div>
                <div class="component ai pulse" onclick="showDetail('anomaly')">
                    <div class="component-icon">🚨</div>
                    <div class="component-title">이상 탐지</div>
                    <div class="component-desc">패턴 이상 감지</div>
                </div>
                <div class="component ai" style="margin-left: 50px;" onclick="showDetail('llm')">
                    <div class="component-icon">🧠</div>
                    <div class="component-title">Phi-3 Mini LLM</div>
                    <div class="component-desc">종합 추론/판단</div>
                </div>
            </div>

            <!-- Layer 4: 출력/서비스 -->
            <div class="layer layer-4">
                <div class="component output" onclick="showDetail('api')">
                    <div class="component-icon">🔌</div>
                    <div class="component-title">예측 API</div>
                    <div class="component-desc">REST/WebSocket</div>
                </div>
                <div class="component output" onclick="showDetail('dashboard')">
                    <div class="component-icon">📊</div>
                    <div class="component-title">실시간 대시보드</div>
                    <div class="component-desc">모니터링/시각화</div>
                </div>
                <div class="component output" onclick="showDetail('alert')">
                    <div class="component-icon">📱</div>
                    <div class="component-title">알림 시스템</div>
                    <div class="component-desc">이상 상황 알림</div>
                </div>
            </div>

            <!-- 플로우 레이블 -->
            <div class="flow-label" style="left: 100px; top: 150px;">실시간 수집</div>
            <div class="flow-label" style="left: 600px; top: 250px;">전처리/변환</div>
            <div class="flow-label" style="left: 300px; top: 450px;">병렬 처리</div>
            <div class="flow-label" style="left: 800px; top: 450px;">종합 판단</div>
            <div class="flow-label" style="left: 600px; top: 650px;">서비스 제공</div>
        </div>

        <!-- 범례 -->
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);"></div>
                <span>데이터 수집</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);"></div>
                <span>데이터 처리</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);"></div>
                <span>AI 모델</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);"></div>
                <span>출력/서비스</span>
            </div>
        </div>
    </div>

    <!-- 상세 정보 팝업 -->
    <div class="overlay" onclick="closeDetail()"></div>
    <div class="detail-popup">
        <span class="popup-close" onclick="closeDetail()">&times;</span>
        <h2 id="detailTitle"></h2>
        <div id="detailContent"></div>
    </div>

    <script>
        const componentDetails = {
            mcs: {
                title: 'MCS 로그 데이터',
                content: `
                    <h3>수집 데이터</h3>
                    <ul>
                        <li>FOUP/FOSB 이동 기록</li>
                        <li>경로별 소요 시간</li>
                        <li>우선순위 정보</li>
                        <li>장비 상태 정보</li>
                    </ul>
                    <h3>수집 주기</h3>
                    <p>실시간 (이벤트 발생 시)</p>
                    <h3>데이터 포맷</h3>
                    <p>JSON/CSV 형식, 초당 1000+ 레코드</p>
                `
            },
            sensor: {
                title: '센서 데이터',
                content: `
                    <h3>모니터링 항목</h3>
                    <ul>
                        <li>온도: 18-22°C 유지</li>
                        <li>습도: 40-45% RH</li>
                        <li>진동/충격 감지</li>
                        <li>파티클 수준</li>
                    </ul>
                    <h3>활용 방안</h3>
                    <p>환경 변수에 따른 이동 시간 보정</p>
                `
            },
            erp: {
                title: 'ERP/MES 연동',
                content: `
                    <h3>연동 데이터</h3>
                    <ul>
                        <li>생산 계획 및 일정</li>
                        <li>PM 스케줄</li>
                        <li>Lot 정보 및 우선순위</li>
                        <li>장비 가동률</li>
                    </ul>
                    <h3>업데이트 주기</h3>
                    <p>15분마다 동기화</p>
                `
            },
            history: {
                title: '과거 이력 데이터',
                content: `
                    <h3>데이터 범위</h3>
                    <p>최근 3년간 물류 이동 기록</p>
                    <h3>활용 목적</h3>
                    <ul>
                        <li>계절별 패턴 분석</li>
                        <li>장기 트렌드 파악</li>
                        <li>이상 패턴 학습</li>
                    </ul>
                `
            },
            etl: {
                title: 'ETL 파이프라인',
                content: `
                    <h3>처리 단계</h3>
                    <ol>
                        <li>데이터 수집 (Extract)</li>
                        <li>이상치 제거 (Transform)</li>
                        <li>정규화 및 집계</li>
                        <li>저장소 적재 (Load)</li>
                    </ol>
                    <h3>처리 성능</h3>
                    <p>초당 10,000 레코드 처리</p>
                `
            },
            feature: {
                title: '특징 추출',
                content: `
                    <h3>시계열 특징</h3>
                    <ul>
                        <li>이동평균 (MA)</li>
                        <li>지수평활 (EMA)</li>
                        <li>계절성 지표</li>
                        <li>트렌드 성분</li>
                    </ul>
                    <h3>도메인 특징</h3>
                    <ul>
                        <li>경로별 혼잡도</li>
                        <li>시간대별 패턴</li>
                        <li>긴급도 가중치</li>
                    </ul>
                `
            },
            embedding: {
                title: '임베딩 생성',
                content: `
                    <h3>임베딩 모델</h3>
                    <p>BGE-small-en-v1.5 (384차원)</p>
                    <h3>처리 내용</h3>
                    <ul>
                        <li>경로 정보 텍스트화</li>
                        <li>패턴 설명 벡터화</li>
                        <li>유사도 계산용 인덱싱</li>
                    </ul>
                `
            },
            storage: {
                title: '데이터 저장소',
                content: `
                    <h3>벡터 DB (ChromaDB)</h3>
                    <ul>
                        <li>임베딩 벡터 저장</li>
                        <li>유사도 검색 지원</li>
                        <li>메타데이터 관리</li>
                    </ul>
                    <h3>시계열 DB (InfluxDB)</h3>
                    <ul>
                        <li>시계열 데이터 최적화</li>
                        <li>빠른 범위 쿼리</li>
                        <li>자동 다운샘플링</li>
                    </ul>
                `
            },
            rag: {
                title: 'RAG 엔진',
                content: `
                    <h3>검색 전략</h3>
                    <ul>
                        <li>코사인 유사도 기반</li>
                        <li>Top-5 유사 패턴 추출</li>
                        <li>시간 가중치 적용</li>
                    </ul>
                    <h3>응답 시간</h3>
                    <p>평균 50ms 이내</p>
                `
            },
            prophet: {
                title: 'Prophet 시계열 모델',
                content: `
                    <h3>예측 특징</h3>
                    <ul>
                        <li>일간/주간 계절성</li>
                        <li>휴일 효과 반영</li>
                        <li>트렌드 변화점 감지</li>
                    </ul>
                    <h3>예측 범위</h3>
                    <p>1-7일 중기 예측</p>
                `
            },
            arima: {
                title: 'ARIMA 모델',
                content: `
                    <h3>모델 설정</h3>
                    <p>ARIMA(1,1,1) - 자동 최적화</p>
                    <h3>특화 영역</h3>
                    <ul>
                        <li>1-24시간 단기 예측</li>
                        <li>급격한 변동 포착</li>
                        <li>실시간 업데이트</li>
                    </ul>
                `
            },
            anomaly: {
                title: '이상 탐지',
                content: `
                    <h3>탐지 방법</h3>
                    <ul>
                        <li>Isolation Forest</li>
                        <li>통계적 이상치</li>
                        <li>패턴 기반 탐지</li>
                    </ul>
                    <h3>알림 기준</h3>
                    <p>95% 신뢰구간 벗어난 경우</p>
                `
            },
            llm: {
                title: 'Phi-3 Mini LLM',
                content: `
                    <h3>역할</h3>
                    <ul>
                        <li>다중 모델 결과 종합</li>
                        <li>컨텍스트 기반 판단</li>
                        <li>자연어 설명 생성</li>
                    </ul>
                    <h3>성능</h3>
                    <ul>
                        <li>파라미터: 3.8B</li>
                        <li>응답시간: 500ms</li>
                        <li>메모리: 4GB</li>
                    </ul>
                `
            },
            api: {
                title: '예측 API',
                content: `
                    <h3>엔드포인트</h3>
                    <ul>
                        <li>POST /predict/route</li>
                        <li>GET /status/realtime</li>
                        <li>WebSocket /stream</li>
                    </ul>
                    <h3>응답 형식</h3>
                    <pre>{
  "prediction": 15.5,
  "confidence": 0.92,
  "method": "ensemble",
  "factors": ["rush_hour", "pm_schedule"]
}</pre>
                `
            },
            dashboard: {
                title: '실시간 대시보드',
                content: `
                    <h3>주요 기능</h3>
                    <ul>
                        <li>Fab 레이아웃 시각화</li>
                        <li>실시간 FOUP 추적</li>
                        <li>병목 구간 히트맵</li>
                        <li>예측 정확도 차트</li>
                    </ul>
                    <h3>기술 스택</h3>
                    <p>React + D3.js + WebSocket</p>
                `
            },
            alert: {
                title: '알림 시스템',
                content: `
                    <h3>알림 조건</h3>
                    <ul>
                        <li>예측 시간 20% 초과</li>
                        <li>이상 패턴 감지</li>
                        <li>시스템 장애</li>
                    </ul>
                    <h3>알림 채널</h3>
                    <ul>
                        <li>SMS/Email</li>
                        <li>Slack/Teams</li>
                        <li>모바일 푸시</li>
                    </ul>
                `
            }
        };

        function showDetail(component) {
            const detail = componentDetails[component];
            document.getElementById('detailTitle').textContent = detail.title;
            document.getElementById('detailContent').innerHTML = detail.content;
            document.querySelector('.overlay').classList.add('show');
            document.querySelector('.detail-popup').classList.add('show');
        }

        function closeDetail() {
            document.querySelector('.overlay').classList.remove('show');
            document.querySelector('.detail-popup').classList.remove('show');
        }

        // ESC 키로 팝업 닫기
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeDetail();
            }
        });

        // 페이지 로드 애니메이션
        window.addEventListener('load', () => {
            const components = document.querySelectorAll('.component');
            components.forEach((comp, index) => {
                comp.style.opacity = '0';
                comp.style.transform = 'translateY(20px)';
                setTimeout(() => {
                    comp.style.transition = 'all 0.5s ease';
                    comp.style.opacity = '1';
                    comp.style.transform = 'translateY(0)';
                }, index * 100);
            });
        });
    </script>
</body>
</html>