# AI Studio 사용법 (텍스트 전용 버전 — 이미지 모두 텍스트로 변환)

본 문서는 [`AI_STUDIO_사용법.md`](AI_STUDIO_사용법.md)의 **텍스트 전용 버전**입니다.
원본 문서의 80개 PNG 화면 캡처를 모두 직접 보고 **각 이미지의 내용을 상세 텍스트로 풀어 옮겼습니다.**
LLM Knowledge Base / RAG / 일반 문서 도구에 그대로 넣어도 시각 정보 손실 없이 활용 가능합니다.

각 이미지가 있던 자리는 다음 블록으로 대체됩니다:

> 🖼️ **\[화면설명: 원본파일명]**
> (이미지에 표시된 UI / 표 / 코드 / 다이얼로그 내용 텍스트 풀어쓰기)

> 슬로건: **"Serve your MODEL on AI ocean"** — *From Modeling to Deployment, All in One Place*

---

## 목차

1. [AI Studio 소개](#1-ai-studio-소개)
2. [AI Studio 전체 아키텍처](#2-ai-studio-전체-아키텍처)
3. [AI Studio Main 화면](#3-ai-studio-main-화면)
4. [HCP 프로젝트 생성하기](#4-hcp-프로젝트-생성하기)
5. [HCP에서 프로젝트 멤버 추가하기](#5-hcp에서-프로젝트-멤버-추가하기)
6. [\[Home\] Overview & Info](#6-home-overview--info)
7. [AI Studio 좌측 메뉴 구조](#7-ai-studio-좌측-메뉴-구조)
8. [\[Modeling > Run\] Training을 Run(Job)으로 실행하기](#8-modeling--run--training을-runjob으로-실행하기)
9. [\[Modeling > Experiment\] 실험 결과 확인하기 (MLflow)](#9-modeling--experiment--실험-결과-확인하기-mlflow)
10. [\[Serving > Inference Service\] Single Inference 하기](#10-serving--inference-service--single-inference-하기)
11. [\[Serving > Inference Service\] Group(Ensemble) Inference 하기](#11-serving--inference-service--groupensemble-inference-하기)
12. [\[Serving > Inference Service\] Debugging 하기 (VS Code)](#12-serving--inference-service--debugging-하기-vs-code)
13. [\[Serving > Static Endpoint\] URL 고정하기 + Rollback](#13-serving--static-endpoint--url-고정하기--rollback)
14. [\[Monitoring > Resource\] 리소스 모니터링](#14-monitoring--resource--리소스-모니터링)
15. [\[Monitoring > Model\] 모델 모니터링 (Grafana)](#15-monitoring--model--모델-모니터링-grafana)
16. [Model Endpoint 모니터링 차트 해석](#16-model-endpoint-모니터링-차트-해석)
17. [Model Endpoint "잘" 모니터링하기 — 개발자 Tip](#17-model-endpoint-잘-모니터링하기--개발자-tip)
18. [Model Endpoint 모니터링 차트 FAQ](#18-model-endpoint-모니터링-차트-faq)
19. [각 서비스 접속 시 로그인 정보](#19-각-서비스-접속-시-로그인-정보)
20. [부록 — 참고 링크](#20-부록--참고-링크)

---

## 1. AI Studio 소개

**AI Studio**는 **AI/ML DevOps 시스템**입니다.
> 그러면.. AI/ML DevOps는 왜 필요할까요..?
> **한번 만들고 끝이 아닌, 지속적인 관리/배포 체계가 필요하기 때문**입니다.

### 1.1 AI 모델, 한번 만들면 끝일까?

> 🖼️ **\[화면설명: AI_Studio_란1.png — 슬라이드 형식의 안내 페이지]**
>
> - 페이지 제목 (좌상단): **"AI 모델, 한번 만들면 끝일까?"**
> - 부제: **"AI 모델의 숨겨 보이지 않는 과업이 있다."**
> - 좌측 본문:
>   - "Model이 처음에 만든 모델은 출시 후 시간이 지남에 따라 데이터의 변화 등에 따라 Model의 '결과'의 정확도 또한 떨어지기 때문에 지속적으로 재학습/재배포 등의 모니터링·배포 등의 과업이 필요합니다."
> - 가운데: AI 모델을 형상화한 일러스트 (전구·로봇·"AI MODEL" 텍스트가 그려진 그래픽)
> - 우상단 박스 — **"신규 모델 생성"**: 첫 배포 시 성능이 좋음
> - 우하단 박스 — **"출시 후 성능 저하 요인"**:
>   - 데이터 변화로 인한 열화
>   - 환경의 변경으로 인한 열화
>   - 요건 변경으로 인한 열화
> - 하단 푸터: "Make IT Intelligent" / "We Do Technology | SK hynix" / 페이지 번호 1

### 1.2 AI/ML DevOps를 요리에 비유한다면…

이해를 돕기 위해 **분석 → 요리** 로 비유한 매핑입니다.

> 🖼️ **\[화면설명: AI_Studio_란2.png — 비교 표 슬라이드]**
>
> - 페이지 제목: **"AI/ML DevOps를 요리에 비유 한다면…"**
> - 부제: **"이해 돕기 위하여 '분석=요리'에 비유해 본다면..."**
> - 표 컬럼: `요소` / `분석` / `데일리한 가족식 운영하는 카페 (As-Is)` / `시스템화 한 인근 보다 큰 호텔 (To-Be)`
> - 행 매핑(요약):
>   - **데이터** — 재료
>   - **모델** — 레시피
>   - **머신** — 주방
>   - **MLOps** — 주방장
>   - **Pipeline** — 조리도구
>   - **CI-CD** — 배달
>   - **모니터링/Logging** — 맛집평가
>   - **Tracking** — 주문이력
> - 가운데에 사람이 요리하는 사진 일러스트 삽입
> - 우측 셀에는 각 행에 대한 As-Is / To-Be 차이를 한 줄씩 설명
> - 페이지 번호 2

### 1.3 AS-IS vs To-Be (DevOps 시스템 도입 효과)

> 🖼️ **\[화면설명: AI_Studio_란3.png — AS-IS vs To-Be 비교 표 슬라이드]**
>
> - 페이지 제목: **"AI/ML DevOps 시스템 없이 개발(As-Is) vs AI Studio 사용 후(To-Be)"**
> - 7행 비교 표:

| 영역 | AS-IS (시스템 없이 개발) | To-Be (AI Studio 사용 후) |
|---|---|---|
| **Model Analysis** | • 모델 Training 시 결과를 기록하고 Logging 및 Tracking되는 툴이 없음<br>• 모델 성능 엑셀로 기록 및 관리<br>• 모델 비교 불가능 | • Parameter, Metric을 로깅하고 Model 저장<br>• 실험 로깅 및 모델 비교를 통하여 분석 과정 효율적으로 진행<br>• 재현성 확보 |
| **Testing and debugging** | • Training을 VM 혹은 Local에서 수행하다가, 리셋이 되는 경우, 원인을 알 수 없음<br>• 문제 발생 시 수동으로 모델을 조사하여 오류 근본 원인 파악에 많은 시간이 소요됨 | • Training 결과에 대하여 Log 통하여 확인 가능<br>• 자동화된 모니터링을 통하여 문제를 신속히 발견 및 해결 가능 |
| **Process Management** | • 여러 팀 간 협업이 비효율적이며 프로세스가 표준화 되지 않아 기술 부채가 증가 | • CI/CD 파이프라인을 통해 개발 및 운영 사이의 갭을 줄일 수 있음<br>• 프로세스 자동화로 효율성 증대 |
| **Serving Infrastructure** | • 프로젝트마다 별도로 개발<br>• 비전문가가 개발하는 경우 IT적 요소의 고려사항을 놓쳐 운영 Risk 증가<br>• 유지보수 어려움 | • 별도 Serving 영역 구축 필요 없음<br>• Container화 및 자동화 된 Serving Infra를 통하여 안정적으로 배포 가능<br>• Scale-out/in 에 대한 고려 |
| **Resource Management** | • 수동적 리소스 관리<br>• 개별 프로젝트 마다 리소스를 점유하여 활용 하여 자원 낭비 가능<br>• 혹은 추가 리소스가 필요할 때 마다 재구축 등의 리스크 발생 가능 | • K8S Container 기반의 자동 Provisioning을 통하여 다수의 Share 가능<br>• 리소스가 추가로 필요한 경우라도 재구축 없이 Scale-up 가능 |
| **Monitoring** | • 모델 성능 Tracking에 대한 모니터링 요소 별도 개발 필요 | • Infra적 요소부터 Application 요소까지 모두 고려된 Logging을 통하여 별도 작업 없이 Monitoring 화면을 제공받을 수 있음 |
| **Automation** | • 대부분의 작업을 수작업으로 진행 시 반복 작업 및 오류 발생 가능성 높음<br>• 자동화 개발을 진행하더라도, 모델러가 서버 환경에서 안정적으로 운영 가능한 요소를 모두 고려 한 개발이 쉽지 않음 | • 자동화 된 파이프라인을 통하여 Training시의 로깅, 배포 등 자동화 및 최적화를 통하여 모델러 입장에서 유지보수가 간소화 됨 |

> (페이지 번호 6)

### 1.4 AI/ML DevOps를 위한 AI Studio

모델의 시스템화를 위하여, **AI Studio**를 통하여 표준화된 **AI/ML DevOps Tool**을 제공합니다.

> 🖼️ **\[화면설명: AI_Studio_란4.png — AI Studio 도입 효과 슬라이드]**
>
> - 페이지 제목: **"AI/ML DevOps를 위한 AI Studio"**
> - 헤드라인: **"모델의 시스템화를 위하여, AI Studio를 통하여 표준화된 AI/ML DevOps Tool을 제공 합니다."**
> - 좌측: 4분할 원형 다이어그램 (시계방향 순환)
>   1. **Training** (회색 화살표)
>   2. **Model Tracking & 모델등록** (회색 화살표)
>   3. **Serving & Inference** (파란 화살표 — 강조)
>   4. **Monitoring & Feedback** (회색 화살표)
>   - 가운데에 "AI/ML DevOps" 라벨
> - → 화살표
> - 우측: 실제 AI Studio Portal 스크린샷 — 헤더 "AI Studio - SK Hynix - From Modeling to Deployment - All in One Place", My Project 카드 4개(`dev-gson`, `dev-aiu-0725`, `dev-aiu-0723`, `dev-picaso`) 노출
> - 페이지 번호 4

```
        Training
           ↓
   Model Tracking & 모델등록
           ↓
   Serving & Inference
           ↓
   Monitoring & Feedback
           ↓
       (loop back to Training)
```

---

## 2. AI Studio 전체 아키텍처

> 🖼️ **\[화면설명: Project_생성_하기1.png — 시스템 아키텍처 다이어그램]**
>
> 두 개의 Layer로 구분된 다이어그램:
>
> **상단 — AI Studio Layer (파란색 배경 박스):**
> 좌→우 5단계 박스, 각각 점선 화살표로 연결
> 1. `Training Data Preparation/등록`
> 2. `AI Model`
> 3. `Training`
> 4. `모델 배포 (Serving)`
> 5. `Inference`
>
> 5개 박스 모두 아래로 점선 화살표 → **하단 큰 박스: `모니터링 및 알람 ( Metric, Logging, Tracing )`**
> 그 박스에서 우측으로 점선 화살표 → `Dashboard`
>
> **하단 — HCP Layer (회색 배경 박스):**
> 좌→우 박스: `IDE` / `S3` / `Job` / `SRE` / `…`
> 그 아래에 가로로 길게: `K8S`
>
> **연결선 (빨간 점선)** "HCP 연계":
> AI Studio 박스들 → HCP 박스로 연결
> - `Training Data Preparation` → `IDE`
> - `AI Model` → `S3`
> - `Training` → `Job`
> - `Dashboard` → `SRE`
>
> 좌측 범례:
> - 파란 점선 화살표 = "AI Studio 자동화"
> - 빨간 점선 화살표 = "HCP 연계"

```
┌─────────────────────────── AI Studio ───────────────────────────┐
│  Training Data       AI       Training   →  모델 배포  →  Inference│
│  Preparation/등록 → Model →               (Serving)               │
│       ↓             ↓        ↓                ↓          ↓        │
│  ───────────────────────────────────────────────────  Dashboard   │
│      모니터링 및 알람 ( Metric, Logging, Tracing )                    │
└──────────────────────────────────────────────────────────────────┘
                              │ HCP 연계
┌─────────────────────────── HCP ─────────────────────────────────┐
│   IDE       S3        Job        SRE       ...                   │
│   ─────────────────────────────────────────────                  │
│                       K8S                                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. AI Studio Main 화면

| 번호 | 메뉴 | 설명 |
|:---:|---|---|
| (1) | **ALL PROJECTS** | 전체 프로젝트 확인 및 검색 가능한 화면을 띄워 줍니다. |
| (2) | **CREATE PROJECT** | 프로젝트 생성 가능하도록, **HCP 프로젝트 생성 사이트**와 연결합니다. |
| (3) | **My Project** | 내가 멤버로 속해 있는 프로젝트들을 카드 형태로 보여 줍니다. |
| (4) | **NOTICE** | AI Studio iflow의 공지사항과 연결되어 있습니다. |
| (5) | **GUIDE** | AI Studio iflow의 가이드와 연결되어 있습니다. |
| (6) | **VIDEO** | AI Studio HyTube 영상과 연결되어 있습니다. |

> 🖼️ **\[화면설명: \[Main\]Main_화면1.png — AI Studio 메인 화면]**
>
> - URL: `aistudio.skhynix.com/apps/ai-studio-fe/main`
> - 좌측 사이드바: AI Studio 로고
> - 본문 헤더: 큰 글자 **"AI Studio"** + 부제 **"Serve your MODEL on AI ocean"**
> - 본문 좌측 — **My Project** 영역:
>   - 카드 4개 가로 배치 (각 카드: 별표 즐겨찾기 표시, 프로젝트 코드, 설명, 작성자(예: 박지영(2068298)), 날짜(2025-09-XX), 사용 중 서비스 아이콘)
>   - 프로젝트들: `aiu-guide-pjt` ★ AI Studio Guide Project, `aiu-goerp-poc`, `aiu-project`, `aiu-taq-ai`
>   - 우측 상단 (1) `ALL PROJECTS` 버튼, (2) `CREATE PROJECT` 버튼 (red highlight)
> - 본문 좌측 하단 — **PROJECT STATUS** 통계 카드:
>   - Project: **11**
>   - Experiment: **24**
>   - Endpoint: **17**
> - 본문 우측 — (4) `NOTICE` 패널 (AI Studio iflow 항목 리스트, 2025-08 날짜)
> - 본문 우측 상단 — (5) `GUIDE` 버튼, (6) `VIDEO` 버튼
> - 우측 상단 헤더: 알림 / 메모 / 설정 / 프로필 아이콘

> 🖼️ **\[화면설명: \[Main\]Main_화면2.png — ALL PROJECTS 모달]**
>
> - 모달 제목: **"ALL PROJECTS"** / 우측 상단 X 버튼
> - 부제: `Total : 11`
> - 우측 상단: `즐겨찾기만 보기` 체크박스, `프로젝트명, 프로젝트 생성자 입력` 검색창
> - 카드 8개를 4열 × 2행 그리드로 표시 (각 카드: 별표 토글, 승인 상태(녹색 점), 프로젝트 코드, 설명, 작성자, 날짜, 서비스 아이콘):
>   - 1행: `aiu-guide-pjt` ★ (AI Studio Guide Project, 박지영, 2025-09-16) / `aiu-goerp-poc` (goerp poc 입니다, 박지영, 2025-09-19) / `aiu-project` (문제해결용, 김영주, 2025-09-19) / `aiu-taq-ai` (실시간 PM Source 정합성 검증 시스템 TAQ에 AI Model Handler, 강해진, 2025-09-16)
>   - 2행: `aiu-lcsa-demandreg` (lcsa(출고예측수량), 박지영, 2025-09-15) / `aiu-aipccb-mlops` (aipccb ml ops, 김상봉, 2025-09-15) / `aiu-gmdm-llm` (G-MDM의 마스터 정보 및 규칙에 대해 답변하는 LLM툴 개발합니다, 김지수, 2025-09-10) / `aiu-pm-part-test` (pm-part ai test, 임기수, 2025-09-09)
> - 하단: 페이징 `< 1 / 2 >`, 페이지 크기 `8 ▾`, 우측 `CREATE PROJECT` 버튼

> 🖼️ **\[화면설명: \[Main\]Main_화면3.png — 프로젝트 생성 안내 다이얼로그]**
>
> - 다이얼로그 제목 (느낌표 아이콘): **"프로젝트 생성 안내"**
> - 본문 텍스트:
>   - "프로젝트 생성이 처음이신가요?"
>   - "프로젝트는 각 머신러닝 과제의 수행 단위이며, 하나의 프로젝트 내에서 다양한 업무를 체계적으로 관리할 수 있어요."
>   - "진행 전 [프로젝트 생성 가이드]를 확인해 주세요."
> - 하단 버튼: `CANCEL` / `GUIDE ↗` / `CREATE PROJECT ↗` (보라색 강조)

---

## 4. HCP 프로젝트 생성하기

1. **AI Studio (AIU)** 는 HCP의 서비스로서, HCP의 서비스를 기반으로 AI/ML DevOps를 조금 더 쉽게 적용할 수 있도록 도움을 주는 시스템입니다.
2. AI Studio를 활용하기 위해서는 **HCP에서 프로젝트를 먼저 신청**해야 합니다.
3. <http://cloud.skhynix.com> 접속.
4. 좌측 메뉴 트리에서 **`Project → 프로젝트`** 클릭.

> 🖼️ **\[화면설명: Project_생성_하기2.png — HCP App Service 좌측 메뉴]**
>
> - URL: `cloud.skhynix.com/apps/hcp-web-app-service`
> - 페이지 제목: **"Hybrid Cloud Platform : App Service"**
> - 좌측 트리 메뉴 (메뉴검색 입력창 + 트리):
>   - **Common**: 모든 서비스, 모든 리소스, 리소스 만들기
>   - **Project** (펼쳐짐):
>     - **프로젝트** ● (선택됨)
>     - 프로젝트 Dashboard
>     - Metering Dashboard Daily
>     - Metering Dashboard Monthly
>     - 프로젝트 리소스 요청 승인
>     - 프로젝트 생성 요청 승인
>   - **DevOps** (접힘)
> - 우측 본문 일부: "리소스 명 filter" 검색창, `STG 배포` / `PRD 배포` 버튼, 표 헤더 보임

5. 프로젝트 목록 화면 상단의 **"프로젝트 추가"** 클릭.

> 🖼️ **\[화면설명: Project_생성_하기3.png — HCP 프로젝트 목록 화면]**
>
> - 페이지 제목 (큰 글자): **"프로젝트"**
> - 부제: "프로젝트를 통해 리소스를 그룹화하여 관리합니다."
> - 필터 영역: `프로젝트 filter` / `프로젝트 타입 filter` / `북마크 선택 ▾` / `전체조회` 토글
> - 액션 버튼: `🔄 새로고침` / `⬆️ 프로젝트 추가` / `🗑️ 프로젝트 삭제` / `★ 북마크 설정`
> - 표 헤더: `프로젝트` / `Service` / `Resource` / `프로젝트 타입`
> - URL: `cloud.skhynix.com/apps/hcp-web-base-project`

6. 기본 정보 입력 — 다음 항목들을 입력합니다.

> 🖼️ **\[화면설명: Project_생성_하기4.png — STEP1 기본정보 입력 폼]**
>
> - URL: `cloud.skhynix.com/apps/hcp-web-base-project/add`
> - 페이지 제목: **"프로젝트 추가"**
> - 부제: "GIT Repository 및 Jenkins Job 생성, Code initializing 및 Code Push가 제공되며, 배포 및 모니터링·로그분석 환경이 제공됩니다."
> - 상단 단계 표시: ① 기본 정보 (활성, 보라색) → ② 검토+만들기
> - 폼 제목: **"STEP1. 기본정보를 입력해주세요!"**
> - 입력 필드 (★ = 필수):
>   - **프로젝트** ★ : `aiu-myfirst-test`
>   - **프로젝트 타입** ★ : `AI` (드롭다운, **빨간 테두리로 강조됨**)
>   - **프로젝트 미러 구분** ★ : `본사`
>   - **프로젝트 설명** ★ : `myfirst test project`
>   - **HyDesk** : 시스템그룹 미선택 / 시스템 미선택
>   - **ITSM 연계** : ○ Y / ● N (라디오)
>   - **관리자** ★ : `박지영 (2068298)`
>   - **배포승인자** ★ : `박지영 (2068298)`
>   - **개발자** ★ : `박지영 (2068298)` `AI Studio (C0000307)`
>   - **Inference Service 전용 NAS** : (비어있음)
>   - **Notification CUBE 채널** ? : "Notification CUBE 채널을 선택하세요." 드롭다운
>   - **Alarm CUBE 채널** ★ ? : `박지영, 회의 알림 봇 ×`
> - 하단 버튼: `← Prev` / `Next →`

> "프로젝트 타입"을 **반드시 `AI` 로 선택**해야 AI Studio를 활용할 수 있습니다.

7. **"만들기"** 클릭.

> 🖼️ **\[화면설명: Project_생성_하기5.png — STEP2 검토+만들기]**
>
> - 단계 표시: ① 기본 정보 (✓ 녹색) → ② 검토+만들기 (활성, 보라색)
> - 폼 제목: **"STEP2. 입력하신 정보로 리소스가 생성됩니다."**
> - STEP1에서 입력한 모든 정보가 회색 배경의 read-only 형태로 다시 표시됨
> - 하단 버튼: `← Prev` / **`└ 만들기`** (빨간 테두리로 강조됨)

8. **프로젝트 승인 후**에 AI Studio 활용 가능합니다.

---

## 5. HCP에서 프로젝트 멤버 추가하기

AI Studio에서 프로젝트를 활용하다가, 또 다른 멤버를 추가해야 할 일이 생길 수 있습니다.
이런 경우에는 **HCP Portal**에서 해당 프로젝트에 권한을 주면 됩니다.

1. **HCP에 접속**: <http://cloud.skhynix.com/>
2. 좌측 메뉴에서 **"Project > 프로젝트"** 클릭

> 🖼️ **\[화면설명: AI_Studio에멤버추가하기1.png — HCP 좌측 메뉴 클로즈업]**
>
> - 페이지 제목: **"Hybrid Cloud Platform : 프로젝트"**
> - 좌측 트리 메뉴:
>   - **Common**: 모든 서비스, 모든 리소스, 리소스 만들기
>   - **Project** (펼쳐짐):
>     - **프로젝트** ● (선택됨, 보라색)
>     - 프로젝트 Dashboard
>     - Metering Dashboard Daily
>     - Metering Dashboard Monthly
>     - 프로젝트 리소스 요청 승인
>     - 프로젝트 생성 요청 승인
>   - **DevOps** (접힘)

3. 프로젝트 찾기

> 🖼️ **\[화면설명: AI_Studio에멤버추가하기2.png — 프로젝트 검색]**
>
> - 페이지 제목: **"프로젝트"** / 부제: "프로젝트를 통해 리소스를 그룹화하여 관리합니다."
> - 필터: `aiu-lcsa` 검색어 입력됨 (× 클리어 버튼), 프로젝트 타입 filter, 북마크 선택, 전체조회
> - 액션 버튼: 새로고침, 프로젝트 추가, 프로젝트 삭제, 북마크 설정
> - 표 (헤더: 프로젝트, Service, Resource, 프로젝트 타입, 설명)
>   - 행: ☐ `aiu-lcsa-demandreg` / **App DS Job** / Resource / `AI` / `lcsa (출고예측수량)`

4. 해당 프로젝트를 눌러서 **"접근 권한"** 을 눌러 권한 추가 화면 접근

> 🖼️ **\[화면설명: AI_Studio에멤버추가하기3.png — 프로젝트 상세 / 접근 권한 패널]**
>
> - 우측 패널 헤더: **"aiu-lcsa-demandreg"** (빨간 글씨)
> - 부제: "AI | lcsa (출고예측수량)"
> - 브레드크럼: Home / Project / aiu-lcsa-demandreg
> - 좌측 메뉴 (패널 안):
>   - **일반**: 개요 / **접근 권한** (선택됨, 빨간 강조) / 태그
>   - **모니터링**: 로그설정
>   - **Alert**: Alert 설정 / Alert History
> - 상단 액션: `🔄 새로고침` / `📤 저장`
> - 폼 (★ 필수):
>   - **관리자** ★ : `김현수 (2067628)` `박지영 (2068298)` `hcp_aistudio (hcp_aistudio)` `AI Studio (X9903354)` `박상헌 (2068297)` `조병열 (X0100670)`
>   - **배포승인자** ★ : `김현수 (2067628)` `조병열 (X0100670)`
>   - **개발자** ★ : `김현수 (2067628)` `조병열 (X0100670)`

5. 추가하고자 하는 멤버 선택 후 확인 버튼

> 🖼️ **\[화면설명: AI_Studio에멤버추가하기4.png — 조직도 다이얼로그]**
>
> - 다이얼로그 제목: **"조직도"**
> - 상단 탭: **구성원** (선택됨) / 가상그룹 / 나만의 그룹
> - 좌측 — 부서 트리:
>   - 회사: `SK하이닉스 ▾`
>   - 검색창: "부서명 검색"
>   - 트리:
>     - **Data Intelligence** (펼쳐짐)
>       - AI Transformation
>       - 품질지능화
>       - 장비지능화
>       - **AI/Data Platform** (선택됨, 파란 글씨)
>       - AI Solution
>     - CAE
>     - 산업보안
>     - AIX확산TF
>       - 용인Cluster DT TF
>       - CTO Culture Partner
>     - 기반기술센터
>     - Memory Systems Research
>     - 변화추진
> - 가운데 — 검색 결과:
>   - 검색창 입력: `ai studio`
>   - `□ 전체 선택` / 우측 "총 1명 선택"
>   - 행 1: ☐ 🐳 `AI Studio(C0000307)` Intelligent Bot /
>   - 행 2: ☑ ⛵ `AI Studio(X9903354)` 기타 협력사 / (체크됨)
>   - 가운데 우측 화살표 `>` (오른쪽으로 이동 버튼)
> - 우측 — 선택 목록 (`나만의 그룹 추가` / `순서변경` / `전체삭제` 버튼):
>   - 👤 김현수(2067628) 🗑
>   - 👤 조병열(X0100670) 🗑
>   - ⛵ AI Studio(X9903354) 🗑
> - 하단: `취소` / `확인` (보라색 강조)

6. **"저장"** 버튼 클릭

> 🖼️ **\[화면설명: AI_Studio에멤버추가하기5.png — 멤버 추가 후 권한 화면]**
>
> - 페이지 헤더: **"aiu-lcsa-demandreg"** / "AI | lcsa (출고예측수량)"
> - 좌측 메뉴: 일반(개요/접근 권한 선택/태그), 모니터링(로그설정), Alert(Alert 설정/Alert History)
> - 상단 액션: `🔄 새로고침` / `📤 저장`
> - 폼 (저장 후 멤버 추가된 상태):
>   - **관리자** ★ : 김현수, 박지영, hcp_aistudio, AI Studio (X990335...)
>   - **배포승인자** ★ : 김현수, 조병열, AI Studio (X9903354)
>   - **개발자** ★ : 김현수, 조병열

이제 멤버 추가가 완료되었습니다!

---

## 6. \[Home\] Overview & Info

### 6.1 Overview

전체적으로 어떤 모델과 서비스가 있고, 어떤 Run 들이 있는지에 대하여 확인 가능합니다.

| 번호 | 항목 | 설명 |
|:---:|---|---|
| (1) | **Model** | 최근 등록된 모델과 최신 버전을 보여 줍니다. |
| (2) | **Endpoint** | 최근 등록된 Endpoint를 보여 줍니다. |
| (3) | **Run Template** | 최근 등록된 Run Template을 보여 줍니다. |
| (4) | **Run Instance** | 최근 실행된 Run을 보여 줍니다. |

> 🖼️ **\[화면설명: \[Home\]_Overview_Info1.png — 프로젝트 Home Overview 화면]**
>
> - URL: `aistudio.skhynix.com/apps/ai-studio-fe/projects/aiu-guide-pjt`
> - 상단 탭: My Recent Projects → `★ aiu-guide-pjt` (선택, 노란 별), `aiu-goerp-poc`, `aiu-project`, `aiu-taq-ai`, `aiu-lcsa-demandreg`, `+`
> - 본문 헤더: 큰 글자 **"A aiu-guide-pjt ★"**
> - 브레드크럼: 🏠 / aiu-guide-pjt
> - 본문 탭: **Overview** (선택) / Info
> - 좌측 사이드바: 🏠 Home / 🪐 Jupyter (1) / 🟦 VS Code (1) / 🌀 MLflow (1) / 📦 Modeling ▾ / 🔗 Serving ▾ / 📊 Monitoring ▾ / 📋 Management
> - 본문 — 4개 패널 2x2 그리드 (모두 빨간 테두리로 강조):
>
>   **(1) Model** 패널:
>   | Name | Version | Created at | Action |
>   |---|---|---|---|
>   | randomforest | 2 | 2025-09-22 12:33:08 | `mlflow ↗` |
>   | decisiontree | 2 | 2025-09-22 12:30:59 | `mlflow ↗` |
>   | LogisticRegression | 2 | 2025-09-22 12:28:10 | `mlflow ↗` |
>
>   **(2) Endpoint** 패널:
>   | Name | Status | Type | Created at | Action |
>   |---|---|---|---|---|
>   | g3 | ● READY | GROUP | 2025-09-23 17:29:33 | (복사) `Log ↗` |
>   | skl-model-test-v4-s1 | ● READY | SINGLE | 2025-09-23 13:29:31 | (복사) `Log ↗` |
>   | g2 | ● READY | GROUP | 2025-09-22 12:53:07 | (복사) `Log ↗` |
>
>   **(3) Run Template** 패널:
>   | Name | Deploy Status | Created at | Action |
>   |---|---|---|---|
>   | runtest | ● SUCCESS | 2025-09-23 09:29:03 | `Git ↗` |
>   | run-test | ● SUCCESS | 2025-09-22 10:40:33 | `Git ↗` |
>
>   **(4) Run Instance** 패널:
>   | Name | Status | Started at | Ended at |
>   |---|---|---|---|
>   | runtest-n6s59 | ● COMPLETED | 2025-09-23 10:47:35 | 2025-09-23 10:47:49 |
>   | run-test-tf5va | ● COMPLETED | 2025-09-22 11:58:48 | 2025-09-22 11:59:01 |
>   | run-test-1ex5f | ● FAILED | 2025-09-22 10:44:35 | 2025-09-22 10:44:41 |

### 6.2 Info

프로젝트 이름, 멤버 등의 정보를 조회할 수 있습니다.

> 🖼️ **\[화면설명: \[Home\]_Overview_Info2.png — Info 탭 화면]**
>
> - URL: `.../projects/aiu-guide-pjt?tabs=info`
> - 본문 탭: Overview / **Info** (선택)
> - 카드:
>   - 📁 **aiu-guide-pjt**
>   - **Description** : `AI Studio Guide Project`
>   - **System Code [HyDesk]** : `aiu-guide-pjt`
>   - **Cube** : 🔔 `200355409` ↗ (빨간 테두리로 강조됨, 클릭 가능 — HCP의 Cube 페이지로 이동)
> - 👥 **Members** 영역:
>   - **Admin(7)** :
>     - 👤 AI Studio(X9903354)
>     - 👤 나소진(2076567)
>     - 👤 박상헌(2068297)
>     - 👤 박지영(2068298)
>     - 👤 박지혜(2062582)
>     - 👤 조태연(2066763)
>     - 👤 채선율(2069978)
>   - **Deploy Approver(1)** : 👤 박지영(2068298)
>   - **Developer(1)** : 👤 박지영(2068298)
> - 우측 하단: `EDIT ↗` 버튼 (보라색)

Cube 알람을 수정하거나, 프로젝트 멤버 등을 수정할 수 있도록 **"Cube" 링크**나 **"Edit" 버튼**을 누르면 **HCP의 수정 페이지**로 이동 합니다.

> 🖼️ **\[화면설명: \[Home\]_Overview_Info3.png — HCP 프로젝트 수정 페이지]**
>
> - URL: `cloud.skhynix.com/apps/hcp-web-base-project/drawer/Defaults/aiu-guide-pjt`
> - 페이지 제목: **"Hybrid Cloud Platform : 프로젝트"**
> - 좌측 — 프로젝트 목록 (검색창 + 새로고침/추가 버튼, 체크박스로 다수 선택 가능):
>   - aiu-goerp-poc, **aiu-guide-pjt**, aiu-lcsa-demandreg, aiu-mirts, ai-data-live, ai-studio, aipccb, atom, daydemo, dd-sem, defectmastersystem, dramctqusingwt, elsa, epms, euvautomation, hub-platform-confluent, hub-platform-service, ...
> - 우측 패널 헤더 (빨간 글씨): **"aiu-guide-pjt"** / "AI | AI Studio Guide Project"
> - 패널 좌측 메뉴: **일반** (`개요` 선택, 접근 권한, 태그), **모니터링** (로그설정), **Alert** (Alert 설정, Alert History)
> - 상단 액션: 🔄 새로고침 / 📤 저장 / 🗑 삭제 / 모니터링
> - 폼 — 모든 필드 read-only 표시:
>   - 프로젝트 : `aiu-guide-pjt`
>   - 프로젝트 타입 : `AI`
>   - 프로젝트 미러 구분 : `본사`
>   - 프로젝트 설명 ★ : `AI Studio Guide Project`
>   - HyDesk ? : 시스템그룹 미선택 / 시스템 미선택
>   - ITSM 연계 : ○ Y / ● N
>   - HCP H/W서버 투자여부 : ● Y / ○ N
>   - HCP 자원협의내용 : (비어있음)
>   - Inference Service 전용 NAS : (비어있음)
>   - Notification CUBE 채널 ? : "Notification CUBE 채널을 선택하세요."
>   - Alarm CUBE 채널 ★ ? : `2025 AI Studio ×`
>   - 등록자 : 박지영(2068298)
>   - 등록일 : 2025.09.16 06:46:54
> - 우하단 부동 채팅 위젯: "HCP 와 관련해서 궁금한 점을 질문해주세요!"

---

## 7. AI Studio 좌측 메뉴 구조

모든 화면에서 좌측 사이드바는 다음과 같은 구조를 가집니다.

```
🏠 Home
─────────────────────
🪐 Jupyter (1)        ← 카운트 배지
🟦 VS Code (1)
🌀 MLflow  (1)
─────────────────────
📦 Modeling
   ├ Experiment ↗  (별도 탭으로 MLflow 오픈)
   └ Run
🔗 Serving
   ├ Inference Service
   └ Static Endpoint
📊 Monitoring
   ├ Model ↗  (별도 탭으로 Grafana 오픈)
   └ Resource
📋 Management
```

상단에는 `My Recent Projects` 탭(즐겨찾기 별표) + 우측에는 알림/메모/설정/프로필 아이콘이 표시됩니다.

---

## 8. \[Modeling > Run\] Training을 Run(Job)으로 실행하기

### 8.1 Run 화면 진입

`Modeling → Run` 메뉴 클릭. 한 번도 수행한 이력이 없다면 **"Latest Run Instances"** 화면은 나오지 않으며, 수행 이력이 있다면 상단에 최근 수행 이력이 카드 형태로 나타납니다.

> 🖼️ **\[화면설명: \[Modeling_Run\]Training을_Run(Job)으로_실행하기1.png — Run 메뉴 진입 직전 Overview 화면]**
>
> 좌측 사이드바의 `Modeling` 펼침 메뉴 중 **`Run`** 이 빨간 테두리로 강조됨. 본문은 Home Overview 화면(Model/Endpoint/Run Template/Run Instance 4 패널) 표시.

> 🖼️ **\[화면설명: \[Modeling_Run\]Training을_Run(Job)으로_실행하기2.png — Run 메인 화면]**
>
> - URL: `.../modeling/run`
> - 페이지 제목: **"Run"** / 브레드크럼: aiu-guide-pjt / Modeling / Run
> - 상단 — **Latest Run Instances** 영역에 카드 2개:
>
>   **카드 1 — `run-test`**
>   - ● COMPLETED (녹색)
>   - Instance ID : `run-test-tf5va`
>   - Arguments : (비어있음)
>   - Queue : `-`
>   - Duration : `2025-09-22 11:58:48 ~ 2025-09-22 11:59:01` `00:00:13` (녹색 배지)
>   - 👤 박지영 (2068298) / 🕐 2025-09-22 11:58:45
>
>   **카드 2 — `run-test`**
>   - ● FAILED (빨간색)
>   - Instance ID : `run-test-1ex5f`
>   - Arguments : (비어있음)
>   - Queue : `-`
>   - Duration : `2025-09-22 10:44:35 ~ 2025-09-22 10:44:41` `00:00:06` (빨간 배지)
>   - 👤 박지영 (2068298) / 🕐 2025-09-22 10:44:19
>
> - 하단 — **Run Template** 표 (`Total : 1`, 우측에 `CREATE` 버튼):
>
>   | Name | Description | Deploy Status | Ver... | Resources | Arguments | File | Created by | Created at | Ac... |
>   |---|---|---|---|---|---|---|---|---|---|
>   | run-test | run test | ● SUCCESS | 1 | CPU: 2, Memory: 1GiB, GPU: 0 |  | runtest.py | 박지영 (2068298) | 2025-09-22 10:40 | 🗑 ▶ |
>
> - 페이징 `< 1 >`, 페이지 크기 `5 ▾`

### 8.2 RUN Template 생성하기

화면 우측 하단 Run Template 표의 **"CREATE"** 버튼 클릭.

> 🖼️ **\[화면설명: \[Modeling_Run\]Training을_Run(Job)으로_실행하기3.png — CREATE 버튼 강조]**
>
> 위 Run 메인 화면과 동일하지만, 우측 하단 **`CREATE`** 버튼이 빨간 테두리로 강조됨.

### 8.3 기본 정보 입력 (Run Template 생성 폼)

| 번호 | 항목 | 설명 / 예시 |
|:---:|---|---|
| (1) | **Name** ★ | Run Template 이름 (예: `runtest`) |
| (2) | **Python Version** ★ | 생성하고자 하는 파이썬 버전 (예: `3.11`) |
| (3) | **Resource** ★ | 생성하고자 하는 리소스 크기 (예: `CPU: 2, Memory: 1GiB, GPU: 0`) |
| (4) | **Repository** ★ | **`http://`** 형태의 git 주소 (ssh 불가) |
| (5) | **Branch** ★ | 실행하고자 하는 Branch 이름 (예: `master`) |
| (6) | **File** ★ | Bitbucket을 선택 후 나타나는 파일 트리에서 메인 파일 선택 |
| (7) | **Requirements** ★ | RUN 실행 시 설치할 패키지를 한 줄에 하나씩 입력 |
|  | Arguments | (선택) 실행 시 인자 |
|  | Command | (선택) 커스텀 커맨드 |
|  | Description | (선택) 설명 |

> 🖼️ **\[화면설명: \[Modeling_Run\]Training을_Run(Job)으로_실행하기4.png — Run Template 생성 폼 (File 트리 펼침)]**
>
> - URL: `.../modeling/run/create`
> - 페이지 제목: **"Run Template 생성"** / 브레드크럼: aiu-guide-pjt / Modeling / Run / Run Template 생성
> - 우측 상단 버튼: `COPY TEMPLATE`
> - **기본정보** 섹션 폼 (각 필드에 빨간 숫자 1~6 라벨):
>   - **(1) Name** ★ : `runtest` ✔
>   - **(2) Python Version** ★ : `3.11` ✔
>   - **(3) Resource** ★ : `CPU: 2, Memory: 1GiB, GPU: 0` ✔
>   - **(4) Repository** ★ : `http://` + `http://bitbucket.skhynix.com/scm/hcp-aiu-guide-pjt/job-test.git` ✔
>   - **(5) Branch** ★ : `master` ✔
>   - **File** ★ : (입력란 비어있음)
>   - **(6) `파일명 또는 폴더 검색`** 입력란 (드롭다운 펼쳐짐, 빨간 테두리로 강조):
>     - ▶ .ipynb_checkpoints
>     - ▶ \_\_pycache\_\_
>     - ▶ aiu_custom
>     - ▶ config
>     - input_example.json
>     - requirements.txt
>     - **runtest.py** (파란 강조)
>     - ▶ saved_model
>   - Arguments (비어있음)
>   - Command (비어있음)
>   - Requirements ★ (가려짐)
>   - Description (가려짐)
> - 하단: `CANCEL` / `CREATE` (보라색, 빨간 강조)

> 🖼️ **\[화면설명: \[Modeling_Run\]Training을_Run(Job)으로_실행하기5.png — Run Template 생성 폼 (Requirements 입력)]**
>
> - 폼이 모두 채워진 상태:
>   - Name: `runtest`, Python Version: `3.11`, Resource: `CPU: 2, Memory: 1GiB, GPU: 0`
>   - Repository: `http://bitbucket.skhynix.com/scm/hcp-aiu-guide-pjt/job-test.git`
>   - Branch: `master`
>   - File: `runtest.py` (위 박스), 아래 셀렉터에도 `runtest.py`
>   - Arguments: (비어있음, "Arguments를 입력하세요.")
>   - Command: (비어있음)
>   - **(7) Requirements** ★ (빨간 테두리로 강조) ✔ :
>     ```
>     pandas==2.3.0
>     requests==2.32.4
>     scikit-learn==1.7.0
>     ```
>   - Description: (비어있음, "설명을 작성해 주세요.")
> - 하단: `CANCEL` / **`CREATE`** (보라색)

### 8.4 RUN 환경 생성

> 🖼️ **\[화면설명: \[Modeling_Run\]Training을_Run(Job)으로_실행하기6.png — 생성 직후, IN PROGRESS 상태]**
>
> - 우상단에 토스트: ● `생성되었습니다.`
> - **Run Template** 표 (`Total : 2`):
>   - **runtest** | (Description 비어있음) | **● IN PROGRESS...** | 1 | CPU: 2, Memory: 1GiB, GPU: 0 | (Arguments 없음) | runtest.py | 박지영 (2068298) | 2025-09-23 09:29 | 🗑 ▶ ← (행 전체 빨간 테두리로 강조)
>   - run-test | run test | ● SUCCESS | 1 | CPU: 2, Memory: 1GiB, GPU: 0 |  | runtest.py | 박지영 | 2025-09-22 10:40 | 🗑 ▶

> 🖼️ **\[화면설명: \[Modeling_Run\]Training을_Run(Job)으로_실행하기7.png — Deploy TimeLine 다이얼로그 (Double Click)]**
>
> - "Double Click" 라벨이 빨간 화살표로 IN PROGRESS 행을 가리킴
> - 화면 가운데에 모달 다이얼로그 열림:
>   - 제목: **"Run Template Deploy TimeLine"** / 우측 X
>   - 단계 목록:
>     - ⏳ **Docker Image Build & Push**
>     - 진행 중

### 8.5 RUN 실행

> 🖼️ **\[화면설명: \[Modeling_Run\]Training을_Run(Job)으로_실행하기8.png — Build SUCCESS 후 실행 버튼 강조]**
>
> - **Run Template** 표:
>   - **runtest** | | ● SUCCESS | 1 | CPU: 2, Memory: 1GiB, GPU: 0 |  | runtest.py | 박지영 | 2025-09-23 09:29 | 🗑 **▶** (재생 버튼이 빨간 테두리로 강조)
>   - run-test | run test | ● SUCCESS | 1 | ... | runtest.py | 박지영 | 2025-09-22 10:40 | 🗑 ▶

### 8.6 Queue 선택

> 🖼️ **\[화면설명: \[Modeling_Run\]Training을_Run(Job)으로_실행하기9.png — Queue List 다이얼로그]**
>
> 화면 가운데에 모달:
> - 제목: **"Queue List"** / 우측 X
> - `Total : 1`
> - 표 컬럼: ` ` / `Queue` / `Guarantee` / `Wait Count`
> - 행: ○ `cpu-common-queue` / `CPU: 64, Memory: 1000Gi` / `0`
> - 페이징 `< 1 >`, 페이지 크기 `5 ▾`
> - 하단: `CANCEL` / `EXECUTE` (보라색)

> 추후 **GPU** 도 Queue에 추가될 예정입니다.

### 8.7 실행 상태 확인

| 상태 | 색상 / 표시 |
|---|---|
| **WAITING** | 노란색 — 실행 대기 중 |
| **COMPLETED** | 녹색 — 완료 |
| **FAILED** | 빨간색 — 실패 |

> 🖼️ **\[화면설명: \[Modeling_Run\]Training을_Run(Job)으로_실행하기10.png — 3가지 상태 카드 비교]**
>
> Latest Run Instances 영역에 3개 카드 가로로 배치되며 아래에 빨간 라벨로 (1) WAITING / (2) COMPLETED / (3) FAILED 표시:
>
> - **카드 1 — runtest** (노란 좌측 테두리)
>   - ● WAITING (노랑)
>   - Instance ID: `runtest-n6s59`
>   - Arguments: (비어있음)
>   - Queue: `cpu-common-queue`
>   - Duration: `-`
>   - 👤 박지영 (2068298) / 🕐 2025-09-23 10:47:19
>
> - **카드 2 — run-test** (녹색 좌측 테두리)
>   - ● COMPLETED (녹색)
>   - Instance ID: `run-test-tf5va`
>   - Queue: `-`
>   - Duration: `2025-09-22 11:58:48 ~ 2025-09-22 11:59:01` `00:00:13` (녹색 배지)
>   - 박지영 / 2025-09-22 11:58:45
>
> - **카드 3 — run-test** (빨간 좌측 테두리)
>   - ● FAILED (빨강)
>   - Instance ID: `run-test-1ex5f`
>   - Duration: `2025-09-22 10:44:35 ~ 2025-09-22 10:44:41` `00:00:06` (빨간 배지)
>   - 박지영 / 2025-09-22 10:44:19

### 8.8 전체 Run Instances 확인

> 🖼️ **\[화면설명: \[Modeling_Run\]Training을_Run(Job)으로_실행하기11.png — `+` 버튼 위치]**
>
> Run 메인 화면에서, **`Latest Run Instances`** 영역의 **우측 상단 ⊕ (+) 버튼**이 빨간 테두리로 강조됨.

> 🖼️ **\[화면설명: \[Modeling_Run\]Training을_Run(Job)으로_실행하기12.png — Run Instances 페이지]**
>
> - URL: `.../modeling/run/instances`
> - 페이지 제목: **`← Run Instances`** / 브레드크럼: aiu-guide-pjt / Modeling / Run / Run Instances
> - `Total : 3`
> - 표 컬럼: `Run Template` / `Instance ID` / `Status` / `Arguments` / `Started at` / `Ended at` / `Durati...` / `Executed by` / `Action`
> - 행:
>   - runtest | runtest-n6s59 | ● COMPLETED |  | 2025-09-23 10:47:35 | 2025-09-23 10:47:49 | `00:00:14` (녹) | 박지영 (2068298) | `Log ↗` `Metric ↗`
>   - run-test | run-test-tf5va | ● COMPLETED |  | 2025-09-22 11:58:48 | 2025-09-22 11:59:01 | `00:00:13` (녹) | 박지영 | `Log ↗` `Metric ↗`
>   - run-test | run-test-1ex5f | ● FAILED |  | 2025-09-22 10:44:35 | 2025-09-22 10:44:41 | `00:00:06` (빨) | 박지영 | `Log ↗` `Metric ↗`
> - 페이징 `< 1 >`, 페이지 크기 `15 ▾`

---

## 9. \[Modeling > Experiment\] 실험 결과 확인하기 (MLflow)

좌측 메뉴 **`Experiment ↗`** 클릭 시 **MLflow** 화면이 별도 탭으로 열립니다.

> 🖼️ **\[화면설명: \[Modeling_Experiment\]_실험결과_확인하기1.png — Experiment 메뉴 클릭]**
>
> 좌측 사이드바의 `Modeling` 펼침 메뉴 중 **`Experiment ↗`** (외부 링크 아이콘)이 빨간 테두리로 강조됨. 본문은 Home Overview.

> 🖼️ **\[화면설명: \[Modeling_Experiment\]_실험결과_확인하기2.png — MLflow Experiments 화면]**
>
> - URL: `aiu-guide-pjt-mlflow-001.aisp01.skhynix.com/#/experiments/3?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&...`
> - 좌측 상단 로고: **mlflow 2.13.2**
> - 상단 탭: **Experiments** (활성, 파란 밑줄) / Models
> - 우상단: 다크모드 토글 / 설정 / `GitHub` / `Docs`
> - 좌측 패널 — **Experiments** (✚ ◀ 버튼):
>   - 검색창: "Search Experiments"
>   - 목록 (각 실험 옆에 ✏️ 🗑):
>     - ☐ Default
>     - ☐ sklearn_test
>     - ☐ sklearn_job_test
>     - ☑ **ensemble-test** (선택, 굵게)
> - 우측 본문 — `ensemble-test` 상세:
>   - 헤더: **ensemble-test** ⓘ + `Provide Feedback ↗` `Add Description` / 우측 `Share` 버튼
>   - 검색창 (예시 쿼리): `metrics.rmse < 1 and params.model = "tree"`
>   - 필터 버튼: `Time created ▾` / `State: Active ▾` / `Datasets ▾` / 점3개 / 새로고침 / `+ New run`
>   - 정렬·뷰 컨트롤: `Sort: Created ▾` / `Columns ▾` / `Group by ▾`
>   - 탭: **Table** (선택) / Chart / Evaluation / Experimental
>   - 표 (`6 matching runs`) 컬럼: ☐ / Run Name / Created / Dataset / Duration / Source / Models
>     - ☐ ⚫ auspicious-asp-15 | ✔ 18 hours ago | - | 7.4s | randomf... | randomfo...
>     - ☐ 🔵 incongruous-worm-991 | ✔ 18 hours ago | - | 6.9s | decision... | decisiontre...
>     - ☐ 🔴 redolent-lynx-264 | ✔ 18 hours ago | - | 9.5s | logistic_r... | LogisticRe...
>     - ☐ 🟠 classy-seal-999 | ✔ 18 hours ago | - | 9.4s | randomf... | randomfo...
>     - ☐ 🟣 receptive-mink-107 | ✔ 18 hours ago | - | 9.1s | decision... | decisiontre...
>     - ☐ 🟢 trusting-pig-703 | ✔ 18 hours ago | - | 9.6s | logistic_r... | LogisticRe...

---

## 10. \[Serving > Inference Service\] Single Inference 하기

`Serving → Inference Service` 메뉴 진입 → 상단 **`Single`** 탭.

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Single_Inference하기1.png — Inference Service Single 목록]**
>
> - 페이지 제목: **"Inference Service ⓘ"** / 브레드크럼: aiu-guide-pjt / Serving / Inference Service
> - 좌측 메뉴: `Serving > Inference Service` (빨간 테두리 강조), Static Endpoint
> - 본문 탭: **Single** (선택, 밑줄) / Group
> - `Total : 4`, 우측 상단 `CREATE` 버튼
> - 표 컬럼: `Name` / `Tag` / `Deploy Status` / `Status` / `Created by` / `Created at` / `Debug` / `Action`
> - 행 (모두 SUCCESS / READY / OFF):
>   - randomforest-v2-s1 |  | ● SUCCESS | ● READY | 박지영 (2068298) | 2025-09-22 12:49:41 | `🐞 OFF` | 🔗 📋 🗑 `Log ↗` `Metric`
>   - decisiontree-v2-s1 |  | ● SUCCESS | ● READY | 박지영 | 2025-09-22 12:49:14 | `🐞 OFF` | 🔗 📋 🗑 `Log ↗` `Metric`
>   - logisticregression-v2-s1 |  | ● SUCCESS | ● READY | 박지영 | 2025-09-22 12:48:51 | `🐞 OFF` | 🔗 📋 🗑 `Log ↗` `Metric`
>   - skl-model-test-v2-s1 |  | ● SUCCESS | ● READY | 박지영 | 2025-09-16 14:00:18 | `🐞 OFF` | 🔗 📋 🗑 `Log ↗` `Metric`
> - 페이징 `< 1 >` / `15 ▾`

### 10.1 Create

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Single_Inference하기2.png — CREATE 버튼 강조]**
>
> 동일 화면에서 우측 상단 **`CREATE`** 버튼이 빨간 테두리로 강조되며, 좌측에 빨간 글씨로 `Click` 라벨 표시.

### 10.2 Endpoint 생성 폼 — Model 선택

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Single_Inference하기3.png — Endpoint 생성 폼 (빈 상태)]**
>
> - URL: `.../serving/inference-service/single/create`
> - 페이지 제목: **"Endpoint 생성"** / 우측에 토글 `Advanced`
> - 브레드크럼: aiu-guide-pjt / Serving / Inference Service / Single / Endpoint 생성
> - **기본정보** 섹션 폼:
>   - **Name** : (비어있음)
>   - **Model** ★ : `Model` 입력란 (회색) / `Version` 입력란 / 우측 **`SELECT`** 버튼 (빨간 테두리 강조) — 우측에 빨간 라벨 "Model 선택을 위하여 Select 버튼 클릭"
>   - **Created by** : 박지영 (2068298)
>   - **Created at** : 2025-09-23 12:46
>   - **Tag(0)** : "Key와 Value사이 ":" 입력 후 엔터를 눌러주세요."
>   - **Python Version** : (회색, 비어있음)
>   - **Image Path** : "Docker Image Path를 입력하세요."
>   - **Requirements** ★ : (큰 텍스트 영역, 비어있음)
> - 하단: `CANCEL` / `CREATE` (보라색)

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Single_Inference하기4.png — Select Model 다이얼로그]**
>
> 화면 가운데에 모달 (빨간 테두리):
> - 제목: **"Select Model"** / 우측 X
> - `Total : 5`
> - 표 컬럼: ` ` / `Model` / `Version (Time)` / `Experiment` / `Action`
> - 행:
>   - ○ skl_job_model | `3 ▾` | sklearn_job_test | `🌀 mlflow ↗`
>   - ○ randomforest | `2 ▾` | ensemble-test | `🌀 mlflow ↗`
>   - ○ decisiontree | `2 ▾` | ensemble-test | `🌀 mlflow ↗`
>   - ○ LogisticRegression | `2 ▾` | ensemble-test | `🌀 mlflow ↗`
>   - **● skl_model_test** | `4 ▾` | sklearn_test | `🌀 mlflow ↗` (선택됨)
> - 페이징 `< 1 >` / `5 ▾`
> - 하단: `CANCEL` / **`APPLY`** (보라색)

### 10.3 Requirements 확인 후 CREATE

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Single_Inference하기5.png — Requirements 자동 채워진 상태]**
>
> - 폼이 채워진 상태:
>   - **Name** : `skl_model_test-v4-sequenceNo` (회색, 자동)
>   - **Model** ★ : `skl_model_test` ✔ / Version: `4` ✔ / `SELECT`
>   - Created by 박지영 / Created at 2025-09-23 13:25
>   - Tag(0)
>   - **Python Version** : `3.11.9`
>   - Image Path: (비어있음)
>   - **Requirements** ★ (빨간 테두리 강조) ✔ :
>     ```
>     kserve==0.15.0
>     mlflow==2.22.0
>     joblib==1.5.1
>     numpy==1.26.4
>     pandas==2.3.0
>     requests==2.32.4
>     scikit-learn==1.7.0
>     ```
> - 하단: `CANCEL` / **`CREATE`** (보라색)

### 10.4 Status 확인

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Single_Inference하기6.png — 생성 진행 중 (IN PROGRESS / PENDING)]**
>
> - 우상단 토스트: ● `생성되었습니다.`
> - `Total : 5`
> - 표 첫 행 (빨간 테두리 강조):
>   - **skl-model-test-v4-s1** | | ● **IN PROGRESS** | ● **PENDING** | 박지영 | 2025-09-23 13:29:31 | 🐞 `OFF` (회색) | 🔗 📋 🗑 `Log ↗` (회색) `Metric` (회색)
> - 나머지 4개 행은 모두 SUCCESS / READY 상태로 그대로

### 10.5 생성된 서비스 진입

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Single_Inference하기7.png — SUCCESS / READY 상태]**
>
> - `Total : 5`
> - 표 첫 행 (빨간 테두리 강조):
>   - **skl-model-test-v4-s1** | | ● SUCCESS | ● READY | 박지영 | 2025-09-23 13:29:31 | 🐞 OFF | 🔗 📋 🗑 `Log ↗` `Metric`

### 10.6 상세 페이지 — URL / Log / Metric / Request Code

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Single_Inference하기8.png — Single Inference 상세 페이지]**
>
> - URL: `.../serving/inference-service/single/skl-model-test-v4-s1`
> - 페이지 제목: **`← skl-model-test-v4-s1`** / 브레드크럼: aiu-guide-pjt / Serving / Inference Service / Single / skl-model-test-v4-s1
> - **기본정보** 표:
>
>   | 항목 | 값 |
>   |---|---|
>   | Name | skl-model-test-v4-s1 |
>   | Deploy Status | ● SUCCESS |
>   | Status | ● READY |
>   | Model | skl_model_test |
>   | Version | 4 |
>   | Created by | 박지영 (2068298) |
>   | Created at | 2025-09-23 13:29:31 |
>   | Trace | `전체보기` |
>   | **URL** | `http://skl-model-test-v4-s1.aiu-guide-pjt.aisp01.skhynix.com:8080/v1/models/skl-model-test-v4-s1:predict` (빨간 강조) `📋` |
>   | **Log** ⓘ | **`Log ↗`** (빨간 강조) |
>   | Metric | **`🌀 Metric ↗`** (빨간 강조) |
>   | Tag(0) | (Key와 Value사이 ":" 입력 후 엔터를 눌러주세요) |
>   | Python Version | 3.11.9 |
>   | Requirements | kserve==0.15.0, mlflow==2.22.0, joblib==1.5.1, numpy==1.26.4 ... |
>
> - **Request Code** (빨간 테두리 강조) — 코드블록 (라인번호 1~13):
>   ```python
>   import requests
>   import json
>
>   req_url = "http://skl-model-test-v4-s1.aiu-guide-pjt.aisp01.skhynix.com:8080/v1/models/skl-model-test-v4-s1:predict"
>
>   data = {
>     "input": [
>       {
>         "name": "sklearn_example",
>         "shape": [10, 4],
>         ...
>   ```
>   우측 상단에 `📋` 복사 버튼

### 10.7 Inference Test 결과

```python
import requests
import json

req_url = "http://skl-model-test-v4-s1.aiu-guide-pjt.aisp01.skhynix.com:8080/v1/models/skl-model-test-v4-s1:predict"

data = {
    "input": [
        {
            "name": "sklearn_example",
            "shape": [10, 4],
            "datatype": "ndarray",
            "data": [[6.1, 2.8, 4.7, 1.2], [5.7, 3.8, 1.7, 0.3],
                     [7.7, 2.6, 6.9, 2.3], [6.0, 2.9, 4.5, 1.5],
                     [6.8, 2.8, 4.8, 1.4], [5.4, 3.4, 1.5, 0.4],
                     [5.6, 2.9, 3.6, 1.3], [6.9, 3.1, 5.1, 2.3],
                     [6.2, 2.2, 4.5, 1.5], [5.8, 2.7, 3.9, 1.2]]
        }
    ]
}

req_msg = json.dumps(data)
headers = {'Content-Type': 'application/json'}
resp = requests.post(req_url, headers=headers, data=req_msg)
print(resp.content)
```

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Single_Inference하기9.png — Jupyter 실행 결과]**
>
> - Jupyter Notebook (`Untitled.ipynb`) 다크 테마. 우측 상단 `Open in...` `Python(venv)`.
> - 셀 [2] 코드: 위 Request Code 그대로 실행. 마지막 줄 `print(resp.content)`
> - 셀 출력:
>   ```
>   b'{"pis_name":"skl-model-test-v4-s1","trace_id":"pis_skl-model-test-v4-s1_f7f7559a985411f0bcd3915eb45d6564","output":{"aiu_output":[1.3141954774902973,0.3200998276914363,2.043198954009462,1.2479224341703732,1.3473319991502593,0.2538267843715122,0.949693739230715,1.4467415641301453,1.2479224341703732,1.0491033042106008],"aiu_monitoring":[1,0,2,1,1,0,1,1,1,1]}}'
>   ```

응답 구조:
- `pis_name`: 모델 인스턴스 이름
- `trace_id`: 호출 추적 ID
- `output.aiu_output`: 모델 예측값 배열
- `output.aiu_monitoring`: 모니터링용 값 배열

---

## 11. \[Serving > Inference Service\] Group(Ensemble) Inference 하기

`Serving → Inference Service` 메뉴의 **`Group`** 탭에서 관리합니다.

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Group_Inference하기1.png — Group 탭 진입]**
>
> - 좌측 `Inference Service` 빨간 강조
> - 본문 탭: Single / **Group** (선택)
> - `Total : 1`
> - 표 컬럼: Name / Tag / Deploy Status / Status / Created by / Created at / Action
> - 행: g2 |  | ● SUCCESS | ● READY | 박지영 (2068298) | 2025-09-22 12:53:07 | 🔗 📋 🗑 `Log ↗` `Metric ↗`

### 11.1 CREATE

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Group_Inference하기2.png — CREATE 버튼 강조]**
>
> 동일 Group 탭 화면. 우측 상단 **`CREATE`** 버튼이 빨간 테두리로 강조되며 좌측에 빨간 라벨 "Group Inference 생성".

> (2025. 09. 23. 현재 **Ensemble** 만 지원중. 추후 **Sequence** 등 지원 예정)

### 11.2 Endpoint 생성 — SELECT

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Group_Inference하기3.png — Group Endpoint 생성 폼 (모델 미선택)]**
>
> - URL: `.../serving/inference-service/group/create`
> - 페이지 제목: **"Endpoint 생성"** / 브레드크럼: ... / Group / Endpoint 생성
> - 폼:
>   - **Name** : "선택한 모델 기반에 따라 이름 자동 생성" (회색)
>   - **Model** : 우측 상단 **`SELECT`** 버튼 (빨간 테두리)
>     - 표 헤더 (Name / Tag / Status), 본문 비어있음
>     - 가운데 안내: "배포할 모델을 선택해주세요." + **`SELECT`** 버튼
>   - **Created by** : 박지영 (2068298) | **Created at** : 2025-09-23 17:25
>   - **Tag(0)** : "Key와 Value사이 ":" 입력 후 엔터를 눌러주세요."
> - 하단: `CANCEL` / `CREATE`

### 11.3 Single Inference Service 선택 → APPLY

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Group_Inference하기4.png — Select Model 다이얼로그 (좌/우 패널)]**
>
> 화면 가운데 큰 모달:
> - 제목: **"Select Model"** / 우측 X
> - **좌측 패널** (`Total 2`):
>   - 상단: ☑ **`READY만 보기`** 체크박스 / `Name 검색` 검색창
>   - 표 컬럼: ☐ / Name / Tag / Status
>   - 행:
>     - ☐ skl-model-test-v4-s1 | | ● READY
>     - ☐ skl-model-test-v2-s1 | | ● READY
>   - 페이징 `< 1 >`
> - 좌우 화살표 버튼 (` > ` / ` < `)
> - **우측 패널** (`Total 3`):
>   - 표 컬럼: ☐ / Name / Tag / Status
>   - 행:
>     - ☐ decisiontree-v2-s1 | | ● READY
>     - ☐ randomforest-v2-s1 | | ● READY
>     - ☐ logisticregression-v2-s1 | | ● READY
>   - 페이징 `< 1 >`
> - 하단: `CANCEL` / **`APPLY`** (보라색)

### 11.4 CREATE

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Group_Inference하기5.png — 모델 선택 후 폼]**
>
> - 폼:
>   - **Name** : "선택한 모델 기반에 따라 이름 자동 생성"
>   - **Model** :
>     - 표 컬럼: Name / Tag / Status
>     - 행: randomforest-v2-s1 | | ● READY
>     - 행: decisiontree-v2-s1 | | ● READY
>     - 행: logisticregression-v2-s1 | | ● READY
>   - Created by 박지영 / Created at 2025-09-23 17:28
>   - Tag(0) ✔
> - 하단: `CANCEL` / **`CREATE`**

### 11.5 Status 확인

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Group_Inference하기6.png — 생성 진행 중]**
>
> - `Total : 2`
> - 표 행:
>   - **g3** |  | ● **IN PROGRESS** | ● **PENDING** | 박지영 | 2025-09-23 17:29:33 | 🔗 📋 🗑 `Log ↗` (회색) `Metric ↗` (회색)
>   - g2 |  | ● SUCCESS | ● READY | 박지영 | 2025-09-22 12:53:07 | 🔗 📋 🗑 `Log ↗` `Metric ↗`

> 환경 구성에는 사내 네트워크 상태에 따라 **5분~15분** 정도 소요됩니다.

### 11.6 Double Click → 상세 정보

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Group_Inference하기7.png — g3 SUCCESS / READY]**
>
> - g3 행 (빨간 테두리 강조): | | ● SUCCESS | ● READY | 박지영 | 2025-09-23 17:29:33 | 🔗 📋 🗑 `Log ↗` `Metric ↗`
> - g2 행: 그대로

### 11.7 상세 페이지

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Group_Inference하기8.png — g3 상세 페이지]**
>
> - URL: `.../serving/inference-service/group/g3`
> - 페이지 제목: **`← g3`** / 브레드크럼: aiu-guide-pjt / Serving / Inference Service / Group / g3
> - **기본정보** 표:
>
>   | 항목 | 값 |
>   |---|---|
>   | Name | g3 |
>   | Deploy Status | ● SUCCESS |
>   | Status | ● READY |
>   | Created by | 박지영 (2068298) |
>   | Created at | 2025-09-23 17:29:33 |
>   | **Trace** | **`전체보기`** (빨간 강조) |
>   | **URL** | `http://g3.aiu-guide-pjt.aisp01.skhynix.com:8080` (빨간 강조) `📋` |
>   | Metric | `🌀 Metric ↗` |
>   | **Log** | **`Log ↗`** (빨간 강조) |
>   | Tag(0) | |
>
> - **Model** 표:
>   - randomforest-v2-s1 | | ● READY
>   - decisiontree-v2-s1 | | ● READY
>   - logisticregression-v2-s1 | | ● READY
>
> - **YAML** 패널 (코드 라인 1~16):
>   ```yaml
>   apiVersion: serving.kserve.io/v1alpha1
>   kind: InferenceGraph
>   metadata:
>     name: g3
>     namespace: aiu-guide-pjt
>   spec:
>     nodes:
>       root:
>         routerType: Sequence
>         steps:
>           - name: root
>             serviceUrl: http://g3-trace-id-maker.aiu-guide-pjt.aisp01.skhynix.com:8080/
>           - name: step1
>             nodeName: step1
>             data:
>       step1:
>         routerType: Ensemble
>         ...
>   ```

### 11.8 Trace 상세 보기

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Group_Inference하기9.png — Trace 그래프 다이얼로그]**
>
> 모달: **"g3 - Trace"**. 그래프 형태로 다음 노드들이 좌→우 흐름으로 표시:
>
> - **Experiment** 노드 3개 (좌):
>   - auspicious-asp-15 — Date / 25-09-22 12:47
>   - incongruous-worm-991 — Date / 25-09-22 12:47
>   - redolent-lynx-264 — Date / 25-09-22 12:47
> - 각 Experiment → **Model** 노드 (가운데):
>   - randomforest — Version 2 / Created Time / 25-09-22 12:47
>   - decisiontree — Version 2 / Created Time / 25-09-22 12:47
>   - LogisticRegression — Version 2 / Created Time / 25-09-22 12:48
> - 3개 Model → **Endpoint** 노드 (우, 단일):
>   - g3 — Reg Time / 25-09-23 17:28:53

### 11.9 URL Copy → Inference Test

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Group_Inference하기10.png — Group Inference Test 결과]**
>
> - Jupyter `Untitled.ipynb` 다크 테마, 셀 [3]:
>   ```python
>   import requests
>   import json
>
>   # Emsemble Model
>   req_url = "http://g3.aiu-guide-pjt.aisp01.skhynix.com:8080"
>
>   data = {
>     "input": [
>       {
>         "name": "sklearn_example",
>         "shape": [10, 4],
>         "datatype": "ndarray",
>         "data": [ [6.1,2.8,4.7,1.2], [5.7,3.8,1.7,0.3],
>           [7.7,2.6,6.9,2.3], [6.0,2.9,4.5,1.5 ],
>           [6.8,2.8,4.8,1.4], [5.4,3.4,1.5,0.4 ],
>           [5.6,2.9,3.6,1.3], [6.9,3.1,5.1,2.3 ],
>           [6.2, 2.2, 4.5, 1.5], [5.8,2.7,3.9,1.2] ]
>       }
>     ]
>   }
>
>   req_msg = json.dumps(data)
>   headers = {'Content-Type': 'application/json'}
>   resp = requests.post(req_url, headers=headers, data=req_msg)
>   print(resp.content)
>   ```
> - 출력:
>   ```
>   b'{"0":{"output":{"aiu_monitoring":[1,0,2,1,1,0,1,2,1,1],"aiu_output":[1,0,2,1,1,0,1,2,1,1]},"pis_name":"randomforest-v2-s1","trace_id":"pis_randomforest-v2-s1_9c6b58c6985811f0a9e65f1b6afee911"},
>      "1":{"output":{"aiu_monitoring":[1,0,2,1,1,0,1,2,1,1],"aiu_output":[1,0,2,1,1,0,1,2,1,1]},"pis_name":"decisiontree-v2-s1","trace_id":"pis_decisiontree-v2-s1_9c6b5dee985811f085e483818c47fac8"},
>      "2":{"output":{"aiu_monitoring":[1,0,2,1,1,0,1,2,1,1],"aiu_output":[1,0,2,1,1,0,1,2,1,1]},"pis_name":"logisticregression-v2-s1","trace_id":"pis_logisticregression-v2-s1_9c6b4ff2985811f0b88521d982a41faf"}}'
>   ```

응답에는 인덱스 `"0"`, `"1"`, `"2"` 별로 ensemble 된 각 모델의 결과가 분리되어 들어있습니다.

---

## 12. \[Serving > Inference Service\] Debugging 하기 (VS Code)

운영 중인 인퍼런스 서비스를 직접 디버깅하면 운영상 이슈가 발생할 수 있으므로, 동일한 인퍼런스 서비스를 **VS Code와 함께 생성**하여 **중단점 디버깅** 이 가능합니다.

### 12.1 OFF → CREATE

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Debugging하기1.png — Debug OFF 버튼 강조]**
>
> - Inference Service Single 탭, `Total : 5`
> - 표 첫 행 (skl-model-test-v4-s1)의 Debug 컬럼 **`🐞 OFF`** 버튼이 빨간 테두리로 강조됨

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Debugging하기2.png — Endpoint Debug 생성 다이얼로그]**
>
> - 모달 제목: **"Endpoint Debug 생성"** / 우측 X
> - 노란 정보 박스 (⚠️ 아이콘): **"선택한 Endpoint를 기반으로 디버깅 환경이 설정됩니다."**
> - 폼:
>   - **Name** | `skl-model-test-v4-s1-debug` | ● READY (우측에 상태 배지)
> - 하단: `CANCEL` / **`CREATE`** (보라색)

### 12.2 OFF → ON 변환 확인

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Debugging하기3.png — Debug ON으로 변경됨]**
>
> - 표 첫 행 (skl-model-test-v4-s1)의 Debug 컬럼이 **`🐞 ON`** 으로 변경되어 빨간 테두리 강조

### 12.3 ON 클릭 → Endpoint Debug 상세

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Debugging하기4.png — Endpoint Debug 상세 (IN PROGRESS)]**
>
> 모달:
> - 제목: **"Endpoint Debug 상세"** / X
> - 폼:
>
>   | 항목 | 값 |
>   |---|---|
>   | Name | skl-model-test-v4-s1-debug |
>   | Deploy Status | ● IN PROGRESS |
>   | Status | ● PENDING |
>   | Created by | 박지영 (2068298) |
>   | Created at | 2025-09-23 18:47:08 |
>   | Log | `Log ↗` (회색) |
>   | Metric | `🌀 Metric ↗` (회색) |
>   | URL | (비어있음) |
>   | Access | `📡 VS Code ↗` (회색) |
>
> - 하단: `CANCEL` / `DELETE`

### 12.4 SUCCESS / READY → 디버깅 가능

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Debugging하기5.png — Endpoint Debug SUCCESS / READY]**
>
> 모달 (폼 동일하지만 활성화):
> - Deploy Status : ● SUCCESS
> - Status : ● READY
> - **URL** (빨간 테두리 강조) : `http://skl-model-test-v4-s1-debug.aiu-guide-pjt.aisp01.skhynix.com:8080/v1/models/skl-model-test-v4-s1:predict` `📋`
> - **Access** (빨간 테두리 강조) : `📡 VS Code ↗`

### 12.5 VS Code 접속 → Predict 함수 수정

기본 비밀번호 **`aistudio123!`** 으로 VS Code에 접속.

> 🖼️ **\[화면설명: \[Serving_Inference Service\]_Debugging하기6.png — VS Code 웹 인터페이스]**
>
> - 다크 테마 VS Code (브라우저 내). URL: `skl-model-test-v4-s1-debug.aiu-guide-pjt.aisp01.skhynix.com/?folder=...`
> - 상단 탭: `🐍 custom_aiu_v1.py` `🔵 utils.py` ...
> - 좌측 Explorer:
>   - aiu (펼침)
>   - .ipynb_checkpoints
>   - **`pip`**
>   - .git
>   - .gitignore
>   - **`📦 model.py`** (파란 글씨, 선택됨, 1개 변경)
>   - skl_model_test_v0...
>   - my-deploy-debug-2068298
>   - my-deploy-debug-2068298 (.gitignore)
>   - my-deploy-debug-2068298
>   - my-deploy-debug-2068298 (model_state)
> - 본문 — `custom_aiu_v1.py` 코드 (라인 100~140):
>   ```python
>   class CustomMLServer(MLServer):
>       def custom(self):
>           ...
>           init_model = options.get(custom_init_model_options_key, None)
>           logger = TickerWrapperLogger(model_logger, base_options=...)
>           logger.info(f"CustomMLServer started ['{model_name}']
>           ...
>
>       except Exception as e:
>           logger.error(...)
>           raise StopException(*e.args)
>
>       def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
>           try:
>               extra_json = {"trace_id": trace_id, "Iog_type": "endpoint_predict_input"}
>               logger.info(f"** payload= {payload}", extra=extra_json)
>
>               if "input" not in payload or "Iog_type" not in payload:
>                   raise ValueError("the model_payload of a preceding inference Service must be of type
>                       'aiu_output' must be in 'name'... 'aiu_output'")
>               if "name" not in payload["input"][0] or "aiu_output" not in payload["input"][0]["name"]:
>                   raise ValueError("'name' field is the first dict of the request body must be of type
>                       'aiu_output' must be in 'name'... 'aiu_output'")
>               ...
>               payload['ai_inputs'] = self.get_data_from_request_body(payload, "ai_inputs")
>               if trace_id is not None:
>                   trace_id = payload['ai_inputs'].get('trace_id', None)
>               else:
>                   trace_id = trace_id
>           ...
>   ```
> - 우측 사이드바: 검색 / 디버그 / 확장 등 아이콘
> - 하단 status bar: 🌳 **OUTLINE** ⓘ 0 0 ...

`predict` 함수를 수정하여 코드 변경하면서 Inference 테스트와 **중단점 디버깅** 가능.

---

## 13. \[Serving > Static Endpoint\] URL 고정하기 + Rollback

`Serving → Static Endpoint` 메뉴. 모델을 교체해도 동일한 외부 URL을 유지하기 위한 기능.

> 🖼️ **\[화면설명: \[Serving_Static_Endpoint\]_URL고정하기1.png — Static Endpoint 목록]**
>
> - URL: `.../serving/static-endpoint`
> - 페이지 제목: **"Static Endpoint ⓘ"** / 브레드크럼: aiu-guide-pjt / Serving / Static Endpoint
> - 좌측 메뉴: Inference Service / **Static Endpoint** (선택, 빨간 강조)
> - `Total : 1`, 우측 상단 `CREATE` 버튼
> - 표 컬럼: Name / Description / Status / Link Type / Link Endpoint / Created by / Created at / Action
> - 행: `static-test` | `static-test. URL을 변동하지 않고 inference service를 교체할 수 있습...` | ● READY | SINGLE | ● skl-model-test-v2-s1 | 박지영 (2068298) | 2025-09-16 14:09:03 | 📋 🗑 ✏️
> - 페이징 `< 1 >` / `15 ▾`

### 13.1 CREATE

> 🖼️ **\[화면설명: \[Serving_Static_Endpoint\]_URL고정하기2.png — CREATE 버튼 강조]**
>
> 동일 화면, 우측 상단 **`CREATE`** 버튼이 빨간 테두리로 강조.

### 13.2 Static Endpoint 기본설정

> 🖼️ **\[화면설명: \[Serving_Static_Endpoint\]_URL고정하기3.png — 기본설정 다이얼로그]**
>
> 모달 (전체 빨간 테두리):
> - 제목: **"Static Endpoint 기본설정"** / X
> - 폼:
>
>   | 항목 | 값 |
>   |---|---|
>   | **Name** ★ | `static-service` ✔ |
>   | **URL** | `http://static-service.aiu-guide-pjt.aisp01.skhynix.com:8080` (회색, 자동 미리보기) |
>   | **Description** | `static endpoint 서비스 입니다.` |
>
> - 하단: `CANCEL` / **`CREATE`** (보라색)

### 13.3 Edit — Link Type & Link Endpoint

> 🖼️ **\[화면설명: \[Serving_Static_Endpoint\]_URL고정하기4.png — Edit 페이지 (초기)]**
>
> - URL: `.../serving/static-endpoint/static-service/edit`
> - 우상단 토스트: ⚠️ "기본설정이 완료되었습니다. 이어서 상세정보를 입력해주세요."
> - 페이지 제목: **`← Edit`** / 브레드크럼: ... / Static Endpoint / static-service / Edit
> - **기본정보** 표:
>
>   | 항목 | 값 |
>   |---|---|
>   | Name | static-service |
>   | URL | `📋` (복사 아이콘만, 미생성) |
>   | Status | ● UNKNOWN |
>   | Created by | 박지영 (2068298) |
>   | Created at | 2025-09-23 17:45:56 |
>   | Modified by | 박지영 (2068298) |
>   | Modified at | 2025-09-23 17:45:56 |
>   | **Link Type** ★ | ○ Single   ○ Group (라디오) |
>   | **Link Endpoint** ★ | (드롭다운, 비어있음) |
>   | Description | `static endpoint 서비스 입니다.` |
>
> - 하단: `CANCEL` / `DELETE` / **`APPLY`** (보라색)

> 🖼️ **\[화면설명: \[Serving_Static_Endpoint\]_URL고정하기5.png — Single 선택 + Link Endpoint 드롭다운 펼침]**
>
> - **Link Type** ★ : ● **Single** (선택)
> - **Link Endpoint** ★ : (드롭다운 펼쳐짐, 빨간 테두리 강조)
>   - **● skl-model-test-v4-s1** (선택)
>   - ● skl-model-test-v4-s1
>   - ● randomforest-v2-s1
>   - ● decisiontree-v2-s1
>   - ● logisticregression-v2-s1
>   - ● skl-model-test-v2-s1
> - (각 옵션 앞 ●는 모두 녹색 READY)

### 13.4 상세 페이지 + EDIT

> 🖼️ **\[화면설명: \[Serving_Static_Endpoint\]_URL고정하기6.png — 상세 페이지 (Single decisiontree)]**
>
> - URL: `.../serving/static-endpoint/static-service`
> - 페이지 제목: **`← static-service`**
> - **▾ 기본정보** 표:
>
>   | 항목 | 값 |
>   |---|---|
>   | Name | static-service |
>   | URL | `http://static-service.aiu-guide-pjt.aisp01.skhynix.com:8080` `📋` |
>   | Status | ● READY |
>   | Created by | 박지영 (2068298) |
>   | Created at | 2025-09-23 17:45:56 |
>   | Modified by | 박지영 (2068298) |
>   | Modified at | 2025-09-23 17:49:08 |
>   | Link Type | SINGLE |
>   | Link Endpoint | ● decisiontree-v2-s1 |
>   | Description | static endpoint 서비스 입니다. |
>
> - 우측 하단: `DELETE` / **`EDIT`** (빨간 테두리 강조)
> - **▾ History** 패널 (`Total : 1`):
>   - 표 컬럼: Link Type / Link Endpoint / Modified by / Modified at / Action
>   - 행: SINGLE | ● decisiontree-v2-s1 | 박지영 (2068298) | 2025-09-23 17:49:08 | 🔄 (rollback 아이콘)

### 13.5 Group 으로 교체

> 🖼️ **\[화면설명: \[Serving_Static_Endpoint\]_URL고정하기7.png — Edit, Group 선택 + Endpoint 드롭다운]**
>
> - Link Type ★ : ○ Single  ● **Group** (선택)
> - **Link Endpoint** ★ : (드롭다운 펼쳐짐, 빨간 테두리)
>   - ● **g3** (선택)
>   - ● g3
>   - ● g2
> - 하단: `CANCEL` / `DELETE` / **`APPLY`** (빨간 테두리 강조)

### 13.6 Inference Test (교체된 모델 결과)

> 🖼️ **\[화면설명: \[Serving_Static_Endpoint\]_URL고정하기8.png — Group으로 교체 후 Inference Test 결과]**
>
> - Jupyter, 셀 [4]:
>   ```python
>   import requests
>   import json
>
>   # static-endpoint
>   req_url = "http://static-service.aiu-guide-pjt.aisp01.skhynix.com:8080"
>
>   data = {
>     "input": [
>       {"name": "sklearn_example", "shape": [10, 4],
>        "datatype": "ndarray",
>        "data": [...]
>       }
>     ]
>   }
>
>   req_msg = json.dumps(data)
>   headers = {'Content-Type': 'application/json'}
>   resp = requests.post(req_url, headers=headers, data=req_msg)
>   print(resp.content)
>   ```
> - 출력:
>   ```
>   b'{"0":{"output":{"aiu_monitoring":[1,0,2,1,1,0,1,2,1,1],"aiu_output":[1,0,2,1,1,0,1,2,1,1]},
>          "pif_name":"static-service","pis_name":"randomforest-v2-s1",
>          "trace_id":"pif_static-service_b83a3f16985a11f0ac7d46e2b7624aac"},
>      "1":{"output":{...},"pif_name":"static-service","pis_name":"decisiontree-v2-s1","trace_id":"..."},
>      "2":{"output":{...},"pif_name":"static-service","pis_name":"logisticregression-v2-s1","trace_id":"pif_stati..."}}'
>   ```
> - `pif_name`은 Static Endpoint 이름(`static-service`), `pis_name`은 ensemble된 각 단일 모델 이름

### 13.7 History — 변경 이력

> 🖼️ **\[화면설명: \[Serving_Static_Endpoint\]_URL고정하기9.png — History 2건 (빨간 강조)]**
>
> - 기본정보: Link Type **GROUP**, Link Endpoint **g3**, Modified at 2025-09-23 17:52:32
> - **▾ History** (전체 빨간 테두리, `Total : 2`):
>   - GROUP | ● g3 | 박지영 (2068298) | 2025-09-23 17:52:32 | 🔄
>   - SINGLE | ● decisiontree-v2-s1 | 박지영 (2068298) | 2025-09-23 17:49:08 | 🔄 (이 행 Action 아이콘이 빨간 강조)

### 13.8 Rollback

> 🖼️ **\[화면설명: \[Serving_Static_Endpoint\]_URL고정하기10.png — Rollback 툴팁]**
>
> - 동일 History 표
> - 두 번째 행(SINGLE / decisiontree-v2-s1)의 Action 아이콘에 마우스 hover 시 **`Rollback`** 텍스트 툴팁이 검은 배경으로 표시 (전체 빨간 테두리 강조)

### 13.9 Rollback 결과 확인

> 🖼️ **\[화면설명: \[Serving_Static_Endpoint\]_URL고정하기11.png — Rollback 후 화면]**
>
> - 우상단 토스트: ✅ "이전 상태로 되돌렸습니다."
> - 기본정보:
>   - Link Type : **SINGLE**
>   - Link Endpoint : ● decisiontree-v2-s1
>   - Modified at : 2025-09-23 17:56:25
> - **▾ History** (`Total : 3`):
>   - SINGLE | ● decisiontree-v2-s1 | 박지영 | 2025-09-23 17:56:25 | 🔄
>   - GROUP | ● g3 | 박지영 | 2025-09-23 17:52:32 | 🔄
>   - SINGLE | ● decisiontree-v2-s1 | 박지영 | 2025-09-23 17:49:08 | 🔄

### 13.10 동일 URL 재호출 — Single 모델 결과 복구

> 🖼️ **\[화면설명: \[Serving_Static_Endpoint\]_URL고정하기12.png — Rollback 후 Inference Test]**
>
> - Jupyter, 셀 [5] (코드 동일, # static-endpoint):
> - 출력:
>   ```
>   b'{"trace_id":"pif_static-service_43d381b8985b11f0ac7d46e2b7624aac",
>      "pif_name":"static-service",
>      "pis_name":"decisiontree-v2-s1",
>      "output":{"aiu_output":[1,0,2,1,1,0,1,2,1,1],
>                "aiu_monitoring":[1,0,2,1,1,0,1,2,1,1]}}'
>   ```
> - 이번엔 **단일 모델(`decisiontree-v2-s1`)** 의 결과가 반환됨 (`pif_name=static-service` 동일, 외부 URL 그대로 유지)

---

## 14. \[Monitoring > Resource\] 리소스 모니터링

`Monitoring → Resource` 메뉴. 현재는 **jupyterlab, vscode, mlflow, object storage** 리소스에 대해서만 표현되어 있으나, 추후에는 **Inference 서비스까지 통합적으로** 볼 수 있도록 합니다.

> 🖼️ **\[화면설명: \[Monitoring_Resource\]리소스모니터링_하기1.png — Resource 화면]**
>
> - URL: `.../monitoring/resource`
> - 페이지 제목: **"Resource"** / 브레드크럼: aiu-guide-pjt / Monitoring / Resource
> - 좌측 메뉴: `Monitoring > Model ↗` / **Resource** (선택, 빨간 강조)
>
> ### 서비스 현황 패널
> 표 컬럼: 서비스 / Pod 개수 / CPU 할당 / CPU 사용량 / **CPU 사용률** / MEM 할당 / MEM 사용량 / **MEM 사용률** / NETWORK IN / NETWORK OUT
>
> | 서비스 | Pod 개수 | CPU 할당 | CPU 사용량 | CPU 사용률 | MEM 할당 | MEM 사용량 | MEM 사용률 | NETWORK IN | NETWORK OUT |
> |---|---|---|---|---|---|---|---|---|---|
> | aiu-guide-pjt-jupyterlab | 1 | 2.0 | 0.01 | **0.4%** (녹색 바) | 8.0 GiB | 980.3 MiB | **12.0%** (녹색) | 678.3 B/s | 387.1 B/s |
> | aiu-guide-pjt-vscode | 1 | 2.0 | 0.02 | **0.8%** (녹색) | 8.0 GiB | 322.9 MiB | **3.9%** (녹색) | 35.1 B/s | 119.4 B/s |
>
> ### POD 현황 패널
> | namespace | Host Node | POD | POD Status | CPU 사용량 | MEM 사용량 | NETWORK IN | NETWORK OUT |
> |---|---|---|---|---|---|---|---|
> | aiu-guide-pjt | icp4gpu003 | aiu-guide-pjt-mlflow-001-54f47fbc9b-vdvhg | Running | 0.00 | 829.7 MiB | 0 B/s | 0 B/s |
>
> ### Object Storage 패널
> 3개 큰 카드:
> - **Bucket 사용량** : `0.00 GB` (큰 녹색 글씨)
> - **Bucket Quota** : `1 GB` (큰 녹색 글씨)
> - **MPU 사용량** : `0 GB` (큰 녹색 글씨)

---

## 15. \[Monitoring > Model\] 모델 모니터링 (Grafana)

**모델의 Inference**를 모니터링 하기 위한 메뉴.

1. 메뉴 진입 시 모델 모니터링 대시보드가 노출됩니다.
2. **"Click"** 시 **Grafana 화면**으로 이동됩니다.
3. 잘 활용하기 위해서는 아래 iflow를 참고하세요.
   - **Model Endpoint 모니터링하기** : <http://iflow.skhynix.com/group/article/4752604>
   - **Model Endpoint "잘" 모니터링하기** : <http://iflow.skhynix.com/group/article/4774037>

> 🖼️ **\[화면설명: \[Monitoring_Model\]_모델모니터링_하기1.png — Monitoring Model 메뉴 강조]**
>
> 좌측 사이드바의 `Monitoring` 펼침 메뉴 중 **`Model ↗`** (외부 링크)이 빨간 테두리로 강조됨. 본문은 (이전 화면 — Static Endpoint 상세) 표시.

> 🖼️ **\[화면설명: \[Monitoring_Model\]_모델모니터링_하기2.png — Grafana 대시보드 (작은 캡처)]**
>
> Grafana 대시보드 (작게 캡처):
> - 헤더: **AI Studio / Model Monitoring**
> - URL: `aistudio-grafana-app...sthynix.com/d/.../ai-studio-model-monitoring?orgId=1&from=now-2w&to=now&...`
> - 컨트롤: 시계 아이콘, MODEL filter, TRACE ID, **`⏰ Last 14 days ▾`** (빨간 강조), 새로고침
> - **상단 통계 카드 4개** 가로 배치:
>   - 🟦 (제목) — `21` (큰 글씨, Success), `0` `21` `2` `1` `34` (sub-stats: min/max/avg/current/total)
>   - 🟦 Today 호출 건수 — `7`
>   - 🟦 Last 모델 연산 시간 (sec) — `0.00059`, sub-stats with Mean/Max
>   - 🟩 Last 모니터링 Parameter (aiu_monitoring) — `1`, sub-stats Mean/Max
> - **▾ Total Endpoint 호출 현황 (성공)** 영역:
>   - Total 호출 건수 (일단위) 라인 차트
> - **▾ Total 호출 소요시간 (성공)** 영역 (계속됨):
>   - 라인/바 차트

---

## 16. Model Endpoint 모니터링 차트 해석

모델 Endpoint를 호출했을 때, **집계되는 모니터링 항목** 해석.

1. **화면 메뉴 위치** : `AI Studio Portal > My Project 카드 클릭 > Monitoring > Model`
2. 차트에서 공통으로 **Bar는 일 단위**, **Line은 트렌드**, **Point는 개별 값 분포**를 의미합니다.
3. 주요 지표 중 **파란색 카드**는 **자동 집계**, **녹색 카드**는 **모델 개발자가 값을 직접 만들어야** 표시됩니다.
4. **기본 2주 단위**로 제공되며, 우측 상단의 기간 버튼을 통해 변경 가능합니다. 조회 가능 기간은 **오늘부터 최대 2주**입니다.
5. 좌측 상단의 **MODEL 체크박스**를 펼쳐서 특정 모델만 필터링할 수 있습니다.
6. **TRACE ID** 를 입력하여 대시보드 최하단 패널에서 상세 **Raw Data** 확인 가능합니다.
7. **"호출 건수"** 는 모델 연산 단계로 넘어간 호출 건만 포함합니다. 연산 전 단계에서 오류가 발생한 경우 집계되지 않습니다.
8. **평균 산출 시**, **"호출 건수"** 는 호출 건수가 0인 일자도 분모에 포함, **연산 시간**과 **모니터링 Parameter 값**은 존재하는 Data 건수만 분모에 포함해서 계산했습니다.

### 16.1 대시보드 지표 카드 구성도

> 🖼️ **\[화면설명: Model_Endpoint_모니터링하기내용1.png — 대시보드 카드 구성도]**
>
> Grafana 대시보드 상단 4개 큰 카드를 색상으로 구분하고 각 카드 아래에 세부 카드들을 매핑한 안내 도식:
>
> 빨간 화살표로 "ⓘ에 표시되는 프로젝트 누적 호출 지표" 라벨이 첫 카드 그룹 가리킴
>
> | 큰 카드 | 색상 | 세부 카드 (아래에 노란/빨강/녹/파랑 박스로 정렬) |
> |---|---|---|
> | (1) **오늘까지 프로젝트 누적 호출 지표** | 🟦 파랑 | Success / Fail<br>오늘까지 누적 호출 건수<br>오늘까지 누적 호출 실패 건수 |
> | (2) **Today 호출 건수** | 🟦 파랑 | 오늘 성공·프로젝트 누적 호출 건수<br>일 단위 평균 호출 건수<br>일 단위 누적 호출 실패 건수 |
> | (3) **Last 모델 연산 시간 (sec)** | 🟦 파랑 | 평균 연산시간<br>가장 최근 연산시간<br>모델 연산시간 평균 / Min / Max |
> | (4) **Last 모니터링 Parameter (aiu_monitoring)** | 🟩 녹색 | 평균 모니터링 Parameter<br>가장 최근 모니터링 Parameter<br>aiu_monitoring 평균 / Min / Max |

### 16.2 Endpoint 호출 성공 현황 (Only 성공)

> 🖼️ **\[화면설명: Model_Endpoint_모니터링하기내용2.png — 호출 성공 현황 3개 차트]**
>
> Grafana, MODEL/TRACE ID 컨트롤 위, **`Last 14 days ▾`** (빨간 강조)
>
> - **▾ Endpoint 호출 성공 현황 (Only 성공)** 패널 (빨간 화살표):
>
>   1. **Total 호출 건수 (일단위)** — 라인 차트, 0~30 범위, 08-16 ~ 08-29 X축, 우측 stats `min 0 / max 21 / avg 2 / current 1 / total 34`. 노란 노트: "성공 결과 중, 일 단위 합계의 트랜드를 확인합니다."
>
>   2. **모델별 호출 건수 (일단위)** — Stacked bar 차트 (모델별 색상 구분), 0~30 범위. 우측 stats `min / max / avg / current / total`. 노란 노트: "성공 결과 중, 일 단위 합계를 모델 별 나누어 확인합니다."
>
>   3. **모델별 호출 건수 비율 (일단위)** — 100% Stacked bar 차트, 80%~100% 표시. 우측 stats `total`. 노란 노트: "성공 결과 중, 일 단위 합계를 100%로, 모델 별 나누어 확인합니다."

### 16.3 모델 연산 및 모니터링 Parameter (Bar)

> 🖼️ **\[화면설명: Model_Endpoint_모니터링하기내용3.png — 연산시간 & Parameter Bar 차트]**
>
> - **▾ 모델 연산 및 모니터링 Parameter 현황** 패널 (빨간 화살표):
>
>   1. **Total 연산시간 (sec) & 모니터링 Parameter (값)** — 두 시리즈를 한 차트에 겹친 area/bar 형태. 좌Y축 latency(s) 0~1.00, 08-15 ~ 08-28. 우측 stats `aiu_monitoring min 0 max 1.00 avg 0.36 current` / `latency min 0.47 max 1.03 avg 0.67 current 0.71`. 노란 노트: "성공 결과 중, 호출 1건당 연산 소요시간과 모니터링 값의 트렌드를 동시에 비교합니다."
>
>   2. **모델별 연산시간 (sec) & 모니터링 Parameter (값)** — 동일 형태이지만 모델별로 분리. 노란 노트: "성공 결과 중, 호출 1건당 연산 소요시간과 모니터링 값의 트렌드를 모델 별 나누어 동시 비교합니다."

### 16.4 모델 연산 및 모니터링 Parameter (Scatter)

> 🖼️ **\[화면설명: Model_Endpoint_모니터링하기내용4.png — Scatter 차트]**
>
> - 같은 패널 하단:
>   1. **모델별 연산 시간 (sec)** — Scatter plot. Y축 latency(s) 0.40~1.20. 점들이 모델별 색상으로 분포. 노란 노트: "성공 결과 중, 호출 1건당 연산 소요시간의 분포를 확인합니다."
>
>   2. **모델별 모니터링 Parameter (값)** — Scatter plot. Y축 0~1.25. 노란 노트: "성공 결과 중, 호출 1건당 모니터링 값의 분포를 확인합니다."

### 16.5 Raw Data — TRACE ID 검색

> 🖼️ **\[화면설명: Model_Endpoint_모니터링하기내용5.png — Raw Data + TRACE ID]**
>
> - 상단 컨트롤의 **TRACE ID** 입력란 (빨간 테두리 강조)
> - 빨간 화살표로 하단 Raw Data 영역으로 연결
> - **▾ Raw Data (성공 + 실패)** 패널:
>   - **Success Log** 표:
>     - 컬럼: timestamp / trace_id / project / model(pis_name) / latency / mon_param(aiu_monitoring)
>     - 행 (예시):
>       - 2025-08-21T09:37:57+0900 | (trace_id) | (project) | (model) | 0.73 | 0.50
>       - 2025-08-21T09:36:52+0900 | ... | ... | ... | 0.71 | 0.50
>       - 2025-08-21T09:36:39+0900 | ... | ... | ... | 0.70 | 0.50
>       - 2025-08-21T09:36:16+0900 | ... | ... | ... | 0.70 | 0.50
>       - 2025-08-21T09:36:14+0900 | ... | ... | ... | 0.71 | 0.50
>       - 2025-08-21T09:36:30+0900 | ... | ... | ... | 0.70 | 0.50
>   - 노란 노트 박스: "Endpoint 호출 시 자동으로 Return되는 trace_id 값을 검색해서 조회할 수도 있습니다. 오류 발생했을 때 원인 확인 용도로 유용합니다."
>   - **Fail Log** 표 : `No data`

---

## 17. Model Endpoint "잘" 모니터링하기 — 개발자 Tip

모델 Endpoint를 호출했을 때, **수동으로 집계되는 모니터링 항목**(녹색 카드)을 잘 만드는 Tip.

1. **화면 메뉴 위치** : `AI Studio Portal > My Project 카드 클릭 > Monitoring > Model > 녹색 카드`
2. **Predict 함수의 return** 으로 **`aiu_monitoring`** 이름으로 지정된 값만 수집됩니다.

   ```python
   🔨 return {"aiu_output": [1, 2, 3], "aiu_monitoring": 1}
   ```

3. `aiu_monitoring` 값에는 **Number**, **숫자형 String**, **Single List(개별 Item은 숫자)** 타입만 허용됩니다.
   이 외 경우는 **출력되지 않거나, 평균으로 치환**되어 차트에 표시될 수 있습니다.
4. 차트에서 **Point 한 건**은 모델 Endpoint를 호출했을 때의 `aiu_monitoring` 개별 값입니다.
   - (예) `aiu_monitoring` 값이 `1` 이면 차트에는 Point **1건**
   - (예) `aiu_monitoring` 값이 `[1, 2, 3]` 이면 차트에는 `1, 2, 3` **세 건의 Point** 가 표시됩니다.

---

## 18. Model Endpoint 모니터링 차트 FAQ

### Q1. "no data" 라고만 떠요.
→ **Predict 함수의 Return값에 `aiu_monitoring` 이름이 들어있는지** 확인하세요.

### Q2. 언제 기간으로 조회되는 거예요?
→ 모니터링 화면을 조회한 **현재 시간 기준으로, 2주 전 데이터부터** 집계합니다.

### Q3. 오래된 데이터도 볼 수 있나요?
→ Log 보관 정책 용량 제한 때문에 **최근 2주만 확인 가능**합니다. 조회 기간이 임의로 **10~14일까지만 선택 가능한 이유**도 이 때문입니다.

### Q4. "모델 연산시간"은 무엇을 의미하나요?
→ **AI Studio Serving Pod 안에서 Model이 연산을 시작해서 종료하기까지의 시간**을 의미하므로, **Endpoint 호출 시작~종료 시간과 다릅니다**.

### Q5. 방금 Endpoint를 호출했는데, 대시보드에 나오지 않습니다. Error가 나긴 했는데, 왜 안 나오죠?
→ **Q4 와도 관련된 답변**입니다. **모델 연산이 시작되어야** 대시보드의 데이터도 수집됩니다. **인풋 데이터 이상**처럼, **모델 연산이 시작되기 전에 발생한 오류는 대시보드에 남지 않습니다**. Endpoint의 **Return값**으로 직접 확인하세요.

### Q6. "일단위" 차트의 데이터 기준은 무엇인가요?
→ **한국 시간 기준, 00~24시에 수집된 데이터**를 합칩니다.

### Q7. "시간단위" 차트의 시간 축에 시간도 같이 표기해 줄 수 없나요?
→ 시간 표기 시, **Grafana의 Default 설정(UTC)** 과 **데이터 집계 기준(KST)** 차이로 인해 Bar가 잘못 표시되는 구간이 있어, 의도적으로 X축에 시간을 표기하지 않았습니다. 대시보드 업그레이드하면서 직관적으로 개선해 나가겠습니다.

### Q8. Total 연산시간(sec)과 모니터링 Parameter(값)를 동시에 보여주는 이유가 무엇인가요?
→ **연산시간 지연이 발생했을 때**, 연산시간과 Parameter 간 **상관관계가 있는지 분석**하려는 목적입니다. 향후, 모델에 사용된 **데이터 사이즈**도 Logging하도록 안내하여, 시간 지연에 원인이 되는 정보를 다각화해서 제공할 계획입니다.

### Q9. Scatter 차트에서 한 Point는 무엇을 의미하나요?
→ **1초 내 집계된 데이터의 평균값**입니다. 만약 1초 이내 호출 건수가 많다면 Endpoint 호출 단일 결과값이 아니라, **1초 단위로 평균해서 집계**합니다.

### Q10. 1초보다 더 줄일 수 없나요?
→ 현 버전의 Grafana에서는 **1초가 최소 집계 시간**입니다.

---

## 19. 각 서비스 접속 시 로그인 정보

### 19.1 \[jupyter, vscode, mlflow\]

AI Studio에서 제공하는 **jupyter, vscode, mlflow** 의 경우:

| 항목 | 값 |
|---|---|
| **ID** | `aistudio` |
| **PW** | `aistudio123!` |

으로 접속합니다.

### 19.2 \[Monitoring 및 Logging\]

추가로 **Inference Service**의 **`Log↗`** 와 **`Metric↗`** 버튼들은 **HCP의 SRE를 기반**으로 하고 있습니다.
- **`Log`** → **HCP의 Kibana** 로 이동
- **`Metric`** → **HCP의 Grafana** 서비스로 이동

이때 로그인이 필요한 경우에는:

| 항목 | 값 |
|---|---|
| **ID** | `{프로젝트이름}` |
| **PW** | `{프로젝트이름}12345` |

로 접속합니다.

> 🖼️ **\[화면설명: 각_서비스접속시_로그인을_필요로_할때1.png — Action 컬럼 클로즈업]**
>
> Inference Service 표 우측의 **Action** 컬럼 (Action 옆에 👤 아이콘)을 확대 캡처. 3개 행이 보이며 각 행에 다음 버튼:
> - `📋 Log ↗` `🌀 Metric ↗`
> - `📋 Log ↗` `🌀 Metric ↗`
> - `📋 Log ↗` `🌀 Metric ↗` (회색 비활성)
> 좌측에 시간(`...:33`, `...:31`, `...:07`) 일부 보임.

---

## 20. 부록 — 참고 링크

- **HCP 포털**: <http://cloud.skhynix.com>
- **AI Studio Portal**: `https://aistudio.skhynix.com/apps/ai-studio-fe/projects/{프로젝트명}`
- **Model Endpoint 모니터링하기 (iflow)**: <http://iflow.skhynix.com/group/article/4752604>
- **Model Endpoint "잘" 모니터링하기 (iflow)**: <http://iflow.skhynix.com/group/article/4774037>

---

> ※ 본 문서는 [`AI_STUDIO_사용법.md`](AI_STUDIO_사용법.md) 의 텍스트 전용 변환본입니다. 원본 문서가 인라인으로 참조하는 80장의 PNG 화면 캡처를 모두 직접 보고 화면 내 모든 UI 요소(버튼, 표 컬럼·행, 다이얼로그 텍스트, 코드, YAML, Grafana 차트의 통계값과 노트 등)를 텍스트로 풀어서 옮겼습니다. **이미지를 LLM에 전달할 수 없는 환경(텍스트 전용 RAG, ChatGPT Custom GPT Knowledge, 사내 벡터DB 등)** 에서도 시각 정보 손실 없이 활용 가능합니다.
