"""
Demos(민중) 프로젝트 Alpha 0.8 - Flask 웹앱 (고도화 버전)ㅁ
=====================================================
cla-main + zircote/.claude 통합: 355개 스킬 (과학 174 + 개발도구 52 + 에이전트 117 + 가이드 12)

사용법:
  1. scientific-skills 폴더를 이 파일과 같은 위치에 복사
  2. TOKEN.TXT 파일에 API 키를 넣어두기 (같은 폴더)
  3. pip install flask requests
  4. python app.py
  5. 브라우저에서 http://localhost:10009 접속

폴더 구조:
  app.py
  TOKEN.TXT              ← API 키 (한 줄)
  scientific-skills/     ← 355개 스킬 (과학+개발+에이전트+가이드)
    ├── biopython/SKILL.md          (과학 스킬)
    ├── agent-python-pro/SKILL.md   (에이전트 스킬)
    ├── guide-python/SKILL.md       (개발 가이드)
    └── ... (나머지 스킬 폴더들)
"""

import os
import sys
import io
import csv
import json
import glob
import requests as req
from flask import Flask, request, jsonify, render_template_string, send_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 제한

# GGUF 모델 (llama-cpp-python)
gguf_model = None  # Llama 인스턴스
gguf_loaded_path = None  # 현재 로드된 모델 파일 경로

# 업로드된 CSV 데이터 (세션별 - 단순 메모리 저장)
uploaded_csv_data = {
    "filename": "",
    "headers": [],
    "rows": [],
    "summary": "",
    "raw_preview": "",
}

# 업로드된 파일 (CSV 외 범용)
uploaded_files = []  # [{filename, type, size, summary, content_preview}]
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================
# 설정
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(BASE_DIR, "scientific-skills")
TOKEN_FILE = os.path.join(BASE_DIR, "TOKEN.TXT")
PROMPTS_DIR = os.path.join(BASE_DIR, "saved-prompts")
os.makedirs(PROMPTS_DIR, exist_ok=True)

# 멀티에이전트 모델 레지스트리 (capability 태그 기반 자동 라우팅)
MODEL_REGISTRY = {
    "glm-4.7": {
        "env_id": "dev",
        "model": "GLM-4.7",
        "url": "http://dev.hcp.llm.skhynix.com/v1/chat/completions",
        "name": "4.7",
        "capabilities": {"text", "code", "fast"},
        "context_window": 128000,
        "priority": 3,
        "cost_tier": "low",
    },
    "qwen3.5-397b": {
        "env_id": "prod",
        "model": "Qwen3.5-397B-A17B",
        "url": "http://dev.hcp.llm.skhynix.com/v1/chat/completions",
        "name": "PROD (397B)",
        "capabilities": {"text", "code", "analysis", "large"},
        "context_window": 128000,
        "priority": 1,
        "cost_tier": "high",
    },
    "gpt-oss-120b": {
        "env_id": "common",
        "model": "gpt-oss-120b",
        "url": "http://dev.hcp.llm.skhynix.com/v1/chat/completions",
        "name": "COMMON (120B)",
        "capabilities": {"text", "code", "medium"},
        "context_window": 128000,
        "priority": 2,
        "cost_tier": "medium",
    },
    "qwen3-vl-235b": {
        "env_id": "vl-large",
        "model": "Qwen3-VL-235B-A22B-Instruct",
        "url": "http://dev.hcp.llm.skhynix.com/v1/chat/completions",
        "name": "VL-235B (Vision)",
        "capabilities": {"text", "code", "vision", "analysis", "large"},
        "context_window": 128000,
        "priority": 1,
        "cost_tier": "high",
    },
    "qwen2.5-vl-72b": {
        "env_id": "vl-medium",
        "model": "Qwen2.5-VL-72B-Instruct",
        "url": "http://dev.hcp.llm.skhynix.com/v1/chat/completions",
        "name": "VL-72B (Vision)",
        "capabilities": {"text", "vision", "medium"},
        "context_window": 128000,
        "priority": 2,
        "cost_tier": "medium",
    },
    "qwen3-vl-30b": {
        "env_id": "vl-fast",
        "model": "Qwen3-VL-30B-A3B-Instruct",
        "url": "http://dev.hcp.llm.skhynix.com/v1/chat/completions",
        "name": "VL-30B (Vision/Fast)",
        "capabilities": {"text", "vision", "fast"},
        "context_window": 128000,
        "priority": 3,
        "cost_tier": "low",
    },
    "bge-reranker": {
        "env_id": "reranker",
        "model": "bge-reranker-v2-m3",
        "url": "http://dev.hcp.llm.skhynix.com/v1/chat/completions",
        "name": "Reranker",
        "capabilities": {"rerank"},
        "context_window": 8192,
        "priority": 1,
        "cost_tier": "low",
    },
}

# MODEL_REGISTRY에서 ENV_CONFIG 자동 생성 (하위 호환)
ENV_CONFIG = {
    v["env_id"]: {"url": v["url"], "model": v["model"], "name": v["name"]}
    for v in MODEL_REGISTRY.values()
    if "rerank" not in v["capabilities"]
}
# gguf-N 환경은 앱 시작 시 .gguf 파일 자동 감지되면 추가됨

# env_id → registry key 역매핑
ENV_TO_REGISTRY = {v["env_id"]: k for k, v in MODEL_REGISTRY.items()}

# 폴백 체인: 모델 실패 시 순서대로 시도
FALLBACK_CHAINS = {
    "qwen3-vl-235b":  ["qwen2.5-vl-72b", "qwen3.5-397b", "gpt-oss-120b"],
    "qwen3.5-397b":   ["gpt-oss-120b", "glm-4.7"],
    "gpt-oss-120b":   ["glm-4.7"],
    "glm-4.7":        ["gpt-oss-120b"],
    "qwen2.5-vl-72b": ["qwen3-vl-30b", "gpt-oss-120b"],
    "qwen3-vl-30b":   ["qwen2.5-vl-72b", "gpt-oss-120b"],
}

# Reranker 기능 플래그 (bge-reranker 엔드포인트 안정화 후 활성화)
RERANKER_ENABLED = False


def load_token():
    """TOKEN.TXT 파일에서 API 키 읽기"""
    if os.path.isfile(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8-sig") as f:
            token = f.read().strip()
            if not token:
                print(f"  ⚠️  TOKEN.TXT 파일이 비어있습니다 - API 키를 입력하세요: {TOKEN_FILE}")
                return ""
            # ASCII만 허용 (한글 플레이스홀더 무시)
            try:
                token.encode("ascii")
                return token
            except UnicodeEncodeError:
                print(f"  ⚠️  TOKEN.TXT에 비영문 문자 포함 - 실제 API 키로 교체하세요")
                return ""
    else:
        print(f"  ⚠️  TOKEN.TXT 파일을 찾을 수 없습니다: {TOKEN_FILE}")
    return ""


API_TOKEN = load_token()

# 372개 스킬 한글 설명 (과학 174 + 개발도구 55 + 에이전트 117 + 가이드 15 + 커맨드 11)
SKILL_DESC_KO = {
    "biopython":"생물 서열/구조 분석","scanpy":"단일세포 RNA-seq 분석","pydeseq2":"차등 유전자 발현 분석",
    "bioservices":"40+ 생물정보학 DB 통합","anndata":"단일세포 주석 행렬","arboreto":"유전자 조절 네트워크 추론",
    "cellxgene-census":"6100만+ 단일세포 데이터","deeptools":"NGS BAM/bigWig 분석","gget":"20+ 생물DB 빠른 조회",
    "geniml":"유전체 구간 ML","gtars":"유전체 구간 고성능 분석","pysam":"SAM/BAM/VCF 파일 처리",
    "scikit-bio":"서열/다양성/마이크로바이옴","scvelo":"RNA velocity 분석","scvi-tools":"단일세포 딥러닝",
    "tiledbvcf":"유전체 변이 저장/조회","flowio":"유세포분석 FCS 파싱","phylogenetics":"계통수 구축",
    "etetoolkit":"계통수 조작/시각화","cobrapy":"대사 모델링 FBA/FVA","glycoengineering":"글리코실화 분석",
    "esm":"단백질 언어모델 구조예측","gene-database":"NCBI Gene 조회","ensembl-database":"Ensembl 250+ 종 유전체",
    "uniprot-database":"UniProt 단백질 검색","geo-database":"GEO 유전자 발현","clinvar-database":"ClinVar 변이 의미",
    "gnomad-database":"gnomAD 대립유전자 빈도","gtex-database":"GTEx 조직별 발현","gwas-database":"GWAS SNP-형질 연관",
    "ena-database":"유럽 뉴클레오타이드 아카이브","biorxiv-database":"bioRxiv 프리프린트","string-database":"단백질 상호작용",
    "reactome-database":"Reactome 경로 분석","kegg-database":"KEGG 대사경로","interpro-database":"단백질 도메인 주석",
    "jaspar-database":"전사인자 결합 프로파일","monarch-database":"질병-유전자 연관","alphafold-database":"AI 단백질 구조",
    "pdb-database":"PDB 3D 구조","cosmic-database":"COSMIC 암 돌연변이","cbioportal-database":"암 유전체",
    "depmap":"암 유전자 의존성","opentargets-database":"치료 표적 발굴",
    "rdkit":"분자 처리 SMILES/SDF","datamol":"RDKit 래퍼/분자기술자","deepchem":"분자 ML",
    "molfeat":"100+ 분자 특성화기","matchms":"질량스펙트럼 유사도","medchem":"약물유사성/PAINS필터",
    "diffdock":"분자 도킹 결합예측","molecular-dynamics":"분자동역학 OpenMM","torchdrug":"분자 그래프NN",
    "chembl-database":"ChEMBL 생활성 분자","drugbank-database":"약물 정보/상호작용","pubchem-database":"PubChem 화합물",
    "bindingdb-database":"약물-표적 친화도","zinc-database":"2.3억 화합물 스크리닝","hmdb-database":"22만 대사체 DB",
    "clinpgx-database":"약물유전체학","brenda-database":"효소 동역학","metabolomics-workbench-database":"대사체학 API",
    "primekg":"정밀의학 지식그래프","pytdc":"신약 벤치마크","rowan":"클라우드 양자화학",
    "pymatgen":"재료과학 결정/상도","astropy":"천문학/천체물리","fluidsim":"전산유체역학","sympy":"기호 수학",
    "qiskit":"IBM 양자 컴퓨팅","cirq":"Google 양자 회로","pennylane":"양자 ML","qutip":"양자계 시뮬레이션",
    "matplotlib":"과학 시각화","seaborn":"통계 시각화","plotly":"인터랙티브 차트","scikit-learn":"ML 학습/평가",
    "pytorch-lightning":"딥러닝 멀티GPU","polars":"고속 DataFrame","dask":"분산 컴퓨팅","vaex":"대규모 테이블",
    "networkx":"네트워크 분석","shap":"모델 해석 SHAP","umap-learn":"UMAP 차원축소",
    "statsmodels":"통계 모델 OLS/GLM","statistical-analysis":"통계 분석 가이드",
    "exploratory-data-analysis":"탐색적 데이터 분석","torch-geometric":"그래프 신경망",
    "stable-baselines3":"강화학습","pufferlib":"고성능 RL","transformers":"트랜스포머 NLP/CV",
    "simpy":"이산사건 시뮬레이션","pymoo":"다목적 최적화","pymc":"베이지안 MCMC",
    "aeon":"시계열 ML","timesfm-forecasting":"시계열 예측","geopandas":"지리공간 분석",
    "geomaster":"GIS/원격탐사","fred-economic-data":"FRED 경제 데이터","datacommons-client":"공공 통계",
    "clinical-decision-support":"임상 의사결정","clinical-reports":"임상보고서 작성",
    "clinicaltrials-database":"임상시험 조회","fda-database":"FDA 의약품/기기","treatment-plans":"치료 계획 생성",
    "pydicom":"DICOM 의료영상","pyhealth":"의료 AI 예측","pathml":"전산병리학",
    "histolab":"조직영상 타일추출","imaging-data-commons":"암 영상 데이터",
    "iso-13485-certification":"의료기기 품질인증","neurokit2":"생체신호 처리","neuropixels-analysis":"신경 기록 분석",
    "scientific-writing":"논문 작성 IMRAD","literature-review":"체계적 문헌 검토",
    "citation-management":"인용/BibTeX 관리","peer-review":"논문 심사 평가",
    "research-grants":"연구비 제안서","scientific-brainstorming":"연구 아이디어 발상",
    "scientific-critical-thinking":"과학적 근거 평가","hypothesis-generation":"가설 수립/실험 설계",
    "scholar-evaluation":"학술 업적 평가","scientific-visualization":"출판용 그림",
    "scientific-schematics":"과학 다이어그램 AI","scientific-slides":"발표 슬라이드",
    "venue-templates":"학술지 LaTeX 템플릿","latex-posters":"LaTeX 포스터",
    "pptx-posters":"연구 포스터 HTML/PDF","infographics":"인포그래픽 AI",
    "markdown-mermaid-writing":"마크다운/Mermaid","paper-2-web":"논문→웹사이트",
    "pubmed-database":"PubMed 논문 검색","openalex-database":"2.4억 학술문헌",
    "adaptyv":"클라우드랩 단백질 검증","benchling-integration":"Benchling R&D",
    "ginkgo-cloud-lab":"Ginkgo 클라우드랩","opentrons-integration":"Opentrons 로봇",
    "pylabrobot":"랩 자동화 프레임워크","labarchive-integration":"전자실험노트",
    "lamindb":"생물 데이터 관리","latchbio-integration":"서버리스 생물정보",
    "dnanexus-integration":"클라우드 유전체","omero-integration":"현미경 영상 관리",
    "protocolsio-integration":"과학 프로토콜","pyzotero":"Zotero 참고문헌",
    "alpha-vantage":"주식/외환/암호화폐","edgartools":"SEC 재무제표",
    "hedgefundmonitor":"헤지펀드 리스크","usfiscaldata":"미국 재정 데이터",
    "market-research-reports":"시장조사 보고서","uspto-database":"특허/상표 검색",
    "docx":"Word 문서 처리","xlsx":"Excel 스프레드시트","pdf":"PDF 처리/OCR",
    "pptx":"PowerPoint 처리","markitdown":"파일→마크다운","matlab":"MATLAB/Octave",
    "modal":"클라우드 GPU 실행","generate-image":"AI 이미지 생성",
    "get-available-resources":"시스템 자원 감지","bgpt-paper-search":"논문/실험데이터 검색",
    "research-lookup":"연구 정보 검색","perplexity-search":"Perplexity 웹검색",
    "parallel-web":"웹검색/딥리서치","open-notebook":"AI 연구 노트북",
    "consciousness-council":"다관점 숙의","what-if-oracle":"What-If 시나리오",
    "hypogenic":"자동 가설 생성","dhdna-profiler":"텍스트 저자 분석",
    "offer-k-dense-web":"K-Dense Web 안내","denario":"AI 연구 자동화",
    "drawio-diagram":"Draw.io 다이어그램 생성/저장",
    "zarr-python":"청크 N-D 배열 저장",
    "pyopenms":"질량분석 데이터 처리",
    "scikit-survival":"생존 분석 ML",
    # ===== cla-main 직접 스킬 (52개) =====
    "aesthetic":"UI/UX 미학 디자인","ai-multimodal":"Gemini 멀티모달 AI 처리",
    "anthropic-architect":"Anthropic 아키텍처 설계","anthropic-prompt-engineer":"Anthropic 프롬프트 엔지니어링",
    "artifacts-builder":"코드 아티팩트 빌더","backend-development":"백엔드 개발 가이드",
    "better-auth":"인증/보안 구현","brand-guidelines":"브랜드 가이드라인 작성",
    "canvas-design":"캔버스 디자인 도구","changelog-generator":"변경사항 자동 문서화",
    "chrome-devtools":"크롬 개발자도구 활용","claude-code":"Claude Code 활용 가이드",
    "code-review":"코드 리뷰 프로세스","competitive-ads-extractor":"경쟁사 광고 분석",
    "content-research-writer":"콘텐츠 리서치 작성","databases":"데이터베이스 설계/최적화",
    "datadog-entity-generator":"Datadog 엔티티 생성","developer-growth-analysis":"개발자 성장 분석",
    "devops":"DevOps CI/CD 파이프라인","docs-seeker":"문서 검색/정리",
    "domain-name-brainstormer":"도메인 네임 발상","engineer-skill-creator":"엔지니어 스킬 제작",
    "file-organizer":"파일 정리/관리","frontend-design":"프론트엔드 디자인",
    "frontend-development":"프론트엔드 개발","github-ecosystem":"GitHub 생태계 활용",
    "google-adk-python":"Google ADK Python","image-enhancer":"이미지 향상/편집",
    "internal-comms":"내부 커뮤니케이션","invoice-organizer":"청구서 정리",
    "lead-research-assistant":"리드 리서치 어시스턴트","mcp-builder":"MCP 서버 구축",
    "media-processing":"미디어 처리/변환","meeting-insights-analyzer":"회의 인사이트 분석",
    "notebooklm-skill":"NotebookLM 스타일 분석","openai-prompt-engineer":"OpenAI 프롬프트 엔지니어링",
    "python-deprecation-fixer":"Python 폐기 API 수정","python-project-skel":"Python 프로젝트 스캐폴딩",
    "raffle-winner-picker":"추첨/선정 도구","repomix":"레포지토리 분석/통합",
    "sequential-thinking":"순차적 사고 추론","shopify":"Shopify 스토어 개발",
    "skill-creator":"스킬 제작 도구","skill-share":"스킬 공유 도구",
    "slack-gif-creator":"Slack GIF 생성","template-skill":"스킬 템플릿",
    "theme-factory":"테마/스타일 생성","ui-styling":"UI 스타일링 가이드",
    "video-downloader":"비디오 다운로드","web-artifacts-builder":"웹 아티팩트 빌더",
    "web-frameworks":"웹 프레임워크 가이드","webapp-testing":"웹앱 테스팅",
    "react-best-practices":"React/Next.js 성능 최적화 가이드",
    "web-design-guidelines":"웹 UI 디자인 100+ 규칙",
    "owasp-security":"OWASP Top 10 보안 코드리뷰 체크리스트",
    "logpresso-query":"로그프레소 LPQL 쿼리 작성 전문가",
    # ===== cla-main 에이전트 (117개) =====
    "agent-api-designer":"API 설계 전문가","agent-backend-developer":"백엔드 개발 전문가",
    "agent-electron-pro":"Electron 데스크톱 앱","agent-frontend-developer":"프론트엔드 개발 전문가",
    "agent-fullstack-developer":"풀스택 개발 전문가","agent-graphql-architect":"GraphQL 아키텍처",
    "agent-microservices-architect":"마이크로서비스 설계","agent-mobile-developer":"모바일 개발 전문가",
    "agent-ui-designer":"UI 디자인 전문가","agent-websocket-engineer":"WebSocket 엔지니어",
    "agent-wordpress-master":"WordPress 전문가",
    "agent-angular-architect":"Angular 아키텍처","agent-cpp-pro":"C++ 전문가",
    "agent-csharp-developer":"C# 개발 전문가","agent-django-developer":"Django 개발",
    "agent-dotnet-core-expert":".NET Core 전문가","agent-dotnet-framework-4.8-expert":".NET Framework 전문가",
    "agent-flutter-expert":"Flutter 크로스플랫폼","agent-golang-pro":"Go 언어 전문가",
    "agent-java-architect":"Java 아키텍처","agent-javascript-pro":"JavaScript 전문가",
    "agent-kotlin-specialist":"Kotlin 전문가","agent-laravel-specialist":"Laravel PHP",
    "agent-nextjs-developer":"Next.js 풀스택","agent-php-pro":"PHP 전문가",
    "agent-python-pro":"Python 전문가","agent-rails-expert":"Ruby on Rails",
    "agent-react-specialist":"React 프론트엔드","agent-rust-engineer":"Rust 시스템 프로그래밍",
    "agent-spring-boot-engineer":"Spring Boot 서버","agent-sql-pro":"SQL 쿼리 전문가",
    "agent-swift-expert":"Swift iOS/macOS","agent-typescript-pro":"TypeScript 전문가",
    "agent-vue-expert":"Vue.js 프론트엔드",
    "agent-cloud-architect":"클라우드 아키텍처","agent-database-administrator":"DB 관리 전문가",
    "agent-deployment-engineer":"배포 엔지니어","agent-devops-engineer":"DevOps 엔지니어",
    "agent-devops-incident-responder":"DevOps 인시던트 대응","agent-incident-responder":"인시던트 대응",
    "agent-kubernetes-specialist":"Kubernetes 전문가","agent-network-engineer":"네트워크 엔지니어",
    "agent-platform-engineer":"플랫폼 엔지니어","agent-security-engineer":"보안 엔지니어",
    "agent-sre-engineer":"SRE 엔지니어","agent-terraform-engineer":"Terraform IaC",
    "agent-accessibility-tester":"접근성 테스트","agent-architect-reviewer":"아키텍처 리뷰어",
    "agent-chaos-engineer":"카오스 엔지니어링","agent-code-reviewer":"코드 리뷰 전문가",
    "agent-compliance-auditor":"컴플라이언스 감사","agent-debugger":"디버깅 전문가",
    "agent-error-detective":"에러 탐지/분석","agent-penetration-tester":"침투 테스트",
    "agent-performance-engineer":"성능 엔지니어","agent-qa-expert":"QA 전문가",
    "agent-security-auditor":"보안 감사","agent-test-automator":"테스트 자동화",
    "agent-ai-engineer":"AI 엔지니어","agent-data-analyst":"데이터 분석가",
    "agent-data-engineer":"데이터 엔지니어","agent-data-scientist":"데이터 사이언티스트",
    "agent-database-optimizer":"DB 최적화","agent-llm-architect":"LLM 아키텍처",
    "agent-machine-learning-engineer":"ML 엔지니어","agent-ml-engineer":"ML 엔지니어 (생산)",
    "agent-mlops-engineer":"MLOps 엔지니어","agent-nlp-engineer":"NLP 전문가",
    "agent-postgres-pro":"PostgreSQL 전문가","agent-prompt-engineer":"프롬프트 엔지니어",
    "agent-build-engineer":"빌드 엔지니어","agent-cli-developer":"CLI 도구 개발",
    "agent-dependency-manager":"의존성 관리","agent-documentation-engineer":"문서화 엔지니어",
    "agent-dx-optimizer":"개발자 경험 최적화","agent-git-workflow-manager":"Git 워크플로 관리",
    "agent-legacy-modernizer":"레거시 현대화","agent-mcp-developer":"MCP 개발",
    "agent-refactoring-specialist":"리팩토링 전문가","agent-tooling-engineer":"도구 엔지니어",
    "agent-api-documenter":"API 문서 작성","agent-blockchain-developer":"블록체인 개발",
    "agent-embedded-systems":"임베디드 시스템","agent-fintech-engineer":"핀테크 엔지니어",
    "agent-game-developer":"게임 개발","agent-iot-engineer":"IoT 엔지니어",
    "agent-mobile-app-developer":"모바일 앱 개발","agent-payment-integration":"결제 시스템 통합",
    "agent-quant-analyst":"퀀트 분석","agent-risk-manager":"리스크 관리",
    "agent-seo-specialist":"SEO 전문가",
    "agent-business-analyst":"비즈니스 분석가","agent-content-marketer":"콘텐츠 마케터",
    "agent-customer-success-manager":"고객성공 매니저","agent-legal-advisor":"법률 자문",
    "agent-product-manager":"프로덕트 매니저","agent-project-manager":"프로젝트 매니저",
    "agent-sales-engineer":"세일즈 엔지니어","agent-scrum-master":"스크럼 마스터",
    "agent-technical-writer":"테크니컬 라이터","agent-ux-researcher":"UX 리서처",
    "agent-agent-organizer":"에이전트 조직화","agent-context-manager":"컨텍스트 관리",
    "agent-error-coordinator":"에러 코디네이터","agent-knowledge-synthesizer":"지식 통합",
    "agent-multi-agent-coordinator":"멀티에이전트 조율","agent-performance-monitor":"성능 모니터링",
    "agent-task-distributor":"작업 배분","agent-workflow-orchestrator":"워크플로 오케스트레이션",
    "agent-competitive-analyst":"경쟁사 분석","agent-data-researcher":"데이터 리서처",
    "agent-market-researcher":"시장 조사","agent-research-analyst":"리서치 분석가",
    "agent-search-specialist":"검색 전문가","agent-trend-analyst":"트렌드 분석",
    "agent-datadog-api-expert":"Datadog API 전문가","agent-datadog-pro":"Datadog 전문가",
    # ===== cla-main 가이드 (12개) =====
    "guide-documentation":"문서화 표준 가이드","guide-git":"Git 워크플로 가이드",
    "guide-github-actions":"GitHub Actions CI/CD","guide-golang":"Go 언어 가이드",
    "guide-hmhco":"HMHCO 조직 가이드","guide-mcp-reference":"MCP 프로토콜 참조",
    "guide-opus-4-5-agent":"Opus 4.5 에이전트 가이드","guide-opus-4-5":"Opus 4.5 모델 가이드",
    "guide-python":"Python 코딩 표준","guide-react":"React 개발 가이드",
    "guide-testing":"테스팅 표준 가이드","guide-version-discovery":"버전 관리 탐색",
    # ===== 추가 누락분 (17개) =====
    "common":"공통 유틸리티/API 키 관리","debugging":"체계적 디버깅 방법론",
    "problem-solving":"고급 문제 해결 프레임워크",
    "cmd-cr-fx":"코드 리뷰 수정 커맨드","cmd-cr":"코드 리뷰 커맨드",
    "cmd-deep-research":"딥 리서치 커맨드","cmd-explore":"코드베이스 탐색 커맨드",
    "cmd-git-cm":"Git 커밋 커맨드","cmd-git-cp":"Git 체리픽 커맨드",
    "cmd-git-ff":"Git 패스트포워드 커맨드","cmd-git-fr":"Git 프레시 브랜치 커맨드",
    "cmd-git-pr":"Git PR 생성 커맨드","cmd-git-prune":"Git 정리 커맨드",
    "cmd-git-sync":"Git 동기화 커맨드",
    "guide-opus-migration":"Opus 4.5 마이그레이션 가이드","guide-hooks":"Claude 훅 시스템 가이드",
    "guide-claude-md":"CLAUDE.md 글로벌 설정 가이드",
}

# 분야별 스킬 매핑 (355개 전체 분류: 24개 카테고리)
DOMAIN_SKILLS = {
    "bioinformatics": {
        "label": "생물정보학",
        "icon": "🧬",
        "color": "#ef4444",
        "skills": [
            "biopython","scanpy","pydeseq2","bioservices","anndata",
            "arboreto","cellxgene-census","deeptools","gget","geniml",
            "gtars","pysam","scikit-bio","scvelo","scvi-tools",
            "tiledbvcf","flowio","phylogenetics","etetoolkit","cobrapy",
            "glycoengineering","esm",
        ]
    },
    "bio-databases": {
        "label": "생물 DB",
        "icon": "🗄️",
        "color": "#f97316",
        "skills": [
            "gene-database","ensembl-database","uniprot-database",
            "geo-database","clinvar-database","gnomad-database",
            "gtex-database","gwas-database","ena-database",
            "biorxiv-database","string-database","reactome-database",
            "kegg-database","interpro-database","jaspar-database",
            "monarch-database","alphafold-database","pdb-database",
            "cosmic-database","cbioportal-database","depmap",
            "opentargets-database",
        ]
    },
    "cheminformatics": {
        "label": "화학/신약",
        "icon": "⚗️",
        "color": "#06b6d4",
        "skills": [
            "rdkit","datamol","deepchem","molfeat","matchms",
            "medchem","diffdock","molecular-dynamics","torchdrug",
            "chembl-database","drugbank-database","pubchem-database",
            "bindingdb-database","zinc-database","hmdb-database",
            "clinpgx-database","brenda-database",
            "metabolomics-workbench-database","primekg","pytdc","rowan",
            "pyopenms",
        ]
    },
    "materials-physics": {
        "label": "재료/물리/양자",
        "icon": "⚛️",
        "color": "#8b5cf6",
        "skills": [
            "pymatgen","astropy","fluidsim","sympy",
            "qiskit","cirq","pennylane","qutip",
        ]
    },
    "data-ml": {
        "label": "데이터/ML",
        "icon": "📊",
        "color": "#f59e0b",
        "skills": [
            "matplotlib","seaborn","plotly","scikit-learn",
            "pytorch-lightning","polars","dask","vaex","networkx",
            "shap","umap-learn","statsmodels","statistical-analysis",
            "exploratory-data-analysis","torch-geometric",
            "stable-baselines3","pufferlib","transformers","simpy",
            "pymoo","pymc","aeon","timesfm-forecasting",
            "geopandas","geomaster","scikit-survival",
        ]
    },
    "finance": {
        "label": "금융/경제",
        "icon": "💰",
        "color": "#10b981",
        "skills": [
            "alpha-vantage","edgartools","hedgefundmonitor",
            "fred-economic-data","usfiscaldata","datacommons-client",
            "market-research-reports","uspto-database",
        ]
    },
    "clinical": {
        "label": "임상/의학",
        "icon": "🏥",
        "color": "#ec4899",
        "skills": [
            "clinical-decision-support","clinical-reports",
            "clinicaltrials-database","fda-database","treatment-plans",
            "pydicom","pyhealth","pathml","histolab",
            "imaging-data-commons","iso-13485-certification",
            "neurokit2","neuropixels-analysis",
        ]
    },
    "writing-comm": {
        "label": "논문/연구",
        "icon": "📝",
        "color": "#3b82f6",
        "skills": [
            "scientific-writing","literature-review","citation-management",
            "peer-review","research-grants","scientific-brainstorming",
            "scientific-critical-thinking","hypothesis-generation",
            "scholar-evaluation","scientific-visualization",
            "scientific-schematics","scientific-slides",
            "venue-templates","latex-posters","pptx-posters",
            "infographics","markdown-mermaid-writing","paper-2-web",
            "pubmed-database","openalex-database",
        ]
    },
    "lab-automation": {
        "label": "랩 자동화",
        "icon": "🤖",
        "color": "#14b8a6",
        "skills": [
            "adaptyv","benchling-integration","ginkgo-cloud-lab",
            "opentrons-integration","pylabrobot","labarchive-integration",
            "lamindb","latchbio-integration","dnanexus-integration",
            "omero-integration","protocolsio-integration","pyzotero",
        ]
    },
    "utilities": {
        "label": "유틸리티",
        "icon": "🔧",
        "color": "#6b7280",
        "skills": [
            "drawio-diagram","docx","xlsx","pdf","pptx","markitdown","matlab",
            "modal","generate-image","get-available-resources",
            "bgpt-paper-search","research-lookup","perplexity-search",
            "parallel-web","open-notebook","consciousness-council",
            "what-if-oracle","hypogenic","dhdna-profiler","denario",
            "offer-k-dense-web","zarr-python",
        ]
    },
    # ===== cla-main 확장 카테고리 =====
    "dev-tools": {
        "label": "개발 도구",
        "icon": "🛠️",
        "color": "#2563eb",
        "skills": [
            "aesthetic","artifacts-builder","backend-development","better-auth",
            "brand-guidelines","canvas-design","changelog-generator","chrome-devtools",
            "claude-code","code-review","databases","devops",
            "docs-seeker","domain-name-brainstormer","engineer-skill-creator",
            "file-organizer","frontend-design","frontend-development","github-ecosystem",
            "google-adk-python","mcp-builder","python-deprecation-fixer","python-project-skel",
            "repomix","sequential-thinking","skill-creator","skill-share",
            "template-skill","theme-factory","ui-styling","web-artifacts-builder",
            "web-frameworks","webapp-testing",
            "react-best-practices","web-design-guidelines","owasp-security",
            "logpresso-query",
            "common","debugging","problem-solving",
        ]
    },
    "ai-prompt": {
        "label": "AI/프롬프트",
        "icon": "🤖",
        "color": "#7c3aed",
        "skills": [
            "anthropic-architect","anthropic-prompt-engineer","openai-prompt-engineer",
            "notebooklm-skill","ai-multimodal","content-research-writer",
            "lead-research-assistant","image-enhancer","media-processing",
            "meeting-insights-analyzer","video-downloader","slack-gif-creator",
        ]
    },
    "business-ops": {
        "label": "비즈니스",
        "icon": "💼",
        "color": "#dc2626",
        "skills": [
            "competitive-ads-extractor","datadog-entity-generator","developer-growth-analysis",
            "internal-comms","invoice-organizer","raffle-winner-picker","shopify",
        ]
    },
    "agent-core-dev": {
        "label": "코어 개발 에이전트",
        "icon": "⚙️",
        "color": "#0891b2",
        "skills": [
            "agent-api-designer","agent-backend-developer","agent-electron-pro",
            "agent-frontend-developer","agent-fullstack-developer","agent-graphql-architect",
            "agent-microservices-architect","agent-mobile-developer","agent-ui-designer",
            "agent-websocket-engineer","agent-wordpress-master",
        ]
    },
    "agent-lang-spec": {
        "label": "언어 전문 에이전트",
        "icon": "📜",
        "color": "#ea580c",
        "skills": [
            "agent-angular-architect","agent-cpp-pro","agent-csharp-developer",
            "agent-django-developer","agent-dotnet-core-expert","agent-dotnet-framework-4.8-expert",
            "agent-flutter-expert","agent-golang-pro","agent-java-architect",
            "agent-javascript-pro","agent-kotlin-specialist","agent-laravel-specialist",
            "agent-nextjs-developer","agent-php-pro","agent-python-pro",
            "agent-rails-expert","agent-react-specialist","agent-rust-engineer",
            "agent-spring-boot-engineer","agent-sql-pro","agent-swift-expert",
            "agent-typescript-pro","agent-vue-expert",
        ]
    },
    "agent-infra": {
        "label": "인프라 에이전트",
        "icon": "☁️",
        "color": "#0d9488",
        "skills": [
            "agent-cloud-architect","agent-database-administrator","agent-deployment-engineer",
            "agent-devops-engineer","agent-devops-incident-responder","agent-incident-responder",
            "agent-kubernetes-specialist","agent-network-engineer","agent-platform-engineer",
            "agent-security-engineer","agent-sre-engineer","agent-terraform-engineer",
        ]
    },
    "agent-quality": {
        "label": "품질/보안 에이전트",
        "icon": "🛡️",
        "color": "#15803d",
        "skills": [
            "agent-accessibility-tester","agent-architect-reviewer","agent-chaos-engineer",
            "agent-code-reviewer","agent-compliance-auditor","agent-debugger",
            "agent-error-detective","agent-penetration-tester","agent-performance-engineer",
            "agent-qa-expert","agent-security-auditor","agent-test-automator",
        ]
    },
    "agent-data-ai": {
        "label": "데이터/AI 에이전트",
        "icon": "🧠",
        "color": "#7c3aed",
        "skills": [
            "agent-ai-engineer","agent-data-analyst","agent-data-engineer",
            "agent-data-scientist","agent-database-optimizer","agent-llm-architect",
            "agent-machine-learning-engineer","agent-ml-engineer","agent-mlops-engineer",
            "agent-nlp-engineer","agent-postgres-pro","agent-prompt-engineer",
        ]
    },
    "agent-dx": {
        "label": "개발자경험 에이전트",
        "icon": "🔧",
        "color": "#ca8a04",
        "skills": [
            "agent-build-engineer","agent-cli-developer","agent-dependency-manager",
            "agent-documentation-engineer","agent-dx-optimizer","agent-git-workflow-manager",
            "agent-legacy-modernizer","agent-mcp-developer","agent-refactoring-specialist",
            "agent-tooling-engineer",
        ]
    },
    "agent-domain": {
        "label": "특수 도메인 에이전트",
        "icon": "🎯",
        "color": "#be185d",
        "skills": [
            "agent-api-documenter","agent-blockchain-developer","agent-embedded-systems",
            "agent-fintech-engineer","agent-game-developer","agent-iot-engineer",
            "agent-mobile-app-developer","agent-payment-integration","agent-quant-analyst",
            "agent-risk-manager","agent-seo-specialist",
        ]
    },
    "agent-business": {
        "label": "비즈니스 에이전트",
        "icon": "📊",
        "color": "#9333ea",
        "skills": [
            "agent-business-analyst","agent-content-marketer","agent-customer-success-manager",
            "agent-legal-advisor","agent-product-manager","agent-project-manager",
            "agent-sales-engineer","agent-scrum-master","agent-technical-writer",
            "agent-ux-researcher",
        ]
    },
    "agent-meta": {
        "label": "오케스트레이션 에이전트",
        "icon": "🎼",
        "color": "#4f46e5",
        "skills": [
            "agent-agent-organizer","agent-context-manager","agent-error-coordinator",
            "agent-knowledge-synthesizer","agent-multi-agent-coordinator",
            "agent-performance-monitor","agent-task-distributor","agent-workflow-orchestrator",
        ]
    },
    "agent-research": {
        "label": "리서치 에이전트",
        "icon": "🔍",
        "color": "#059669",
        "skills": [
            "agent-competitive-analyst","agent-data-researcher","agent-market-researcher",
            "agent-research-analyst","agent-search-specialist","agent-trend-analyst",
            "agent-datadog-api-expert","agent-datadog-pro",
        ]
    },
    "guides": {
        "label": "개발 가이드",
        "icon": "📖",
        "color": "#6366f1",
        "skills": [
            "guide-documentation","guide-git","guide-github-actions","guide-golang",
            "guide-hmhco","guide-mcp-reference","guide-opus-4-5-agent","guide-opus-4-5",
            "guide-python","guide-react","guide-testing","guide-version-discovery",
            "guide-opus-migration","guide-hooks","guide-claude-md",
        ]
    },
    "commands": {
        "label": "CLI 커맨드",
        "icon": "⌨️",
        "color": "#71717a",
        "skills": [
            "cmd-cr-fx","cmd-cr","cmd-deep-research","cmd-explore",
            "cmd-git-cm","cmd-git-cp","cmd-git-ff","cmd-git-fr",
            "cmd-git-pr","cmd-git-prune","cmd-git-sync",
        ]
    },
}


# ============================================
# 스킬 파일 읽기
# ============================================
def scan_skills():
    """scientific-skills 폴더를 스캔해서 사용 가능한 스킬 목록 반환 (scripts/references/assets 포함)"""
    result = {}
    if not os.path.isdir(SKILLS_DIR):
        return result

    for folder_name in os.listdir(SKILLS_DIR):
        skill_dir = os.path.join(SKILLS_DIR, folder_name)
        skill_md = os.path.join(skill_dir, "SKILL.md")
        if not os.path.isfile(skill_md):
            continue

        # scripts 폴더 스캔
        scripts = []
        scripts_dir = os.path.join(skill_dir, "scripts")
        if os.path.isdir(scripts_dir):
            for root, dirs, files in os.walk(scripts_dir):
                for fn in files:
                    if fn.endswith(".py"):
                        full = os.path.join(root, fn)
                        rel = os.path.relpath(full, skill_dir)
                        scripts.append({
                            "name": fn,
                            "path": rel,
                            "size": os.path.getsize(full),
                        })

        # references 폴더 스캔
        references = []
        refs_dir = os.path.join(skill_dir, "references")
        if os.path.isdir(refs_dir):
            for fn in os.listdir(refs_dir):
                full = os.path.join(refs_dir, fn)
                if os.path.isfile(full):
                    references.append({
                        "name": fn,
                        "path": os.path.join("references", fn),
                        "size": os.path.getsize(full),
                    })

        # assets 폴더 스캔
        assets = []
        assets_dir = os.path.join(skill_dir, "assets")
        if os.path.isdir(assets_dir):
            for fn in os.listdir(assets_dir):
                full = os.path.join(assets_dir, fn)
                if os.path.isfile(full):
                    assets.append({
                        "name": fn,
                        "path": os.path.join("assets", fn),
                        "size": os.path.getsize(full),
                    })

        result[folder_name] = {
            "name": folder_name,
            "path": skill_md,
            "has_content": True,
            "scripts": scripts,
            "references": references,
            "assets": assets,
        }
    return result


# ============================================
# 오토 스킬 라우터 (키워드 매칭)
# ============================================
SKILL_KEYWORDS = {
    # === 생물정보학 ===
    "biopython": ["생물","서열","DNA","RNA","단백질","FASTA","GenBank","BLAST","alignment","서열분석"],
    "scanpy": ["단일세포","single-cell","scRNA","10x","세포군집","UMAP","leiden"],
    "pydeseq2": ["차등발현","DEG","differential expression","RNA-seq","유전자발현"],
    "anndata": ["AnnData","h5ad","단일세포행렬"],
    "scvelo": ["RNA velocity","RNA벨로시티","세포분화궤적"],
    "scvi-tools": ["scVI","VAE","단일세포딥러닝"],
    "cellxgene-census": ["CellxGene","6100만","세포데이터"],
    "deeptools": ["NGS","BAM","bigWig","ChIP-seq"],
    "gget": ["gget","유전자조회","Ensembl조회"],
    "pysam": ["SAM","BAM","VCF","시퀀싱"],
    "phylogenetics": ["계통수","phylogenetic","진화","계통분석"],
    "etetoolkit": ["계통수시각화","ete3","newick"],
    "cobrapy": ["대사모델","FBA","FVA","flux","대사경로모델링"],
    "esm": ["단백질언어모델","ESM","구조예측","protein language"],
    "flowio": ["유세포","FCS","flow cytometry","FACS"],
    "glycoengineering": ["글리코실화","glyco","당화"],
    # === 생물 DB ===
    "gene-database": ["NCBI","유전자검색","gene ID","gene search"],
    "ensembl-database": ["Ensembl","종유전체","genome browser"],
    "uniprot-database": ["UniProt","단백질DB","protein database"],
    "geo-database": ["GEO","유전자발현DB","microarray"],
    "clinvar-database": ["ClinVar","변이의미","pathogenic","variant"],
    "gnomad-database": ["gnomAD","대립유전자빈도","allele frequency"],
    "gwas-database": ["GWAS","SNP","형질연관","genome-wide"],
    "biorxiv-database": ["bioRxiv","프리프린트","preprint"],
    "pubmed-database": ["PubMed","논문검색","medical literature","PMID"],
    "string-database": ["STRING","단백질상호작용","PPI","protein interaction"],
    "reactome-database": ["Reactome","경로분석","pathway"],
    "kegg-database": ["KEGG","대사경로","metabolic pathway"],
    "alphafold-database": ["AlphaFold","AI구조","protein structure prediction"],
    "pdb-database": ["PDB","3D구조","crystal structure","단백질구조"],
    "cosmic-database": ["COSMIC","암돌연변이","cancer mutation"],
    "opentargets-database": ["OpenTargets","치료표적","drug target"],
    # === 화학/신약 ===
    "rdkit": ["RDKit","SMILES","분자","molecule","화합물","compound","SDF","molecular"],
    "deepchem": ["DeepChem","분자ML","molecular ML"],
    "datamol": ["datamol","분자기술자","descriptor"],
    "molfeat": ["분자특성","fingerprint","분자지문"],
    "matchms": ["질량스펙트럼","mass spec","MS/MS"],
    "medchem": ["약물유사성","PAINS","Lipinski","drug-like"],
    "diffdock": ["도킹","docking","결합예측","binding"],
    "molecular-dynamics": ["분자동역학","MD시뮬레이션","OpenMM","GROMACS"],
    "chembl-database": ["ChEMBL","생활성","bioactivity"],
    "drugbank-database": ["DrugBank","약물정보","drug interaction"],
    "pubchem-database": ["PubChem","화합물DB"],
    "zinc-database": ["ZINC","화합물스크리닝","virtual screening"],
    # === 데이터/ML ===
    "scikit-learn": ["sklearn","머신러닝","분류","회귀","클러스터링","SVM","random forest","machine learning","이상치","outlier","anomaly"],
    "pytorch-lightning": ["PyTorch","딥러닝","신경망","neural network","GPU학습","deep learning"],
    "transformers": ["트랜스포머","HuggingFace","NLP","BERT","GPT","LLM","언어모델"],
    "polars": ["Polars","DataFrame","데이터프레임","빠른테이블","CSV","csv","TSV","데이터로드"],
    "dask": ["Dask","분산컴퓨팅","대용량","distributed"],
    "matplotlib": ["시각화","그래프","차트","plot","figure","matplotlib"],
    "seaborn": ["seaborn","통계시각화","heatmap","히트맵"],
    "plotly": ["plotly","인터랙티브","interactive chart","대시보드"],
    "networkx": ["네트워크","그래프이론","graph","node","edge"],
    "shap": ["SHAP","모델해석","feature importance","설명가능"],
    "statsmodels": ["회귀분석","OLS","GLM","통계모델","regression"],
    "statistical-analysis": ["통계","t-test","ANOVA","검정","p-value","유의성"],
    "umap-learn": ["UMAP","차원축소","embedding","t-SNE"],
    "torch-geometric": ["그래프신경망","GNN","GCN","graph neural"],
    "pymc": ["베이지안","MCMC","사후분포","Bayesian"],
    "sympy": ["수학","미적분","방정식","적분","미분","symbolic math"],
    "aeon": ["시계열","time series","forecast","예측"],
    "timesfm-forecasting": ["시계열예측","TimesFM","forecasting"],
    # === 임상/의학 ===
    "clinical-decision-support": ["임상","진단","의사결정","clinical decision"],
    "clinical-reports": ["임상보고서","clinical report","환자보고"],
    "clinicaltrials-database": ["임상시험","clinical trial","NCT"],
    "fda-database": ["FDA","의약품","승인","drug approval"],
    "treatment-plans": ["치료계획","treatment plan","처방"],
    "pydicom": ["DICOM","의료영상","CT","MRI","X-ray"],
    "pyhealth": ["의료AI","electronic health","EHR","환자예측"],
    "pathml": ["전산병리","pathology","조직학","H&E"],
    "neurokit2": ["생체신호","ECG","EEG","심전도","뇌전도"],
    # === 논문/연구 ===
    "scientific-writing": ["논문작성","IMRAD","학술논문","paper writing","scientific paper"],
    "literature-review": ["문헌검토","문헌 검토","systematic review","메타분석","literature","리뷰논문","선행연구"],
    "citation-management": ["인용","BibTeX","참고문헌","citation","Zotero"],
    "peer-review": ["심사","peer review","리뷰어"],
    "research-grants": ["연구비","grant","제안서","proposal","연구과제"],
    "hypothesis-generation": ["가설","hypothesis","실험설계","experiment design"],
    "scientific-visualization": ["과학시각화","출판용그림","publication figure"],
    # === 개발도구 ===
    "code-review": ["코드리뷰","코드 리뷰","code review","PR리뷰","코드검토"],
    "debugging": ["디버깅","debug","에러","error","버그","bug"],
    "sequential-thinking": ["단계별사고","체계적분석","step by step","논리적"],
    "problem-solving": ["문제해결","problem solving","브레인스토밍"],
    "databases": ["데이터베이스","DB설계","SQL","PostgreSQL","MySQL","MongoDB"],
    "devops": ["DevOps","CI/CD","파이프라인","배포","deploy","Docker","Jenkins"],
    "github-ecosystem": ["GitHub","git","레포","repository","PR","이슈"],
    "mcp-builder": ["MCP","서버구축","프로토콜"],
    "frontend-development": ["프론트엔드","React","Vue","Angular","CSS","HTML","UI"],
    "backend-development": ["백엔드","API","서버","Flask","Django","FastAPI","Express"],
    "python-project-skel": ["Python프로젝트","스캐폴딩","프로젝트구조","setup.py"],
    # === AI/프롬프트 ===
    "anthropic-prompt-engineer": ["프롬프트","prompt engineering","프롬프트작성"],
    "ai-multimodal": ["멀티모달","이미지분석","vision","Gemini"],
    # === 에이전트 (자주 쓰이는 것만) ===
    "agent-python-pro": ["Python","파이썬","python코드","스크립트","py"],
    "agent-data-scientist": ["데이터분석","data science","분석가","EDA","탐색적분석","데이터","탐색","탐색하"],
    "agent-ml-engineer": ["MLOps","모델배포","학습파이프라인"],
    "agent-fullstack-developer": ["풀스택","fullstack","웹앱","web app"],
    "logpresso-query": ["로그프레소","logpresso","LPQL","lpql","로그프레소쿼리","secs_data","M14_DATA","M14B","로그 조회","로그조회","로그검색"],
    "agent-sql-pro": ["SQL","테이블조회"],
    "agent-code-reviewer": ["코드검토","리팩토링","코드품질"],
    "agent-debugger": ["디버그","traceback","스택트레이스","에러추적"],
    "agent-technical-writer": ["기술문서","문서작성","API문서","documentation"],
    "agent-research-analyst": ["리서치","연구분석","정보수집","조사"],
    # === 금융 ===
    "alpha-vantage": ["주식","stock","주가","시세","외환","forex"],
    "edgartools": ["SEC","재무제표","10-K","10-Q","annual report"],
    "fred-economic-data": ["FRED","경제데이터","GDP","실업률","금리"],
    # === 유틸리티 ===
    "docx": ["워드","Word","docx","문서작성"],
    "xlsx": ["엑셀","Excel","xlsx","스프레드시트","spreadsheet"],
    "pdf": ["PDF","pdf","문서변환"],
    "pptx": ["파워포인트","PPT","pptx","프레젠테이션","발표","슬라이드","슬라이드 만들","PPT 만들","PPT 제작","PPT로","ppt로","발표자료","피피티","덱","deck","pitch deck","피치덱","발표 준비","슬라이드 제작"],
    "generate-image": ["이미지생성","AI그림","image generation","그림그려"],
    # === 가이드 ===
    "guide-python": ["Python가이드","코딩표준","PEP","파이썬스타일"],
    "guide-git": ["git가이드","브랜치","merge","rebase"],
    "guide-react": ["React가이드","리액트","컴포넌트"],
    # ===== 자동 생성 키워드 (288개) =====
    "adaptyv": ['클라우드랩', '단백질검증', 'protein engineering', 'Adaptyv'],
    "aesthetic": ['미학', 'UI디자인', 'UX', '비주얼', '디자인시스템'],
    "agent-accessibility-tester": ['접근성', 'a11y', 'WCAG', '스크린리더', '웹접근성'],
    "agent-agent-organizer": ['에이전트조직', 'agent organize', '에이전트관리'],
    "agent-ai-engineer": ['AI엔지니어', 'AI개발', '모델개발', 'AI시스템'],
    "agent-ai-ethics-advisor": ['AI윤리', '편향', 'fairness', '책임AI', 'bias'],
    "agent-ai-infrastructure": ['AI인프라', 'GPU', '모델서빙', 'MLOps', '학습환경'],
    "agent-angular-architect": ['Angular', 'TypeScript', '컴포넌트', 'RxJS', 'NgModule'],
    "agent-api-designer": ['API설계', 'REST', 'OpenAPI', 'Swagger', '엔드포인트'],
    "agent-api-documenter": ['API문서', 'Swagger', 'OpenAPI문서', 'API스펙'],
    "agent-architect-reviewer": ['아키텍처리뷰', '설계검토', '시스템리뷰', '구조분석'],
    "agent-backend-developer": ['백엔드', '서버개발', 'API서버', 'Node.js', 'Express'],
    "agent-blockchain-developer": ['블록체인', '스마트컨트랙트', 'Solidity', 'Web3', '암호화폐'],
    "agent-build-engineer": ['빌드', 'build', 'CI', '컴파일', '빌드시스템'],
    "agent-business-analyst": ['비즈니스분석', '요구사항', 'BRD', '비즈니스로직'],
    "agent-chaos-engineer": ['카오스', 'chaos engineering', '장애주입', 'resilience'],
    "agent-cli-developer": ['CLI', '커맨드라인', '터미널', '명령어도구'],
    "agent-cloud-architect": ['클라우드', 'AWS', 'Azure', 'GCP', '아키텍처', '인프라'],
    "agent-competitive-analyst": ['경쟁분석', '경쟁사', 'competitive analysis', '벤치마킹'],
    "agent-compliance-auditor": ['컴플라이언스', '감사', 'audit', '규정준수', 'GDPR'],
    "agent-content-marketer": ['콘텐츠마케팅', 'SEO', '블로그', '소셜미디어', '마케팅'],
    "agent-context-manager": ['컨텍스트', 'context', '상태관리', '세션'],
    "agent-cpp-pro": ['C++', 'CPP', '시스템프로그래밍', '메모리관리', '포인터'],
    "agent-csharp-developer": ['C#', 'csharp', 'Unity', 'WPF', '.NET'],
    "agent-customer-success-manager": ['고객성공', 'CS', '고객만족', '고객관리'],
    "agent-data-analyst": ['데이터분석가', '리포트', 'KPI', '지표', '대시보드'],
    "agent-data-engineer": ['데이터엔지니어', 'ETL', '파이프라인', 'Airflow', 'Spark'],
    "agent-data-researcher": ['데이터리서치', '데이터수집', '크롤링', '데이터소스'],
    "agent-database-administrator": ['DBA', '데이터베이스관리', '인덱스', '튜닝', '백업'],
    "agent-database-optimizer": ['DB최적화', '인덱스최적화', '쿼리튜닝', '슬로우쿼리'],
    "agent-datadog-api-expert": ['Datadog API', '모니터링API', '메트릭API'],
    "agent-datadog-pro": ['Datadog', 'APM', '로그관리', '인프라모니터링'],
    "agent-dependency-manager": ['의존성', 'dependency', '패키지관리', 'npm', 'pip'],
    "agent-deployment-engineer": ['배포', 'deploy', 'CI/CD', '릴리즈', 'rollback'],
    "agent-devops-engineer": ['DevOps', '파이프라인', '자동화', 'Jenkins', 'GitHub Actions'],
    "agent-devops-incident-responder": ['인시던트', '장애대응', '모니터링', '알림', 'SLA'],
    "agent-django-developer": ['Django', 'ORM', 'MTV', '파이썬웹', 'admin'],
    "agent-documentation-engineer": ['문서엔지니어', '기술문서화', 'docs-as-code'],
    "agent-documentation-writer": ['문서작성', '기술문서', 'README', 'API문서'],
    "agent-dotnet-core-expert": ['.NET Core', 'ASP.NET', 'Blazor', 'Entity Framework'],
    "agent-dotnet-framework-4.8-expert": ['.NET Framework', 'WinForms', 'WCF', '레거시닷넷'],
    "agent-dx-optimizer": ['DX', '개발자경험', '개발환경', '개발도구', '생산성'],
    "agent-electron-pro": ['Electron', '데스크톱앱', 'desktop app', '크로스플랫폼'],
    "agent-embedded-systems": ['임베디드', '펌웨어', 'firmware', 'MCU', 'RTOS', 'IoT'],
    "agent-error-coordinator": ['에러조율', '에러통합', '멀티에러', '에러워크플로'],
    "agent-error-detective": ['에러분석', '에러탐지', '버그추적', '로그분석', 'exception'],
    "agent-fintech-engineer": ['핀테크', '결제', '금융기술', 'PG', 'fintech'],
    "agent-flutter-expert": ['Flutter', 'Dart', '크로스플랫폼', '위젯', '모바일UI'],
    "agent-frontend-developer": ['프론트엔드', 'HTML', 'CSS', 'JavaScript', 'UI개발'],
    "agent-game-developer": ['게임개발', 'Unity', 'Unreal', '게임로직', '2D', '3D'],
    "agent-git-specialist": ['git', '브랜치', 'merge', 'rebase', '충돌해결', '커밋'],
    "agent-git-workflow-manager": ['Git워크플로', '브랜치전략', 'gitflow', 'trunk-based'],
    "agent-golang-pro": ['Go', 'Golang', '고루틴', 'goroutine', '채널', '동시성'],
    "agent-graphql-architect": ['GraphQL', '쿼리언어', '스키마', 'resolver', 'Apollo'],
    "agent-incident-responder": ['장애', 'incident', '복구', 'RCA', '포스트모템'],
    "agent-iot-engineer": ['IoT엔지니어', '센서', 'MQTT', '엣지컴퓨팅', '스마트'],
    "agent-iot-specialist": ['IoT', '사물인터넷', '센서', '아두이노', '라즈베리파이', 'MQTT'],
    "agent-java-architect": ['Java', 'Spring', 'JVM', 'Maven', 'Gradle', '엔터프라이즈'],
    "agent-javascript-pro": ['JavaScript', 'JS', 'ES6', '비동기', 'async', 'Promise'],
    "agent-knowledge-synthesizer": ['지식통합', '정보종합', 'knowledge synthesis', '요약통합'],
    "agent-kotlin-specialist": ['Kotlin', '코틀린', 'Android', 'JetBrains', '코루틴'],
    "agent-kubernetes-specialist": ['Kubernetes', 'K8s', '쿠버네티스', 'Pod', 'Helm', '컨테이너'],
    "agent-laravel-specialist": ['Laravel', 'PHP프레임워크', 'Eloquent', 'Blade'],
    "agent-legacy-modernizer": ['레거시', '현대화', '마이그레이션', '리라이트', 'monolith'],
    "agent-legal-advisor": ['법률', '계약서', '이용약관', '개인정보', 'GDPR'],
    "agent-llm-architect": [
        'LLM', '대규모언어모델', 'RAG', '파인튜닝', '프롬프트설계',
        'LLM 설계자', 'llm architect', '모델 라우팅', '컨텍스트 엔지니어링',
        '토큰 예산', '프롬프트 아키텍처', '평가 파이프라인', 'guardrail', 'hallucination',
    ],
    "agent-low-code-builder": ['로우코드', 'no-code', 'Bubble', 'Retool', '자동화'],
    "agent-machine-learning-engineer": ['ML엔지니어', '모델학습', '하이퍼파라미터', '피처'],
    "agent-market-researcher": ['시장조사', '마켓리서치', 'TAM', '시장규모', '경쟁'],
    "agent-mcp-developer": ['MCP', 'Model Context Protocol', 'MCP서버', 'MCP개발'],
    "agent-microservices-architect": ['마이크로서비스', 'MSA', '서비스분리', 'API게이트웨이'],
    "agent-ml-ops-specialist": ['MLOps', '모델관리', '실험추적', 'MLflow', 'Kubeflow'],
    "agent-mlops-engineer": ['MLOps엔지니어', '모델배포', '실험추적', '모델레지스트리'],
    "agent-mobile-app-developer": ['모바일앱', '앱스토어', 'iOS앱', '안드로이드앱'],
    "agent-mobile-developer": ['모바일', '앱개발', 'iOS', 'Android', 'React Native'],
    "agent-multi-agent-coordinator": ['멀티에이전트', '에이전트조율', '오케스트레이션', '협업'],
    "agent-network-engineer": ['네트워크', 'TCP', 'DNS', '방화벽', '로드밸런서', 'VPN'],
    "agent-nextjs-developer": ['Next.js', 'SSR', 'SSG', 'App Router', 'Vercel'],
    "agent-nlp-engineer": ['NLP', '자연어처리', '텍스트분석', '토큰화', '임베딩'],
    "agent-payment-integration": ['결제연동', 'PG연동', 'Stripe', '토스페이먼츠'],
    "agent-penetration-tester": ['침투테스트', 'pentest', '모의해킹', '취약점스캔'],
    "agent-performance-engineer": ['성능최적화', 'performance', '벤치마크', '프로파일링', 'latency'],
    "agent-performance-monitor": ['성능모니터링', 'APM', '메트릭수집', '대시보드'],
    "agent-php-pro": ['PHP', 'Laravel', 'Composer', 'WordPress개발'],
    "agent-platform-engineer": ['플랫폼', 'IDP', '개발자플랫폼', '셀프서비스'],
    "agent-postgres-pro": ['PostgreSQL', 'Postgres', 'PSQL', 'PG튜닝', '확장'],
    "agent-product-manager": ['프로덕트매니저', 'PM', '로드맵', 'PRD', '사용자스토리'],
    "agent-project-manager": ['프로젝트관리', '일정', 'WBS', '간트', '마일스톤'],
    "agent-prompt-engineer": ['프롬프트엔지니어', '프롬프트최적화', 'few-shot', 'chain-of-thought'],
    "agent-qa-engineer": ['QA', '품질보증', '테스트케이스', '테스트자동화'],
    "agent-qa-expert": ['QA전문가', '테스트전략', '버그리포트', '회귀테스트'],
    "agent-quant-analyst": ['퀀트', '알고리즘트레이딩', '금융모델', '리스크모델'],
    "agent-rails-expert": ['Rails', 'Ruby', 'ActiveRecord', 'MVC', 'Gem'],
    "agent-react-specialist": ['React', '리액트', 'JSX', 'Hooks', '상태관리', 'Redux'],
    "agent-refactoring-guru": ['리팩토링', '클린코드', '디자인패턴', 'SOLID', '코드개선'],
    "agent-refactoring-specialist": ['리팩토링전문', '코드개선', '기술부채', '클린코드'],
    "agent-regex-master": ['정규식', 'regex', '정규표현식', '패턴매칭', 'regexp'],
    "agent-risk-manager": ['리스크', '위험관리', 'risk management', '위험평가'],
    "agent-robotics-engineer": ['로보틱스', 'ROS', '로봇', '자율주행', '모션플래닝'],
    "agent-rust-engineer": ['Rust', '러스트', '소유권', 'ownership', 'borrow', '시스템'],
    "agent-sales-engineer": ['세일즈엔지니어', '기술영업', 'POC', '데모'],
    "agent-scrum-master": ['스크럼', 'Scrum', '스프린트', '애자일', 'Agile', '칸반'],
    "agent-search-specialist": ['검색', 'search', '정보검색', '쿼리최적화'],
    "agent-security-auditor": ['보안감사', '취약점감사', '코드보안', 'OWASP'],
    "agent-security-engineer": ['보안', 'security', '취약점', '암호화', '방화벽', 'CVE'],
    "agent-seo-specialist": ['SEO', '검색최적화', '메타태그', '사이트맵', '구글'],
    "agent-spring-boot-engineer": ['Spring Boot', '스프링', 'JPA', 'Gradle', 'Maven'],
    "agent-sre-engineer": ['SRE', '신뢰성', '가용성', 'SLO', 'SLI', '모니터링'],
    "agent-startup-advisor": ['스타트업', 'MVP', '비즈니스모델', '투자', '피칭'],
    "agent-swift-expert": ['Swift', '스위프트', 'iOS', 'SwiftUI', 'Xcode', 'macOS'],
    "agent-systems-programmer": ['시스템프로그래밍', 'OS', '커널', '드라이버', 'C', '어셈블리'],
    "agent-task-distributor": ['작업분배', '태스크배분', '로드밸런싱', '스케줄링'],
    "agent-tech-lead": ['테크리드', '기술리더', '아키텍처결정', '팀리딩'],
    "agent-terraform-engineer": ['Terraform', 'IaC', '인프라코드', 'HCL', '프로비저닝'],
    "agent-test-automator": ['테스트자동화', 'CI테스트', '자동화테스트', '파이프라인'],
    "agent-test-engineer": ['단위테스트', 'unit test', 'pytest', 'jest', '테스트코드'],
    "agent-tooling-engineer": ['도구개발', '내부도구', 'tooling', 'CLI도구'],
    "agent-trend-analyst": ['트렌드', '동향분석', '기술트렌드', '시장동향'],
    "agent-typescript-pro": ['TypeScript', 'TS', '타입', '인터페이스', '제네릭', 'type'],
    "agent-ui-designer": ['UI디자인', 'Figma', '목업', '와이어프레임', '프로토타입'],
    "agent-ux-researcher": ['UX리서치', '사용자조사', '사용성', '유저테스트'],
    "agent-vue-expert": ['Vue', '뷰', 'Vuex', 'Pinia', 'Composition API', 'Nuxt'],
    "agent-websocket-engineer": ['WebSocket', '실시간', '소켓', '채팅', 'push'],
    "agent-wordpress-master": ['WordPress', '워드프레스', '플러그인', '테마', 'CMS'],
    "agent-workflow-orchestrator": ['워크플로', '자동화', '파이프라인', '태스크관리'],
    "anthropic-architect": ['Anthropic', '아키텍처', 'Claude API', '시스템설계'],
    "arboreto": ['유전자네트워크', 'GRN', 'gene regulatory', 'GENIE3', 'GRNBoost'],
    "artifacts-builder": ['아티팩트', '코드생성', 'code artifact', '빌더'],
    "astropy": ['astropy', '천문학', '천체', 'astronomy', 'FITS', 'celestial'],
    "benchling-integration": ['Benchling', '실험노트', 'LIMS', '분자생물학'],
    "better-auth": ['인증', 'authentication', 'OAuth', 'JWT', '로그인', '보안'],
    "bgpt-paper-search": ['논문검색', '실험데이터', 'paper search', '학술검색'],
    "bindingdb-database": ['BindingDB', '약물표적', 'binding affinity', 'Ki', 'Kd'],
    "bioservices": ['BioServices', 'DB통합', 'UniProt', 'KEGG', 'NCBI', '생물DB'],
    "brand-guidelines": ['브랜드', '가이드라인', 'CI', '로고', '스타일가이드'],
    "brenda-database": ['BRENDA', '효소', 'enzyme kinetics', 'Km', 'Vmax'],
    "canvas-design": ['캔버스', 'Canvas', '그래픽', '드로잉'],
    "cbioportal-database": ['cBioPortal', '암유전체', 'cancer genomics', 'mutation'],
    "changelog-generator": ['변경로그', 'changelog', '릴리즈노트', '버전관리'],
    "chrome-devtools": ['크롬개발자도구', 'DevTools', '디버깅', '네트워크탭', '콘솔'],
    "cirq": ['Cirq', '양자회로', 'quantum circuit', 'Google quantum'],
    "claude-code": ['Claude Code', 'CLI', '에이전트코딩', '클로드'],
    "clinpgx-database": ['약물유전체', 'pharmacogenomics', 'PGx', '약물반응'],
    "cmd-cr": ['코드리뷰', '커밋리뷰', 'PR분석'],
    "cmd-cr-fx": ['코드수정', '자동수정', 'fix', '패치'],
    "cmd-deep-research": ['딥리서치', '심층조사', '종합분석', 'deep research'],
    "cmd-dr": ['디렉토리분석', '폴더구조', '프로젝트분석'],
    "cmd-explore": ['탐색', 'explore', '코드베이스', '디렉토리탐색'],
    "cmd-gen-mcp": ['MCP생성', 'MCP서버', '프로토콜'],
    "cmd-git-cm": ['커밋', 'commit', 'git commit', '커밋메시지'],
    "cmd-git-cp": ['체리픽', 'cherry-pick', '커밋선택', 'git cp'],
    "cmd-git-ff": ['패스트포워드', 'fast-forward', 'git merge ff'],
    "cmd-git-fr": ['프레시브랜치', 'fresh branch', '새브랜치'],
    "cmd-git-pr": ['PR생성', '풀리퀘스트', 'git PR', '코드제출'],
    "cmd-git-prune": ['정리', 'prune', 'git clean', '불필요제거'],
    "cmd-git-sync": ['동기화', 'sync', 'git pull', 'git fetch', '원격동기화'],
    "cmd-implement": ['구현', 'implement', '기능개발', '코딩'],
    "cmd-init": ['초기화', 'init', '프로젝트생성', 'setup'],
    "cmd-memory": ['메모리', '기억', '컨텍스트', 'context'],
    "cmd-think": ['사고', 'think', '분석', '추론', 'reasoning'],
    "cmd-user-prompt": ['사용자프롬프트', '커스텀', 'user prompt'],
    "common": ['공통', '유틸리티', '설정', 'configuration', '환경변수'],
    "competitive-ads-extractor": ['경쟁사', '광고분석', 'competitive', 'ads'],
    "consciousness-council": ['다관점', '숙의', '다각도분석', 'council', '관점비교'],
    "content-research-writer": ['콘텐츠', '리서치', '글쓰기', 'content writing'],
    "datacommons-client": ['공공통계', 'DataCommons', '인구', 'GDP', '통계데이터'],
    "datadog-entity-generator": ['Datadog', '모니터링', '엔티티', '메트릭'],
    "denario": ['AI연구', '자동화연구', 'research automation'],
    "depmap": ['DepMap', '암유전자의존성', 'CRISPR screen', 'cancer dependency'],
    "developer-growth-analysis": ['개발자성장', '성장분석', '커리어', '역량'],
    "dhdna-profiler": ['저자분석', '텍스트분석', 'authorship', '문체분석'],
    "dnanexus-integration": ['DNAnexus', '클라우드유전체', 'WGS', 'WES', '유전체분석'],
    "docs-seeker": ['문서검색', '문서정리', 'documentation', '문서'],
    "document-skills": ['문서스킬', '문서변환', '문서처리', 'document'],
    "domain-name-brainstormer": ['도메인', 'domain name', '네이밍', '사이트이름'],
    "ena-database": ['ENA', '유럽뉴클레오타이드', '시퀀싱데이터', 'FASTQ'],
    "engineer-skill-creator": ['스킬제작', 'skill builder', '엔지니어스킬'],
    "exploratory-data-analysis": ['EDA', '탐색적분석', '데이터탐색', '분포', '상관관계', 'describe'],
    "file-organizer": ['파일정리', '파일관리', '폴더정리', 'file organize'],
    "fluidsim": ['유체역학', 'CFD', 'fluid dynamics', 'Navier-Stokes', '유동'],
    "frontend-design": ['프론트엔드디자인', 'UI구현', 'CSS디자인', '레이아웃'],
    "geniml": ['유전체구간', 'genomic interval', 'BED', '유전체ML'],
    "geomaster": ['GIS', '원격탐사', 'remote sensing', '위성영상', '지리정보'],
    "geopandas": ['GeoPandas', '지리공간', 'GIS', 'shapefile', '좌표', '지도'],
    "get-available-resources": ['시스템자원', 'CPU', '메모리', '디스크', 'GPU감지'],
    "ginkgo-cloud-lab": ['Ginkgo', '합성생물학', 'synthetic biology', '클라우드랩'],
    "google-adk-python": ['Google ADK', 'Agent Development Kit', '구글에이전트'],
    "gtars": ['유전체구간', 'genomic region', '고성능분석', 'interval'],
    "gtex-database": ['GTEx', '조직발현', 'tissue expression', 'RNA발현'],
    "guide-claude-md": ['CLAUDE.md', '프로젝트설정', 'Claude설정'],
    "guide-documentation": ['문서화', 'documentation', '표준문서', 'README'],
    "guide-github-actions": ['GitHub Actions', 'CI/CD', '워크플로', '자동배포'],
    "guide-golang": ['Go가이드', 'Golang패턴', 'Go관례', '고루틴패턴'],
    "guide-hmhco": ['HMHCO', '조직가이드', '회사가이드'],
    "guide-hooks": ['훅', 'hook', '커스텀훅', 'pre-commit', '이벤트'],
    "guide-mcp-reference": ['MCP참조', 'MCP프로토콜', 'MCP문서'],
    "guide-opus-4-5": ['Opus 4.5', '모델가이드', 'Claude모델'],
    "guide-opus-4-5-agent": ['Opus에이전트', 'Agent가이드', '에이전트모델'],
    "guide-opus-migration": ['Opus', '마이그레이션', '모델전환', '업그레이드'],
    "guide-pydantic": ['Pydantic', '데이터검증', '모델정의', 'validation'],
    "guide-ruby": ['Ruby', '루비', 'Gem', 'Rails가이드'],
    "guide-rust": ['Rust가이드', '러스트패턴', '소유권', 'lifetime'],
    "guide-testing": ['테스트가이드', 'TDD', '테스트전략', '커버리지'],
    "guide-typescript": ['TypeScript가이드', '타입설계', 'TS패턴'],
    "guide-version-discovery": ['버전관리', 'version discovery', '버전탐색'],
    "hedgefundmonitor": ['헤지펀드', '리스크', 'fund monitoring', '포트폴리오'],
    "histolab": ['조직영상', 'histology', '타일추출', 'H&E', 'WSI', '병리', 'pathology', '조직학'],
    "hmdb-database": ['HMDB', '대사체', 'metabolite', 'metabolomics'],
    "hypogenic": ['가설생성', 'hypothesis generation', '자동가설'],
    "image-enhancer": ['이미지향상', '이미지편집', 'image enhance', '보정'],
    "imaging-data-commons": ['암영상', 'IDC', '의료영상데이터', 'TCIA'],
    "infographics": ['인포그래픽', 'infographic', '데이터시각화', '정보디자인'],
    "internal-comms": ['내부소통', '커뮤니케이션', '공지', '사내'],
    "interpro-database": ['InterPro', '단백질도메인', 'domain annotation', 'Pfam'],
    "invoice-organizer": ['청구서', '인보이스', 'invoice', '결제'],
    "iso-13485-certification": ['ISO 13485', '의료기기', '품질인증', 'QMS'],
    "jaspar-database": ['JASPAR', '전사인자', 'transcription factor', 'motif'],
    "labarchive-integration": ['전자실험노트', 'ELN', 'LabArchives', '실험기록'],
    "lamindb": ['LaminDB', '데이터관리', '생물데이터', 'data lineage'],
    "latchbio-integration": ['LatchBio', '서버리스', '워크플로우', 'bioinformatics pipeline'],
    "latex-posters": ['LaTeX포스터', '학회포스터', 'poster', '연구포스터'],
    "lead-research-assistant": ['리드리서치', '연구보조', 'research assistant'],
    "drawio-diagram": ['drawio', 'draw.io', '드로우IO', '드로우io', '드로우아이오', 'DrawIO', '다이어그램', '플로우차트', '아키텍처', '구조도', 'UML', '클래스다이어그램', '시퀀스다이어그램', 'ERD', '흐름도', '시스템구조', '데이터흐름도', 'DFD', '배치도'],
    "markdown-mermaid-writing": ['마크다운', 'Mermaid', '다이어그램', 'flowchart', '시퀀스', 'MD파일', 'MD 파일', 'md파일', '마크다운 보고서', '마크다운 문서', 'MD로', 'md로', '문서화', '보고서 작성', '정리해줘', '회의록', '미팅노트', '기술문서', 'MD 보고서', 'markdown report', 'markdown document', 'MD만들어', 'MD다운로드'],
    "market-research-reports": ['시장조사', 'market research', '산업분석', '보고서'],
    "markitdown": ['마크다운변환', '파일변환', '문서변환', 'markdown conversion'],
    "matlab": ['MATLAB', 'Octave', '매트랩', '행렬', '수치해석'],
    "media-processing": ['미디어', '동영상', '오디오', '변환', 'ffmpeg'],
    "meeting-insights-analyzer": ['회의', '미팅', '인사이트', 'meeting', '회의록'],
    "metabolomics-workbench-database": ['대사체학', 'metabolomics', 'Metabolomics Workbench'],
    "modal": ['Modal', '클라우드GPU', '서버리스GPU', 'GPU실행'],
    "monarch-database": ['Monarch', '질병유전자', 'disease gene', 'phenotype'],
    "neuropixels-analysis": ['Neuropixels', '신경기록', 'electrophysiology', '뉴런', 'spike'],
    "notebooklm-skill": ['NotebookLM', '노트북', '문서분석', '요약'],
    "offer-k-dense-web": ['K-Dense', '웹가이드', '안내'],
    "omero-integration": ['OMERO', '현미경', 'microscopy', '영상관리', 'imaging'],
    "open-notebook": ['연구노트북', 'AI노트북', '실험기록', 'notebook'],
    "openai-prompt-engineer": ['OpenAI', 'GPT', '프롬프트', 'prompt', 'ChatGPT'],
    "openalex-database": ['OpenAlex', '학술문헌', '논문검색', 'academic search'],
    "opentrons-integration": ['Opentrons', '로봇', '피펫팅', '자동화', 'liquid handling'],
    "paper-2-web": ['논문웹사이트', 'paper to web', '연구홍보', '랜딩페이지'],
    "parallel-web": ['병렬검색', '딥리서치', 'deep research', '웹크롤링'],
    "pennylane": ['PennyLane', '양자ML', 'quantum machine learning', 'variational'],
    "perplexity-search": ['Perplexity', '웹검색', 'AI검색', 'web search'],
    "pptx-posters": ['연구포스터', 'poster', 'HTML포스터', '학회'],
    "primekg": ['PrimeKG', '정밀의학', 'precision medicine', 'knowledge graph'],
    "protocolsio-integration": ['protocols.io', '실험프로토콜', 'method', '실험방법'],
    "pufferlib": ['PufferLib', '고성능RL', '게임AI', '환경병렬화'],
    "pylabrobot": ['랩자동화', '로봇실험', 'laboratory automation', 'Hamilton'],
    "pymatgen": ['pymatgen', '재료과학', '결정구조', 'crystal', '상도', 'phase diagram', 'Materials Project'],
    "pymoo": ['pymoo', '다목적최적화', 'multi-objective', 'NSGA', 'Pareto'],
    "pyopenms": ['OpenMS', '질량분석', 'mass spectrometry', 'LC-MS', 'proteomics'],
    "pytdc": ['PyTDC', '신약벤치마크', 'drug benchmark', 'ADMET', 'TDC'],
    "python-deprecation-fixer": ['deprecated', '폐기API', '버전업', '마이그레이션'],
    "pyzotero": ['Zotero', '참고문헌', 'bibliography', '문헌관리', '인용'],
    "qiskit": ['Qiskit', '양자컴퓨팅', 'quantum computing', 'IBM quantum', '큐비트', 'qubit'],
    "qutip": ['QuTiP', '양자시뮬레이션', 'quantum simulation', '양자역학'],
    "raffle-winner-picker": ['추첨', '랜덤선정', 'raffle', '당첨'],
    "repomix": ['레포분석', 'repository analysis', '코드분석', '코드베이스'],
    "research-lookup": ['연구정보', '리서치', '정보검색', 'research lookup'],
    "rowan": ['Rowan', '양자화학', 'quantum chemistry', 'DFT', 'ab initio'],
    "scholar-evaluation": ['학술업적', 'h-index', '논문평가', '인용', 'impact factor'],
    "scientific-brainstorming": ['연구아이디어', '브레인스토밍', '실험설계', '연구주제'],
    "scientific-critical-thinking": ['비판적사고', '과학적근거', '논증', 'evidence-based'],
    "scientific-schematics": ['다이어그램', 'schematic', '과학그림', '모식도'],
    "scientific-slides": ['발표', '슬라이드', '학회발표', 'presentation', '학술발표'],
    "scikit-bio": ['scikit-bio', '마이크로바이옴', 'microbiome', '다양성지수', '서열정렬'],
    "scikit-survival": ['생존분석', 'survival analysis', 'Kaplan-Meier', 'Cox', '생존곡선'],
    "shopify": ['Shopify', '쇼핑몰', '이커머스', 'e-commerce', '스토어'],
    "simpy": ['SimPy', '시뮬레이션', '이산사건', 'discrete event', '큐잉'],
    "skill-creator": ['스킬생성', 'skill create', '스킬템플릿'],
    "skill-share": ['스킬공유', 'skill share', '스킬배포'],
    "slack-gif-creator": ['Slack', 'GIF', '애니메이션', '슬랙'],
    "stable-baselines3": ['강화학습', 'RL', 'reinforcement learning', 'PPO', 'DQN', 'A2C'],
    "template-skill": ['템플릿', '스킬템플릿', 'boilerplate'],
    "theme-factory": ['테마', 'theme', '스타일', '다크모드', '색상'],
    "tiledbvcf": ['TileDB', 'VCF', '유전체변이', 'variant storage', 'genotype'],
    "torchdrug": ['TorchDrug', '분자그래프', 'molecular graph', 'GNN', 'drug discovery'],
    "ui-styling": ['UI스타일', 'CSS', '스타일링', '디자인시스템', '색상'],
    "usfiscaldata": ['미국재정', 'fiscal data', '국채', 'government debt'],
    "uspto-database": ['USPTO', '특허', 'patent', '상표', 'trademark', '지식재산'],
    "vaex": ['vaex', '대규모데이터', 'out-of-core', 'lazy evaluation', '빅데이터'],
    "venue-templates": ['LaTeX', '학술지', '템플릿', 'journal template', '논문포맷'],
    "video-downloader": ['비디오', '영상다운로드', 'video download', 'YouTube'],
    "web-artifacts-builder": ['웹아티팩트', '웹앱생성', 'HTML생성', '인터랙티브'],
    "web-frameworks": ['웹프레임워크', 'Express', 'Next.js', 'Nuxt', 'SvelteKit'],
    "webapp-testing": ['웹테스트', 'E2E', 'Playwright', 'Cypress', '테스팅'],
    "react-best-practices": ['React', 'Next.js', 'NextJS', '리액트', 'RSC', 'Server Component', 'SSR', 'CSR', 'App Router', 'hooks', '컴포넌트', 'useState', 'useEffect'],
    "web-design-guidelines": ['웹디자인', 'UI설계', 'UX', '레이아웃', '타이포그래피', '색상', '반응형', 'responsive', '접근성', 'a11y', 'WCAG', '다크모드', 'dark mode'],
    "owasp-security": ['보안', 'OWASP', 'XSS', 'SQL인젝션', 'CSRF', '취약점', 'vulnerability', '인증', '인가', 'authentication', 'authorization', 'injection', '보안점검', '코드보안', 'security'],
    "what-if-oracle": ['What-If', '시나리오', '가정분석', 'hypothetical'],
    "zarr-python": ['Zarr', '청크배열', 'N-D array', '대용량배열', 'cloud storage'],
}

def _score_query(query_lower):
    """쿼리에 대한 스킬별 점수 계산 (내부 공통 함수)"""
    import re
    scores = {}
    for skill_id, keywords in SKILL_KEYWORDS.items():
        score = 0
        for kw in keywords:
            kw_lower = kw.lower()
            if len(kw_lower) <= 4 and kw_lower.isascii() and kw_lower.isalpha():
                pattern = r'(?<![a-zA-Z])' + re.escape(kw_lower) + r'(?![a-zA-Z])'
                if re.search(pattern, query_lower):
                    score += len(kw_lower) * 2
            elif kw_lower in query_lower:
                score += len(kw_lower)
        if score > 0:
            scores[skill_id] = score
    return scores


def auto_select_skills(query, max_skills=5):
    """사용자 질문을 분석하여 관련 스킬을 자동 선택"""
    scores = _score_query(query.lower())
    sorted_skills = sorted(scores.items(), key=lambda x: -x[1])
    return [(sid, sc) for sid, sc in sorted_skills[:max_skills]]


def context_aware_skill_select(query, history, max_skills=7):
    """대화 컨텍스트 + 현재 질문 기반 스킬 자동 선택 (멀티에이전트 Router)

    [Router 역할]
    1. 현재 질문에서 직접 매칭되는 스킬 → 높은 가중치
    2. 최근 대화 3턴에서 매칭되는 스킬 → 중간 가중치 (컨텍스트 유지)
    3. 질문 유형 분석 → 보조 스킬 자동 추가
    """
    # 1단계: 현재 질문 스코어링 (가중치 1.0)
    current_scores = _score_query(query.lower())

    # 2단계: 최근 대화 컨텍스트 스코어링 (가중치 0.3, 최근 3턴)
    context_scores = {}
    recent_msgs = [m for m in history if m.get("role") == "user"][-3:]
    for msg in recent_msgs:
        msg_scores = _score_query(msg.get("content", "").lower())
        for sid, sc in msg_scores.items():
            context_scores[sid] = context_scores.get(sid, 0) + sc * 0.3

    # 3단계: 합산
    combined = {}
    all_sids = set(list(current_scores.keys()) + list(context_scores.keys()))
    for sid in all_sids:
        combined[sid] = current_scores.get(sid, 0) + context_scores.get(sid, 0)

    # 4단계: 질문 유형별 보조 스킬 자동 추가
    q = query.lower()
    boosted = set()
    # 코드 수정/디버깅 감지
    if any(kw in q for kw in ["에러", "error", "버그", "bug", "수정", "고쳐", "안돼", "안되", "traceback", "exception"]):
        for sid in ["debugging", "agent-debugger", "agent-error-detective"]:
            combined[sid] = combined.get(sid, 0) + 5
            boosted.add(sid)
    # 데이터 분석 감지
    if any(kw in q for kw in ["분석", "데이터", "csv", "그래프", "시각화", "차트", "통계"]):
        for sid in ["exploratory-data-analysis", "matplotlib", "statistical-analysis"]:
            combined[sid] = combined.get(sid, 0) + 3
            boosted.add(sid)
    # Draw.io / 다이어그램 감지 (코드 작성보다 우선)
    if any(kw in q for kw in ["drawio", "draw.io", "드로우io", "드로우IO", "드로우아이오", "드로잉io", "드로잉IO", "drawingio", "다이어그램", "구조도", "흐름도", "아키텍처도", "배치도", "dfd"]):
        combined["drawio-diagram"] = combined.get("drawio-diagram", 0) + 15
        boosted.add("drawio-diagram")
    # PPT / 프레젠테이션 감지
    if any(kw in q for kw in ["ppt", "피피티", "파워포인트", "슬라이드", "발표자료", "프레젠테이션", "발표 만들", "deck", "피치덱"]):
        combined["pptx"] = combined.get("pptx", 0) + 12
        boosted.add("pptx")
    # 코드 작성 감지
    if any(kw in q for kw in ["코드", "함수", "클래스", "구현", "작성", "만들어", "코딩"]):
        for sid in ["agent-python-pro", "debugging"]:
            combined[sid] = combined.get(sid, 0) + 3
            boosted.add(sid)

    # LLM/에이전트 설계 감지
    if any(kw in q for kw in ["llm", "rag", "파인튜닝", "프롬프트", "모델 라우팅", "에이전트 설계", "llm 설계", "llm 아키텍처"]):
        for sid in ["agent-llm-architect", "agent-prompt-engineer", "agent-ai-engineer"]:
            combined[sid] = combined.get(sid, 0) + 6
            boosted.add(sid)

    # 5단계: 업로드 파일 기반 부스트
    # 파일 확장자 → 관련 스킬 매핑
    file_ext_skill_map = {
        "py": [("agent-python-pro", 5), ("debugging", 3)],
        "js": [("agent-nextjs-developer", 5)],
        "csv": [("exploratory-data-analysis", 8), ("statistical-analysis", 5), ("matplotlib", 3)],
        "tsv": [("exploratory-data-analysis", 8), ("statistical-analysis", 5)],
        "xlsx": [("exploratory-data-analysis", 5)],
        "docx": [("docx", 5)],
        "pdf": [("pdf", 5)],
        "pptx": [("pptx", 5)],
        "drawio": [("drawio-diagram", 15)],
        "md": [("markdown-mermaid-writing", 3)],
        "html": [("web-artifacts-builder", 3)],
        "json": [("agent-python-pro", 3)],
        "xml": [("drawio-diagram", 5)],
    }
    for uf in uploaded_files:
        ext = uf.get("ext", "").lower()
        fname = uf.get("filename", "").lower()
        # 확장자 매칭
        if ext in file_ext_skill_map:
            for sid, boost in file_ext_skill_map[ext]:
                combined[sid] = combined.get(sid, 0) + boost
                boosted.add(sid)
        # 파일명에 키워드가 있으면 추가 매칭
        fname_scores = _score_query(fname)
        for sid, sc in fname_scores.items():
            combined[sid] = combined.get(sid, 0) + sc * 0.5

    # CSV 데이터가 로드되어 있으면 데이터 분석 스킬 부스트
    if uploaded_csv_data.get("filename"):
        for sid in ["exploratory-data-analysis", "statistical-analysis", "matplotlib"]:
            combined[sid] = combined.get(sid, 0) + 5
            boosted.add(sid)

    # 6단계: 정렬 후 반환 (현재 질문 직접 매칭 우선)
    sorted_skills = sorted(combined.items(), key=lambda x: -x[1])

    # 6.5단계: Reranker 선택적 적용 (feature flag)
    if RERANKER_ENABLED:
        top_candidates = [sid for sid, _ in sorted_skills[:20]]
        try:
            reranked = rerank_skills(query, top_candidates, top_k=max_skills)
            result = reranked
        except Exception:
            result = [(sid, sc) for sid, sc in sorted_skills[:max_skills] if sc > 0]
    else:
        result = [(sid, sc) for sid, sc in sorted_skills[:max_skills] if sc > 0]

    return result, list(boosted)


def rerank_skills(query, candidate_skill_ids, top_k=7):
    """bge-reranker-v2-m3로 스킬 후보를 재정렬 (RERANKER_ENABLED=True일 때만 호출)

    Returns:
        [(skill_id, score), ...] 상위 top_k개
    """
    if not candidate_skill_ids:
        return []
    reg = MODEL_REGISTRY.get("bge-reranker")
    if not reg:
        return [(sid, 0) for sid in candidate_skill_ids[:top_k]]

    # query-document 쌍 구성
    pairs = []
    for sid in candidate_skill_ids:
        desc = SKILL_DESC_KO.get(sid, sid)
        pairs.append({"text": [query, desc]})

    headers = {"Content-Type": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"

    try:
        resp = req.post(
            reg["url"].replace("/chat/completions", "/rerank"),  # rerank 엔드포인트 시도
            headers=headers,
            json={
                "model": reg["model"],
                "query": query,
                "documents": [SKILL_DESC_KO.get(sid, sid) for sid in candidate_skill_ids],
                "top_n": top_k,
            },
            timeout=10,
            verify=False,
        )
        resp.raise_for_status()
        result = resp.json()
        # reranker 응답 형식: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
        if "results" in result:
            ranked = sorted(result["results"], key=lambda x: -x.get("relevance_score", 0))
            return [(candidate_skill_ids[r["index"]], r.get("relevance_score", 0))
                    for r in ranked[:top_k] if r["index"] < len(candidate_skill_ids)]
    except Exception:
        pass  # 실패 시 caller에서 키워드 점수로 폴백

    return [(sid, 0) for sid in candidate_skill_ids[:top_k]]


def build_orchestration_prompt(query, skill_ids, loaded_skills_content):
    """멀티에이전트 오케스트레이션 프롬프트 생성

    [Expert 역할] 각 스킬을 전문가로 취급
    [Synthesizer 역할] 여러 전문가 지식을 조합하는 지시문
    """
    if len(loaded_skills_content) <= 1:
        return ""  # 단일 스킬이면 오케스트레이션 불필요

    expert_names = [f"[{SKILL_DESC_KO.get(sid, sid)}]" for sid in skill_ids if sid in loaded_skills_content]

    orchestration = f"""
[멀티에이전트 오케스트레이션 모드]
현재 {len(expert_names)}명의 전문가 지식이 로드되었습니다: {', '.join(expert_names)}

당신은 이 전문가들의 지식을 조합하여 최적의 답변을 생성하는 통합 전문가입니다.

[오케스트레이션 원칙]
1. 각 SKILL의 전문 지식을 해당 영역에 맞게 활용
2. 서로 다른 SKILL 간의 시너지를 찾아 조합 (예: biopython + matplotlib → 서열분석+시각화)
3. 답변을 하나의 통합된 흐름으로 제시 (분절되지 않게)
4. 각 SKILL에서 가져온 핵심 기법/코드를 명시하되, 자연스럽게 녹여내기
"""
    return orchestration


# ===================== 멀티에이전트 라우터: 작업 분류 & 모델 자동 선택 =====================
VISION_SIGNALS = [
    "이미지", "사진", "그림 분석", "사진 속", "보이는", "스크린샷",
    "screenshot", "그림에서", "화면", "figure", "diagram", "차트 읽",
    "이 그림", "이 사진", "이미지에서", "사진에서", "이미지를", "사진을",
    "이미지 속", "보여주는", "캡처", "화면에",
]
COMPLEX_SIGNALS = [
    "분석", "비교", "설계", "아키텍처", "최적화", "리팩토링", "구현해줘",
    "전체 코드", "시스템 설계", "파이프라인", "종합", "심층", "상세히",
    "비교 분석", "장단점", "트레이드오프", "벤치마크",
]
PPT_SIGNALS = [
    "ppt", "피피티", "파워포인트", "슬라이드", "발표자료", "프레젠테이션",
    "발표 만들", "deck", "피치덱",
]
DATA_SIGNALS = [
    "데이터 분석", "csv 분석", "통계 분석", "회귀", "상관관계",
    "데이터셋", "데이터프레임", "pandas", "히스토그램", "분포",
]
SIMPLE_MAX_LEN = 50  # 이 길이 이하의 짧은 질문은 간단한 Q&A로 간주


def classify_and_route(query, history, uploaded_files_list):
    """작업 유형을 분류하고 최적 모델(env_id)을 자동 선택하는 휴리스틱 라우터

    Returns:
        (env_id, route_reason): 선택된 환경 ID와 선택 이유
    """
    q = query.lower() if query else ""
    has_images = any(f.get("type") == "image" for f in uploaded_files_list)
    has_csv = any(f.get("ext", "").lower() in ("csv", "tsv", "xlsx") for f in uploaded_files_list)
    has_vision_kw = any(kw in q for kw in VISION_SIGNALS)

    # 만약 온라인 모델용 토큰이 없으면, 강제로 로컬 GGUF 모델 중에서 자동 선택
    if not API_TOKEN:
        gguf_envs = {k: v for k, v in ENV_CONFIG.items() if str(k).startswith("gguf-")}
        if not gguf_envs:
            return "common", "토큰 없음 & 로컬 모델 없음 → 실패 예상"
        
        vl_ggufs = [k for k, v in gguf_envs.items() if "vl" in v["name"].lower()]
        normal_ggufs = [k for k, v in gguf_envs.items() if "vl" not in v["name"].lower()]
        
        if has_images and vl_ggufs:
            return vl_ggufs[0], f"로컬 이미지 분석 → {ENV_CONFIG[vl_ggufs[0]]['name']}"
        elif normal_ggufs:
            complex_count = sum(1 for kw in COMPLEX_SIGNALS if kw in q)
            if complex_count >= 2 or len(q) > 200:
                return normal_ggufs[0], f"로컬 복잡한 분석 → {ENV_CONFIG[normal_ggufs[0]]['name']}"
            else:
                return normal_ggufs[-1], f"로컬 간단한 요청 → {ENV_CONFIG[normal_ggufs[-1]]['name']}"
        else:
            first_key = list(gguf_envs.keys())[0]
            return first_key, f"로컬 기본 모델 → {ENV_CONFIG[first_key]['name']}"

    # 1순위: 이미지 첨부 → VL 모델
    if has_images:
        # 복잡한 분석 요청 → 대형 VL
        if any(kw in q for kw in COMPLEX_SIGNALS) or len(q) > 200:
            return "vl-large", "이미지+복잡 분석 → VL-235B"
        # 보통 요청 → 중형 VL
        elif len(q) > SIMPLE_MAX_LEN:
            return "vl-medium", "이미지 분석 → VL-72B"
        # 간단한 요청 → 소형 VL
        else:
            return "vl-fast", "간단 이미지 → VL-30B"

    # 비전 키워드는 있지만 이미지가 없는 경우 (이미지 업로드 유도)
    if has_vision_kw and not has_images:
        return "vl-medium", "비전 키워드 감지 → VL-72B (이미지 업로드 권장)"

    # 2순위: PPT 생성 → 중형 모델
    if any(kw in q for kw in PPT_SIGNALS):
        return "common", "PPT 생성 → 120B"

    # 3순위: 복잡한 분석/코드/데이터 → 대형 모델
    complex_count = sum(1 for kw in COMPLEX_SIGNALS if kw in q)
    if complex_count >= 2 or (complex_count >= 1 and len(q) > 200):
        return "prod", "복잡한 분석 → 397B"

    # 4순위: 데이터 분석 (CSV 로드 + 분석 키워드)
    if has_csv or any(kw in q for kw in DATA_SIGNALS):
        return "prod", "데이터 분석 → 397B"

    # 5순위: 코드 작성 요청 (중간~긴 쿼리)
    code_kw = ["코드", "함수", "클래스", "구현", "작성", "코딩", "스크립트", "프로그래밍"]
    if any(kw in q for kw in code_kw) and len(q) > 80:
        return "prod", "코드 작성 → 397B"

    # 6순위: 간단한 Q&A → 빠른 모델
    if len(q) <= SIMPLE_MAX_LEN:
        return "dev", "간단 Q&A → GLM-4.7 (fast)"

    # 기본값: 중형 모델
    return "common", "일반 요청 → 120B"


def classify_format_and_style(query, history, uploaded_files_list, skill_ids):
    """채팅 내용을 분석하여 최적의 출력형식과 작성 스타일을 자동 선택

    Returns:
        (format_id, style_value, reason): 출력형식, 스타일 텍스트, 선택 이유
    """
    q = query.lower() if query else ""
    has_csv = any(f.get("ext", "").lower() in ("csv", "tsv", "xlsx") for f in uploaded_files_list)
    has_code_file = any(f.get("ext", "").lower() in ("py", "js", "java", "c", "cpp", "go", "rs", "html", "css") for f in uploaded_files_list)
    has_image = any(f.get("type") == "image" for f in uploaded_files_list)

    # 비전/이미지 키워드
    vision_kw = ["이미지", "사진", "그림", "스크린샷", "screenshot", "화면", "figure",
                 "diagram", "차트 읽", "캡처", "보이는", "보여주는"]
    has_vision_kw = any(kw in q for kw in vision_kw)

    # 스킬 기반 힌트
    data_skills = {"exploratory-data-analysis", "statistical-analysis", "matplotlib",
                   "seaborn", "plotly", "polars", "statsmodels", "scikit-learn"}
    debug_skills = {"debugging", "agent-debugger", "agent-error-detective"}
    writing_skills = {"scientific-writing", "literature-review", "peer-review",
                      "research-grants", "clinical-reports"}
    has_data_skill = bool(set(skill_ids) & data_skills)
    has_debug_skill = bool(set(skill_ids) & debug_skills)
    has_writing_skill = bool(set(skill_ids) & writing_skills)

    # === 출력형식 분류 ===
    fmt = "code"  # 기본값

    # 이미지 분석 요청 → 코드 아닌 분석/설명으로 (최우선)
    if has_image or has_vision_kw:
        # 이미지 + 코드 요청이 명시적이면 코드
        code_explicit = any(kw in q for kw in ["코드", "코딩", "구현", "스크립트", "import", "def "])
        if code_explicit:
            fmt = "code"
        else:
            fmt = "analysis"

    # 보고서/리포트 요청
    elif any(kw in q for kw in ["보고서", "리포트", "report", "요약해줘", "요약 작성", "정리해줘",
                 "문서 작성", "문서화", "보고", "브리핑", "개요"]) or has_writing_skill:
        fmt = "report"

    # 데이터 분석 요청
    elif has_csv or has_data_skill or any(kw in q for kw in ["분석해줘", "분석해 줘", "데이터 분석", "인사이트", "통계 분석", "상관관계", "추세"]):
        fmt = "analysis"

    # LLM/아키텍처 설계 요청
    elif any(kw in q for kw in ["llm", "rag", "아키텍처", "시스템 설계", "트레이드오프", "모델 라우팅"]):
        fmt = "analysis"

    # 단계별 설명 요청
    elif any(kw in q for kw in ["방법", "어떻게", "절차", "과정", "단계별", "step by step",
                                 "가르쳐", "알려줘", "설명해", "튜토리얼", "가이드"]):
        fmt = "step-by-step"

    # 디버깅/코드 수정
    elif has_debug_skill or has_code_file or any(kw in q for kw in ["에러", "error", "버그", "bug", "수정", "고쳐", "안돼", "안되", "traceback", "exception", "오류"]):
        fmt = "code-fix"

    # 코드 작성 요청
    elif any(kw in q for kw in ["코드", "함수", "클래스", "구현", "작성", "코딩", "스크립트",
                                 "만들어", "프로그래밍", "import", "def ", "class "]):
        fmt = "code"

    # 일반 질문/대화 → 단계별 (코드보다 설명 우선)
    elif not any(kw in q for kw in ["코드", "코딩", "구현", "함수", "클래스"]):
        fmt = "step-by-step"

    # === 스타일 분류 ===
    style = ""  # 기본값: 없음 (시스템 기본)

    # 이미지 분석 모드
    if (has_image or has_vision_kw) and fmt == "analysis":
        style = "이미지 내용을 자연어로 설명하세요. 코드 없이 분석 결과만 제시. 핵심 내용→세부 관찰→해석 순서."

    # 디버깅 모드
    elif fmt == "code-fix" or has_debug_skill:
        style = "에러 원인 분석 중심. traceback 해석, 재현 조건, 해결책 순서."

    # LLM 설계 모드
    elif "agent-llm-architect" in skill_ids or any(kw in q for kw in ["llm", "rag", "아키텍처", "시스템 설계"]):
        style = "설계 의사결정 중심. 요구사항→옵션 비교→트레이드오프→권장안→실행계획 순서."

    # 데이터 분석 모드
    elif fmt == "analysis" or has_data_skill:
        style = "데이터 스토리텔링. 숫자→의미→액션 순서로 해석."

    # 학술/논문 모드
    elif has_writing_skill or any(kw in q for kw in ["논문", "학술", "연구", "인용", "레퍼런스", "paper"]):
        style = "학술적 톤. 정확한 용어, 인용, 근거 제시."

    # 실용적 모드 (기본 코드 작성)
    elif fmt in ("code", "code-fix"):
        style = "실전에서 바로 활용 가능하게. 핵심 요점과 구체적 방법 위주. 선택된 출력형식에 맞춰 답변하세요."

    reason = f"형식:{fmt}"
    if style:
        reason += " / 스타일 자동"

    return fmt, style, reason


def get_registry_key_for_env(env_id):
    """env_id → MODEL_REGISTRY 키 반환. GGUF 등 미등록 환경은 None."""
    return ENV_TO_REGISTRY.get(env_id)


def get_model_capabilities(env_id):
    """env_id의 capability set 반환"""
    reg_key = get_registry_key_for_env(env_id)
    if reg_key and reg_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[reg_key].get("capabilities", set())
    return set()


def load_skill_content(skill_name):
    """스킬 SKILL.md 내용 읽기"""
    skill_md = os.path.join(SKILLS_DIR, skill_name, "SKILL.md")
    if os.path.isfile(skill_md):
        with open(skill_md, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def get_skill_catalog():
    """프론트엔드에 보낼 스킬 카탈로그 생성"""
    available = scan_skills()
    catalog = {}

    for domain_id, domain_info in DOMAIN_SKILLS.items():
        skills_list = []
        for skill_id in domain_info["skills"]:
            info = available.get(skill_id, {})
            skills_list.append({
                "id": skill_id,
                "name": skill_id.replace("-", " ").title(),
                "desc": SKILL_DESC_KO.get(skill_id, ""),
                "available": skill_id in available,
                "scripts": len(info.get("scripts", [])),
                "references": len(info.get("references", [])),
                "assets": len(info.get("assets", [])),
            })
        catalog[domain_id] = {
            "label": domain_info["label"],
            "icon": domain_info["icon"],
            "color": domain_info["color"],
            "skills": skills_list,
        }

    # 매핑에 없지만 폴더에 존재하는 스킬도 추가
    mapped_ids = set()
    for d in DOMAIN_SKILLS.values():
        mapped_ids.update(d["skills"])

    extra = []
    for sid in available:
        if sid not in mapped_ids:
            info = available[sid]
            extra.append({
                "id": sid,
                "name": sid.replace("-", " ").title(),
                "desc": SKILL_DESC_KO.get(sid, ""),
                "available": True,
                "scripts": len(info.get("scripts", [])),
                "references": len(info.get("references", [])),
                "assets": len(info.get("assets", [])),
            })

    if extra:
        catalog["etc"] = {
            "label": "기타 스킬",
            "icon": "📦",
            "color": "#6b7280",
            "skills": extra,
        }

    return catalog


# ============================================
# API 라우트
# ============================================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/uio")
@app.route("/uio/")
def uio_page():
    """UIO 2D Pixel Office 페이지 서빙 (base href 주입으로 상대경로 해결)"""
    uio_path = os.path.join(BASE_DIR, "UIO", "index.html")
    if os.path.exists(uio_path):
        with open(uio_path, "r", encoding="utf-8") as f:
            html = f.read()
        # <head> 바로 뒤에 <base href="/uio/"> 삽입 → 상대경로(img/, sound/)가 /uio/img/ 등으로 해석됨
        html = html.replace("<head>", '<head>\n<base href="/uio/">', 1)
        return html
    return "UIO index.html not found", 404


@app.route("/uio/<path:filename>")
def uio_static(filename):
    """UIO 정적 파일 서빙 (img, sound 등)"""
    from flask import send_from_directory
    uio_dir = os.path.join(BASE_DIR, "UIO")
    return send_from_directory(uio_dir, filename)


@app.route("/api/config")
def api_config():
    """환경 설정 및 토큰 상태 반환"""
    return jsonify({
        "envs": {k: {"url": v["url"], "model": v["model"], "name": v["name"]} for k, v in ENV_CONFIG.items()},
        "has_token": bool(API_TOKEN),
        "token_file": TOKEN_FILE,
        "token_optional": True,
    })


# ============================================
# GGUF 모델 관리 (llama-cpp-python)
# ============================================
def find_gguf_files():
    """app.py 주변에서 GGUF 파일 검색"""
    patterns = [
        os.path.join(BASE_DIR, "*.gguf"),
        os.path.join(BASE_DIR, "models", "*.gguf"),
        os.path.join(BASE_DIR, "model", "*.gguf"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return [{"path": f, "name": os.path.basename(f), "size_gb": round(os.path.getsize(f) / 1e9, 1)} for f in files]


def load_gguf_model(model_path, n_ctx=32768, n_gpu_layers=99, n_batch=128):
    """llama-cpp-python으로 GGUF 모델 로드 (이미 같은 모델이면 스킵)"""
    global gguf_model, gguf_loaded_path

    # 이미 같은 모델이 로드되어 있으면 스킵
    if gguf_loaded_path == model_path and gguf_model is not None:
        print(f"     ℹ️  이미 로드됨: {os.path.basename(model_path)}")
        return True

    try:
        from llama_cpp import Llama
        print(f"     모델 로딩 중: {os.path.basename(model_path)}...")

        # 기존 모델 해제
        if gguf_model is not None:
            print(f"     🔄 기존 모델 해제: {os.path.basename(gguf_loaded_path or '')}")
            del gguf_model
            gguf_model = None
            gguf_loaded_path = None

        # Windows: llama.cpp C 라이브러리가 stdout/stderr 핸들을 건드려서
        # Flask(click/colorama) 콘솔 출력이 깨지는 문제 방지
        saved_stdout = sys.stdout
        saved_stderr = sys.stderr
        try:
            # 일부 환경에서 n_batch를 크게 잡으면 디코드 실패가 증가해 보수적으로 설정
            try:
                gguf_model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    n_batch=n_batch,
                    verbose=False,
                )
            except TypeError:
                # 구버전 llama-cpp-python 호환
                gguf_model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                )
        finally:
            # 핸들 복원
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr

        gguf_loaded_path = model_path
        return True
    except ImportError:
        print(f"     ❌ llama-cpp-python 패키지 없음")
        print(f"        → pip install llama-cpp-python")
        return False
    except Exception as e:
        print(f"     ❌ 모델 로드 실패: {e}")
        return False


def gguf_chat(messages, temperature=0.5, max_tokens=4096, stop_flag=None):
    """로드된 GGUF 모델로 채팅 (스트리밍으로 중단 가능)"""
    global gguf_model
    if gguf_model is None:
        return None, "GGUF 모델이 로드되지 않았습니다."
    try:
        # 중단 플래그가 있으면 스트리밍 모드로 토큰별 체크
        if stop_flag is not None:
            chunks = []
            for chunk in gguf_model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            ):
                if stop_flag.get("stop", False):
                    # 중단 요청 → 지금까지 생성된 부분 반환
                    partial = "".join(chunks)
                    return (partial + "\n\n⏹️ (응답이 중단되었습니다)") if partial else None, None
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    chunks.append(content)
            return "".join(chunks), None
        else:
            resp = gguf_model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if resp and "choices" in resp and len(resp["choices"]) > 0:
                return resp["choices"][0].get("message", {}).get("content", ""), None
            return None, f"예상치 못한 응답: {resp}"
    except Exception as e:
        return None, f"GGUF 추론 오류: {str(e)}"


@app.route("/api/auto-skills", methods=["POST"])
def api_auto_skills():
    """사용자 질문에 맞는 스킬을 자동 추천 (컨텍스트 인식)"""
    data = request.json
    query = data.get("query", "")
    max_skills = data.get("max_skills", 7)
    history = data.get("history", [])  # 대화 히스토리

    if not query:
        return jsonify({"skills": [], "message": "질문을 입력해주세요."})

    # 대화 히스토리가 있으면 컨텍스트 인식 모드
    if history and len(history) > 0:
        results, boosted = context_aware_skill_select(query, history, max_skills=max_skills)
    else:
        results = auto_select_skills(query, max_skills=max_skills)
        boosted = []

    # 실제 SKILL.md가 있는 스킬만 필터링
    available = scan_skills()
    skills = []
    for sid, score in results:
        if sid not in available:
            continue  # SKILL.md 없는 스킬은 제외
        desc = SKILL_DESC_KO.get(sid, "")
        skills.append({
            "id": sid,
            "desc": desc,
            "score": score,
            "boosted": sid in boosted,  # 질문유형으로 자동 추가된 것
        })

    mode = "🧠 컨텍스트" if history else "🔍 키워드"
    return jsonify({
        "skills": skills,
        "query": query,
        "mode": mode,
        "boosted_count": len(boosted),
        "message": f"{mode} 매칭: {len(skills)}개 스킬" if skills else "매칭되는 스킬이 없습니다.",
    })


# ===================== 시스템 프롬프트 저장/불러오기 (TXT) =====================

# 기본 프리셋 프롬프트 (파일로 저장 안 됨, 코드 내장)
PRESET_PROMPTS = {
    "coding": {
        "name": "코딩 전문가",
        "icon": "💻",
        "content": """당신은 숙련된 소프트웨어 개발자입니다. 다음 원칙을 따르세요:

[코드 작성 규칙]
- 코드는 반드시 즉시 실행 가능하게 작성 (import 포함, 필요 패키지 명시)
- 에러 처리(try/except)를 반드시 포함
- 함수/클래스 단위로 구조화, 재사용 가능하게
- 주석은 한국어로 '왜' 이렇게 했는지 설명
- 타입 힌트 사용 권장
- 변수명은 명확하고 의미 있게

[응답 구조]
1. 요구사항 분석 (1~2줄)
2. 핵심 코드 (주석 포함)
3. 사용법/실행 방법
4. 필요 패키지: pip install xxx"""
    },
    "code-fix": {
        "name": "코드 수정/디버깅",
        "icon": "🔧",
        "content": """당신은 코드 디버깅 및 리팩토링 전문가입니다.

[분석 절차]
1. 원본 코드를 먼저 분석하고 문제점을 정확히 진단
2. 에러 메시지가 있으면 원인과 해결책을 명확히 설명
3. 수정 전/후 코드를 비교해서 보여주기
4. 왜 그렇게 고쳤는지 이유 설명

[코드 수정 원칙]
- 최소 변경 원칙: 문제를 고치는 데 필요한 최소한만 수정
- 기존 코딩 스타일 유지
- 성능 개선이 가능하면 제안 (하지만 강요하지 않음)
- 잠재적 버그나 취약점이 보이면 경고

[응답 형식]
- 문제 원인 → 수정 코드 → 설명 순서"""
    },
    "data-analysis": {
        "name": "데이터 분석",
        "icon": "📊",
        "content": """당신은 데이터 분석 전문가입니다.

[분석 절차]
1. 데이터 형태 파악 (shape, dtypes, 결측치, 이상치)
2. 탐색적 분석 (EDA): 분포, 상관관계, 패턴
3. 시각화: matplotlib/seaborn으로 핵심 차트 생성
4. 인사이트 도출: 데이터가 말해주는 것을 명확히 해석

[코드 원칙]
- pandas/polars 활용, 코드와 해석을 함께 제공
- 차트에는 반드시 한글 제목/라벨 (plt.rcParams 설정 포함)
- 큰 데이터는 샘플링 전략 제시
- 통계적 검증이 필요하면 scipy/statsmodels 활용

[응답 구조]
데이터 개요 → 분석 코드 → 시각화 → 인사이트 요약"""
    },
    "general-dev": {
        "name": "개발 올라운더",
        "icon": "🛠️",
        "content": """당신은 풀스택 개발 어시스턴트입니다. 코딩, 디버깅, 아키텍처, DevOps를 모두 지원합니다.

[핵심 원칙]
- 질문의 맥락을 파악하고 가장 적합한 방식으로 답변
- 코드는 실행 가능하게, 설명은 간결하게
- 한국어 주석, 필요 패키지 명시
- 모르는 건 모른다고 솔직히 말하기
- 여러 해결책이 있으면 장단점 비교 후 추천

[스킬 활용]
- 로드된 스킬의 방법론, 코드 패턴, API를 적극 활용
- 여러 스킬이 관련되면 조합해서 최적의 답변 생성"""
    },
    "llm-system-architect": {
        "name": "LLM 시스템 아키텍트",
        "icon": "🧠",
        "content": """당신은 LLM 시스템 아키텍트입니다.

[우선 원칙]
1. 먼저 요구사항과 제약(성능/비용/보안/운영)을 분석하세요.
2. 분석 후 아키텍처 옵션 2~3개를 비교하고 최종안을 추천하세요.
3. 반드시 트레이드오프(장단점, 리스크, 운영 난이도)를 명시하세요.

[설계 범위]
- 모델 선택/라우팅 전략 (단일 vs 멀티모델)
- RAG 파이프라인 (임베딩, 검색, 리랭킹, 캐시)
- 프롬프트/가드레일/평가(Eval) 체계
- 서빙(vLLM/TGI), 모니터링, 비용 최적화
- 장애 대응(폴백, 재시도, 회로차단, 관측성)

[응답 구조]
1. 요구사항 분석
2. 아키텍처 옵션 비교표
3. 추천 아키텍처(이유 포함)
4. 단계별 구현 계획(POC→운영)
5. 운영 지표(SLO/비용/품질)"""
    },
    "semiconductor": {
        "name": "반도체 엔지니어",
        "icon": "🔬",
        "content": """당신은 반도체 공정 및 소자 전문가입니다.

[전문 영역]
- DRAM/NAND Flash 구조, 공정 흐름, 불량 분석
- 웨이퍼 공정 데이터 분석 (수율, SPC, FDC)
- 장비 데이터 분석 및 공정 최적화
- 신뢰성 분석, 불량 메커니즘

[응답 원칙]
- 전문 용어를 정확히 사용하되, 필요시 설명 추가
- 데이터 분석 코드가 필요하면 pandas + matplotlib 활용
- 공정 파라미터 관계는 시각화로 보여주기
- 양산 환경을 고려한 실용적 제안"""
    },
}


@app.route("/api/prompts", methods=["GET"])
def api_list_prompts():
    """저장된 프롬프트 목록 (프리셋 + 사용자 저장분)"""
    result = []

    # 1) 프리셋
    for pid, p in PRESET_PROMPTS.items():
        result.append({
            "id": f"preset:{pid}",
            "name": p["name"],
            "icon": p["icon"],
            "type": "preset",
            "preview": p["content"][:80] + "...",
        })

    # 2) 사용자 저장 TXT 파일
    if os.path.isdir(PROMPTS_DIR):
        for fname in sorted(os.listdir(PROMPTS_DIR)):
            if fname.endswith(".txt"):
                fpath = os.path.join(PROMPTS_DIR, fname)
                name = fname[:-4]  # .txt 제거
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    result.append({
                        "id": f"saved:{name}",
                        "name": name,
                        "icon": "💾",
                        "type": "saved",
                        "preview": content[:80] + "..." if len(content) > 80 else content,
                    })
                except Exception:
                    pass

    return jsonify({"prompts": result})


@app.route("/api/prompts/<prompt_id>", methods=["GET"])
def api_get_prompt(prompt_id):
    """프롬프트 내용 불러오기"""
    if prompt_id.startswith("preset:"):
        key = prompt_id[7:]
        if key in PRESET_PROMPTS:
            p = PRESET_PROMPTS[key]
            return jsonify({"id": prompt_id, "name": p["name"], "content": p["content"]})
        return jsonify({"error": "프리셋을 찾을 수 없습니다."}), 404

    elif prompt_id.startswith("saved:"):
        name = prompt_id[6:]
        fpath = os.path.join(PROMPTS_DIR, f"{name}.txt")
        if os.path.isfile(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            return jsonify({"id": prompt_id, "name": name, "content": content})
        return jsonify({"error": "저장된 프롬프트를 찾을 수 없습니다."}), 404

    return jsonify({"error": "잘못된 ID 형식"}), 400


@app.route("/api/prompts", methods=["POST"])
def api_save_prompt():
    """시스템 프롬프트를 TXT로 저장"""
    data = request.json
    name = data.get("name", "").strip()
    content = data.get("content", "").strip()

    if not name or not content:
        return jsonify({"error": "이름과 내용을 모두 입력해주세요."}), 400

    # 파일명 안전하게
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_', '.', 'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ') or ('\uac00' <= c <= '\ud7a3')).strip()
    if not safe_name:
        safe_name = "custom-prompt"

    fpath = os.path.join(PROMPTS_DIR, f"{safe_name}.txt")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)

    return jsonify({"success": True, "id": f"saved:{safe_name}", "name": safe_name, "path": fpath})


@app.route("/api/prompts/<prompt_id>", methods=["DELETE"])
def api_delete_prompt(prompt_id):
    """저장된 프롬프트 삭제 (프리셋은 삭제 불가)"""
    if prompt_id.startswith("preset:"):
        return jsonify({"error": "프리셋은 삭제할 수 없습니다."}), 400

    if prompt_id.startswith("saved:"):
        name = prompt_id[6:]
        if ".." in name or "/" in name or "\\" in name:
            return jsonify({"error": "잘못된 프롬프트 ID입니다."}), 400
        fpath = os.path.join(PROMPTS_DIR, f"{name}.txt")
        if os.path.isfile(fpath):
            try:
                os.remove(fpath)
            except Exception as e:
                return jsonify({"error": f"삭제 실패: {str(e)}"}), 500
            return jsonify({"success": True})
        return jsonify({"error": "파일을 찾을 수 없습니다."}), 404

    return jsonify({"error": "잘못된 ID 형식"}), 400


@app.route("/api/skills")
def api_skills():
    """스킬 카탈로그 반환"""
    return jsonify(get_skill_catalog())


@app.route("/api/skill/<skill_name>")
def api_skill_content(skill_name):
    """개별 스킬 상세 반환 (SKILL.md + scripts/references/assets 목록)"""
    content = load_skill_content(skill_name)
    available = scan_skills()
    info = available.get(skill_name, {})
    return jsonify({
        "name": skill_name,
        "content": content,
        "scripts": info.get("scripts", []),
        "references": info.get("references", []),
        "assets": info.get("assets", []),
    })


@app.route("/api/skill/<skill_name>/file")
def api_skill_file(skill_name):
    """스킬 내부 파일(scripts/references/assets) 내용 읽기"""
    file_path = request.args.get("path", "")
    if not file_path:
        return jsonify({"error": "path 파라미터가 필요합니다."}), 400

    # 보안: 상위 디렉토리 접근 차단
    if ".." in file_path or file_path.startswith("/"):
        return jsonify({"error": "잘못된 경로"}), 400

    full_path = os.path.join(SKILLS_DIR, skill_name, file_path)
    if not os.path.isfile(full_path):
        return jsonify({"error": f"파일 없음: {file_path}"}), 404

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return jsonify({
            "name": os.path.basename(file_path),
            "path": file_path,
            "content": content,
            "size": os.path.getsize(full_path),
        })
    except UnicodeDecodeError:
        return jsonify({"error": "바이너리 파일은 읽을 수 없습니다."}), 400


@app.route("/api/skill/<skill_name>/run", methods=["POST"])
def api_skill_run(skill_name):
    """스킬 내부 파이썬 스크립트 실행"""
    import subprocess

    data = request.json or {}
    script_path = data.get("script", "")
    args = data.get("args", [])

    if not script_path:
        return jsonify({"error": "script 파라미터가 필요합니다."}), 400
    if ".." in script_path or script_path.startswith("/"):
        return jsonify({"error": "잘못된 경로"}), 400
    if not script_path.endswith(".py"):
        return jsonify({"error": "파이썬 파일만 실행 가능합니다."}), 400

    full_path = os.path.join(SKILLS_DIR, skill_name, script_path)
    if not os.path.isfile(full_path):
        return jsonify({"error": f"스크립트 없음: {script_path}"}), 404

    try:
        result = subprocess.run(
            [sys.executable, full_path] + args,
            capture_output=True, text=True,
            timeout=60,
            cwd=os.path.join(SKILLS_DIR, skill_name),
        )
        return jsonify({
            "stdout": result.stdout[:8000] if len(result.stdout) > 8000 else result.stdout,
            "stderr": result.stderr[:4000] if len(result.stderr) > 4000 else result.stderr,
            "returncode": result.returncode,
            "script": script_path,
        })
    except subprocess.TimeoutExpired:
        return jsonify({"error": "스크립트 실행 시간 초과 (60초)"}), 504
    except Exception as e:
        return jsonify({"error": f"실행 오류: {str(e)}"}), 500


@app.route("/api/upload_csv", methods=["POST"])
def api_upload_csv():
    """CSV 파일 업로드 및 파싱"""
    global uploaded_csv_data

    if 'file' not in request.files:
        return jsonify({"error": "파일이 없습니다."}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "파일명이 없습니다."}), 400

    fname = file.filename.lower()
    if not (fname.endswith('.csv') or fname.endswith('.tsv') or fname.endswith('.txt')):
        return jsonify({"error": "CSV, TSV, TXT 파일만 지원합니다."}), 400

    try:
        # 파일 읽기 (여러 인코딩 시도)
        raw_bytes = file.read()
        content = None
        for enc in ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]:
            try:
                content = raw_bytes.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            return jsonify({"error": "파일 인코딩을 인식할 수 없습니다."}), 400

        # 구분자 자동 감지
        sample = content[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',\t;|')
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = '\t' if fname.endswith('.tsv') else ','

        # CSV 파싱
        reader = csv.reader(io.StringIO(content), delimiter=delimiter)
        all_rows = []
        for row in reader:
            all_rows.append(row)
            if len(all_rows) > 10000:  # 최대 10000행
                break

        if len(all_rows) < 1:
            return jsonify({"error": "빈 파일입니다."}), 400

        headers = all_rows[0]
        data_rows = all_rows[1:]

        # 통계 요약 생성
        total_rows = len(data_rows)
        total_cols = len(headers)

        # 각 컬럼 타입 추정 및 기본 통계
        col_stats = []
        for ci, h in enumerate(headers):
            vals = [r[ci] for r in data_rows if ci < len(r) and r[ci].strip()]
            # 숫자 판별
            nums = []
            for v in vals:
                try:
                    nums.append(float(v.replace(',', '')))
                except ValueError:
                    pass

            if len(nums) > len(vals) * 0.5 and nums:
                col_stats.append(f"  {h}: 숫자형 ({len(vals)}건, 범위 {min(nums):.4g}~{max(nums):.4g}, 평균 {sum(nums)/len(nums):.4g})")
            else:
                unique = len(set(vals))
                col_stats.append(f"  {h}: 문자형 ({len(vals)}건, 고유값 {unique}개)")

        summary = f"파일: {file.filename}\n행: {total_rows}개, 열: {total_cols}개\n컬럼:\n" + "\n".join(col_stats)

        # 미리보기 (상위 5행)
        preview_rows = data_rows[:5]
        preview_text = delimiter.join(headers) + "\n"
        for r in preview_rows:
            preview_text += delimiter.join(r) + "\n"
        if total_rows > 5:
            preview_text += f"... ({total_rows - 5}행 더 있음)"

        # 저장
        uploaded_csv_data = {
            "filename": file.filename,
            "headers": headers,
            "rows": data_rows,
            "summary": summary,
            "raw_preview": preview_text,
        }

        return jsonify({
            "success": True,
            "filename": file.filename,
            "rows": total_rows,
            "cols": total_cols,
            "headers": headers,
            "summary": summary,
            "preview": preview_text,
            "sample_rows": [dict(zip(headers, r)) for r in preview_rows[:3]],
        })

    except Exception as e:
        return jsonify({"error": f"파일 처리 오류: {str(e)}"}), 500


@app.route("/api/clear_csv", methods=["POST"])
def api_clear_csv():
    """업로드된 CSV 데이터 삭제"""
    global uploaded_csv_data
    uploaded_csv_data = {"filename": "", "headers": [], "rows": [], "summary": "", "raw_preview": ""}
    return jsonify({"success": True})


# ===================== 범용 파일 업로드 =====================
ALLOWED_EXTENSIONS = {
    'csv', 'tsv', 'txt', 'log', 'ini', 'cfg', 'conf', 'toml',
    'docx', 'xlsx', 'pdf', 'pptx',
    'md', 'markdown', 'rst',
    'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg',
    'py', 'js', 'ts', 'tsx', 'jsx', 'vue', 'svelte',
    'html', 'css', 'scss', 'less', 'json', 'xml', 'yaml', 'yml',
    'c', 'cpp', 'h', 'hpp', 'java', 'kt', 'scala', 'go', 'rs', 'sh', 'bat',
    'rb', 'php', 'swift', 'r', 'sql', 'lua', 'dart', 'zig',
}


def _detect_drm(filepath, ext):
    """파일의 DRM/암호화 보호 여부를 감지. 보호된 경우 설명 문자열 반환, 아니면 None."""
    try:
        if ext == 'pdf':
            with open(filepath, 'rb') as f:
                head = f.read(4096)
            # PDF 암호화 플래그 감지
            if b'/Encrypt' in head:
                return "PDF 파일에 암호화(DRM)가 설정되어 있어 내용을 읽을 수 없습니다."

        elif ext in ('docx', 'xlsx', 'pptx'):
            import zipfile
            # DRM 걸린 Office 파일은 ZIP이 아닌 OLE2(CFB) 형식인 경우가 많음
            if not zipfile.is_zipfile(filepath):
                with open(filepath, 'rb') as f:
                    magic = f.read(8)
                # Microsoft OLE2 Compound File (암호화된 Office)
                if magic[:4] == b'\xd0\xcf\x11\xe0':
                    return f"{ext.upper()} 파일에 DRM 또는 암호 보호가 설정되어 있어 내용을 읽을 수 없습니다."
                return None
            # ZIP 기반이어도 EncryptedPackage가 있으면 DRM
            try:
                with zipfile.ZipFile(filepath) as z:
                    names = z.namelist()
                    if 'EncryptedPackage' in names or 'EncryptedInfo' in names:
                        return f"{ext.upper()} 파일에 DRM 또는 암호 보호가 설정되어 있어 내용을 읽을 수 없습니다."
                    # Office 365 IRM(Information Rights Management) 감지
                    if any('irm' in n.lower() or 'rights' in n.lower() for n in names):
                        return f"{ext.upper()} 파일에 IRM(정보 권한 관리)이 설정되어 있어 내용을 읽을 수 없습니다."
            except zipfile.BadZipFile:
                pass

        elif ext in ('epub',):
            import zipfile
            if zipfile.is_zipfile(filepath):
                try:
                    with zipfile.ZipFile(filepath) as z:
                        names = z.namelist()
                        if any('encryption.xml' in n.lower() or 'rights.xml' in n.lower() for n in names):
                            return "EPUB 파일에 DRM이 설정되어 있어 내용을 읽을 수 없습니다."
                except zipfile.BadZipFile:
                    pass

    except Exception:
        pass
    return None

def extract_file_text(filepath, filename):
    """파일에서 텍스트 추출 (가능한 경우)"""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    text = ""
    file_type = "unknown"

    # DRM/암호화 보호 감지
    drm_msg = _detect_drm(filepath, ext)
    if drm_msg:
        return drm_msg, "drm_protected"

    try:
        # 텍스트 기반 파일
        if ext in ('md', 'markdown', 'rst', 'txt', 'log', 'ini', 'cfg', 'conf', 'toml',
                    'py', 'js', 'ts', 'tsx', 'jsx', 'vue', 'svelte',
                    'html', 'css', 'scss', 'less', 'json', 'xml', 'yaml', 'yml',
                    'c', 'cpp', 'h', 'hpp', 'java', 'kt', 'scala', 'go', 'rs', 'sh', 'bat',
                    'rb', 'php', 'swift', 'r', 'sql', 'lua', 'dart', 'zig'):
            for enc in ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]:
                try:
                    with open(filepath, "r", encoding=enc) as f:
                        text = f.read()
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            file_type = "text"

        # DOCX
        elif ext == 'docx':
            try:
                import zipfile
                with zipfile.ZipFile(filepath) as z:
                    with z.open('word/document.xml') as doc:
                        import xml.etree.ElementTree as ET
                        tree = ET.parse(doc)
                        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                        paragraphs = tree.findall('.//w:p', ns)
                        texts = []
                        for p in paragraphs:
                            runs = p.findall('.//w:t', ns)
                            para_text = ''.join(r.text or '' for r in runs)
                            if para_text.strip():
                                texts.append(para_text)
                        text = '\n'.join(texts)
                file_type = "docx"
            except Exception as e:
                text = f"(DOCX 텍스트 추출 실패: {e})"
                file_type = "docx"

        # XLSX
        elif ext == 'xlsx':
            try:
                import zipfile
                import xml.etree.ElementTree as ET
                with zipfile.ZipFile(filepath) as z:
                    # shared strings
                    shared = []
                    if 'xl/sharedStrings.xml' in z.namelist():
                        with z.open('xl/sharedStrings.xml') as ss:
                            tree = ET.parse(ss)
                            ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                            for si in tree.findall('.//s:si', ns):
                                t_el = si.find('.//s:t', ns)
                                shared.append(t_el.text if t_el is not None and t_el.text else '')
                    # sheet1
                    with z.open('xl/worksheets/sheet1.xml') as ws:
                        tree = ET.parse(ws)
                        ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                        rows = tree.findall('.//s:row', ns)
                        lines = []
                        for row in rows[:100]:  # 최대 100행
                            cells = []
                            for c in row.findall('s:c', ns):
                                v_el = c.find('s:v', ns)
                                t_attr = c.get('t', '')
                                if v_el is not None and v_el.text is not None:
                                    if t_attr == 's' and v_el.text.isdigit():
                                        idx = int(v_el.text)
                                        cells.append(shared[idx] if idx < len(shared) else v_el.text)
                                    else:
                                        cells.append(v_el.text)
                                else:
                                    cells.append('')
                            lines.append('\t'.join(cells))
                        text = '\n'.join(lines)
                file_type = "xlsx"
            except Exception as e:
                text = f"(XLSX 텍스트 추출 실패: {e})"
                file_type = "xlsx"

        # PDF (기본 텍스트 추출)
        elif ext == 'pdf':
            try:
                with open(filepath, 'rb') as f:
                    content = f.read()
                # 간단한 PDF 텍스트 추출 (stream 기반)
                import re as re_mod
                texts = re_mod.findall(rb'\((.*?)\)', content)
                decoded = []
                for t in texts[:500]:
                    try:
                        decoded.append(t.decode('utf-8', errors='ignore'))
                    except:
                        pass
                text = ' '.join(decoded)
                if len(text.strip()) < 50:
                    text = f"(PDF 텍스트 추출 제한 - 바이너리 PDF. 파일크기: {os.path.getsize(filepath)}바이트)"
                file_type = "pdf"
            except Exception as e:
                text = f"(PDF 텍스트 추출 실패: {e})"
                file_type = "pdf"

        # PPTX
        elif ext == 'pptx':
            try:
                import zipfile
                import xml.etree.ElementTree as ET
                with zipfile.ZipFile(filepath) as z:
                    slide_texts = []
                    for name in sorted(z.namelist()):
                        if name.startswith('ppt/slides/slide') and name.endswith('.xml'):
                            with z.open(name) as sl:
                                tree = ET.parse(sl)
                                ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
                                texts_in_slide = []
                                for t in tree.findall('.//a:t', ns):
                                    if t.text and t.text.strip():
                                        texts_in_slide.append(t.text)
                                if texts_in_slide:
                                    slide_num = name.split('slide')[-1].split('.')[0]
                                    slide_texts.append(f"[슬라이드 {slide_num}] {' | '.join(texts_in_slide)}")
                    text = '\n'.join(slide_texts)
                file_type = "pptx"
            except Exception as e:
                text = f"(PPTX 텍스트 추출 실패: {e})"
                file_type = "pptx"

        # 이미지
        elif ext in ('png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg'):
            import base64
            with open(filepath, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('ascii')
            text = f"[이미지 파일: {filename}, 크기: {os.path.getsize(filepath)}바이트, base64 길이: {len(img_data)}]"
            file_type = "image"
            # img_data는 caller에서 file_info에 저장 (VL 모델용)

        else:
            text = f"(지원하지 않는 파일 형식: .{ext})"
            file_type = ext

    except Exception as e:
        text = f"(파일 읽기 오류: {e})"

    return text, file_type


@app.route("/api/upload_file", methods=["POST"])
def api_upload_file():
    """범용 파일 업로드 (docx, xlsx, pdf, pptx, md, 이미지, 코드 등)"""
    global uploaded_files

    if 'file' not in request.files:
        return jsonify({"error": "파일이 없습니다."}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "파일명이 없습니다."}), 400

    fname = file.filename
    ext = fname.rsplit('.', 1)[-1].lower() if '.' in fname else ''

    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"지원하지 않는 파일 형식입니다: .{ext}"}), 400

    try:
        # 파일 저장
        import time as _time
        safe_name = f"{int(_time.time())}_{fname}"
        filepath = os.path.join(UPLOAD_DIR, safe_name)
        file.save(filepath)
        file_size = os.path.getsize(filepath)

        # 텍스트 추출
        text, file_type = extract_file_text(filepath, fname)

        # DRM 보호 파일 감지 → 업로드는 정상 진행, 경고만 추가
        drm_warning = None
        if file_type == "drm_protected":
            drm_warning = text  # DRM 안내 메시지
            file_type = ext  # 원래 확장자를 타입으로 사용

        # 미리보기 (최대 3000자)
        preview = text[:3000]
        if len(text) > 3000:
            preview += f"\n... (총 {len(text)}자 중 3000자 표시)"

        # 파일 정보 저장
        file_info = {
            "filename": fname,
            "safe_name": safe_name,
            "type": file_type,
            "ext": ext,
            "size": file_size,
            "content_preview": preview,
            "content_full": text[:60000],  # 시스템 프롬프트용 최대 60000자
            "path": filepath,
        }
        # VL 모델용: 이미지 base64 데이터 저장 (최대 10MB)
        if file_type == "image" and ext in ('png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg'):
            import base64 as _b64
            if file_size <= 10 * 1024 * 1024:  # 10MB 제한
                with open(filepath, 'rb') as _imgf:
                    file_info["img_base64"] = _b64.b64encode(_imgf.read()).decode('ascii')
            else:
                file_info["img_base64"] = None  # 너무 큰 이미지
        uploaded_files.append(file_info)

        # 아이콘 매핑
        icon_map = {
            "docx": "📄", "xlsx": "📊", "pdf": "📕", "pptx": "📽️",
            "text": "📝", "image": "🖼️",
        }
        icon = icon_map.get(file_type, "📎")
        if drm_warning:
            icon = "🔒"

        resp = {
            "success": True,
            "filename": fname,
            "type": file_type,
            "ext": ext,
            "size": file_size,
            "icon": icon,
            "preview": preview[:500],
            "total_files": len(uploaded_files),
        }
        if drm_warning:
            resp["drm_warning"] = drm_warning
        return jsonify(resp)

    except Exception as e:
        return jsonify({"error": f"파일 처리 오류: {str(e)}"}), 500


@app.route("/api/uploaded_files", methods=["GET"])
def api_uploaded_files():
    """업로드된 파일 목록"""
    return jsonify({
        "files": [{
            "filename": f["filename"],
            "type": f["type"],
            "ext": f["ext"],
            "size": f["size"],
        } for f in uploaded_files]
    })


@app.route("/api/remove_file", methods=["POST"])
def api_remove_file():
    """업로드된 파일 제거"""
    global uploaded_files
    data = request.json
    fname = data.get("filename", "")

    new_files = []
    removed = False
    for f in uploaded_files:
        if f["filename"] == fname and not removed:
            # 파일 삭제
            try:
                os.remove(f["path"])
            except Exception:
                pass
            removed = True
        else:
            new_files.append(f)
    uploaded_files = new_files
    return jsonify({"success": removed, "total_files": len(uploaded_files)})


@app.route("/api/clear_files", methods=["POST"])
def api_clear_files():
    """모든 업로드 파일 제거"""
    global uploaded_files
    for f in uploaded_files:
        try:
            os.remove(f["path"])
        except Exception:
            pass
    uploaded_files = []
    return jsonify({"success": True})


# ===================== Draw.io XML 검증/수정 =====================
def _sanitize_drawio_xml(xml_str: str) -> str:
    """Draw.io XML의 흔한 오류를 자동 수정"""
    import xml.etree.ElementTree as ET
    import re as _re

    if not xml_str or not xml_str.strip():
        return xml_str

    # 1) mxfile 래퍼가 없으면 추가
    if '<mxfile' not in xml_str and '<mxGraphModel' in xml_str:
        xml_str = f'<mxfile host="app.diagrams.net" type="device">\n<diagram id="d1" name="Page-1">\n{xml_str}\n</diagram>\n</mxfile>'
    elif '<mxfile' not in xml_str and '<mxCell' in xml_str:
        xml_str = (
            '<mxfile host="app.diagrams.net" type="device">\n'
            '<diagram id="d1" name="Page-1">\n'
            '<mxGraphModel dx="1422" dy="762" grid="1" pageWidth="1169" pageHeight="827">\n'
            '<root>\n<mxCell id="0"/>\n<mxCell id="1" parent="0"/>\n'
            f'{xml_str}\n</root>\n</mxGraphModel>\n</diagram>\n</mxfile>'
        )

    # 2) XML 파싱 시도
    try:
        ET.fromstring(xml_str)
        return xml_str  # 정상이면 그대로 반환
    except ET.ParseError:
        pass  # 파싱 실패 → 수정 시도

    # 3) value/label 속성 안의 & 이스케이프
    xml_str = _re.sub(
        r'((?:value|label)=")([^"]*?)(")',
        lambda m: m.group(1) + _re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;|#)', '&amp;', m.group(2)) + m.group(3),
        xml_str
    )

    # 4) 구조 재구성: <root>...</root> 안의 요소들을 추출해서 올바르게 재조립
    try:
        ET.fromstring(xml_str)
        return xml_str
    except ET.ParseError:
        pass

    # 5) 최후 수단: root 내부의 모든 완전한 mxCell/UserObject 요소만 추출하여 재조립
    wrap_match = _re.search(r'([\s\S]*?<root\b[^>]*>)([\s\S]*?)(</root>[\s\S]*)', xml_str)
    if wrap_match:
        header, body, footer = wrap_match.group(1), wrap_match.group(2), wrap_match.group(3)
        elements = []
        # mxCell with children (mxGeometry 포함)
        for m in _re.finditer(r'<mxCell\s[^>]*?>(?:\s*<(?:mxGeometry|Array|mxPoint)[\s\S]*?(?:/>|</(?:mxGeometry|Array|mxPoint)>))*\s*</mxCell>', body):
            elements.append(m.group(0))
        # self-closing mxCell
        for m in _re.finditer(r'<mxCell\s[^>]*?/>', body):
            if not any(m.group(0) in e for e in elements):
                elements.append(m.group(0))
        # UserObject
        for m in _re.finditer(r'<UserObject[\s\S]*?</UserObject>', body):
            elements.append(m.group(0))

        if elements:
            xml_str = header + '\n' + '\n'.join(elements) + '\n' + footer

    # 최종 검증
    try:
        ET.fromstring(xml_str)
    except ET.ParseError:
        pass  # 최선을 다했으나 여전히 문제가 있으면 그냥 반환

    return xml_str


# ===================== Draw.io 다운로드 =====================
@app.route("/api/download_drawio", methods=["POST"])
def api_download_drawio():
    """Draw.io XML을 .drawio 파일로 다운로드"""
    data = request.json
    xml_content = data.get("xml", "")
    filename = data.get("filename", "diagram.drawio")
    if not filename.endswith(".drawio"):
        filename += ".drawio"

    # --- XML 구조 검증 및 자동 수정 ---
    xml_content = _sanitize_drawio_xml(xml_content)

    from io import BytesIO
    buf = BytesIO(xml_content.encode("utf-8"))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="application/xml")


@app.route("/api/save_md", methods=["POST"])
def api_save_md():
    """마크다운 내용을 .md 파일로 다운로드"""
    data = request.json
    content = data.get("content", "")
    filename = data.get("filename", "report.md")
    if not filename.endswith(".md"):
        filename += ".md"

    from io import BytesIO
    buf = BytesIO(content.encode("utf-8"))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="text/markdown")




def _prepare_pptx_code_for_copy(code):
    """복사/실행용 python-pptx 코드를 손상 없이 정리."""
    import re as _re

    if not isinstance(code, str):
        return ""

    code = code.strip()

    # 0) 바깥 코드펜스 제거
    code = _re.sub(r'^\s*```(?:python|py)?\s*\n', '', code, flags=_re.IGNORECASE)
    code = _re.sub(r'\n```\s*$', '', code)

    # 1) 특수 따옴표/전각 문자 정리
    _quote_map = {
        '\u201c': '"', '\u201d': '"',
        '\u2018': "'", '\u2019': "'",
        '\u300c': "'", '\u300d': "'",
        '\u300e': '"', '\u300f': '"',
        '\uff02': '"', '\uff07': "'",
    }
    for _old, _new in _quote_map.items():
        code = code.replace(_old, _new)

    _fullwidth_map = {
        '\uff08': '(', '\uff09': ')',
        '\uff1a': ':', '\uff1b': ';',
        '\uff0c': ',', '\uff0e': '.',
        '\uff1d': '=', '\uff0b': '+',
    }
    for _old, _new in _fullwidth_map.items():
        code = code.replace(_old, _new)

    # 2) 이어붙은 "from ... import ..."만 안전하게 줄바꿈
    code = _re.sub(
        r'(\bfrom\s+[A-Za-z_][\w\.]*\s+import\s+[^\n#;]+?)\s+(?=from\s+[A-Za-z_][\w\.]*\s+import\s+)',
        r'\1\n',
        code
    )

    def _looks_like_code_line(s):
        return bool(_re.match(
            r'^(from\s+|import\s+|def\s+|class\s+|if\s+|elif\s+|else:|for\s+|while\s+|try:|except\b|finally:|with\s+|return\b|yield\b|pass\b|break\b|continue\b|@|#|\"\"\"|\'\'\'|[A-Za-z_]\w*\s*=|[A-Za-z_][\w\.]*\s*\(|\)|\]|\})',
            s
        ))

    # 3) 명백한 메타/설명 라인만 주석 처리 (실제 코드는 보존)
    cleaned = []
    for ln in code.splitlines():
        s = ln.strip()
        if not s:
            cleaned.append(ln)
            continue

        if any(k in s for k in ('자동 스킬', '[LOCAL', 'prompt~', 'ctx=', 'max_tokens=')):
            continue

        if _re.match(r'^[=\-]{3,}$', s):
            cleaned.append('# ' + s)
            continue

        if _re.match(r'^\d+\.\s+', s) and not _looks_like_code_line(s):
            cleaned.append('# ' + s)
            continue

        if _re.search(r'[가-힣]', s) and not _looks_like_code_line(s):
            cleaned.append('# ' + s)
            continue

        cleaned.append(ln)

    code = '\n'.join(cleaned)

    # 4) 자주 틀리는 python-pptx 패턴 보정
    code = code.replace('.add_text_box(', '.add_textbox(').replace('.add_text_box (', '.add_textbox(')
    code = _re.sub(
        r'\b(?:content|body|placeholder|text_frame|title_shape|subtitle|body_shape|content_placeholder)\s*\.\s*shapes\s*\.',
        'slide.shapes.',
        code
    )

    # 5) 문법 실패 시, 문제 라인이 비코드로 보이면 그 라인만 주석 처리
    lines = code.splitlines()
    for _ in range(10):
        code_try = '\n'.join(lines)
        try:
            compile(code_try, '<pptx_copy_code>', 'exec')
            return code_try
        except SyntaxError as e:
            idx = (e.lineno or 1) - 1
            if idx < 0 or idx >= len(lines):
                break
            s = lines[idx].strip()
            if (not s) or s.startswith('#'):
                break
            if _looks_like_code_line(s):
                break
            lines[idx] = '# ' + lines[idx]

    return '\n'.join(lines)


@app.route("/api/prepare_pptx_code", methods=["POST"])
def api_prepare_pptx_code():
    """PPT 코드 복사 전에 실행 가능하도록 정리된 Python 코드를 반환"""
    data = request.json or {}
    code = data.get("code", "")
    if not isinstance(code, str) or not code.strip():
        return jsonify({"error": "code가 비어있습니다."}), 400

    try:
        prepared = _prepare_pptx_code_for_copy(code)
        return jsonify({"code": prepared})
    except Exception as e:
        return jsonify({"error": f"코드 정리 실패: {str(e)}"}), 500
# ===================== PPT 생성 (python-pptx) =====================
@app.route("/api/generate_pptx", methods=["POST"])
def api_generate_pptx():
    """LLM이 생성한 python-pptx 코드를 실행해 .pptx 파일 반환"""
    data = request.json
    code = data.get("code", "")
    filename = data.get("filename", "presentation.pptx") or "presentation.pptx"
    if not filename.endswith(".pptx"):
        filename += ".pptx"

    try:
        from pptx import Presentation as _PptxCheck
    except ImportError:
        return jsonify({"error": "python-pptx 패키지가 설치되어 있지 않습니다.\npip install python-pptx"}), 500

    import tempfile, subprocess, json as _json

    # --- LLM 생성 코드 자동 정리 (SyntaxError 방지) ---
    import re as _re

    # 1) 특수 따옴표 → ASCII 따옴표
    _quote_map = {
        '\u201c': '"', '\u201d': '"',   # " " (curly double)
        '\u2018': "'", '\u2019': "'",   # ' ' (curly single)
        '\u300c': "'", '\u300d': "'",   # 「 」
        '\u300e': '"', '\u300f': '"',   # 『 』
        '\uff02': '"', '\uff07': "'",   # ＂ ＇ (fullwidth)
    }
    for _old, _new in _quote_map.items():
        code = code.replace(_old, _new)

    # 2) 전각 문자 → 반각 (코드 안에서 자주 문제됨)
    _fullwidth_map = {
        '\uff08': '(', '\uff09': ')',   # （ ）
        '\uff1a': ':', '\uff1b': ';',   # ： ；
        '\uff0c': ',', '\uff0e': '.',   # ， ．
        '\uff1d': '=', '\uff0b': '+',   # ＝ ＋
    }
    for _old, _new in _fullwidth_map.items():
        code = code.replace(_old, _new)

    # 2.5) LLM이 자주 틀리는 pptx 메서드/속성 자동 교정
    _method_typo_map = {
        '.add_text_box(': '.add_textbox(',
        '.add_text_box (': '.add_textbox(',
        'XLCT.XL_LEGEND_POSITION': 'XL_LEGEND_POSITION',
        'XL_CHART_TYPE.XL_LEGEND_POSITION': 'XL_LEGEND_POSITION',
        'chart.plot_area.shapes': '[]',
        'chart.plot_area.data_labels': 'chart.plots[0].data_labels',
    }
    for _old, _new in _method_typo_map.items():
        code = code.replace(_old, _new)
        
    if 'XL_LEGEND_POSITION' in code and 'from pptx.enum.chart import XL_LEGEND_POSITION' not in code:
        code = "from pptx.enum.chart import XL_LEGEND_POSITION\n" + code

    # 3) placeholder.shapes → slide.shapes 자동 수정
    # LLM이 content/body/placeholder 등에 .shapes를 호출하는 실수 수정
    code = _re.sub(
        r'\b(?:content|body|placeholder|text_frame|title_shape|subtitle|body_shape|content_placeholder)\s*\.\s*shapes\s*\.',
        'slide.shapes.',
        code
    )

    # 3.5) 주석 형식 자동 수정: # === 내용 === 또는 === 내용 === → ## 내용
    def _fix_comments(code_str):
        lines = code_str.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            # 패턴1: "=== 내용 ===" (주석 기호 없이 === 로만 된 줄)
            m = _re.match(r'^([ \t]*)(=+\s*)(.+?)\s*=*\s*$', stripped)
            if m and not any(kw in stripped for kw in ('import ', 'from ', 'def ', 'class ', '=========')):
                content = m.group(3).strip().rstrip('=').strip()
                if content and not '=' in content.replace('==', ''):
                    indent = line[:len(line) - len(line.lstrip())]
                    lines[i] = f"{indent}## {content}"
                    continue
            # 패턴2: "# === 내용 ===" → "## 내용"
            m = _re.match(r'^([ \t]*)#\s*(=+\s*)(.+?)\s*=*\s*$', stripped)
            if m:
                content = m.group(3).strip().rstrip('=').strip()
                if content:
                    indent = line[:len(line) - len(line.lstrip())]
                    lines[i] = f"{indent}## {content}"
                    continue
            # 패턴3: "# --- 내용 ---" → "## 내용"
            m = _re.match(r'^([ \t]*)#\s*(-+\s*)(.+?)\s*-*\s*$', stripped)
            if m:
                content = m.group(3).strip().rstrip('-').strip()
                if content:
                    indent = line[:len(line) - len(line.lstrip())]
                    lines[i] = f"{indent}## {content}"
                    continue
            # 패턴4: 단일 "#" 주석을 "##"로 변환 (코드가 아닌 순수 주석 라인)
            m = _re.match(r'^([ \t]*)#\s+([^#!].+)$', stripped)
            if m and not stripped.startswith('#!'):
                content = m.group(2)
                # 코드 라인이 아닌 순수 주석인지 확인
                if not any(kw in content for kw in ('coding:', 'type:', 'noqa', 'pylint', 'pragma')):
                    indent = line[:len(line) - len(line.lstrip())]
                    lines[i] = f"{indent}## {content}"
        return '\n'.join(lines)

    code = _fix_comments(code)

    # 4) 중첩 따옴표 패턴 수정: p.text = ""내용"" → p.text = "내용"
    code = _re.sub(r'= ""([^"]*?)""', r"= '\1'", code)
    code = _re.sub(r"= ''([^']*?)''", r"= '\1'", code)
    # "text" 안에 이스케이프 안 된 "가 있는 패턴도 수정
    code = _re.sub(r'""([^"]{2,}?)""', r"'\1'", code)

    # 4.5) f-string 숫자 포맷팅 시도 시 변수를 _safe_float()로 강제 캐스팅 (문자열 등 포맷불가 값 대비)
    code = _re.sub(r'\{([^:}]+?):([^{}]*?\.?\d+[fF%])\}', r'{_safe_float(\1):\2}', code)
    code = code.replace("_safe_float(float(", "_safe_float(")
    code = code.replace("_safe_float(_safe_float(", "_safe_float(")

    # 5) 괄호 불일치 자동 수정 (예: slide_layouts[6) → slide_layouts[6])
    def _fix_bracket_mismatch(code_str):
        lines = code_str.split('\n')
        for i, line in enumerate(lines):
            # 각 라인에서 괄호 스택 추적
            stack = []
            pairs = {'(': ')', '[': ']', '{': '}'}
            reverse = {')': '(', ']': '[', '}': '{'}
            in_str = None
            chars = list(line)
            fixed = False
            for j, ch in enumerate(chars):
                # 문자열 안에서는 괄호 무시
                if ch in ('"', "'") and (j == 0 or chars[j-1] != '\\' or (j >= 2 and chars[j-2] == '\\')):
                    if in_str is None:
                        in_str = ch
                    elif in_str == ch:
                        in_str = None
                    continue
                if in_str:
                    continue
                if ch in pairs:
                    stack.append((ch, j))
                elif ch in reverse:
                    expected_open = reverse[ch]
                    if stack and stack[-1][0] == expected_open:
                        stack.pop()
                    elif stack and stack[-1][0] != expected_open:
                        # 불일치: 올바른 닫는 괄호로 교체
                        chars[j] = pairs[stack[-1][0]]
                        stack.pop()
                        fixed = True
            if fixed:
                lines[i] = ''.join(chars)
        return '\n'.join(lines)

    # 6) 미종료 문자열 리터럴 자동 수정
    def _fix_unterminated_strings(code_str):
        lines = code_str.split('\n')
        for i, line in enumerate(lines):
            stripped = line.rstrip()
            if not stripped:
                continue
            # 코드 라인에서 문자열이 열렸는데 닫히지 않은 경우 감지
            in_str = None
            escaped = False
            for ch in stripped:
                if escaped:
                    escaped = False
                    continue
                if ch == '\\':
                    escaped = True
                    continue
                if ch in ('"', "'"):
                    if in_str is None:
                        # 삼중 따옴표 시작 여부는 무시 (복잡)
                        in_str = ch
                    elif in_str == ch:
                        in_str = None
            # 문자열이 닫히지 않았으면 끝에 닫는 따옴표 추가
            if in_str is not None:
                lines[i] = stripped + in_str
        return '\n'.join(lines)

    # 7) 구문 검증 → 실패 시 추가 정리 시도
    try:
        compile(code, '<pptx_code>', 'exec')
    except SyntaxError as _e:
        # 미종료 문자열 수정 시도
        code = _fix_unterminated_strings(code)
        # 괄호 불일치 자동 수정 시도
        code = _fix_bracket_mismatch(code)
        # 문자열 안 따옴표 충돌 최후 수단: 문제 라인의 바깥 따옴표를 삼중따옴표로 교체
        lines = code.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            # .text = "..." 또는 .text = '...' 패턴에서 내부 따옴표 충돌 수정
            m = _re.match(r'''(.*(?:\.text|\.value)\s*=\s*)(['"])(.*)\2\s*$''', stripped)
            if m:
                prefix, quote, content = m.groups()
                alt_quote = "'" if quote == '"' else '"'
                if quote in content and alt_quote not in content:
                    indent = line[:len(line) - len(stripped)]
                    lines[i] = f"{indent}{prefix}{alt_quote}{content}{alt_quote}"
                elif quote in content:
                    content_escaped = content.replace(quote, '\\' + quote)
                    indent = line[:len(line) - len(stripped)]
                    lines[i] = f"{indent}{prefix}{quote}{content_escaped}{quote}"
        code = '\n'.join(lines)
        # 수정 후 재검증
        try:
            compile(code, '<pptx_code>', 'exec')
        except SyntaxError:
            pass  # 자동 수정 실패 — 원래 에러 그대로 출력될 것

    # 임시 파일에 코드 작성 → 별도 프로세스에서 실행
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, filename)
        # 코드에서 writeFile/save 경로를 out_path로 강제 교체
        wrapped_code = (
            "import sys, os\n"
            "def _safe_float(val):\n"
            "    try: return float(val)\n"
            "    except: return 0.0\n"
            f"_OUTPUT_PATH = {repr(out_path)}\n"
            "# --- monkey-patch: fix default.pptx template path resolution ---\n"
            "import pptx.api as _pptx_api\n"
            "def _fixed_default_pptx_path():\n"
            "    import pptx as _pptx_pkg\n"
            "    _pkg_dir = os.path.dirname(_pptx_pkg.__file__)\n"
            "    return os.path.join(_pkg_dir, 'templates', 'default.pptx')\n"
            "_pptx_api._default_pptx_path = _fixed_default_pptx_path\n"
            "# --- monkey-patch: auto-convert float to int for pptx coordinates ---\n"
            "import pptx.oxml.simpletypes as _st\n"
            "_orig_validate_int = _st.BaseSimpleType.validate_int\n"
            "@classmethod\n"
            "def _patched_validate_int(cls, value):\n"
            "    if isinstance(value, float):\n"
            "        value = int(round(value))\n"
            "    return _orig_validate_int.__func__(cls, value)\n"
            "_st.BaseSimpleType.validate_int = _patched_validate_int\n"
            "# --- monkey-patch: fix vertical_anchor using PP_ALIGN instead of MSO_ANCHOR ---\n"
            "import pptx.text.text as _pptx_text\n"
            "_orig_va_setter = _pptx_text.TextFrame.vertical_anchor.fset\n"
            "def _patched_va_setter(self, value):\n"
            "    try:\n"
            "        _orig_va_setter(self, value)\n"
            "    except (ValueError, TypeError):\n"
            "        from pptx.enum.text import MSO_ANCHOR\n"
            "        _map = {0: MSO_ANCHOR.TOP, 1: MSO_ANCHOR.MIDDLE, 2: MSO_ANCHOR.MIDDLE, 3: MSO_ANCHOR.BOTTOM}\n"
            "        v = _map.get(int(value), MSO_ANCHOR.MIDDLE) if hasattr(value, '__int__') else MSO_ANCHOR.MIDDLE\n"
            "        _orig_va_setter(self, v)\n"
            "_pptx_text.TextFrame.vertical_anchor = property(_pptx_text.TextFrame.vertical_anchor.fget, _patched_va_setter)\n"
            "# --- monkey-patch: safe table.cell() to prevent IndexError ---\n"
            "import pptx.table as _pptx_table\n"
            "_orig_table_cell = _pptx_table.Table.cell\n"
            "def _safe_table_cell(self, row_idx, col_idx):\n"
            "    row_count = len(self._tbl.tr_lst)\n"
            "    col_count = len(self._tbl.tr_lst[0].tc_lst) if row_count > 0 else 0\n"
            "    if row_idx >= row_count:\n"
            "        for _ in range(row_idx - row_count + 1):\n"
            "            from pptx.oxml.ns import qn as _qn\n"
            "            from lxml import etree as _etree\n"
            "            new_tr = _etree.SubElement(self._tbl, _qn('a:tr'))\n"
            "            new_tr.set('h', self._tbl.tr_lst[0].get('h', '370840'))\n"
            "            for c in range(col_count):\n"
            "                tc = _etree.SubElement(new_tr, _qn('a:tc'))\n"
            "                txBody = _etree.SubElement(tc, _qn('a:txBody'))\n"
            "                _etree.SubElement(txBody, _qn('a:bodyPr'))\n"
            "                p = _etree.SubElement(txBody, _qn('a:p'))\n"
            "                tc_pr = _etree.SubElement(tc, _qn('a:tcPr'))\n"
            "    return _orig_table_cell(self, row_idx, col_idx)\n"
            "_pptx_table.Table.cell = _safe_table_cell\n"
            "# --- monkey-patch: allow GraphicFrame.rows and GraphicFrame.cell --- \n"
            "import pptx.shapes.graphfrm as _gf\n"
            "if hasattr(_gf, 'GraphicFrame'):\n"
            "    _gf.GraphicFrame.rows = property(lambda self: self.table.rows if self.has_table else None)\n"
            "    _gf.GraphicFrame.columns = property(lambda self: self.table.columns if self.has_table else None)\n"
            "    _gf.GraphicFrame.cell = lambda self, r, c: self.table.cell(r, c) if self.has_table else None\n"
            "# --- user code ---\n"
            f"{code}\n"
        )
        # .save() 호출이 없으면 자동 추가
        if ".save(" not in code:
            wrapped_code += f"\nprs.save(_OUTPUT_PATH)\n"
        else:
            # 사용자 코드 부분에서만 save 경로를 _OUTPUT_PATH로 교체
            import re as _save_re
            marker = "# --- user code ---\n"
            idx = wrapped_code.find(marker)
            if idx >= 0:
                prefix = wrapped_code[:idx + len(marker)]
                user_part = wrapped_code[idx + len(marker):]
                user_part = _save_re.sub(r'\.save\([^)]*\)', '.save(_OUTPUT_PATH)', user_part)
                wrapped_code = prefix + user_part
            else:
                wrapped_code = _save_re.sub(r'\.save\([^)]*\)', '.save(_OUTPUT_PATH)', wrapped_code)

        script_path = os.path.join(tmpdir, "_gen_pptx.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(wrapped_code)

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=30,
            cwd=tmpdir,
            encoding="utf-8", errors="replace"
        )

        if result.returncode != 0:
            err_msg = (result.stderr.strip() if result.stderr else "알 수 없는 오류")[-1000:]
            return jsonify({
                "error": f"PPT 생성 코드 실행 실패:\n{err_msg}"
            }), 500

        if not os.path.exists(out_path):
            # save 경로가 다를 수 있으니 tmpdir에서 .pptx 파일 탐색
            pptx_files = [f for f in os.listdir(tmpdir) if f.endswith(".pptx")]
            if pptx_files:
                out_path = os.path.join(tmpdir, pptx_files[0])
            else:
                return jsonify({"error": "PPT 파일이 생성되지 않았습니다."}), 500

        import shutil, datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"presentation_{timestamp}.pptx"

        # uploads 폴더에 저장 후 브라우저 다운로드 URL 반환
        uploads_dir = os.path.join(BASE_DIR, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        save_path = os.path.join(uploads_dir, final_filename)
        shutil.copy2(out_path, save_path)

        return jsonify({
            "download_url": f"/api/download_static/presentation.pptx?id={timestamp}",
            "message": f"'{final_filename}' PPT 생성 완료!"
        })

@app.route("/api/download_static/presentation.pptx", methods=["GET"])
def api_download_static():
    file_id = request.args.get('id', '')
    if not file_id: return "Invalid ID", 400
    save_name = f"presentation_{file_id}.pptx"
    static_dir = os.path.join(BASE_DIR, 'uploads')
    return send_file(
        os.path.join(static_dir, save_name), 
        as_attachment=True, 
        download_name="presentation.pptx",
        mimetype="application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )


# ===================== 응답 중지 =====================
chat_stop_flag = {"stop": False}

@app.route("/api/chat/stop", methods=["POST"])
def api_chat_stop():
    """진행 중인 LLM 응답을 중지"""
    chat_stop_flag["stop"] = True
    return jsonify({"stopped": True})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """LLM API 프록시 - 스킬을 시스템 프롬프트에 넣어서 회사 API로 전달"""
    chat_stop_flag["stop"] = False  # 새 요청 시작 시 플래그 초기화
    data = request.json
    # 환경 선택 시 서버에서 URL/모델 결정, 토큰은 TOKEN.TXT에서 읽음
    user_env = data.get("env", "")
    auto_routed = False
    route_reason = ""

    # AUTO 모드: 질문 분석 후 최적 모델 자동 선택
    if user_env == "auto":
        last_query = ""
        msgs = data.get("messages", [])
        if msgs:
            last_msg = msgs[-1]
            last_query = last_msg.get("content", "") if isinstance(last_msg.get("content"), str) else ""
        env_id, route_reason = classify_and_route(last_query, msgs, uploaded_files)
        auto_routed = True
    elif user_env and user_env in ENV_CONFIG:
        env_id = user_env
    else:
        env_id = user_env

    if env_id and env_id in ENV_CONFIG:
        api_url = ENV_CONFIG[env_id]["url"]
        model = ENV_CONFIG[env_id]["model"]
    else:
        api_url = data.get("api_url", "")
        model = data.get("model", "")
    api_key = API_TOKEN or data.get("api_key", "")
    messages = data.get("messages", [])
    skill_ids = data.get("skills", [])
    effort = data.get("effort", 2)
    output_format = data.get("format", "code")
    writing_style = data.get("writing_style", "")
    custom_system_prompt = data.get("system_prompt", "")
    max_tokens = data.get("max_tokens", 8192)
    think_mode = data.get("think_mode", False)
    requested_output_format = output_format
    is_gguf = env_id.startswith("gguf-") if env_id else False

    # 출력형식/스타일 자동 분류 (format=auto 또는 writing_style=auto일 때)
    auto_format = False
    auto_style = False
    auto_fmt_reason = ""
    if output_format == "auto" or writing_style == "auto":
        last_q = ""
        if messages:
            last_m = messages[-1]
            last_q = last_m.get("content", "") if isinstance(last_m.get("content"), str) else ""
        auto_fmt, auto_sty, auto_fmt_reason = classify_format_and_style(
            last_q, messages, uploaded_files, skill_ids
        )
        if output_format == "auto":
            output_format = auto_fmt
            auto_format = True
        if writing_style == "auto":
            writing_style = auto_sty
            auto_style = True

    # GGUF 전용: PPT/Draw.io 생성 요청은 code 형식으로 강제 (API 경로 영향 없음)
    last_user_query = ""
    if messages:
        last_m = messages[-1]
        if isinstance(last_m.get("content"), str):
            last_user_query = last_m.get("content", "")
    ql = last_user_query.lower()
    gguf_artifact_request = is_gguf and any(kw in ql for kw in [
        "ppt", "피피티", "파워포인트", "프레젠테이션", "슬라이드", "deck",
        "draw.io", "drawio", "드로우", "다이어그램", "mxfile", "mxgraphmodel",
    ])
    if gguf_artifact_request and requested_output_format == "auto" and output_format in ("report", "analysis", "step-by-step"):
        output_format = "code"
        auto_format = True
        auto_fmt_reason = (auto_fmt_reason + " | " if auto_fmt_reason else "") + "GGUF artifact request -> force code"

    if (not api_url or not model) and not env_id.startswith("gguf-"):
        return jsonify({"error": "API URL과 모델 이름을 설정해주세요."}), 400

    # 시스템 프롬프트 구성
    # 출력 형식에 따라 기본 규칙 분기
    non_code_formats = ("report", "analysis", "step-by-step")
    if output_format in non_code_formats:
        code_rules = """- 사용자가 코드를 명시적으로 요청한 경우에만 코드를 포함하세요
- 기본적으로 자연어(글)로 설명, 분석, 보고하세요"""
    else:
        code_rules = """- 코드는 즉시 실행 가능하게 (import 포함, 필요 패키지 명시)
- 에러 처리(try/except)를 포함하세요
- 코드 주석은 반드시 ## 형식으로 작성하세요. 단일 #이 아니라 ##을 사용하고, 구간 설명용으로만 간결하게 작성하세요. 줄마다 주석을 달지 마세요. 예: ## 데이터 전처리, ## API 호출"""

    # 사고 모드 on/off
    if think_mode:
        think_rule = "- 답변 전에 반드시 <think>...</think> 태그 안에서 충분히 사고한 후 답변하세요. 사고 내용도 반드시 한국어로 작성하세요."
    else:
        think_rule = "- <think> 태그를 사용하지 마세요. 사고 과정 없이 바로 답변하세요."

    default_prompt = f"""당신은 Demos(민중) Alpha 0.8 - 과학 연구와 소프트웨어 개발을 돕는 전문 AI 어시스턴트입니다.
370개+ 전문 스킬(과학/개발/AI/인프라/비즈니스)을 활용할 수 있습니다.

[기본 규칙]
- 반드시 한국어(한글)로 답변하세요. 코드 주석도 한글로 작성하세요.
{think_rule}
{code_rules}

[스킬 활용]
- 아래에 로드된 SKILL 내용은 당신의 전문 지식입니다
- 해당 스킬의 방법론, 코드 패턴, API, 도구를 적극 활용하세요
- 여러 스킬이 로드된 경우 관련된 것끼리 조합하여 최적의 답변을 생성하세요

"""

    if custom_system_prompt:
        system_prompt = custom_system_prompt + "\n\n" + default_prompt
    else:
        system_prompt = default_prompt

    # ===== 토큰 예산 관리 =====
    # API 대형 모델 → 제한 거의 없음 (128K+ 컨텍스트)
    # GGUF 로컬 → 32K 컨텍스트이므로 스마트하게 제한
    is_gguf = env_id.startswith("gguf-") if env_id else False
    history_chars = sum(len(m.get("content", "")) for m in messages)

    if is_gguf:
        gguf_ctx = 32768
        # GGUF: 토큰 ≈ 글자수 * 0.7 (한글 기준), 시스템+히스토리+응답 여유
        available_chars = int(gguf_ctx / 0.7) - len(system_prompt) - history_chars - 3000
        available_chars = max(available_chars, 3000)
    else:
        available_chars = 999999  # API는 사실상 무제한

    # 스킬 로드
    loaded = []
    available = scan_skills()
    total_skill_chars = 0
    num_skills = max(len(skill_ids), 1)
    per_skill_limit = available_chars // num_skills if is_gguf else 999999

    for sid in skill_ids:
        content = load_skill_content(sid)
        if content:
            # GGUF만 자르기, API는 전체 로드
            if is_gguf and total_skill_chars + len(content) > available_chars:
                remaining = available_chars - total_skill_chars
                if remaining > 500:
                    content = content[:remaining] + f"\n... ({len(content)}자 중 {remaining}자 로드)\n"
                else:
                    system_prompt += f"[⚠️ GGUF 컨텍스트 한도 → 스킬 생략: {', '.join(skill_ids[len(loaded):])}]\n"
                    break

            if is_gguf and len(content) > per_skill_limit:
                content = content[:per_skill_limit] + f"\n... ({len(content)}자 중 {per_skill_limit}자 로드)\n"

            system_prompt += f"=== SKILL: {sid} ===\n{content}\n\n"
            total_skill_chars += len(content)
            loaded.append(sid)

            # scripts/references 목록 (짧으니 항상 포함)
            info = available.get(sid, {})
            scripts = info.get("scripts", [])
            if scripts:
                system_prompt += f"[{sid} 스크립트: {', '.join(s['name'] for s in scripts)}]\n"
            refs = info.get("references", [])
            if refs:
                system_prompt += f"[{sid} 참고: {', '.join(r['name'] for r in refs)}]\n"
            system_prompt += "\n"

    if loaded:
        system_prompt += f"[로드된 스킬: {', '.join(loaded)}]\n\n"
        if "agent-llm-architect" in loaded:
            system_prompt += (
                "=== LLM 시스템 설계 규칙 ===\n"
                "사용자가 LLM/에이전트/RAG 설계를 요청하면 다음 순서로 답변하세요:\n"
                "1. 먼저 요구사항과 제약을 분석하세요(트래픽, 지연시간, 예산, 보안, 데이터 민감도).\n"
                "2. 아키텍처 옵션 2~3개를 비교하고 트레이드오프를 명확히 제시하세요.\n"
                "3. 최종 추천안을 제시하고, 선택 근거를 성능/비용/운영성 관점으로 설명하세요.\n"
                "4. 구현 단계를 POC→파일럿→프로덕션으로 나눠 체크리스트 형태로 제시하세요.\n"
                "5. 필수 운영 지표(SLO, 토큰비용, 오류율, 환각률, 안전성)를 포함하세요.\n\n"
            )
        # pptx 스킬이 로드됐으면 python-pptx 코드 생성 지시 추가
        ppt_requested = 'ql' in locals() and any(kw in ql for kw in ["ppt", "피피티", "파워포인트", "프레젠테이션", "슬라이드", "deck"])
        if "pptx" in loaded or ppt_requested:
            system_prompt += (
                "=== PPT 생성 규칙 ===\n"
                "사용자가 PPT/프레젠테이션/슬라이드 생성을 요청하면:\n"
                "1. 반드시 ```python 코드블록 안에 python-pptx 코드를 작성하세요.\n"
                "2. 코드는 `from pptx import Presentation`으로 시작하세요.\n"
                "3. 대화 내용, 분석 결과, 업로드 파일 내용을 PPT에 반영하세요.\n"
                "4. 각 슬라이드에 제목, 내용, 표, 차트 등을 포함하세요.\n"
                "5. 마지막에 `prs.save('presentation.pptx')` 호출을 포함하세요.\n"
                "6. 디자인 가이드: 볼드 색상 팔레트, 다양한 레이아웃, 데이터 시각화 활용.\n"
                "7. 중요: 좌표/크기 값은 반드시 정수(int)로 전달하세요. "
                "Inches(), Cm(), Pt(), Emu() 결과를 직접 사용하면 됩니다. "
                "나눗셈(/) 등으로 float이 생기면 반드시 int()로 감싸세요.\n"
                "8. 중요: tf.vertical_anchor에는 PP_ALIGN이 아니라 MSO_ANCHOR를 사용하세요. "
                "예: `from pptx.enum.text import MSO_ANCHOR` 후 `tf.vertical_anchor = MSO_ANCHOR.MIDDLE`. "
                "PP_ALIGN.CENTER를 vertical_anchor에 쓰면 에러가 납니다.\n"
                "9. 중요: 문자열 안에 한국어 특수 따옴표(\u201c \u201d \u2018 \u2019 \u300c \u300d)를 절대 사용하지 마세요. "
                "반드시 일반 ASCII 따옴표(\" ' )만 사용하세요. "
                "인용문이 필요하면 작은따옴표(')로 감싸세요. 예: p.text = '기술은 훌륭하지만, 보안에 구멍이 많다.'\n"
                "10. 코드 주석은 반드시 `## 설명내용` 형식으로 작성하세요. "
                "단일 #이 아니라 ##을 사용하고, 구간 설명용으로만 간결하게 작성하세요. "
                "예: `## 슬라이드 1: 표지`, `## 차트 데이터 설정`. 줄마다 주석을 달지 마세요.\n"
                "11. 중요: 괄호 매칭에 극도로 주의하세요! "
                "[인덱스]는 반드시 ]로 닫고, (함수호출)은 반드시 )로 닫으세요. "
                "예: prs.slide_layouts[6] (O) / prs.slide_layouts[6) (X — SyntaxError). "
                "코드 작성 후 모든 괄호 쌍([]/()/{})이 올바르게 매칭되는지 반드시 검증하세요.\n"
                "12. 중요: 색상은 반드시 `from pptx.dml.color import RGBColor`를 사용하세요! "
                "RgbColor(X), rgbColor(X) 등은 존재하지 않습니다. 반드시 RGBColor (대문자 RGB)만 사용하세요. "
                "예: `from pptx.dml.color import RGBColor` → `fill.fore_color.rgb = RGBColor(0x1A, 0x73, 0xE8)` "
                "또는 `run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)`. "
                "절대 `from pptx.dml.color import RgbColor`를 쓰지 마세요 — ImportError가 발생합니다.\n"
                "13. 중요: 대화에서 데이터/수치/통계가 있으면 반드시 python-pptx 차트로 시각화하세요! "
                "텍스트만 나열하지 말고, 차트 슬라이드를 추가하세요.\n"
                "14. 중요: 차트/도형/표를 추가할 때 반드시 `slide.shapes.add_chart(...)`, `slide.shapes.add_table(...)` 등 "
                "slide 객체의 shapes를 사용하세요! placeholder나 content 변수에는 .shapes 속성이 없습니다. "
                "예: `content = slide.placeholders[1]`한 뒤 `content.shapes.add_chart(...)` (X — AttributeError). "
                "반드시 `slide.shapes.add_chart(...)` (O)로 작성하세요.\n"
                "15. 중요: 표(table) 생성 시 add_table(rows, cols, ...)의 rows 수를 데이터 행 수 + 헤더 행에 정확히 맞추세요! "
                "데이터가 N개이면 add_table(N+1, cols, ...)로 헤더 포함 N+1행을 만드세요. "
                "행 수가 부족하면 table.cell(i, j)에서 IndexError가 발생합니다. "
                "예: 데이터 5개 → add_table(6, 3, ...) (헤더 1행 + 데이터 5행 = 6행).\n"
                "16. 중요: 문자열 리터럴은 반드시 같은 줄에서 열고 닫으세요! "
                "줄바꿈이 필요하면 삼중따옴표(triple quotes)를 사용하세요. "
                "예: p.text = '긴 텍스트' (O) / p.text = '긴 텍스트 (X — SyntaxError: unterminated string literal).\n"
                "17. 중요 (필수): 파이썬 문법에 맞게 들여쓰기(Indentation)와 줄바꿈을 철저히 지키고 한 줄에 여러 명령어(`slide = ... title = ...`)를 이어서 쓰지 마세요. SyntaxError가 발생합니다.\n"
                "18. 중요 (필수): 반드시 단 1개의 완성된 파이썬 스크립트만 ```python 과 ``` 사이에 출력하세요. 코드 앞뒤로 인사말, <think> 태그, 요약 설명 등 부가적인 텍스트를 절대 쓰지 마세요. 오직 파이썬 코드만 출력해야 작동합니다.\n"
                "python-pptx 차트 코드 예시:\n"
                "```python\n"
                "from pptx.chart.data import CategoryChartData\n"
                "from pptx.enum.chart import XL_CHART_TYPE\n"
                "from pptx.util import Inches\n"
                "\n"
                "## 차트 데이터 설정\n"
                "chart_data = CategoryChartData()\n"
                "chart_data.categories = ['1분기', '2분기', '3분기', '4분기']\n"
                "chart_data.add_series('매출', (120, 190, 300, 250))\n"
                "chart_data.add_series('비용', (80, 100, 150, 130))\n"
                "\n"
                "## 차트 슬라이드 추가\n"
                "slide = prs.slides.add_slide(prs.slide_layouts[6])\n"
                "chart_frame = slide.shapes.add_chart(\n"
                "    XL_CHART_TYPE.COLUMN_CLUSTERED,\n"
                "    Inches(1), Inches(1.5), Inches(8), Inches(5),\n"
                "    chart_data\n"
                ")\n"
                "chart = chart_frame.chart\n"
                "chart.has_legend = True\n"
                "chart.legend.include_in_layout = False\n"
                "```\n"
                "지원 차트: XL_CHART_TYPE.COLUMN_CLUSTERED(세로막대), LINE(꺾은선), PIE(원형), "
                "BAR_CLUSTERED(가로막대), DOUGHNUT(도넛), AREA(영역), RADAR(방사형), "
                "XY_SCATTER(산점도), LINE_MARKERS(꺾은선+마커)\n"
                "여러 시리즈 비교: chart_data.add_series()를 여러 번 호출\n"
                "원형 차트: CategoryChartData에 시리즈 1개만 추가, XL_CHART_TYPE.PIE 사용\n"
                "프론트엔드가 코드를 감지하여 '📽️ PPT 생성 & 다운로드' 버튼을 자동 표시합니다.\n\n"
            )

        drawio_requested = 'ql' in locals() and any(kw in ql for kw in ["drawio", "draw.io", "드로우", "드로잉", "drawingio", "다이어그램", "구조도", "흐름도", "아키텍처", "배치도", "dfd"])
        if "drawio-diagram" in loaded or drawio_requested:
            system_prompt += (
                "=== Draw.io 다이어그램 생성 규칙 ===\n"
                "사용자가 Draw.io, 다이어그램, 아키텍처, 구조도 생성을 요청하면:\n"
                "1. 반드시 ```drawio 코드블록 안에 완전한 XML 코드를 작성하세요.\n"
                "2. XML 구조 예시: ```drawio\n<mxfile><diagram><mxGraphModel><root><mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>...</root></mxGraphModel></diagram></mxfile>\n```\n"
                "3. 중요 (GGUF 필수): 다이어그램에 대한 부가적인 설명, 설치 가이드, 인사말, <think> 태그 등은 **절대 출력하지 마세요**.\n"
                "4. 오직 단 1개의 완성된 XML 코드 블록만 출력해야 프론트엔드가 다이어그램 렌더러를 띄웁니다.\n"
                "5. 노드의 위치(x, y), 크기(width, height) 속성을 적절히 지정해 겹치지 않게 하세요.\n\n"
            )
        # === 차트/그래프 생성 가이드 (항상 포함 - 스킬 불필요) ===
        system_prompt += (
            "=== 차트/그래프 생성 규칙 ===\n"
            "사용자가 그래프, 차트, 시각화를 요청하거나 데이터 분석 결과를 보여줄 때:\n"
            "1. ```chart 코드블록 안에 Chart.js JSON config를 작성하세요.\n"
            "2. 프론트엔드가 자동으로 인터랙티브 차트를 렌더링합니다.\n"
            "3. 지원 차트 유형: bar, line, pie, doughnut, radar, scatter, bubble, polarArea\n"
            "4. JSON 형식 예시:\n"
            "```chart\n"
            '{\n'
            '  "type": "bar",\n'
            '  "data": {\n'
            '    "labels": ["1월", "2월", "3월"],\n'
            '    "datasets": [{\n'
            '      "label": "매출",\n'
            '      "data": [120, 190, 300],\n'
            '      "backgroundColor": ["#6366f1", "#8b5cf6", "#a78bfa"]\n'
            '    }]\n'
            '  },\n'
            '  "options": {\n'
            '    "plugins": {"title": {"display": true, "text": "월별 매출"}}\n'
            '  }\n'
            '}\n'
            "```\n"
            "5. 여러 데이터셋 비교: datasets 배열에 여러 객체 추가\n"
            "6. 색상: backgroundColor, borderColor 사용. 배열로 항목별 색상 지정 가능\n"
            "7. 추천 색상 팔레트: #6366f1(인디고), #8b5cf6(보라), #ec4899(핑크), #f59e0b(앰버), #10b981(에메랄드), #3b82f6(블루), #ef4444(레드), #84cc16(라임)\n"
            "8. 데이터 분석/CSV 분석 결과를 보여줄 때 적극 활용하세요.\n"
            "9. 한 응답에 여러 ```chart 블록을 사용해 다양한 관점의 차트를 보여줄 수 있습니다.\n\n"
        )

        # 멀티에이전트 오케스트레이션 (스킬 2개 이상)
        if len(loaded) >= 2:
            loaded_set = {sid: True for sid in loaded}
            orch_prompt = build_orchestration_prompt(
                messages[-1].get("content", "") if messages else "",
                loaded, loaded_set
            )
            system_prompt += orch_prompt

    # CSV 데이터가 업로드되어 있으면 시스템 프롬프트에 포함
    include_csv = data.get("include_csv", True)
    if include_csv and uploaded_csv_data["filename"]:
        csv_info = uploaded_csv_data["summary"]
        # 데이터 미리보기 (최대 50행)
        preview_limit = min(50, len(uploaded_csv_data["rows"]))
        csv_rows_text = ",".join(uploaded_csv_data["headers"]) + "\n"
        for row in uploaded_csv_data["rows"][:preview_limit]:
            csv_rows_text += ",".join(str(c) for c in row) + "\n"
        if len(uploaded_csv_data["rows"]) > preview_limit:
            csv_rows_text += f"... (총 {len(uploaded_csv_data['rows'])}행 중 {preview_limit}행만 표시)\n"

        system_prompt += f"=== 업로드된 CSV 데이터 ===\n{csv_info}\n\n데이터 미리보기:\n{csv_rows_text}\n\n"
        system_prompt += "사용자가 이 데이터에 대해 질문하면 위 CSV 데이터를 기반으로 분석해주세요.\n\n"

    # 업로드된 파일 내용 주입
    if uploaded_files:
        system_prompt += f"=== 업로드된 파일 ({len(uploaded_files)}개) ===\n"
        for uf in uploaded_files:
            system_prompt += f"\n--- 파일: {uf['filename']} ({uf['type']}, {uf['size']}바이트) ---\n"
            content = uf.get("content_full", "")
            if is_gguf:
                # GGUF: 파일당 최대 8000자로 제한
                content = content[:8000]
                if len(uf.get("content_full", "")) > 8000:
                    content += f"\n... (총 {len(uf['content_full'])}자 중 8000자 표시)"
            else:
                content = content[:50000]  # API: 파일당 최대 50000자
                if len(uf.get("content_full", "")) > 50000:
                    content += f"\n... (총 {len(uf['content_full'])}자 중 50000자 표시)"
            system_prompt += content + "\n"
        system_prompt += "\n사용자가 업로드된 파일에 대해 질문하면 위 내용을 기반으로 답변하세요.\n\n"

    # 응답 수준
    effort_map = [
        "매우 간결하게 핵심만 답변하세요.",
        "간결하게 답변하세요.",
        "표준적인 깊이로 설명하세요.",
        "매우 상세하고 전문적으로 분석하세요. 코드에 주석을 자세히 달고, 원리를 설명하세요.",
    ]
    format_map = {
        "code": "답변을 Python 코드 중심으로 작성하세요. import 포함, 즉시 실행 가능하게.",
        "code-fix": "문제 진단 → 수정 전/후 비교 → 수정 이유 설명 순서로 답변하세요. 최소 변경 원칙.",
        "analysis": "분석 중심으로 답변하세요. 문제 정의 → 핵심 가정 → 옵션 비교(장단점/트레이드오프) → 결론/권장안 순서. 코드보다 근거와 의사결정을 우선하고, 코드는 꼭 필요한 경우에만 최소한으로 포함하세요.",
        "report": "반드시 보고서 형식의 글(텍스트)로 작성하세요. 코드를 생성하지 마세요. 구조: 제목 → 요약(Executive Summary) → 배경/목적 → 본문(핵심 내용, 분석, 근거) → 결론 및 제언. 전문적이고 읽기 쉬운 문서 형태로 답변하세요.",
        "step-by-step": "답변을 단계별(1,2,3...)로 작성하세요. 각 단계마다 무엇을 하는지, 왜 필요한지를 글로 설명하세요. 코드보다 설명과 이해를 우선하세요.",
    }

    system_prompt += effort_map[min(effort, 3)] + "\n"
    fmt_instruction = format_map.get(output_format, "")
    if fmt_instruction:
        system_prompt += fmt_instruction + "\n"
    # 코드가 아닌 출력 형식일 때는 코드 중심 답변을 억제
    if output_format in ("report", "analysis", "step-by-step"):
        system_prompt += "중요: 사용자가 코드를 명시적으로 요청하지 않는 한, 코드 블록 없이 자연어(글)로 답변하세요.\n"
    if writing_style:
        system_prompt += f"작성 스타일: {writing_style}\n"

    # API 요청 구성
    api_messages = [{"role": "system", "content": system_prompt}] + messages
    temperature_map = [0.1, 0.3, 0.5, 0.7]

    # ===== VL 모델: 이미지 첨부 시 OpenAI Vision API 포맷 변환 (GGUF / API 공통) =====
    # env_id가 vl-로 시작하거나, gguf- 계열 중 모델명에 vl이 포함된 경우
    is_gguf_vl = env_id.startswith("gguf-") and "vl" in ENV_CONFIG.get(env_id, {}).get("name", "").lower()
    has_vision = "vision" in get_model_capabilities(env_id) or is_gguf_vl
    
    image_files = [f for f in uploaded_files if f.get("type") == "image" and f.get("img_base64")]
    
    if has_vision and image_files and api_messages:
        # 마지막 user 메시지를 멀티모달 포맷으로 변환
        for i in range(len(api_messages) - 1, -1, -1):
            if api_messages[i].get("role") == "user":
                text_content = api_messages[i].get("content", "")
                if isinstance(text_content, str):
                    content_parts = [{"type": "text", "text": text_content}]
                    for img_f in image_files:
                        ext = img_f.get("ext", "png").lower()
                        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif",
                                "bmp": "bmp", "webp": "webp", "svg": "svg+xml"}.get(ext, "png")
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{mime};base64,{img_f['img_base64']}"
                            }
                        })
                    api_messages[i]["content"] = content_parts
                break

    def _normalize_gguf_artifact_answer(answer_text, artifact_request=False):
        """GGUF 출력의 코드블록 라벨을 보정해 프론트 버튼 감지를 안정화."""
        if not artifact_request or not isinstance(answer_text, str) or not answer_text.strip():
            return answer_text

        import re as _re_norm
        normalized = answer_text
        # ```py, ```python3 -> ```python
        normalized = _re_norm.sub(r"```(?:py|python3)\s*\n", "```python\n", normalized, flags=_re_norm.IGNORECASE)

        # 무라벨 코드블록에서 PPT/Draw.io 패턴을 감지해 라벨 부여
        def _upgrade_unlabeled(m):
            body = m.group(1)
            low = body.lower()
            if any(tok in low for tok in ("from pptx", "import pptx", "presentation(", "add_slide")):
                return f"```python\n{body}```"
            if any(tok in low for tok in ("<mxfile", "<mxgraphmodel", "<mxcell")):
                return f"```drawio\n{body}```"
            return m.group(0)

        normalized = _re_norm.sub(r"```[ \t]*\n([\s\S]*?)```", _upgrade_unlabeled, normalized)

        # 코드블록이 전혀 없지만 python-pptx 코드가 섞여 있으면 후행 코드블록으로 보강
        low_all = normalized.lower()
        if ("from pptx" in low_all or "import pptx" in low_all) and "```python" not in low_all:
            lines = normalized.splitlines()
            start = -1
            for i, line in enumerate(lines):
                ll = line.lower()
                if "from pptx" in ll or "import pptx" in ll or "presentation(" in ll:
                    start = i
                    break
            if start >= 0:
                candidate = "\n".join(lines[start:]).strip()
                if candidate:
                    normalized += "\n\n```python\n" + candidate + "\n```"

        return normalized

    # ===== GGUF 로컬 모델: Python에서 직접 추론 =====
    if env_id.startswith("gguf-"):
        # 선택된 환경의 모델 경로로 동적 로드/스왑
        gguf_path = ENV_CONFIG.get(env_id, {}).get("_gguf_path")
        if gguf_path:
            if not load_gguf_model(gguf_path):
                return jsonify({"error": f"GGUF 모델 로드 실패: {os.path.basename(gguf_path)}"}), 500
        if gguf_model is None:
            return jsonify({"error": "GGUF 모델이 로드되지 않았습니다. .gguf 파일과 llama-cpp-python이 필요합니다."}), 400

        # GGUF 컨텍스트 한도 내에서 max_tokens 자동 조정 (보수적 계산)
        gguf_ctx_attr = getattr(gguf_model, 'n_ctx', None)
        gguf_ctx = gguf_ctx_attr() if callable(gguf_ctx_attr) else (gguf_ctx_attr if gguf_ctx_attr is not None else 32768)
        gguf_reply_cap = max(256, int(os.getenv("GGUF_MAX_TOKENS_CAP", "8192")))
        gguf_ctx_reserve = max(512, int(os.getenv("GGUF_CONTEXT_RESERVE", "1536")))

        def _estimate_gguf_prompt_tokens(msgs):
            # 1) 모델 토크나이저 기반 추정 (가능하면 사용)
            try:
                flat_parts = []
                for m in msgs:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    if isinstance(content, list):
                        text_parts = []
                        for p in content:
                            if isinstance(p, dict) and p.get("type") == "text":
                                text_parts.append(str(p.get("text", "")))
                        content = "\n".join(text_parts)
                    flat_parts.append(f"{role}: {str(content)}")
                flat_text = "\n".join(flat_parts).encode("utf-8", errors="ignore")
                toks = gguf_model.tokenize(flat_text, add_bos=True)
                return len(toks) + 256  # chat template/여유 버퍼
            except Exception:
                # 2) 폴백: 한국어 기준 보수 추정
                total_chars = 0
                for m in msgs:
                    content = m.get("content", "")
                    if isinstance(content, list):
                        text_parts = []
                        for p in content:
                            if isinstance(p, dict) and p.get("type") == "text":
                                text_parts.append(str(p.get("text", "")))
                        content = "\n".join(text_parts)
                    total_chars += len(str(content))
                return int(total_chars * 1.1) + 512

        prompt_tokens_est = _estimate_gguf_prompt_tokens(api_messages)
        safe_max = max(256, gguf_ctx - prompt_tokens_est - gguf_ctx_reserve)
        actual_max_tokens = min(max_tokens, safe_max, gguf_reply_cap)

        if prompt_tokens_est > gguf_ctx - gguf_ctx_reserve:
            return jsonify({"error": f"프롬프트가 너무 깁니다 (~{prompt_tokens_est}토큰). 스킬 수를 줄이거나 히스토리를 초기화해주세요. (GGUF ctx: {gguf_ctx})"}), 400

        answer, err = gguf_chat(
            api_messages,
            temperature=temperature_map[min(effort, 3)],
            max_tokens=actual_max_tokens,
            stop_flag=chat_stop_flag,
        )
        # GGUF 디코드 실패 시: 경량 컨텍스트 + 작은 토큰으로 1회 자동 재시도
        if err and isinstance(err, str) and ("Failed completely even with batch size 1" in err or "llama.eval(decode)" in err):
            compact_system = (
                "당신은 도움이 되는 AI입니다. "
                "핵심만 간결하게 답하고, 코드 요청이면 실행 가능한 코드만 출력하세요."
            )
            compact_messages = [{"role": "system", "content": compact_system}]
            compact_messages.extend(messages[-6:] if len(messages) > 6 else messages)

            compact_prompt_est = _estimate_gguf_prompt_tokens(compact_messages)
            compact_safe_max = max(256, gguf_ctx - compact_prompt_est - gguf_ctx_reserve)
            retry_max = min(2048, max(256, actual_max_tokens // 2), compact_safe_max)

            retry_answer, retry_err = gguf_chat(
                compact_messages,
                temperature=temperature_map[min(effort, 3)],
                max_tokens=retry_max,
                stop_flag=chat_stop_flag,
            )
            if not retry_err and retry_answer:
                answer = retry_answer
                err = None
                prompt_tokens_est = compact_prompt_est
                actual_max_tokens = retry_max

        if err:
            return jsonify({"error": err}), 500

        # GGUF: <think> 사고만 있고 본문 없이 잘린 경우 재시도
        import re as _re
        if answer and answer.strip():
            stripped = answer.strip()
            has_open = "<think>" in stripped
            has_close = "</think>" in stripped
            think_only_gguf = False
            if has_open and not has_close:
                think_only_gguf = True
            elif has_open and has_close:
                after = _re.sub(r'<think>[\s\S]*?</think>\s*', '', stripped).strip()
                if len(after) < 20:
                    think_only_gguf = True
            if think_only_gguf:
                retry_max = min(actual_max_tokens * 2, safe_max)
                if retry_max > actual_max_tokens:
                    retry_answer, retry_err = gguf_chat(
                        api_messages,
                        temperature=temperature_map[min(effort, 3)],
                        max_tokens=retry_max,
                        stop_flag=chat_stop_flag,
                    )
                    if not retry_err and retry_answer:
                        answer = retry_answer

        answer = _normalize_gguf_artifact_answer(answer, gguf_artifact_request)

        return jsonify({
            "content": answer,
            "loaded_skills": loaded,
            "system_prompt_length": len(system_prompt),
            "tokens_budget": f"prompt~{prompt_tokens_est}, max_tokens={actual_max_tokens}, ctx={gguf_ctx}",
        })


    # ===== 회사 API: HTTP 요청 (폴백 체인 지원) =====
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # 폴백 체인 구성: 현재 모델 → 대체 모델들
    primary_reg_key = get_registry_key_for_env(env_id)
    if primary_reg_key:
        fallback_keys = [primary_reg_key] + FALLBACK_CHAINS.get(primary_reg_key, [])
    else:
        fallback_keys = []  # GGUF 등은 폴백 없음

    # 이미지가 있으면 non-vision 폴백에서 이미지 제거 (텍스트만 전송)
    def _prepare_messages_for_model(reg_key, msgs):
        """VL 모델이 아닌 경우 멀티모달 content를 텍스트로 되돌리기"""
        cap = MODEL_REGISTRY.get(reg_key, {}).get("capabilities", set())
        if "vision" not in cap:
            clean = []
            for m in msgs:
                if isinstance(m.get("content"), list):
                    text_parts = [p["text"] for p in m["content"] if p.get("type") == "text"]
                    clean.append({**m, "content": "\n".join(text_parts)})
                else:
                    clean.append(m)
            return clean
        return msgs

    fallback_used = False
    fallback_from = ""
    actual_model_used = model
    last_error = None

    models_tried = []
    if fallback_keys:
        for attempt, reg_key in enumerate(fallback_keys[:3]):  # 최대 3회
            reg = MODEL_REGISTRY[reg_key]
            try_url = reg["url"]
            try_model = reg["model"]
            try_msgs = _prepare_messages_for_model(reg_key, api_messages)
            models_tried.append(try_model)

            try:
                resp = req.post(
                    try_url,
                    headers=headers,
                    json={
                        "model": try_model,
                        "messages": try_msgs,
                        "temperature": temperature_map[min(effort, 3)],
                        "max_tokens": max_tokens,
                        "stream": False,
                    },
                    timeout=120,
                    verify=False,
                )
                resp.raise_for_status()
                result = resp.json()

                # 응답 추출
                truncated = False
                if "choices" in result and len(result["choices"]) > 0:
                    answer = result["choices"][0].get("message", {}).get("content", "")
                    finish_reason = result["choices"][0].get("finish_reason", "")
                    if finish_reason == "length":
                        truncated = True

                    # <think> 사고 과정만 있고 본문이 없이 잘린 경우 자동 재시도
                    import re as _re
                    think_only = False
                    if truncated and answer.strip():
                        # </think> 닫히지 않았거나, </think> 후 본문이 비어있는 경우
                        stripped = answer.strip()
                        has_open_think = "<think>" in stripped
                        has_close_think = "</think>" in stripped
                        if has_open_think and not has_close_think:
                            think_only = True
                        elif has_open_think and has_close_think:
                            after_think = _re.sub(r'<think>[\s\S]*?</think>\s*', '', stripped).strip()
                            if len(after_think) < 20:
                                think_only = True

                    if think_only and max_tokens < 32768:
                        # 토큰 2배로 늘려서 재시도 (최대 32768)
                        retry_max = min(max_tokens * 2, 32768)
                        try:
                            retry_resp = req.post(
                                try_url,
                                headers=headers,
                                json={
                                    "model": try_model,
                                    "messages": try_msgs,
                                    "temperature": temperature_map[min(effort, 3)],
                                    "max_tokens": retry_max,
                                    "stream": False,
                                },
                                timeout=180,
                                verify=False,
                            )
                            retry_resp.raise_for_status()
                            retry_result = retry_resp.json()
                            if "choices" in retry_result and len(retry_result["choices"]) > 0:
                                answer = retry_result["choices"][0].get("message", {}).get("content", "")
                                finish_reason = retry_result["choices"][0].get("finish_reason", "")
                                truncated = finish_reason == "length"
                        except Exception:
                            pass  # 재시도 실패 시 원래 응답 사용

                    if attempt > 0:
                        fallback_used = True
                        fallback_from = models_tried[0]
                        actual_model_used = try_model

                    resp_data = {
                        "content": answer,
                        "loaded_skills": loaded,
                        "system_prompt_length": len(system_prompt),
                        "model_used": try_model,
                    }
                    if auto_routed:
                        resp_data["auto_routed"] = True
                        resp_data["route_reason"] = route_reason
                    if auto_format or auto_style:
                        resp_data["auto_format"] = output_format if auto_format else None
                        resp_data["auto_style"] = writing_style if auto_style else None
                        resp_data["auto_fmt_reason"] = auto_fmt_reason
                    if fallback_used:
                        resp_data["fallback_used"] = True
                        resp_data["fallback_from"] = fallback_from
                    if truncated:
                        resp_data["truncated"] = True
                    return jsonify(resp_data)

                elif "error" in result:
                    last_error = f"API 에러: {result['error']}"
                    continue  # 다음 폴백 시도
                else:
                    last_error = f"예상치 못한 응답: {json.dumps(result, ensure_ascii=False, indent=2)}"
                    continue

            except req.exceptions.Timeout:
                last_error = "API 응답 시간 초과 (120초)"
                continue
            except req.exceptions.ConnectionError as e:
                last_error = f"API 연결 실패: {str(e)}"
                continue
            except req.exceptions.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if code == 401 or code == 403:
                    return jsonify({"error": f"인증 실패 ({code}): TOKEN.TXT의 API 키를 확인하세요."}), code
                last_error = f"HTTP {code}: {str(e)}"
                if e.response is not None:
                    try:
                        detail = json.dumps(e.response.json(), ensure_ascii=False)
                        last_error += f" - {detail[:300]}"
                    except Exception:
                        pass
                continue
            except Exception as e:
                last_error = f"오류: {str(e)}"
                continue

        # 모든 폴백 실패
        return jsonify({
            "error": f"모든 모델 시도 실패 ({', '.join(models_tried)}): {last_error}"
        }), 500

    else:
        # 폴백 체인 없는 경우 (GGUF 등) → 기존 단일 요청
        try:
            resp = req.post(
                api_url,
                headers=headers,
                json={
                    "model": model,
                    "messages": api_messages,
                    "temperature": temperature_map[min(effort, 3)],
                    "max_tokens": max_tokens,
                    "stream": False,
                },
                timeout=120,
                verify=False,
            )
            resp.raise_for_status()
            result = resp.json()

            truncated = False
            if "choices" in result and len(result["choices"]) > 0:
                answer = result["choices"][0].get("message", {}).get("content", "")
                finish_reason = result["choices"][0].get("finish_reason", "")
                if finish_reason == "length":
                    truncated = True

                # <think> 사고 과정만 있고 본문이 없이 잘린 경우 자동 재시도
                import re as _re
                think_only = False
                if truncated and answer.strip():
                    stripped = answer.strip()
                    has_open_think = "<think>" in stripped
                    has_close_think = "</think>" in stripped
                    if has_open_think and not has_close_think:
                        think_only = True
                    elif has_open_think and has_close_think:
                        after_think = _re.sub(r'<think>[\s\S]*?</think>\s*', '', stripped).strip()
                        if len(after_think) < 20:
                            think_only = True

                if think_only and max_tokens < 32768:
                    retry_max = min(max_tokens * 2, 32768)
                    try:
                        retry_resp = req.post(
                            api_url,
                            headers=headers,
                            json={
                                "model": model,
                                "messages": api_messages,
                                "temperature": temperature_map[min(effort, 3)],
                                "max_tokens": retry_max,
                                "stream": False,
                            },
                            timeout=180,
                            verify=False,
                        )
                        retry_resp.raise_for_status()
                        retry_result = retry_resp.json()
                        if "choices" in retry_result and len(retry_result["choices"]) > 0:
                            answer = retry_result["choices"][0].get("message", {}).get("content", "")
                            finish_reason = retry_result["choices"][0].get("finish_reason", "")
                            truncated = finish_reason == "length"
                    except Exception:
                        pass

            elif "error" in result:
                answer = f"API 에러: {result['error']}"
            else:
                answer = f"예상치 못한 응답: {json.dumps(result, ensure_ascii=False, indent=2)}"

            resp_data = {
                "content": answer,
                "loaded_skills": loaded,
                "system_prompt_length": len(system_prompt),
                "model_used": model,
            }
            if auto_routed:
                resp_data["auto_routed"] = True
                resp_data["route_reason"] = route_reason
            if auto_format or auto_style:
                resp_data["auto_format"] = output_format if auto_format else None
                resp_data["auto_style"] = writing_style if auto_style else None
                resp_data["auto_fmt_reason"] = auto_fmt_reason
            if truncated:
                resp_data["truncated"] = True
            return jsonify(resp_data)

        except req.exceptions.Timeout:
            return jsonify({"error": "API 응답 시간 초과 (120초). max_tokens를 줄이거나 API 서버 상태를 확인하세요."}), 504
        except req.exceptions.ConnectionError as e:
            return jsonify({"error": f"API 연결 실패: {str(e)}. URL을 확인하세요."}), 502
        except req.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code == 401 or code == 403:
                return jsonify({"error": f"인증 실패 ({code}): TOKEN.TXT의 API 키를 확인하세요."}), code
            detail = ""
            if e.response is not None:
                try:
                    body = e.response.json()
                    detail = json.dumps(body, ensure_ascii=False, indent=2)
                except Exception:
                    detail = e.response.text[:500] if e.response.text else ""
            prompt_chars = sum(len(m.get("content","")) for m in api_messages)
            err_msg = f"API HTTP 에러 ({code}): {str(e)}"
            if detail:
                err_msg += f"\n서버 응답: {detail}"
            err_msg += f"\n[요청 크기: 전체 메시지={prompt_chars}자, model={model}]"
            return jsonify({"error": err_msg}), code or 500
        except Exception as e:
            return jsonify({"error": f"오류 발생: {str(e)}"}), 500


# ============================================
# HTML 템플릿
# ============================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Demos(민중) Alpha 0.8</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f8f7f4;color:#1a1a1a;display:flex;height:100vh}
.sidebar{width:250px;background:#fff;border-right:1px solid #e5e3de;padding:20px 16px;display:flex;flex-direction:column;overflow-y:auto;transition:width .2s ease,padding .2s ease;flex-shrink:0}
.sidebar.collapsed{width:48px;padding:12px 6px;overflow:hidden}
.sidebar.collapsed .sidebar-inner{display:none}
.sidebar.collapsed .sidebar-file-panel{display:none!important}
.sidebar.collapsed .sidebar-logo{display:none}
.sidebar-toggle{position:absolute;top:12px;left:250px;width:24px;height:24px;border-radius:0 6px 6px 0;border:1px solid #e5e3de;border-left:none;background:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:12px;color:#999;z-index:200;transition:left .2s ease}
.sidebar-toggle:hover{background:#eef2ff;color:#6366f1}
body.sb-collapsed .sidebar-toggle{left:48px}
body.sb-collapsed .chat-box-fixed{left:48px}
.sidebar-toggle-mini{width:100%;padding:6px 0;border:none;background:none;cursor:pointer;font-size:16px;display:none;color:#6366f1}
.sidebar.collapsed .sidebar-toggle-mini{display:block}
.sidebar-logo{font-size:22px;font-weight:700;margin-bottom:24px}.sidebar-logo span{color:#6366f1}
.sidebar-btn{width:100%;padding:10px 14px;border-radius:8px;border:none;cursor:pointer;font-size:14px;text-align:left;margin-bottom:4px;background:transparent;color:#555;transition:all .15s}
.sidebar-btn:hover{background:#f3f2ef}.sidebar-btn.active{background:#6366f1;color:#fff}
.session-item{display:flex;align-items:center;padding:5px 8px;border-radius:6px;cursor:pointer;font-size:12px;color:#555;transition:all .12s;gap:4px;margin:1px 4px}
.session-item:hover{background:#f3f2ef}
.session-item.current{background:#eef2ff;color:#6366f1;font-weight:600}
.session-item .session-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.session-item .session-cb{width:14px;height:14px;cursor:pointer;accent-color:#6366f1;flex-shrink:0}
.session-item .session-time{font-size:9px;color:#bbb;flex-shrink:0}
.sidebar-collapse{display:flex;align-items:center;justify-content:space-between;cursor:pointer;padding:4px 0;user-select:none}
.sidebar-collapse:hover{opacity:.8}
.sidebar-collapse .arrow{font-size:10px;color:#999;transition:transform .15s}
.sidebar-collapse .arrow.open{transform:rotate(90deg)}
.sidebar-collapsible{overflow:hidden;transition:max-height .2s ease}
.sidebar-collapsible.collapsed{max-height:0!important}
.session-actions{display:flex;gap:4px;padding:4px 8px}
.session-actions button{flex:1;padding:3px 0;border:1px solid #e5e3de;border-radius:4px;background:#fafaf8;font-size:10px;cursor:pointer;transition:all .12s}
.session-actions button:hover{background:#eef2ff;border-color:#6366f1}
.session-actions button.danger{color:#ef4444;border-color:#fca5a5}
.session-actions button.danger:hover{background:#fef2f2;border-color:#ef4444}
.sidebar-section{font-size:11px;font-weight:600;color:#999;text-transform:uppercase;letter-spacing:.5px;margin:20px 0 8px 8px}
.skill-count{font-size:11px;color:#6366f1;background:#eef2ff;padding:2px 8px;border-radius:10px;margin-left:6px}
.sidebar-footer{margin-top:auto;padding-top:16px;border-top:1px solid #e5e3de}
.credits{font-size:12px;color:#999}
.main{flex:1;display:flex;flex-direction:column;overflow:hidden}
.header{padding:16px 32px;border-bottom:1px solid #e5e3de;background:#fff;display:flex;align-items:center;justify-content:space-between}
.project-title{font-size:20px;font-weight:600}
.content{flex:1;overflow-y:auto;padding:24px 32px}
.content-inner{max-width:800px;margin:0 auto}
.section-label{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:#555;margin-bottom:10px}
/* Chat - Fixed Bottom */
.chat-box-fixed{position:fixed;bottom:0;left:250px;right:340px;background:#fff;border-top:2px solid #e5e3de;padding:6px 32px;z-index:100;box-shadow:0 -2px 10px rgba(0,0,0,.05);transition:left .2s ease,right .2s ease}
.chat-box-fixed:focus-within{border-top-color:#6366f1}
.chat-box-fixed-inner{max-width:800px;margin:0 auto}
.chat-input{width:100%;border:none;outline:none;font-size:13px;resize:none;min-height:20px;max-height:100px;font-family:inherit;line-height:1.4}
.chat-input::placeholder{color:#aaa;font-size:12px}
.chat-footer{display:flex;justify-content:space-between;align-items:center;margin-top:2px}
.send-btn{width:32px;height:32px;border-radius:50%;border:none;background:#6366f1;color:#fff;cursor:pointer;font-size:14px;transition:background .15s}
.send-btn:hover{background:#4f46e5}
.send-btn:disabled{background:#ccc;cursor:not-allowed}
/* 채팅 첨부파일 */
.attach-btn{width:32px;height:32px;border-radius:50%;border:none;background:transparent;color:#999;cursor:pointer;font-size:18px;transition:all .15s;display:flex;align-items:center;justify-content:center}
.attach-btn:hover{background:#f0f0f0;color:#6366f1}
.chat-dropdown{padding:3px 6px;border-radius:8px;border:1.5px solid #e5e3de;font-size:11px;cursor:pointer;background:#fff;color:#555;outline:none;transition:all .15s;appearance:none;-webkit-appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='8' viewBox='0 0 8 8'%3E%3Cpath d='M0 2l4 4 4-4z' fill='%23999'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 6px center;padding-right:18px}
.chat-dropdown:hover{border-color:#6366f1}.chat-dropdown:focus{border-color:#6366f1;box-shadow:0 0 0 2px rgba(99,102,241,.15)}
.chat-dropdown option{font-size:12px}
.chat-attach-preview{display:flex;flex-wrap:wrap;gap:6px;padding:4px 0}
.chat-attach-badge{display:inline-flex;align-items:center;gap:4px;background:#eef2ff;border:1px solid #c7d2fe;border-radius:8px;padding:3px 10px;font-size:11px;color:#4f46e5;max-width:200px}
.chat-attach-badge .fname{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chat-attach-badge .remove-attach{cursor:pointer;font-size:14px;color:#999;margin-left:2px;line-height:1}
.chat-attach-badge .remove-attach:hover{color:#ef4444}
.content{padding-bottom:80px!important}
/* Tags */
.tag-row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:24px}
.tag{padding:8px 18px;border-radius:20px;border:2px solid #e5e3de;font-size:13px;font-weight:500;cursor:pointer;transition:all .15s;background:#fff;user-select:none}
.tag:hover{border-color:#6366f1}.tag.selected{border-color:var(--c,#6366f1);background:var(--bg,#eef2ff);color:var(--fg,#4f46e5)}
/* Skills */
.skill-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px;margin-bottom:24px;max-height:220px;overflow-y:auto;padding:4px}
.skill-card{padding:10px 14px;border-radius:10px;border:2px solid #e5e3de;background:#fff;cursor:pointer;transition:all .15s;font-size:13px;position:relative}
.skill-card:hover{border-color:#6366f1;transform:translateY(-1px)}
.skill-card.selected{border-color:#6366f1;background:#eef2ff}
.skill-card.auto-loaded{border-color:#f59e0b;background:#fffbeb;animation:autoPulse 1.5s ease-in-out}
@keyframes autoPulse{0%{box-shadow:0 0 0 0 rgba(245,158,11,.4)}70%{box-shadow:0 0 0 8px rgba(245,158,11,0)}100%{box-shadow:none}}
.skill-card.unavailable{opacity:.45;cursor:default}
.skill-card .sn{font-weight:600;margin-bottom:2px}.skill-card .sd{font-size:11px;color:#888}
.skill-card .badge{position:absolute;top:6px;right:8px;font-size:10px;background:#10b981;color:#fff;padding:1px 6px;border-radius:8px}
/* Effort */
.effort-section{margin-bottom:24px}
.effort-slider{width:100%;-webkit-appearance:none;appearance:none;height:6px;border-radius:3px;background:linear-gradient(to right,#6366f1 0%,#6366f1 var(--val,66%),#e5e3de var(--val,66%),#e5e3de 100%);outline:none}
.effort-slider::-webkit-slider-thumb{-webkit-appearance:none;width:20px;height:20px;border-radius:50%;background:#6366f1;cursor:pointer;border:3px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.2)}
.effort-labels{display:flex;justify-content:space-between;font-size:12px;color:#999;margin-top:4px}
.effort-labels span.active{color:#6366f1;font-weight:600}
/* Style */
.style-row{display:flex;gap:16px;margin-bottom:24px}.style-group{flex:1}
.style-group label{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:#999;margin-bottom:8px;display:block}
.style-group textarea{width:100%;border:1px solid #e5e3de;border-radius:10px;padding:10px 12px;font-size:13px;resize:none;height:56px;font-family:inherit;outline:none}
.style-group textarea:focus{border-color:#6366f1}
.fmt-btns{display:flex;flex-wrap:wrap;gap:6px}
.fmt-btn{padding:6px 14px;border-radius:8px;border:2px solid #e5e3de;font-size:12px;cursor:pointer;transition:all .15s;background:#fff}
.fmt-btn:hover{border-color:#6366f1}.fmt-btn.selected{background:#6366f1;color:#fff;border-color:#6366f1}
/* Quick */
.quick-row{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:20px}
.quick-btn{padding:8px 16px;border-radius:20px;border:1px solid #e5e3de;background:#fff;font-size:13px;cursor:pointer;transition:all .15s}
.quick-btn:hover{border-color:#6366f1;background:#eef2ff}
/* Auto Skill */
.auto-skill-toggle{display:flex;align-items:center;gap:8px;margin-bottom:12px;padding:8px 14px;background:#f8f7ff;border:2px solid #e5e3de;border-radius:12px;cursor:pointer;transition:all .15s;user-select:none}
.auto-skill-toggle:hover{border-color:#6366f1}
.auto-skill-toggle.on{border-color:#6366f1;background:#eef2ff}
.auto-skill-toggle .toggle-dot{width:36px;height:20px;border-radius:10px;background:#ccc;position:relative;transition:all .2s}
.auto-skill-toggle.on .toggle-dot{background:#6366f1}
.auto-skill-toggle .toggle-dot::after{content:'';position:absolute;top:2px;left:2px;width:16px;height:16px;border-radius:50%;background:#fff;transition:all .2s}
.auto-skill-toggle.on .toggle-dot::after{left:18px}
.auto-skill-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;background:#eef2ff;color:#6366f1;border:1px solid #c7d2fe;margin:2px}
.auto-skill-preview{margin-top:6px;padding:8px 12px;background:#f0fdf4;border:1px solid #a7f3d0;border-radius:8px;font-size:12px;display:none}
.auto-skill-preview.show{display:block}
/* PPT Suggest Banner */
.ppt-suggest-banner{position:relative;margin:0 auto 8px;max-width:760px;padding:14px 18px 14px 50px;background:linear-gradient(135deg,#eef2ff 0%,#f5f3ff 50%,#fdf2f8 100%);border:1px solid #c7d2fe;border-radius:14px;box-shadow:0 2px 12px rgba(99,102,241,.12);overflow:hidden;animation:pptBannerSlide .5s cubic-bezier(.16,1,.3,1);cursor:default;transition:opacity .4s,transform .4s}
.ppt-suggest-banner.hiding{opacity:0;transform:translateY(-12px)}
.ppt-suggest-banner .ppt-banner-icon{position:absolute;left:14px;top:50%;transform:translateY(-50%);font-size:22px;animation:pptIconPulse 2s ease-in-out infinite}
.ppt-suggest-banner .ppt-banner-title{font-size:13px;font-weight:700;color:#4338ca;margin-bottom:4px}
.ppt-suggest-banner .ppt-banner-hint{font-size:12px;color:#6b7280;line-height:1.5}
.ppt-suggest-banner .ppt-banner-hint em{font-style:normal;color:#6366f1;font-weight:600;cursor:pointer;border-bottom:1px dashed #a5b4fc;transition:color .2s}
.ppt-suggest-banner .ppt-banner-hint em:hover{color:#4338ca}
.ppt-suggest-banner .ppt-banner-close{position:absolute;top:6px;right:10px;background:none;border:none;font-size:14px;color:#a5b4fc;cursor:pointer;padding:2px 6px;border-radius:4px;transition:all .2s}
.ppt-suggest-banner .ppt-banner-close:hover{background:#e0e7ff;color:#4338ca}
@keyframes pptBannerSlide{from{opacity:0;transform:translateY(-20px)}to{opacity:1;transform:translateY(0)}}
@keyframes pptIconPulse{0%,100%{transform:translateY(-50%) scale(1)}50%{transform:translateY(-50%) scale(1.15)}}
/* Messages */
.messages{margin-top:8px}
.msg{margin-bottom:8px;padding:8px 12px;border-radius:10px;line-height:1.5;font-size:13px;word-wrap:break-word}
.msg.user{background:#6366f1;color:#fff;margin-left:120px;border-bottom-right-radius:4px;white-space:pre-wrap}
.msg.assistant{background:#fff;border:1px solid #e5e3de;margin-right:120px;border-bottom-left-radius:4px}
.msg pre{background:#f5f5f0;padding:8px;border-radius:6px;overflow-x:auto;margin:4px 0;font-size:12px;white-space:pre-wrap}
.msg code{font-family:'SF Mono','Fira Code',monospace;font-size:12px}
.msg p{margin:0 0 4px 0}.msg p:last-child{margin-bottom:0}
.msg h1,.msg h2,.msg h3,.msg h4{margin:8px 0 4px 0;font-weight:700}
.msg h1{font-size:1.2em;border-bottom:1px solid #e5e3de;padding-bottom:2px}
.msg h2{font-size:1.1em;border-bottom:1px solid #eee;padding-bottom:2px}
.msg h3{font-size:1em}.msg h4{font-size:.95em}
.msg ul,.msg ol{margin:4px 0 4px 16px;padding:0}.msg li{margin:1px 0}
.msg table{border-collapse:collapse;margin:4px 0;width:100%;font-size:12px}
.msg th,.msg td{border:1px solid #ddd;padding:4px 8px;text-align:left}
.msg th{background:#f5f5f0;font-weight:600}
.msg tr:nth-child(even){background:#fafaf8}
.msg hr{border:none;border-top:1px solid #e5e3de;margin:6px 0}
.msg blockquote{border-left:3px solid #6366f1;margin:4px 0;padding:2px 8px;color:#666;background:#fafaf8;border-radius:0 6px 6px 0}
.msg-label{font-size:10px;font-weight:600;color:#999;margin-bottom:3px;text-transform:uppercase;letter-spacing:.5px}
.msg.user .msg-label{color:rgba(255,255,255,.7)}
.msg .skill-info{font-size:10px;color:#6366f1;margin-top:4px}
.typing{display:inline-flex;gap:4px;padding:8px 14px}
.typing span{width:8px;height:8px;border-radius:50%;background:#ccc;animation:blink 1.4s infinite both}
.typing span:nth-child(2){animation-delay:.2s}.typing span:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:.3}40%{opacity:1}}
/* Modal */
.modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:100;display:flex;align-items:center;justify-content:center}
.modal-bg.hidden{display:none}
.modal{background:#fff;border-radius:16px;padding:28px;width:500px;max-width:90vw;box-shadow:0 20px 60px rgba(0,0,0,.15)}
.modal h2{font-size:18px;margin-bottom:16px}
.modal label{font-size:13px;font-weight:600;color:#555;display:block;margin-bottom:4px;margin-top:12px}
.modal input{width:100%;padding:10px 12px;border:1px solid #e5e3de;border-radius:8px;font-size:14px;outline:none;font-family:inherit}
.modal input:focus{border-color:#6366f1}
.modal-btns{display:flex;justify-content:flex-end;gap:8px;margin-top:20px}
.modal-btns button{padding:10px 20px;border-radius:8px;border:none;font-size:14px;cursor:pointer}
.btn-ok{background:#6366f1;color:#fff}.btn-ok:hover{background:#4f46e5}
.btn-cancel{background:#f3f2ef;color:#555}
.settings-btn{background:none;border:none;cursor:pointer;font-size:20px;color:#999;padding:4px 8px;border-radius:6px}
.settings-btn:hover{background:#f3f2ef;color:#555}
.status{font-size:12px}
.status.on{color:#10b981}.status.off{color:#999}
/* Env selector */
.env-row{display:flex;gap:8px;margin-bottom:24px}
.env-btn{flex:1;padding:12px 10px;border-radius:12px;border:2px solid #e5e3de;background:#fff;cursor:pointer;text-align:center;transition:all .15s;font-size:13px}
.env-btn:hover{border-color:#6366f1;transform:translateY(-1px)}
.env-btn.selected{border-color:#6366f1;background:#eef2ff}
.env-btn .env-name{font-weight:700;font-size:14px;margin-bottom:2px}
.env-btn .env-model{font-size:11px;color:#888}
.env-btn.selected .env-model{color:#6366f1}
.token-status{font-size:12px;padding:8px 12px;border-radius:8px;margin-bottom:16px}
.token-status.ok{background:#ecfdf5;color:#059669}
.token-status.missing{background:#fef2f2;color:#dc2626}
/* System Prompt */
.sysprompt-section{margin-bottom:24px}
.sysprompt-toggle{display:flex;align-items:center;gap:8px;cursor:pointer;margin-bottom:8px}
.sysprompt-toggle .arrow{font-size:12px;transition:transform .2s;color:#999}
.sysprompt-toggle .arrow.open{transform:rotate(90deg)}
.sysprompt-body{display:none}
.sysprompt-body.show{display:block}
.sysprompt-textarea{width:100%;border:1px solid #e5e3de;border-radius:10px;padding:12px;font-size:13px;resize:vertical;min-height:80px;max-height:300px;font-family:'SF Mono','Fira Code',monospace;line-height:1.5;outline:none;background:#fafaf8}
.sysprompt-textarea:focus{border-color:#6366f1;background:#fff}
.sysprompt-info{font-size:11px;color:#999;margin-top:4px}
.prompt-chip{display:inline-flex;align-items:center;gap:3px;padding:3px 10px;border-radius:12px;font-size:11px;cursor:pointer;border:1px solid #e5e3de;background:#fafaf8;transition:all .15s;white-space:nowrap}
.prompt-chip:hover{background:#eef2ff;border-color:#6366f1}
.prompt-chip.active{background:#6366f1;color:#fff;border-color:#6366f1}
.prompt-chip.saved{border-color:#f59e0b;background:#fffbeb}
.prompt-chip.saved:hover{background:#fef3c7}
.prompt-chip.saved.active{background:#f59e0b;color:#fff}
.sysprompt-preview{font-size:11px;color:#6366f1;background:#eef2ff;padding:6px 10px;border-radius:6px;margin-top:6px;max-height:60px;overflow:hidden;cursor:pointer}
.sysprompt-preview:hover{max-height:200px;overflow-y:auto}
/* PPT 생성 블록 */
.pptx-gen-block{background:#f8f7ff;border:2px solid #a5b4fc;border-radius:12px;margin:12px 0;overflow:hidden}
.pptx-gen-header{display:flex;justify-content:space-between;align-items:center;padding:8px 14px;background:#eef2ff;border-bottom:1px solid #a5b4fc}
.pptx-gen-header span{font-weight:600;font-size:13px;color:#4338ca}
.pptx-gen-actions{display:flex;gap:6px;align-items:center}
.pptx-gen-btn{background:#6366f1;color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:12px;cursor:pointer;transition:all .2s;font-weight:600}
.pptx-gen-btn:hover{background:#4f46e5;transform:translateY(-1px)}
.pptx-gen-btn:disabled{background:#d1d5db;cursor:not-allowed;transform:none}
.pptx-gen-btn.secondary{background:#e0e7ff;color:#4338ca}
.pptx-gen-btn.secondary:hover{background:#c7d2fe}
.pptx-gen-status{font-size:11px;color:#6b7280}
.pptx-remake-bar{display:flex;align-items:center;gap:6px;padding:8px 12px;background:#f8fafc;border-top:1px solid #e5e7eb;border-radius:0 0 10px 10px;flex-wrap:wrap}
.pptx-remake-label{font-size:11px;color:#6b7280;font-weight:600;white-space:nowrap}
.pptx-remake-btn{background:#fff;color:#4338ca;border:1px solid #c7d2fe;border-radius:16px;padding:4px 12px;font-size:11px;cursor:pointer;transition:all .2s;font-weight:500;white-space:nowrap}
.pptx-remake-btn:hover{background:#eef2ff;border-color:#818cf8;transform:translateY(-1px)}
/* CSV Upload */
.csv-section{margin-bottom:24px}
/* csv-upload-area 제거됨 → 채팅 📎 첨부로 대체 */
.csv-upload-area .icon{font-size:28px;margin-bottom:6px}
.csv-upload-area .label{font-size:13px;color:#888}
.csv-upload-area .sub{font-size:11px;color:#bbb;margin-top:2px}
.csv-info{background:#fff;border:2px solid #10b981;border-radius:12px;padding:14px 18px;position:relative}
.csv-info .fname{font-weight:700;font-size:14px;color:#059669}
.csv-info .fstats{font-size:12px;color:#666;margin-top:4px}
.csv-info .fremove{position:absolute;top:10px;right:12px;background:#fee2e2;color:#dc2626;border:none;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer}
.csv-info .fremove:hover{background:#fca5a5}
.csv-preview{margin-top:8px;max-height:160px;overflow:auto;font-size:11px;background:#f9fafb;border-radius:8px;padding:8px}
.csv-preview table{border-collapse:collapse;width:100%}
.csv-preview th,.csv-preview td{border:1px solid #e5e7eb;padding:3px 8px;text-align:left;white-space:nowrap}
.csv-preview th{background:#f3f4f6;font-weight:600;position:sticky;top:0}
/* Skill Detail Panel */
.skill-detail-overlay{position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:200;display:flex;align-items:center;justify-content:center}
.skill-detail-panel{background:#fff;border-radius:16px;width:720px;max-width:92vw;max-height:85vh;display:flex;flex-direction:column;box-shadow:0 20px 60px rgba(0,0,0,.2)}
.sdp-header{padding:18px 24px;border-bottom:1px solid #e5e3de;display:flex;justify-content:space-between;align-items:center}
.sdp-header h2{font-size:18px;margin:0}.sdp-close{background:none;border:none;font-size:22px;cursor:pointer;color:#999;padding:4px 8px}
.sdp-close:hover{color:#333}
.sdp-tabs{display:flex;border-bottom:1px solid #e5e3de;padding:0 24px}
.sdp-tab{padding:10px 18px;font-size:13px;font-weight:600;cursor:pointer;border-bottom:3px solid transparent;color:#888;transition:all .15s}
.sdp-tab:hover{color:#333}.sdp-tab.active{color:#6366f1;border-bottom-color:#6366f1}
.sdp-body{flex:1;overflow-y:auto;padding:18px 24px}
.sdp-file-list{list-style:none;padding:0}
.sdp-file-item{display:flex;align-items:center;gap:10px;padding:10px 14px;border:1px solid #e5e3de;border-radius:10px;margin-bottom:6px;transition:all .15s;cursor:pointer}
.sdp-file-item:hover{border-color:#6366f1;background:#f8f7ff}
.sdp-file-icon{font-size:18px}
.sdp-file-name{font-weight:600;font-size:13px;flex:1}
.sdp-file-size{font-size:11px;color:#999}
.sdp-file-actions{display:flex;gap:4px}
.sdp-btn{padding:4px 12px;border-radius:6px;border:1px solid #e5e3de;font-size:11px;cursor:pointer;background:#fff;transition:all .15s}
.sdp-btn:hover{border-color:#6366f1;background:#eef2ff}
.sdp-btn.run{background:#10b981;color:#fff;border-color:#10b981}.sdp-btn.run:hover{background:#059669}
.sdp-btn.inject{background:#6366f1;color:#fff;border-color:#6366f1}.sdp-btn.inject:hover{background:#4f46e5}
.sdp-code{background:#1e1e1e;color:#d4d4d4;padding:14px;border-radius:10px;font-family:'SF Mono','Fira Code',monospace;font-size:12px;line-height:1.5;overflow:auto;max-height:400px;white-space:pre-wrap;word-break:break-all}
.sdp-result{margin-top:12px;padding:12px;border-radius:10px;font-family:monospace;font-size:12px;line-height:1.4;overflow:auto;max-height:300px;white-space:pre-wrap}
.sdp-result.ok{background:#ecfdf5;border:1px solid #a7f3d0}.sdp-result.err{background:#fef2f2;border:1px solid #fca5a5}
.skill-card .extras{font-size:10px;color:#6366f1;margin-top:3px}
/* Copy button */
.copy-btn{position:absolute;top:4px;right:4px;background:#e5e3de;border:none;border-radius:4px;padding:2px 8px;font-size:11px;cursor:pointer;opacity:.7}
.copy-btn:hover{opacity:1}
.drawio-block{border:2px solid #6366f1;border-radius:10px;overflow:hidden;margin:10px 0;background:#fafaff}
.drawio-header{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:#eef2ff;border-bottom:1px solid #d4d8f0;flex-wrap:wrap;gap:6px}
.drawio-header span{font-weight:600;font-size:13px;color:#4338ca}
.drawio-actions{display:flex;gap:6px;flex-wrap:wrap}
.drawio-btn{background:#6366f1;color:#fff;border:none;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;transition:background .2s}
.drawio-btn:hover{background:#4f46e5}
.chart-block{border:2px solid #f59e0b;border-radius:10px;overflow:hidden;margin:10px 0;background:#fffbf0}
.chart-header{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:#fef3c7;border-bottom:1px solid #fcd34d;flex-wrap:wrap;gap:6px}
.chart-header span{font-weight:600;font-size:13px;color:#92400e}
.chart-actions{display:flex;gap:6px;flex-wrap:wrap}
.chart-btn{background:#f59e0b;color:#fff;border:none;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;transition:background .2s}
.chart-btn:hover{background:#d97706}
.chart-canvas-wrap{padding:12px;background:#fff;display:flex;justify-content:center;align-items:center;min-height:250px;max-height:450px}
.chart-canvas-wrap canvas{max-width:100%;max-height:420px}
.chart-raw-toggle{padding:0 12px 8px}
.chart-raw-toggle summary{font-size:11px;color:#999;cursor:pointer}
.chart-raw-toggle pre{font-size:10px;max-height:120px;overflow:auto;background:#f5f5f0;padding:8px;border-radius:6px;margin-top:4px}
.md-report-block{background:#f8faf8;border:2px solid #86efac;border-radius:12px;margin:12px 0;overflow:hidden}
.md-report-header{display:flex;justify-content:space-between;align-items:center;padding:8px 14px;background:#ecfdf5;border-bottom:1px solid #86efac}
.md-report-header span{font-weight:600;font-size:13px;color:#166534}
.md-report-actions{display:flex;gap:6px}
.md-report-btn{background:#22c55e;color:#fff;border:none;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;transition:background .2s}
.md-report-btn:hover{background:#16a34a}
.md-report-btn.md-download{background:#6366f1}.md-report-btn.md-download:hover{background:#4f46e5}
.md-report-preview{padding:14px 18px;font-size:13px;line-height:1.7;max-height:500px;overflow-y:auto}
.md-report-preview h1{font-size:20px;margin:12px 0 8px;border-bottom:2px solid #e5e7eb;padding-bottom:4px}
.md-report-preview h2{font-size:17px;margin:10px 0 6px;color:#374151}
.md-report-preview h3{font-size:15px;margin:8px 0 4px;color:#4b5563}
.md-report-preview table{border-collapse:collapse;width:100%;margin:8px 0}
.md-report-preview th,.md-report-preview td{border:1px solid #d1d5db;padding:4px 8px;font-size:12px;text-align:left}
.md-report-preview th{background:#f3f4f6;font-weight:600}
.md-report-preview pre{background:#1e1e2e;color:#cdd6f4;padding:10px;border-radius:8px;overflow-x:auto;font-size:12px}
.md-report-preview blockquote{border-left:3px solid #6366f1;padding-left:12px;color:#6b7280;margin:8px 0}
.md-raw-toggle{margin:0;padding:4px 14px 8px}
.md-raw-toggle summary{font-size:11px;color:#9ca3af;cursor:pointer}
.md-raw-toggle pre{max-height:300px;overflow:auto;font-size:11px;margin-top:6px}
.drawio-preview{max-height:200px;overflow:auto;margin:0;border-radius:0;border:none;font-size:11px}
.drawio-preview code{font-size:11px}
.send-btn.stop-mode{background:#e74c3c;animation:pulse-stop 1.2s infinite}
@keyframes pulse-stop{0%,100%{opacity:1}50%{opacity:.7}}
.sidebar-file-panel{margin-top:auto;padding:8px 8px 0;border-top:1px solid #e5e3de;max-height:220px;overflow-y:auto}
.uploaded-files-header{font-size:11px;font-weight:600;color:#555;padding:4px 0;border-bottom:1px solid #eee;margin-bottom:2px;display:flex;align-items:center;justify-content:space-between}
.uploaded-file-item{display:flex;align-items:center;gap:4px;padding:3px 2px;border-bottom:1px solid #f0f0f0;font-size:11px}
.ufi-icon{font-size:14px;flex-shrink:0}
.ufi-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#333;cursor:help}
.ufi-size{color:#999;font-size:10px;flex-shrink:0}
.ufi-remove{background:none;border:none;color:#ccc;cursor:pointer;font-size:12px;padding:0 2px}
.ufi-remove:hover{color:#e74c3c}
.ufi-pending{opacity:.6}
.ufi-drm{opacity:.7;border-left:2px solid #e67e22;padding-left:4px}
.ufi-drm .ufi-name{color:#e67e22}
.think-box{margin:8px 0 12px;border:1px solid #d4c8f0;border-radius:8px;background:#f8f5ff;overflow:hidden}
.think-box summary{cursor:pointer;padding:8px 12px;font-size:12px;font-weight:600;color:#7c5cbf;background:#f0ebfa;user-select:none}
.think-box summary:hover{background:#e8e0f6}
.think-box[open] summary{border-bottom:1px solid #d4c8f0}
.think-content{padding:10px 14px;font-size:12px;color:#555;line-height:1.6;max-height:400px;overflow-y:auto}
pre{position:relative}
/* ===== 오른쪽 코드 어시스턴트 패널 ===== */
.right-panel{width:340px;background:#fff;border-left:1px solid #e5e3de;display:flex;flex-direction:column;overflow:hidden;transition:width .2s ease;flex-shrink:0}
.right-panel.collapsed{width:0;border-left:none;overflow:hidden}
.right-panel-toggle{position:fixed;top:12px;right:340px;width:28px;height:28px;border-radius:6px 0 0 6px;border:1px solid #e5e3de;border-right:none;background:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:12px;color:#999;z-index:200;transition:right .2s ease}
.right-panel-toggle:hover{background:#eef2ff;color:#6366f1}
body.rp-collapsed .right-panel-toggle{right:0}
body.rp-collapsed .chat-box-fixed{right:0}
.rp-header{padding:14px 16px;border-bottom:1px solid #e5e3de;display:flex;align-items:center;justify-content:space-between}
.rp-header h3{margin:0;font-size:15px;font-weight:700;color:#1a1a1a}
.rp-body{flex:1;overflow-y:auto;padding:12px 14px}
.rp-section{margin-bottom:16px}
.rp-section-title{font-size:11px;font-weight:600;color:#999;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}
.rp-file-drop{border:2px dashed #d1d5db;border-radius:10px;padding:24px 16px;text-align:center;cursor:pointer;transition:all .15s;background:#fafaf9;margin-bottom:12px}
.rp-file-drop:hover,.rp-file-drop.dragover{border-color:#6366f1;background:#eef2ff}
.rp-file-drop-icon{font-size:28px;margin-bottom:6px}
.rp-file-drop-text{font-size:12px;color:#888}
.rp-file-list{max-height:120px;overflow-y:auto;margin-bottom:8px}
.rp-file-item{display:flex;align-items:center;gap:6px;padding:5px 8px;background:#f8f7f4;border-radius:6px;margin-bottom:4px;font-size:12px}
.rp-file-item .fname{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#333}
.rp-file-item .fsize{color:#aaa;font-size:11px;flex-shrink:0}
.rp-file-item .fremove{background:none;border:none;cursor:pointer;color:#ef4444;font-size:12px;padding:0 2px}
.rp-action-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.rp-action-btn{padding:10px 8px;border-radius:8px;border:1px solid #e5e3de;background:#fff;cursor:pointer;text-align:center;transition:all .15s;font-size:12px}
.rp-action-btn:hover{border-color:#6366f1;background:#eef2ff}
.rp-action-btn.active{border-color:#6366f1;background:#6366f1;color:#fff}
.rp-action-btn .rp-btn-icon{font-size:18px;display:block;margin-bottom:3px}
.rp-action-btn .rp-btn-label{font-weight:600}
.rp-run-btn{width:100%;padding:10px;border-radius:8px;border:none;background:#6366f1;color:#fff;font-size:14px;font-weight:700;cursor:pointer;margin-top:10px;transition:background .15s}
.rp-run-btn:hover{background:#4f46e5}
.rp-run-btn:disabled{background:#d1d5db;cursor:not-allowed}
.rp-result{margin-top:12px;padding:16px;background:#1e1e2e;color:#cdd6f4;border-radius:10px;font-size:13px;line-height:1.7;overflow-y:auto;display:none}
.rp-result.expanded{max-height:none;position:fixed;top:8px;right:8px;bottom:8px;width:calc(var(--rp-width, 340px) - 16px);z-index:1100;box-shadow:0 8px 32px rgba(0,0,0,.4)}
.rp-result h1,.rp-result h2,.rp-result h3{color:#cba6f7;margin:12px 0 6px}
.rp-result h1{font-size:16px} .rp-result h2{font-size:14px} .rp-result h3{font-size:13px}
.rp-result pre{background:#181825;border-radius:6px;padding:10px;overflow-x:auto;margin:8px 0}
.rp-result code{font-family:'Fira Code',monospace;font-size:12px}
.rp-result p{margin:6px 0}
.rp-result ul,.rp-result ol{padding-left:20px;margin:6px 0}
.rp-result table{border-collapse:collapse;width:100%;margin:8px 0;font-size:12px}
.rp-result th,.rp-result td{border:1px solid #45475a;padding:6px 8px;text-align:left}
.rp-result th{background:#313244;color:#cba6f7}
.rp-result-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #45475a}
.rp-result-header .rp-result-title{font-weight:700;color:#a6e3a1;font-size:14px}
.rp-result-header button{background:none;border:1px solid #45475a;color:#cdd6f4;border-radius:6px;padding:4px 10px;cursor:pointer;font-size:12px}
.rp-result-header button:hover{background:#313244}
.rp-skill-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;background:#eef2ff;color:#6366f1;margin-left:6px}
.rp-tree{font-size:12px;max-height:250px;overflow-y:auto;border:1px solid #e5e3de;border-radius:8px;padding:8px;background:#fafaf9;margin-bottom:8px}
.rp-tree-node{padding:2px 0;cursor:pointer;user-select:none;display:flex;align-items:center;gap:2px}
.rp-tree-cb{width:14px;height:14px;cursor:pointer;accent-color:#6366f1;flex-shrink:0}
.rp-select-bar{display:flex;gap:6px;margin-bottom:6px;align-items:center}
.rp-select-btn{background:none;border:1px solid #d1d5db;border-radius:4px;padding:2px 8px;font-size:10px;cursor:pointer;color:#666}
.rp-select-btn:hover{border-color:#6366f1;color:#6366f1}
.rp-checked-count{font-size:11px;color:#6366f1;font-weight:600}
.rp-tree-node:hover{background:#eef2ff;border-radius:4px}
.rp-tree-node-content{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rp-tree-folder{font-weight:600;color:#555}
.rp-tree-file{color:#333}
.rp-tree-file.active{background:#6366f1;color:#fff;border-radius:4px;padding:1px 4px}
.rp-tree-children{padding-left:16px;border-left:1px solid #e5e3de;margin-left:6px}
.rp-tree-toggle{display:inline-block;width:14px;font-size:10px;color:#999;text-align:center}
.rp-tree-icon{margin-right:4px}
.rp-tree-stats{font-size:11px;color:#888;padding:6px 8px;background:#f0f0ed;border-radius:6px;margin-bottom:8px;display:flex;justify-content:space-between}
.rp-tab-row{display:flex;gap:4px;margin-bottom:8px}
.rp-tab{padding:4px 10px;border-radius:6px;border:1px solid #e5e3de;background:#fff;cursor:pointer;font-size:11px;font-weight:600;color:#888;transition:all .15s}
.rp-tab:hover{border-color:#6366f1;color:#6366f1}
.rp-tab.active{background:#6366f1;color:#fff;border-color:#6366f1}
@media(max-width:768px){.sidebar{display:none}.sidebar-toggle{display:none}.right-panel{display:none}.right-panel-toggle{display:none}.chat-box-fixed{left:0;right:0}.style-row{flex-direction:column}.msg.user{margin-left:16px}.msg.assistant{margin-right:16px}.msg{font-size:12px}}
/* 인트로 비디오 오버레이 */
#introOverlay{position:fixed;top:0;left:0;width:100%;height:100%;z-index:9999;background:#000;display:flex;align-items:center;justify-content:center;flex-direction:column;transition:opacity .6s ease}
#introOverlay.fade-out{opacity:0;pointer-events:none}
#introVideo{width:100%;height:100%;object-fit:cover}
#introSkip{position:absolute;bottom:40px;right:40px;background:rgba(255,255,255,.15);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.3);color:#fff;padding:10px 24px;border-radius:24px;font-size:14px;cursor:pointer;transition:all .2s;z-index:10000}
#introSkip:hover{background:rgba(255,255,255,.3)}
#introProgress{position:absolute;bottom:0;left:0;height:3px;background:linear-gradient(90deg,#6366f1,#a855f7);width:0;transition:width .1s linear}
/* 모드 선택 오버레이 */
#modeSelector{position:fixed;top:0;left:0;width:100%;height:100%;z-index:9998;background:radial-gradient(circle at 30% 40%,#1a1a2e,#0f0f1a 70%);display:none;align-items:center;justify-content:center;flex-direction:column;opacity:0;transition:opacity .5s ease}
#modeSelector.visible{display:flex;opacity:1}
#modeSelector h2{color:#fff;font-size:28px;margin-bottom:12px;font-weight:700;letter-spacing:-0.5px}
#modeSelector p{color:rgba(255,255,255,.6);font-size:14px;margin-bottom:40px}
.mode-cards{display:flex;gap:28px;flex-wrap:wrap;justify-content:center}
.mode-card{width:280px;padding:36px 28px;border-radius:16px;border:2px solid rgba(255,255,255,.12);background:rgba(255,255,255,.06);backdrop-filter:blur(12px);cursor:pointer;text-align:center;transition:all .25s ease;position:relative;overflow:hidden}
.mode-card:hover{border-color:#6366f1;background:rgba(99,102,241,.12);transform:translateY(-4px);box-shadow:0 12px 40px rgba(99,102,241,.2)}
.mode-card .mode-icon{font-size:48px;margin-bottom:16px;display:block}
.mode-card .mode-title{color:#fff;font-size:18px;font-weight:700;margin-bottom:8px}
.mode-card .mode-desc{color:rgba(255,255,255,.55);font-size:13px;line-height:1.5}
/* UIO 컨테이너 */
#uioContainer{position:fixed;top:0;left:0;width:100%;height:100%;z-index:9000;display:none;background:#0f151e}
#uioContainer iframe{width:100%;height:100%;border:none}
#uioBackBtn{position:fixed;top:16px;left:16px;z-index:9001;padding:10px 20px;border-radius:10px;border:2px solid rgba(255,255,255,.25);background:rgba(18,27,40,.85);backdrop-filter:blur(8px);color:#fff;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s}
#uioBackBtn:hover{background:rgba(99,102,241,.3);border-color:#6366f1}
</style>
</head>
<body>
<!-- 인트로 비디오 오버레이 -->
<div id="introOverlay">
  <video id="introVideo" autoplay muted playsinline>
    <source src="/static/intro.mp4" type="video/mp4">
  </video>
  <div id="introProgress"></div>
  <button id="introSkip" onclick="dismissIntroAndShowMode()">건너뛰기 ▶</button>
</div>

<!-- 모드 선택 오버레이 -->
<div id="modeSelector">
  <h2>모드를 선택하세요</h2>
  <p>원하는 인터페이스를 선택하여 시작합니다</p>
  <div class="mode-cards">
    <div class="mode-card" onclick="selectMode('default')">
      <span class="mode-icon">💬</span>
      <div class="mode-title">기본 구성</div>
      <div class="mode-desc">Demos LLM 채팅 인터페이스<br>스킬 기반 AI 어시스턴트</div>
    </div>
    <div class="mode-card" onclick="selectMode('uio')">
      <span class="mode-icon">🎮</span>
      <div class="mode-title">UIO 2D 픽셀</div>
      <div class="mode-desc">2D Pixel Office Live<br>픽셀 아트 사무실 시뮬레이션</div>
    </div>
  </div>
</div>

<!-- UIO 2D 픽셀 컨테이너 -->
<div id="uioContainer">
  <iframe id="uioFrame" src="about:blank"></iframe>
  <button id="uioBackBtn" onclick="exitUioMode()">← 메인으로 돌아가기</button>
</div>

<div class="sidebar" id="sidebar">
  <button class="sidebar-toggle-mini" onclick="toggleSidebar()" title="펼치기">☰</button>
  <div class="sidebar-inner">
    <div class="sidebar-logo"><span>D</span>emos <span style="font-size:12px;color:#999">Alpha 0.8</span></div>
    <button class="sidebar-btn active" onclick="createNewSession()">✨ 새 세션</button>
    <div class="sidebar-section">세션 목록 <span class="skill-count" id="sessionCount">0</span></div>
    <div id="sessionList" style="max-height:200px;overflow-y:auto;margin-bottom:4px;"></div>
    <div class="session-actions" id="sessionActions" style="display:none">
      <button onclick="archiveSelectedSessions()">📦 보관</button>
      <button class="danger" onclick="deleteSelectedSessions()">🗑️ 삭제</button>
      <button onclick="clearSessionSelection()">취소</button>
    </div>
    <div class="sidebar-section">불러온 스킬 <span class="skill-count" id="loadedCount">0</span></div>
    <div id="loadedSkillsList" style="font-size:12px;color:#666;padding:0 8px;"></div>
    <div class="sidebar-section" style="margin-top:16px;">안내</div>
    <div style="font-size:12px;color:#888;padding:0 8px;line-height:1.5;">
      1. ⚙️ API 설정<br>
      2. 분야/스킬 선택<br>
      3. 질문 입력 후 전송<br><br>
      <strong>SKILL.md</strong> 파일이 있는 스킬만 ✅ 표시됩니다.
    </div>
  </div>
  <div id="fileListPanel" class="sidebar-file-panel" style="display:none"></div>
  <div class="sidebar-footer">
    <div class="credits">🔬 Demos(민중) Alpha 0.8</div>
  </div>
  </div>
</div>
<button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()">◀</button>

<div class="main">
  <div class="header">
    <div class="project-title">📁 Demos(민중) 프로젝트 <span style="font-size:12px;color:#6366f1;background:#eef2ff;padding:2px 10px;border-radius:10px;margin-left:8px;font-weight:500;">Opus SKILL 4.6 사용중</span></div>
    <div style="display:flex;align-items:center;gap:8px;">
      <span id="tokenBadge" class="status off">⏳ 로딩중...</span>
      <span id="status" class="status off">⚪ 환경 미선택</span>
    </div>
  </div>
  <div class="content">
    <div class="content-inner">
      <!-- 환경 선택 -->
      <div class="section-label">LLM 환경 선택</div>
      <div id="tokenStatus" class="token-status missing">⏳ 토큰 상태 확인중...</div>
      <div class="env-row" id="envRow">
        <!-- JS에서 동적 생성 -->
      </div>

      <!-- 업로드된 파일 목록은 사이드바 하단으로 이동 -->
      <div id="csvInfoPanel" style="display:none"></div>

      <div class="section-label">분야 선택</div>
      <div class="tag-row" id="tagRow"></div>

      <div class="section-label">스킬 선택 <span style="font-weight:400;color:#bbb">( ✅ = SKILL.md 로드됨 )</span></div>
      <div class="skill-grid" id="skillGrid"></div>

      <div class="sysprompt-section">
        <div class="sysprompt-toggle" onclick="toggleSysPrompt()">
          <span class="arrow" id="spArrow">▶</span>
          <div class="section-label" style="margin-bottom:0;cursor:pointer">시스템 프롬프트</div>
          <span id="spPreviewBadge" style="font-size:11px;color:#6366f1;background:#eef2ff;padding:2px 8px;border-radius:10px">기본값</span>
        </div>
        <div class="sysprompt-body" id="spBody">
          <!-- 프리셋/저장 프롬프트 선택 바 -->
          <div id="promptPresets" style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px"></div>
          <textarea class="sysprompt-textarea" id="systemPromptInput" placeholder="시스템 프롬프트를 입력하세요. 비워두면 기본값 사용됩니다.&#10;&#10;아래 프리셋 버튼을 클릭하면 자동으로 채워집니다.&#10;수정 후 '저장' 버튼으로 나만의 프롬프트를 저장하세요."></textarea>
          <div class="sysprompt-info" style="display:flex;align-items:center;flex-wrap:wrap;gap:4px">
            <span>💡 스킬 프롬프트 <strong>앞에</strong> 추가됩니다.</span>
            <button style="background:#6366f1;color:#fff;border:none;border-radius:4px;padding:3px 10px;font-size:11px;cursor:pointer" onclick="saveCurrentPrompt()">💾 저장</button>
            <button style="background:#ef4444;color:#fff;border:none;border-radius:4px;padding:3px 10px;font-size:11px;cursor:pointer" onclick="deleteCurrentPrompt()" id="btnDeletePrompt" title="현재 선택된 저장 프롬프트 삭제">🗑️ 삭제</button>
            <button style="background:#f3f2ef;border:1px solid #e5e3de;border-radius:4px;padding:3px 8px;font-size:11px;cursor:pointer" onclick="resetSysPrompt()">초기화</button>
            <span id="promptSaveStatus" style="font-size:11px;color:#16a34a;display:none">✅ 저장됨!</span>
          </div>
        </div>
      </div>

      <div class="effort-section">
        <div class="section-label">응답 수준</div>
        <input type="range" class="effort-slider" id="effortSlider" min="0" max="3" value="2" oninput="updateEffort()">
        <div class="effort-labels">
          <span id="e0">즉시</span><span id="e1">빠름</span><span id="e2" class="active">표준</span><span id="e3">프로</span>
        </div>
      </div>

      <!-- 작성스타일/출력형식: 채팅창 드롭다운으로 이동, 내부 상태 유지용 hidden -->
      <div style="display:none">
        <div id="styleChips"></div>
        <textarea id="writingStyle"></textarea>
        <div class="fmt-btns">
          <div class="fmt-btn selected" data-f="auto" onclick="selFmt(this)">🤖 자동</div>
          <div class="fmt-btn" data-f="code" onclick="selFmt(this)">코드</div>
          <div class="fmt-btn" data-f="code-fix" onclick="selFmt(this)">코드수정</div>
          <div class="fmt-btn" data-f="analysis" onclick="selFmt(this)">분석</div>
          <div class="fmt-btn" data-f="report" onclick="selFmt(this)">보고서</div>
          <div class="fmt-btn" data-f="step-by-step" onclick="selFmt(this)">단계별</div>
        </div>
      </div>

      <!-- 자동 스킬 선택 -->
      <div class="auto-skill-toggle" id="autoSkillToggle" onclick="toggleAutoSkill()">
        <div class="toggle-dot"></div>
        <div>
          <div style="font-weight:700;font-size:13px">🧠 자동 스킬 선택</div>
          <div style="font-size:11px;color:#888">질문을 분석하여 관련 스킬을 자동으로 로드합니다</div>
        </div>
      </div>
      <div class="auto-skill-preview" id="autoSkillPreview"></div>

      <div class="quick-row">
        <button class="quick-btn" onclick="qp('이 데이터를 분석해줘')">📊 데이터 분석</button>
        <button class="quick-btn" onclick="qp('업로드된 CSV 데이터의 통계 요약을 보여줘')">📋 CSV 요약</button>
        <button class="quick-btn" onclick="qp('이 데이터로 시각화 코드를 작성해줘')">📈 시각화</button>
        <button class="quick-btn" onclick="qp('Python 코드를 작성해줘')">🐍 코드 작성</button>
        <button class="quick-btn" onclick="qp('이 데이터에서 이상치를 찾아줘')">🔍 이상치 탐색</button>
        <button class="quick-btn" onclick="qp('논문 스타일로 정리해줘')">📝 논문 정리</button>
      </div>

      <div id="pptSuggestArea"></div>
      <div class="messages" id="msgs"></div>
    </div>
  </div>

  <!-- 질문 입력 - 하단 고정 -->
  <div class="chat-box-fixed">
    <div class="chat-box-fixed-inner">
      <div class="chat-attach-preview" id="chatAttachPreview" style="display:none"></div>
      <textarea class="chat-input" id="input" placeholder="질문을 하거나 수행하려는 분석을 설명하세요..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();handleSendStop()}" oninput="this.style.height='auto';this.style.height=this.scrollHeight+'px'"></textarea>
      <input type="file" id="chatFileInput" multiple style="display:none" onchange="handleChatFileSelect(this.files)">
      <div class="chat-footer">
        <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
          <button class="attach-btn" onclick="document.getElementById('chatFileInput').click()" title="파일 첨부">📎</button>
          <select class="chat-dropdown" id="chatStyleDrop" onchange="applyChatStyle(this.value)" title="작성 스타일">
            <option value="">🤖 자동</option>
            <option value="간결하고 핵심만. 불필요한 설명 생략.">⚡ 간결</option>
            <option value="상세하고 친절하게. 원리, 배경, 예시 포함.">📖 상세</option>
            <option value="바로 복붙해서 쓸 수 있게. 실전 위주, 이론 최소화.">🔨 실용적</option>
            <option value="학술적 톤. 정확한 용어, 인용, 근거 제시.">🎓 학술</option>
            <option value="코드 주석을 상세한 한국어로. 변수명은 영어 유지.">💬 한글주석</option>
            <option value="시니어 개발자 관점. 왜 이렇게 하는지, 대안은 뭔지, 주의점 포함.">👨‍💻 시니어</option>
            <option value="에러 원인 분석 중심. traceback 해석, 재현 조건, 해결책 순서.">🐛 디버그</option>
            <option value="데이터 스토리텔링. 숫자→의미→액션 순서로 해석.">📊 데이터</option>
          </select>
          <select class="chat-dropdown" id="chatFormatDrop" onchange="applyChatFormat(this.value)" title="출력 형식">
            <option value="auto">🤖 자동</option>
            <option value="code">💻 코드</option>
            <option value="code-fix">🔧 수정</option>
            <option value="analysis">📊 분석</option>
            <option value="report">📄 보고서</option>
            <option value="step-by-step">📝 단계별</option>
          </select>
          <label style="display:flex;align-items:center;gap:3px;font-size:11px;color:#888;cursor:pointer;user-select:none" title="체크하면 모델이 답변 전에 깊이 사고합니다 (응답 느림)">
            <input type="checkbox" id="thinkToggle" style="margin:0;accent-color:#7c5cbf">💭사고
          </label>
          <span style="font-size:11px;color:#bbb">Enter 전송</span>
        </div>
        <button class="send-btn" onclick="handleSendStop()" id="sendBtn">▶</button>
      </div>
    </div>
  </div>
</div>

<!-- 오른쪽 코드 어시스턴트 패널 -->
<div class="right-panel" id="rightPanel">
  <div class="rp-header">
    <h3>🖥️ Code Assistant<span class="rp-skill-badge">code-assistant</span></h3>
  </div>
  <div class="rp-body">
    <div class="rp-section">
      <div class="rp-section-title">프로젝트 / 코드 파일</div>
      <div class="rp-tab-row">
        <button class="rp-tab active" onclick="switchUploadMode('folder',this)">📁 폴더 업로드</button>
        <button class="rp-tab" onclick="switchUploadMode('files',this)">📄 파일 업로드</button>
      </div>
      <div id="rpFolderMode">
        <div class="rp-file-drop" id="rpFolderDrop" onclick="document.getElementById('rpFolderInput').click()">
          <div class="rp-file-drop-icon">📁</div>
          <div class="rp-file-drop-text">클릭하여 <strong>프로젝트 폴더</strong> 통째로 업로드<br><span style="font-size:10px;color:#bbb">폴더 선택 시 하위 파일 전체가 트리로 표시됩니다</span></div>
        </div>
        <input type="file" id="rpFolderInput" webkitdirectory directory multiple style="display:none" onchange="handleRpFolderSelect(this.files)">
      </div>
      <div id="rpFileMode" style="display:none">
        <div class="rp-file-drop" id="rpFileDrop" onclick="document.getElementById('rpFileInput').click()">
          <div class="rp-file-drop-icon">📄</div>
          <div class="rp-file-drop-text">클릭 또는 드래그하여 코드 파일 업로드<br><span style="font-size:10px;color:#bbb">.py .js .html .css .java .c .cpp .go .rs .sh .json .yaml .md</span></div>
        </div>
        <input type="file" id="rpFileInput" multiple accept=".py,.js,.html,.css,.json,.xml,.yaml,.yml,.c,.cpp,.h,.java,.go,.rs,.sh,.bat,.md,.txt" style="display:none" onchange="handleRpFileSelect(this.files)">
      </div>
      <div id="rpTreeStats" class="rp-tree-stats" style="display:none">
        <span id="rpStatsInfo">0 파일 / 0 폴더</span>
        <span id="rpStatsSize">0 KB</span>
        <button onclick="clearRpProject()" style="background:none;border:none;color:#ef4444;cursor:pointer;font-size:11px;font-weight:600">전체 삭제</button>
      </div>
      <div id="rpLangDetect" style="display:none;padding:6px 10px;background:#f0f0ff;border-radius:6px;margin-bottom:6px;font-size:11px;">
        <div style="font-weight:600;color:#4f46e5;margin-bottom:3px;">🔍 감지된 언어/도구</div>
        <div id="rpLangTags" style="display:flex;flex-wrap:wrap;gap:4px;"></div>
        <div id="rpAutoSkills" style="margin-top:4px;color:#6b7280;font-size:10px;"></div>
      </div>
      <div class="rp-select-bar" id="rpSelectBar" style="display:none">
        <button class="rp-select-btn" onclick="rpCheckAll(true)">전체 선택</button>
        <button class="rp-select-btn" onclick="rpCheckAll(false)">전체 해제</button>
        <span class="rp-checked-count" id="rpCheckedCount">0개 선택</span>
      </div>
      <div class="rp-tree" id="rpTree" style="display:none"></div>
      <div class="rp-file-list" id="rpFileList"></div>
    </div>

    <div class="rp-section">
      <div class="rp-section-title">분석 모드 선택</div>
      <div class="rp-action-grid">
        <button class="rp-action-btn" data-mode="explain" onclick="selectRpMode(this)">
          <span class="rp-btn-icon">📖</span>
          <span class="rp-btn-label">코드 설명</span>
        </button>
        <button class="rp-action-btn" data-mode="find_bugs" onclick="selectRpMode(this)">
          <span class="rp-btn-icon">🐛</span>
          <span class="rp-btn-label">버그 탐지</span>
        </button>
        <button class="rp-action-btn" data-mode="improve" onclick="selectRpMode(this)">
          <span class="rp-btn-icon">✨</span>
          <span class="rp-btn-label">개선 제안</span>
        </button>
        <button class="rp-action-btn" data-mode="tests" onclick="selectRpMode(this)">
          <span class="rp-btn-icon">🧪</span>
          <span class="rp-btn-label">테스트 생성</span>
        </button>
        <button class="rp-action-btn" data-mode="docstring" onclick="selectRpMode(this)">
          <span class="rp-btn-icon">📝</span>
          <span class="rp-btn-label">문서화</span>
        </button>
        <button class="rp-action-btn" data-mode="refactor" onclick="selectRpMode(this)">
          <span class="rp-btn-icon">🔧</span>
          <span class="rp-btn-label">리팩토링</span>
        </button>
      </div>
    </div>

    <button class="rp-run-btn" id="rpRunBtn" onclick="runCodeAssistant()" disabled>▶ 분석 실행</button>
    <div class="rp-result" id="rpResult" style="display:none"></div>
  </div>
</div>
<button class="right-panel-toggle" id="rightPanelToggle" onclick="toggleRightPanel()">▶</button>

<!-- 스킬 상세 패널 -->
<div id="skillDetailOverlay" class="skill-detail-overlay" style="display:none" onclick="if(event.target===this)closeSkillDetail()">
  <div class="skill-detail-panel">
    <div class="sdp-header">
      <h2 id="sdpTitle">스킬 상세</h2>
      <button class="sdp-close" onclick="closeSkillDetail()">✕</button>
    </div>
    <div class="sdp-tabs" id="sdpTabs"></div>
    <div class="sdp-body" id="sdpBody">로딩중...</div>
  </div>
</div>

<script>
let envs = {};
let hasToken = false;
let selEnv = 'auto';
let catalog = {};
let selDomains = ['bioinformatics','dev-tools','agent-data-ai'];
let selSkills = [];
let autoSkillMode = true;  // 기본 ON
let selFormat = 'auto';
let formatManualOverride = false;  // 사용자가 수동으로 출력형식을 변경했는지
let styleManualOverride = false;   // 사용자가 수동으로 스타일을 변경했는지
let effort = 2;
let history = [];
let maxTokens = 8192;
let currentSessionId = null;
let sessions = {};

// ===== Init =====
loadSessions();
// Restore last session or create new
const lastSessions = Object.values(sessions).sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0));
if(lastSessions.length > 0){
  currentSessionId = lastSessions[0].id;
  const s = lastSessions[0];
  history = s.history || [];
  setTimeout(()=>{
    document.getElementById('msgs').innerHTML = s.msgsHtml || '';
    if(s.writingStyle){ document.getElementById('writingStyle').value = s.writingStyle; syncStyleDropToSidebar(); }
    if(s.systemPrompt) document.getElementById('systemPromptInput').value = s.systemPrompt;
    if(s.systemPromptId){ currentPromptId=s.systemPromptId; renderPromptChips(); }
    if(s.thinkMode!==undefined) document.getElementById('thinkToggle').checked=s.thinkMode;
    if(s.selFormat){ selFormat=s.selFormat; formatManualOverride=(selFormat!=='auto'); document.querySelectorAll('.fmt-btn').forEach(b=>{b.classList.toggle('selected',b.dataset.f===selFormat);}); syncFormatDropToChat(); }
    if(s.effort!==undefined){ effort=s.effort; document.getElementById('effortSlider').value=effort; }
    if(s.selEnv){ selEnv=s.selEnv; renderEnvs(); updateStatus(); }
    if(s.selDomains && s.selDomains.length>0){ selDomains=s.selDomains; renderTags(); renderSkills(); }
    if(s.selSkills && s.selSkills.length>0){ selSkills=s.selSkills; renderSkills(); updateLoaded(); }
    renderSessionList();
    // 페이지 로드 시 차트 재렌더링 (Chart.js 로드 대기)
    const _initCharts=()=>{
      document.querySelectorAll('#msgs canvas[data-chart-json]').forEach(cv=>{
        try{renderChartBlock(cv.id, b2u(cv.dataset.chartJson));}
        catch(e){cv.parentElement.innerHTML='<p style="color:red;padding:12px;">차트 렌더링 실패: '+e.message+'</p>';}
      });
    };
    if(typeof Chart!=='undefined') _initCharts();
    else setTimeout(_initCharts, 500);
  }, 100);
} else {
  currentSessionId = 'sess_'+Date.now();
  sessions[currentSessionId] = { id:currentSessionId, name:'새 세션', history:[], msgsHtml:'', updatedAt:Date.now() };
  saveSessions();
}

Promise.all([
  fetch('/api/config').then(r=>r.json()),
  fetch('/api/skills').then(r=>r.json()),
]).then(([cfgData, skillData])=>{
  envs = cfgData.envs || {};
  hasToken = cfgData.has_token;
  catalog = skillData || {};
  try{ renderTokenStatus(); }catch(e){ console.error('renderTokenStatus:',e); }
  try{ renderEnvs(); }catch(e){ console.error('renderEnvs:',e); }
  try{ renderTags(); }catch(e){ console.error('renderTags:',e); }
  try{ renderSkills(); }catch(e){ console.error('renderSkills:',e); }
  try{ renderSessionList(); }catch(e){ console.error('renderSessionList:',e); }
}).catch(err=>{
  console.error('설정 로드 실패:', err);
  document.getElementById('tokenStatus').className='token-status missing';
  document.getElementById('tokenStatus').textContent='❌ 서버 연결 실패 - 새로고침 해주세요';
  document.getElementById('tokenBadge').className='status off';
  document.getElementById('tokenBadge').textContent='❌ 연결 실패';
});

// ===== 환경 선택 =====
function renderEnvs(){
  const row = document.getElementById('envRow');
  row.innerHTML = '';
  const icons = {'dev':'🧪','prod':'🚀','common':'🌐','vl-large':'👁️','vl-medium':'👁️','vl-fast':'👁️'};
  // GGUF 로컬 모델은 동적으로 아이콘 매핑
  for(const id of Object.keys(envs)){ if(id.startsWith('gguf-')) icons[id]='💻'; }
  // AUTO 버튼 (맨 앞에 추가)
  const autoBtn = document.createElement('div');
  autoBtn.className = 'env-btn' + (selEnv==='auto'?' selected':'');
  autoBtn.innerHTML = '<div class="env-name">🤖 AUTO</div><div class="env-model">자동 모델 선택</div>';
  autoBtn.onclick = ()=>{ selEnv='auto'; renderEnvs(); updateStatus(); };
  autoBtn.style.borderColor = selEnv==='auto' ? '#6366f1' : '';
  row.appendChild(autoBtn);
  for(const [id, env] of Object.entries(envs)){
    const btn = document.createElement('div');
    btn.className = 'env-btn' + (selEnv===id?' selected':'');
    btn.innerHTML = `<div class="env-name">${icons[id]||'🔗'} ${env.name}</div><div class="env-model">${env.model}</div>`;
    btn.onclick = ()=>{ selEnv=id; renderEnvs(); updateStatus(); };
    row.appendChild(btn);
  }
}
function renderTokenStatus(){
  const el = document.getElementById('tokenStatus');
  const badge = document.getElementById('tokenBadge');
  if(hasToken){
    el.className='token-status ok';
    el.textContent='🔑 TOKEN.TXT 로드됨 - API 키 자동 적용';
    badge.className='status on';
    badge.textContent='🔑 토큰 OK';
  } else {
    el.className='token-status ok';
    el.textContent='ℹ️ TOKEN.TXT 미설정 - 폐쇄망 API는 토큰 없이 사용 가능합니다';
    badge.className='status on';
    badge.textContent='🔗 토큰 선택';
  }
}
function updateStatus(){
  const st = document.getElementById('status');
  if(selEnv === 'auto'){
    st.className='status on';
    st.textContent='🤖 AUTO (자동 선택)';
  } else if(selEnv && envs[selEnv]){
    st.className='status on';
    st.textContent='🟢 ' + envs[selEnv].name;
  } else {
    st.className='status off';
    st.textContent='⚪ 환경 미선택';
  }
}

// ===== Tags =====
function renderTags(){
  const row = document.getElementById('tagRow');
  row.innerHTML = '';
  for(const [id, d] of Object.entries(catalog)){
    const t = document.createElement('div');
    t.className = 'tag' + (selDomains.includes(id)?' selected':'');
    t.style.setProperty('--c', d.color);
    t.style.setProperty('--bg', d.color+'18');
    t.style.setProperty('--fg', d.color);
    t.textContent = d.icon + ' ' + d.label;
    t.onclick = ()=>{ toggleDomain(id); t.classList.toggle('selected'); renderSkills(); };
    row.appendChild(t);
  }
}
function toggleDomain(id){
  if(selDomains.includes(id)) selDomains=selDomains.filter(x=>x!==id);
  else selDomains.push(id);
}

// ===== Skills =====
let autoLoadedSkills = [];  // 자동 로드된 스킬 목록 (시각적 표시용)

function renderSkills(){
  const grid = document.getElementById('skillGrid');
  grid.innerHTML = '';
  selDomains.forEach(did=>{
    const d = catalog[did];
    if(!d) return;
    d.skills.forEach(s=>{
      const c = document.createElement('div');
      const isAutoLoaded = autoLoadedSkills.includes(s.id);
      c.className = 'skill-card' + (selSkills.includes(s.id)?' selected':'') + (isAutoLoaded?' auto-loaded':'') + (!s.available?' unavailable':'');
      const desc = s.desc ? s.desc : '';
      const avail = s.available ? '✅' : '❌';
      // extras: 스크립트/레퍼런스 개수
      let extras = [];
      if(s.scripts > 0) extras.push(`🐍${s.scripts}`);
      if(s.references > 0) extras.push(`📄${s.references}`);
      if(s.assets > 0) extras.push(`📦${s.assets}`);
      const extrasHtml = extras.length ? `<div class="extras">${extras.join(' ')}</div>` : '';
      const autoBadge = isAutoLoaded ? '<span class="badge" style="background:#f59e0b">🧠자동</span>' : '';
      c.innerHTML = `<div class="sn">${avail} ${s.name}</div><div class="sd">${desc}</div>${extrasHtml}${autoBadge}`;
      if(s.available){
        c.onclick = (e)=>{
          // Shift+클릭 = 상세 패널
          if(e.shiftKey || (extras.length > 0 && e.detail === 2)){
            openSkillDetail(s.id);
            return;
          }
          if(selSkills.includes(s.id)) selSkills=selSkills.filter(x=>x!==s.id);
          else selSkills.push(s.id);
          renderSkills();
          updateLoaded();
        };
        // 우클릭 = 상세 패널
        c.oncontextmenu = (e)=>{
          e.preventDefault();
          openSkillDetail(s.id);
        };
      }
      grid.appendChild(c);
    });
  });
  updateLoaded();
}
function updateLoaded(){
  const totalCount = selSkills.length + autoLoadedSkills.length;
  document.getElementById('loadedCount').textContent = totalCount;
  const el = document.getElementById('loadedSkillsList');
  if(totalCount === 0 && dismissedAutoSkills.size === 0){
    el.innerHTML = '<div style="color:#bbb">선택된 스킬 없음</div>';
    return;
  }
  // 수동 선택 스킬 (체크박스로 해제 가능)
  let html = selSkills.map(s =>
    `<label style="display:flex;align-items:center;gap:4px;padding:2px 0;cursor:pointer;">
      <input type="checkbox" checked onchange="unloadSkill('${s}')" style="accent-color:#4f46e5;cursor:pointer;">
      <span style="font-size:12px">✅ ${s}</span>
    </label>`
  ).join('');
  // 자동 추천 스킬 (1회성, 📌으로 수동 고정 / ✕로 해제)
  if(autoLoadedSkills.length > 0){
    html += autoLoadedSkills.map(s =>
      `<div style="display:flex;align-items:center;justify-content:space-between;padding:2px 0;opacity:.8;">
        <span style="font-size:12px">🧠 ${s}</span>
        <span style="display:flex;gap:2px;">
          <button onclick="pinAutoSkill('${s}')" style="background:none;border:none;cursor:pointer;font-size:10px;color:#4f46e5;padding:0 2px;" title="수동 고정">📌</button>
          <button onclick="dismissAutoSkill('${s}')" style="background:none;border:none;cursor:pointer;font-size:10px;color:#ef4444;padding:0 2px;" title="해제 (다시 추천 안함)">✕</button>
        </span>
      </div>`
    ).join('');
  }
  // 해제된 자동 스킬 (블랙리스트) 표시
  if(dismissedAutoSkills.size > 0){
    html += `<div style="margin-top:6px;padding-top:6px;border-top:1px dashed #e5e3de;">
      <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">🚫 해제됨 (자동추천 제외)</div>`;
    html += [...dismissedAutoSkills].map(s =>
      `<div style="display:flex;align-items:center;justify-content:space-between;padding:1px 0;opacity:.5;">
        <span style="font-size:11px;text-decoration:line-through;color:#9ca3af;">${s}</span>
        <button onclick="restoreDismissedSkill('${s}')" style="background:none;border:none;cursor:pointer;font-size:10px;color:#22c55e;padding:0 2px;" title="복구 (자동추천 허용)">♻️</button>
      </div>`
    ).join('');
    if(dismissedAutoSkills.size >= 2){
      html += `<button onclick="clearDismissed()" style="width:100%;margin-top:3px;padding:2px 0;font-size:10px;background:#fef2f2;border:1px solid #fecaca;border-radius:4px;cursor:pointer;color:#ef4444;">전체 복구</button>`;
    }
    html += `</div>`;
  }
  if(selSkills.length + autoLoadedSkills.length >= 2){
    html += `<button onclick="unloadAllSkills()" style="width:100%;margin-top:4px;padding:3px 0;font-size:11px;background:#f5f5f0;border:1px solid #e5e3de;border-radius:4px;cursor:pointer;color:#888;">전체 해제</button>`;
  }
  el.innerHTML = html;
}
let dismissedAutoSkills = new Set();  // 사용자가 해제한 자동 스킬 (세션 내 재로드 방지)

function unloadSkill(id){
  selSkills = selSkills.filter(x => x !== id);
  // 자동 스킬이면 해제 블랙리스트에 추가
  if(autoLoadedSkills.includes(id)){
    autoLoadedSkills = autoLoadedSkills.filter(x => x !== id);
    dismissedAutoSkills.add(id);
  }
  renderSkills();
  updateLoaded();
}
function unloadAllSkills(){
  // 자동 스킬 모두 블랙리스트에 추가
  autoLoadedSkills.forEach(id => dismissedAutoSkills.add(id));
  selSkills = [];
  autoLoadedSkills = [];
  renderSkills();
  updateLoaded();
}
function pinAutoSkill(id){
  // 자동 스킬을 수동 고정으로 전환 → 블랙리스트에서 제거
  autoLoadedSkills = autoLoadedSkills.filter(x => x !== id);
  dismissedAutoSkills.delete(id);
  if(!selSkills.includes(id)) selSkills.push(id);
  renderSkills();
  updateLoaded();
}
function dismissAutoSkill(id){
  // 자동 스킬을 블랙리스트로 이동 (다시 추천 안함)
  autoLoadedSkills = autoLoadedSkills.filter(x => x !== id);
  dismissedAutoSkills.add(id);
  renderSkills();
  updateLoaded();
}
function restoreDismissedSkill(id){
  // 블랙리스트에서 복구 (다음 자동 추천 허용)
  dismissedAutoSkills.delete(id);
  updateLoaded();
}
function clearDismissed(){
  // 블랙리스트 전체 초기화
  dismissedAutoSkills.clear();
  updateLoaded();
}

// ===== Effort =====
function updateEffort(){
  const s = document.getElementById('effortSlider');
  effort = parseInt(s.value);
  s.style.setProperty('--val', (effort/3*100)+'%');
  for(let i=0;i<=3;i++) document.getElementById('e'+i).className = i===effort?'active':'';
}
updateEffort();

// ===== Format =====
function selFmt(el){
  document.querySelectorAll('.fmt-btn').forEach(b=>b.classList.remove('selected'));
  el.classList.add('selected');
  selFormat = el.dataset.f;
  formatManualOverride = (selFormat !== 'auto');  // auto 선택 시 자동 모드로 복귀
  syncFormatDropToChat();
}

// ===== Quick prompt =====
function qp(t){ document.getElementById('input').value=t; }

// ===== Auto Skill =====
function toggleAutoSkill(){
  autoSkillMode = !autoSkillMode;
  const el = document.getElementById('autoSkillToggle');
  el.classList.toggle('on', autoSkillMode);
  if(!autoSkillMode){
    document.getElementById('autoSkillPreview').classList.remove('show');
    document.getElementById('autoSkillPreview').innerHTML = '';
  }
}
// 초기 상태 반영
setTimeout(()=>{
  document.getElementById('autoSkillToggle').classList.toggle('on', autoSkillMode);
}, 200);

// 입력 중 자동 스킬 미리보기 (디바운스)
let autoSkillTimer = null;
document.addEventListener('DOMContentLoaded', ()=>{
  const inp = document.getElementById('input');
  if(inp) inp.addEventListener('input', ()=>{
    if(!autoSkillMode) return;
    clearTimeout(autoSkillTimer);
    autoSkillTimer = setTimeout(async ()=>{
      const q = inp.value.trim();
      if(q.length < 4) { document.getElementById('autoSkillPreview').classList.remove('show'); return; }
      try{
        const r = await fetch('/api/auto-skills',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,max_skills:7,history:history})});
        const d = await r.json();
        const prev = document.getElementById('autoSkillPreview');
        if(d.skills && d.skills.length > 0){
          prev.innerHTML = '🧠 자동 추천: ' + d.skills.map(s=>`<span class="auto-skill-badge" title="${s.desc}">${s.id} (${s.score})</span>`).join('');
          prev.classList.add('show');
        } else {
          prev.classList.remove('show');
        }
      }catch(e){}
    }, 500);
  });
});

// ===== Chat =====
let chatAbort = null;  // AbortController for current request
let isSending = false;

function handleSendStop(){
  if(isSending){
    stopGeneration();
  } else {
    send();
  }
}
function stopGeneration(){
  if(chatAbort){
    chatAbort.abort();
    chatAbort = null;
  }
  // 백엔드 GGUF 중단 요청
  fetch('/api/chat/stop',{method:'POST'}).catch(()=>{});
  isSending = false;
  const btn = document.getElementById('sendBtn');
  btn.textContent = '▶';
  btn.classList.remove('stop-mode');
  btn.disabled = false;
  // typing indicator 제거
  document.querySelectorAll('.typing-wrap').forEach(el=>el.remove());
  addMsg('assistant','⏹️ 응답이 중지되었습니다.');
  saveCurrentSession();
}

/* ── PPT 제안 배너 ── */
let _pptBannerShownCount = 0;
const _pptSuggestItems = [
  {icon:'📊', title:'차트/그래프를 포함할까요?', hint:'"매출 데이터를 막대 차트로 넣어줘"', example:'차트/그래프도 포함해서 PPT 만들어줘'},
  {icon:'📝', title:'그래프 없이 깔끔하게?', hint:'"그래프 없이 텍스트와 표로만 PPT 만들어줘"', example:'그래프 없이 텍스트와 표 위주로 PPT 만들어줘'},
  {icon:'📋', title:'표(Table)를 넣어볼까요?', hint:'"비교 데이터를 표로 정리해서 넣어줘"', example:'데이터를 표로 정리해서 PPT에 넣어줘'},
  {icon:'🖼️', title:'이미지/다이어그램도 가능해요!', hint:'"구조도를 슬라이드에 추가해줘"', example:'다이어그램도 포함해서 PPT 만들어줘'},
  {icon:'📈', title:'데이터 시각화를 추가할까요?', hint:'"트렌드를 꺾은선 그래프로 보여줘"', example:'데이터를 시각화해서 PPT에 넣어줘'},
  {icon:'🎨', title:'심플한 디자인으로?', hint:'"심플하게 핵심 내용만 PPT 만들어줘"', example:'디자인 심플하게 핵심만 PPT 만들어줘'},
  {icon:'🍩', title:'원형/도넛 차트는 어때요?', hint:'"비율을 도넛 차트로 만들어줘"', example:'비율 데이터를 원형 차트로 PPT에 넣어줘'},
  {icon:'📉', title:'비교 차트를 넣어볼까요?', hint:'"전년 대비 성장률을 비교 차트로"', example:'비교 차트를 포함해서 PPT 만들어줘'},
];

const _pptNoVisualItems = [
  {icon:'📝', title:'그래프 없이도 만들 수 있어요!', hint:'"그래프 없이 텍스트와 표로만 PPT 만들어줘"', example:'그래프 없이 텍스트와 표 위주로 PPT 다시 만들어줘'},
  {icon:'🎨', title:'심플한 PPT는 어때요?', hint:'"심플하게 핵심 내용만 PPT로"', example:'디자인 심플하게 핵심만 PPT 만들어줘'},
];
const _pptVisualItems = [
  {icon:'📊', title:'그래프를 추가해볼까요?', hint:'"차트/그래프도 포함해서 PPT 만들어줘"', example:'차트/그래프도 포함해서 PPT 다시 만들어줘'},
  {icon:'📈', title:'데이터 시각화는 어때요?', hint:'"데이터를 시각화해서 PPT에 넣어줘"', example:'데이터를 시각화해서 PPT에 넣어줘'},
];

function _showPptSuggestBanner(mode){
  if(_pptBannerShownCount >= 3) return;
  _pptBannerShownCount++;
  const area = document.getElementById('pptSuggestArea');
  if(!area) return;
  let pool;
  if(mode === 'no-visual') pool = _pptNoVisualItems;
  else if(mode === 'visual') pool = _pptVisualItems;
  else pool = _pptSuggestItems;
  const item = pool[Math.floor(Math.random()*pool.length)];
  const banner = document.createElement('div');
  banner.className = 'ppt-suggest-banner';
  banner.innerHTML = `<span class="ppt-banner-icon">${item.icon}</span>`
    + `<div class="ppt-banner-title">${item.title}</div>`
    + `<div class="ppt-banner-hint"><em onclick="_usePptSuggestion(this,'${item.example.replace(/'/g,"\\'")}')">${item.hint}</em> 처럼 말해보세요!</div>`
    + `<button class="ppt-banner-close" onclick="_closePptBanner(this)" title="닫기">✕</button>`;
  area.innerHTML = '';
  area.appendChild(banner);
  setTimeout(()=>{
    if(banner.parentNode){
      banner.classList.add('hiding');
      setTimeout(()=>{ if(banner.parentNode) banner.remove(); }, 400);
    }
  }, 8000);
}

function _closePptBanner(btn){
  const banner = btn.closest('.ppt-suggest-banner');
  if(banner){ banner.classList.add('hiding'); setTimeout(()=>banner.remove(), 400); }
}

function _usePptSuggestion(el, text){
  const input = document.getElementById('input');
  if(input){ input.value = text; input.focus(); input.style.height='auto'; input.style.height=input.scrollHeight+'px'; }
  const banner = el.closest('.ppt-suggest-banner');
  if(banner){ banner.classList.add('hiding'); setTimeout(()=>banner.remove(), 400); }
}

const _pptKeywords = /PPT|ppt|파워포인트|프레젠테이션|발표자료|슬라이드\s*만들|피피티/i;

async function send(){
  const el=document.getElementById('input');
  const text=el.value.trim();
  if(!text && chatPendingFiles.length === 0) return;

  if(text && _pptKeywords.test(text)){
    const hasVisual = /차트|그래프|표|table|chart|graph|시각화|도넛|원형/i.test(text);
    const hasNoVisual = /그래프\s*없|차트\s*없|텍스트\s*위주|텍스트만|심플하게/i.test(text);
    _showPptSuggestBanner(hasVisual && !hasNoVisual ? 'visual' : hasNoVisual && !hasVisual ? 'no-visual' : null);
  }
  if(!selEnv){alert('먼저 위에서 LLM 환경을 선택해주세요.');return;}

  // 첨부파일이 있으면 먼저 업로드
  let attachedNames = [];
  if(chatPendingFiles.length > 0){
    attachedNames = await uploadChatPendingFiles();
  }

  // 스킬 구성: 수동 선택 + 파일 프리로드 + 질문 기반 자동 추천
  let skillsToUse = [...selSkills];
  let autoLoaded = [...autoLoadedSkills];  // 파일 첨부/세션 로드로 프리로드된 스킬

  // 프리로드 스킬 중 수동/해제 중복 제거
  const manualSet = new Set(skillsToUse);
  autoLoaded = autoLoaded.filter(id => !manualSet.has(id) && !dismissedAutoSkills.has(id));
  skillsToUse = [...skillsToUse, ...autoLoaded];

  // 자동 스킬 모드: 질문 분석으로 추가 보충
  if(autoSkillMode){
    try{
      const currentCount = skillsToUse.length;
      const maxAuto = currentCount === 0 ? 7 : Math.max(0, 10 - currentCount);
      if(maxAuto > 0){
        const ar = await fetch('/api/auto-skills',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:text,max_skills:maxAuto,history:history})});
        const ad = await ar.json();
        if(ad.skills && ad.skills.length > 0){
          const usedSet = new Set(skillsToUse);
          const newAuto = ad.skills.map(s=>s.id).filter(id=>!usedSet.has(id) && !dismissedAutoSkills.has(id));
          autoLoaded = [...autoLoaded, ...newAuto];
          skillsToUse = [...skillsToUse, ...newAuto];
        }
      }
    }catch(e){}
  }

  // 첨부파일 표시 + 메시지
  let displayText = text || '';
  if(attachedNames.length > 0){
    const attachInfo = '📎 ' + attachedNames.join(', ');
    displayText = displayText ? attachInfo + '\n' + displayText : attachInfo;
  }
  addMsg('user', displayText);
  history.push({role:'user',content: displayText});
  el.value=''; el.style.height='auto';
  document.getElementById('autoSkillPreview').classList.remove('show');

  // 자동 로드 안내 (selSkills에는 추가하지 않음 - 1회성 사용)
  if(autoLoaded.length > 0){
    autoLoadedSkills = autoLoaded;
    renderSkills();
    updateLoaded();
    const manualCount = selSkills.length;
    const info = manualCount > 0 ? ` (수동 ${manualCount}개 + 자동 ${autoLoaded.length}개)` : '';
    addMsg('assistant', '🧠 자동 스킬: ' + autoLoaded.join(', ') + info);
  }

  const typing=addTyping();
  isSending = true;
  const btn = document.getElementById('sendBtn');
  btn.textContent = '⏹';
  btn.classList.add('stop-mode');
  btn.disabled = false;

  chatAbort = new AbortController();

  try{
    const resp=await fetch('/api/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      signal: chatAbort.signal,
      body:JSON.stringify({
        env: selEnv,
        messages:history, skills:skillsToUse, effort,
        format: formatManualOverride ? selFormat : 'auto',
        writing_style: styleManualOverride ? document.getElementById('writingStyle').value.trim() : 'auto',
        system_prompt:document.getElementById('systemPromptInput').value.trim(),
        max_tokens:maxTokens,
        think_mode: document.getElementById('thinkToggle').checked,
      })
    });
    const data=await resp.json();
    typing.remove();

    if(data.error){
      addMsg('assistant','❌ '+data.error);
    } else {
      let info = '';
      // 모델 이름 표시 (auto/수동)
      let modelName = data.model_used || (envs[selEnv] ? envs[selEnv].name : selEnv);
      if(data.loaded_skills && data.loaded_skills.length > 0){
        let extra = data.tokens_budget ? ` [${data.tokens_budget}]` : '';
        let mode = autoLoaded.length > 0 ? '🧠자동' : '✅수동';
        info = `\n[${mode} 스킬: ${data.loaded_skills.join(', ')}] [${modelName}] (${data.system_prompt_length}자)${extra}`;
      }
      // 자동 라우팅 표시
      if(data.auto_routed){
        info += ` [🤖 자동: ${data.model_used} (${data.route_reason})]`;
      }
      // 폴백 표시
      if(data.fallback_used){
        info += ` [⚠️ 대체: ${data.fallback_from} → ${data.model_used}]`;
      }
      // 자동 형식/스타일 표시
      if(data.auto_format || data.auto_style){
        let fmtNames = {'code':'코드','code-fix':'코드수정','analysis':'분석','report':'보고서','step-by-step':'단계별'};
        let parts = [];
        if(data.auto_format) parts.push('형식:' + (fmtNames[data.auto_format]||data.auto_format));
        if(data.auto_style) parts.push('스타일:자동');
        info += ` [📝 ${parts.join(' / ')}]`;
      }
      let truncWarn = data.truncated ? '\n\n⚠️ **응답이 토큰 한도('+maxTokens+')에 도달하여 잘렸습니다.** "계속 이어서 작성해줘"라고 입력하면 이어서 받을 수 있습니다.' : '';
      const assistantDisplayText = data.content + truncWarn + info;
      const assistantRawForDetect = data.content + truncWarn;
      addMsg('assistant', assistantDisplayText, assistantRawForDetect);
      history.push({role:'assistant',content:data.content});
      // markdown-mermaid-writing 스킬: ```markdown 블록이 없어도 전체 응답에 MD 다운로드 버튼 추가
      if(selSkills.includes('markdown-mermaid-writing') && !data.content.includes('```markdown')){
        appendMdDownloadBar(data.content);
      }
    }
  }catch(e){
    typing.remove();
    if(e.name === 'AbortError'){
      // 사용자가 중지한 경우 - stopGeneration()에서 이미 처리됨
    } else {
      addMsg('assistant','❌ 서버 연결 실패: '+e.message);
    }
  }
  isSending = false;
  chatAbort = null;
  btn.textContent = '▶';
  btn.classList.remove('stop-mode');
  btn.disabled = false;
  const inp=document.getElementById('input');
  inp.focus();
  inp.style.height='auto';
  saveCurrentSession();
}

function renderMd(text){
  // 0) <think> 블록을 접이식 사고과정 박스로 변환 (HTML escape 전)
  text=text.replace(/<think>([\s\S]*?)<\/think>\s*/g, function(_,content){
    const escaped=content.trim().replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
    return `<details class="think-box"><summary>💭 사고 과정</summary><div class="think-content">${escaped}</div></details>`;
  });
  // 닫히지 않은 <think> 태그도 처리 (응답 잘림 대비)
  text=text.replace(/<think>([\s\S]*)$/g, function(_,content){
    const escaped=content.trim().replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
    return `<details class="think-box"><summary>💭 사고 과정 (진행중...)</summary><div class="think-content">${escaped}</div></details>`;
  });
  // 1) escape HTML (think-box는 이미 처리했으므로 보호)
  const thinkBlocks=[];
  text=text.replace(/<details class="think-box">[\s\S]*?<\/details>/g, m=>{thinkBlocks.push(m);return `__THINK_${thinkBlocks.length-1}__`;});
  let s=text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  // 2) code blocks (```lang\n...``` 또는 ```lang코드...``` 모두 지원)
  // 2a) drawio 코드블록은 특별 처리
  s=s.replace(/```drawio\s*([\s\S]*?)```/g,(_,code)=>{
    const escapedXml = code.trim();
    // HTML escape된 상태를 원본 XML로 복원 (btoa용)
    const realXml = escapedXml.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"');
    const xmlB64 = u2b(realXml);
    return `<div class="drawio-block">
      <div class="drawio-header">
        <span>📐 Draw.io 다이어그램</span>
        <div class="drawio-actions">
          <button class="drawio-btn" onclick="copyDrawioXml(this)" data-xml="${xmlB64}">📋 XML 복사</button>
          <button class="drawio-btn" onclick="downloadDrawio(this)" data-xml="${xmlB64}">💾 .drawio 저장</button>
          <button class="drawio-btn" onclick="openInDrawio(this)" data-xml="${xmlB64}">🔗 Draw.io에서 열기</button>
        </div>
      </div>
      <pre class="drawio-preview"><code class="language-xml">${escapedXml}</code></pre>
    </div>`;
  });
  // 2a-2) chart 코드블록 → Chart.js 인터랙티브 그래프 렌더링
  s=s.replace(/```chart\s*([\s\S]*?)```/g,(_,code)=>{
    const trimmed=code.trim();
    const raw=trimmed.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"');
    const chartId='chart_'+Math.random().toString(36).substr(2,9);
    const b64=u2b(raw);
    return `<div class="chart-block">
      <div class="chart-header">
        <span>📊 인터랙티브 차트</span>
        <div class="chart-actions">
          <button class="chart-btn" onclick="downloadChartPng('${chartId}')">📥 PNG 저장</button>
          <button class="chart-btn" onclick="copyChartJson(this)" data-chart="${b64}">📋 JSON 복사</button>
        </div>
      </div>
      <div class="chart-canvas-wrap"><canvas id="${chartId}" data-chart-json="${b64}"></canvas></div>
      <details class="chart-raw-toggle"><summary>📝 원본 JSON 보기</summary><pre>${trimmed}</pre></details>
    </div>`;
  });
  // 2b) markdown 코드블록 → MD 다운로드 UI
  s=s.replace(/```markdown\s*([\s\S]*?)```/g,(_,code)=>{
    const trimmed=code.trim();
    const mdB64=u2b(trimmed.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"'));
    return `<div class="md-report-block">
      <div class="md-report-header">
        <span>📋 마크다운 문서</span>
        <div class="md-report-actions">
          <button class="md-report-btn" onclick="copyMdContent(this)" data-md="${mdB64}">📋 복사</button>
          <button class="md-report-btn md-download" onclick="downloadMarkdown(this)" data-md="${mdB64}">📥 MD 다운로드</button>
        </div>
      </div>
      <div class="md-report-preview">${renderMdPreview(trimmed)}</div>
      <details class="md-raw-toggle"><summary>📝 원본 마크다운 보기</summary><pre><code class="language-markdown">${trimmed}</code></pre></details>
    </div>`;
  });
  // 2c) 일반 코드블록 — xml 블록 안에 mxfile이 있으면 drawio로 자동 변환
  s=s.replace(/```(\w*)\s*([\s\S]*?)```/g,(_,lang,code)=>{
    const trimmed=code.trim();
    // 코드블록 안에 <mxfile 또는 <mxGraphModel 있으면 drawio UI로 변환
    if(trimmed.includes('&lt;mxfile') || trimmed.includes('&lt;mxGraphModel')){
      const realXml=trimmed.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"');
      const xmlB64=u2b(realXml);
      return `<div class="drawio-block">
        <div class="drawio-header">
          <span>📐 Draw.io 다이어그램</span>
          <div class="drawio-actions">
            <button class="drawio-btn" onclick="copyDrawioXml(this)" data-xml="${xmlB64}">📋 XML 복사</button>
            <button class="drawio-btn" onclick="downloadDrawio(this)" data-xml="${xmlB64}">💾 .drawio 저장</button>
            <button class="drawio-btn" onclick="openInDrawio(this)" data-xml="${xmlB64}">🔗 Draw.io에서 열기</button>
          </div>
        </div>
        <pre class="drawio-preview"><code class="language-xml">${trimmed}</code></pre>
      </div>`;
    }
    const cls=lang?` class="language-${lang}"`:'';
    return `<pre><code${cls}>${trimmed}</code></pre>`;
  });
  // 2c) raw mxfile XML (코드블록 밖) 자동 감지 → drawio UI로 변환
  s=s.replace(/(&lt;mxfile[\s\S]*?&lt;\/mxfile&gt;)/g,(_,xmlEsc)=>{
    const realXml=xmlEsc.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"');
    const xmlB64=u2b(realXml);
    return `<div class="drawio-block">
      <div class="drawio-header">
        <span>📐 Draw.io 다이어그램 (자동 감지)</span>
        <div class="drawio-actions">
          <button class="drawio-btn" onclick="copyDrawioXml(this)" data-xml="${xmlB64}">📋 XML 복사</button>
          <button class="drawio-btn" onclick="downloadDrawio(this)" data-xml="${xmlB64}">💾 .drawio 저장</button>
          <button class="drawio-btn" onclick="openInDrawio(this)" data-xml="${xmlB64}">🔗 Draw.io에서 열기</button>
        </div>
      </div>
      <pre class="drawio-preview"><code class="language-xml">${xmlEsc}</code></pre>
    </div>`;
  });
  // 3) tables
  s=s.replace(/((?:^\|.+\|[ ]*\n){2,})/gm, function(tbl){
    const rows=tbl.trim().split('\n').filter(r=>r.trim());
    if(rows.length<2) return tbl;
    const parseRow=r=>r.replace(/^\|/,'').replace(/\|$/,'').split('|').map(c=>c.trim());
    const hdr=parseRow(rows[0]);
    // skip separator row
    let startIdx=1;
    if(/^[\s|:-]+$/.test(rows[1])) startIdx=2;
    let h='<table><thead><tr>'+hdr.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>';
    for(let i=startIdx;i<rows.length;i++){
      const cells=parseRow(rows[i]);
      h+='<tr>'+cells.map(c=>'<td>'+c+'</td>').join('')+'</tr>';
    }
    return h+'</tbody></table>';
  });
  // 4) headings
  s=s.replace(/^#### (.+)$/gm,'<h4>$1</h4>');
  s=s.replace(/^### (.+)$/gm,'<h3>$1</h3>');
  s=s.replace(/^## (.+)$/gm,'<h2>$1</h2>');
  s=s.replace(/^# (.+)$/gm,'<h1>$1</h1>');
  // 5) hr
  s=s.replace(/^---+$/gm,'<hr>');
  // 6) blockquote
  s=s.replace(/^&gt; (.+)$/gm,'<blockquote>$1</blockquote>');
  // 7) unordered list
  s=s.replace(/(^[\-\*] .+\n?)+/gm, function(block){
    const items=block.trim().split('\n').map(l=>l.replace(/^[\-\*] /,''));
    return '<ul>'+items.map(i=>'<li>'+i+'</li>').join('')+'</ul>';
  });
  // 8) ordered list
  s=s.replace(/(^\d+\. .+\n?)+/gm, function(block){
    const items=block.trim().split('\n').map(l=>l.replace(/^\d+\. /,''));
    return '<ol>'+items.map(i=>'<li>'+i+'</li>').join('')+'</ol>';
  });
  // 9) inline: bold, italic, code
  s=s.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  s=s.replace(/\*(.+?)\*/g,'<em>$1</em>');
  s=s.replace(/`([^`]+)`/g,'<code>$1</code>');
  // 10) paragraphs: double newline -> <p>
  s=s.split(/\n{2,}/).map(block=>{
    const t=block.trim();
    if(!t) return '';
    if(/^<(pre|div|h[1-4]|ul|ol|table|hr|blockquote)/.test(t)) return t;
    return '<p>'+t.replace(/\n/g,'<br>')+'</p>';
  }).join('\n');
  // single newlines inside remaining text
  // think 블록 복원
  thinkBlocks.forEach((block,i)=>{s=s.replace(`__THINK_${i}__`,block);});
  return s;
}
function addMsg(role,text,rawForDetect){
  const c=document.getElementById('msgs');
  const d=document.createElement('div');
  d.className='msg '+role;
  let html = role==='user' ? text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') : renderMd(text);
  d.innerHTML=`<div class="msg-label">${role==='user'?'나':'Demos'}</div>${html}`;
  c.appendChild(d);
  d.querySelectorAll('pre').forEach(pre=>{
    const btn=document.createElement('button');
    btn.className='copy-btn';btn.textContent='복사';
    btn.onclick=()=>{navigator.clipboard.writeText(pre.textContent.replace('복사',''));btn.textContent='✓';setTimeout(()=>btn.textContent='복사',1500);};
    pre.appendChild(btn);
  });
  // PPT 코드 블록 감지 → 생성 & 다운로드 버튼 삽입
  if(role==='assistant') detectAndAddPptxButtons(d, (typeof rawForDetect==='string' ? rawForDetect : text));
  // Chart.js 차트 초기화 (data-chart-json 속성이 있는 canvas 탐색)
  d.querySelectorAll('canvas[data-chart-json]').forEach(cv=>{
    try{renderChartBlock(cv.id, b2u(cv.dataset.chartJson));}
    catch(e){cv.parentElement.innerHTML='<p style="color:red;padding:12px;">차트 렌더링 실패: '+e.message+'</p>';}
  });
  c.scrollIntoView({behavior:'smooth',block:'end'});
}
function addTyping(){
  const c=document.getElementById('msgs');
  const d=document.createElement('div');
  d.className='msg assistant';
  d.innerHTML='<div class="typing"><span></span><span></span><span></span></div>';
  c.appendChild(d);
  c.scrollIntoView({behavior:'smooth',block:'end'});
  return d;
}
// ===== Session Management =====
function loadSessions(){
  try{ sessions = JSON.parse(localStorage.getItem('domos_sessions') || '{}'); }catch(e){ sessions={}; }
}
function saveSessions(){
  try {
    localStorage.setItem('domos_sessions', JSON.stringify(sessions));
  } catch(e){
    console.warn('세션 저장 실패 (localStorage 용량 초과 가능):', e);
    // 오래된 세션 자동 정리 후 재시도
    const ids = Object.keys(sessions).sort((a,b)=>(sessions[a].updatedAt||0)-(sessions[b].updatedAt||0));
    while(ids.length > 20){ const old=ids.shift(); delete sessions[old]; }
    try { localStorage.setItem('domos_sessions', JSON.stringify(sessions)); } catch(e2){}
  }
}
function saveCurrentSession(){
  if(!currentSessionId) return;
  sessions[currentSessionId] = {
    id: currentSessionId,
    name: sessions[currentSessionId]?.name || '새 세션',
    history: history,
    selEnv: selEnv,
    selDomains: selDomains,
    selSkills: selSkills,
    selFormat: selFormat,
    formatManualOverride: formatManualOverride,
    styleManualOverride: styleManualOverride,
    effort: effort,
    writingStyle: document.getElementById('writingStyle')?.value || '',
    writingStyleId: activeStyleId || '',
    systemPrompt: document.getElementById('systemPromptInput')?.value || '',
    systemPromptId: currentPromptId || '',
    thinkMode: document.getElementById('thinkToggle')?.checked || false,
    msgsHtml: document.getElementById('msgs').innerHTML,
    updatedAt: Date.now()
  };
  // Auto-name from first user message
  if(sessions[currentSessionId].name === '새 세션' && history.length > 0){
    const first = history.find(m=>m.role==='user');
    if(first) sessions[currentSessionId].name = first.content.slice(0,30) + (first.content.length>30?'...':'');
  }
  saveSessions();
  renderSessionList();
}
let selectedSessions = new Set();

function renderSessionList(){
  const el=document.getElementById('sessionList');
  if(!el) return;
  const sorted = Object.values(sessions).filter(s=>!s.archived).sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0));
  const archived = Object.values(sessions).filter(s=>s.archived).sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0));

  document.getElementById('sessionCount').textContent = sorted.length;

  if(sorted.length===0 && archived.length===0){
    el.innerHTML='<div style="font-size:11px;color:#bbb;padding:4px 8px;">저장된 세션 없음</div>';
    return;
  }

  let html = sorted.map(s=>{
    const checked = selectedSessions.has(s.id) ? 'checked' : '';
    const isCurrent = s.id===currentSessionId;
    const time = s.updatedAt ? new Date(s.updatedAt).toLocaleDateString('ko',{month:'short',day:'numeric'}) : '';
    return `<div class="session-item${isCurrent?' current':''}" onclick="loadSession('${s.id}')">
      <input type="checkbox" class="session-cb" ${checked} onclick="event.stopPropagation();toggleSessionCheck('${s.id}',this)">
      <span class="session-name">${s.name||'새 세션'}</span>
      <span class="session-time">${time}</span>
    </div>`;
  }).join('');

  // 보관된 세션 (항상 펼쳐서 보여줌)
  if(archived.length > 0){
    html += `<div style="padding:4px 8px;margin-top:6px;border-top:1px solid #e5e3de;padding-top:8px;">
      <div style="font-size:11px;color:#999;margin-bottom:4px">📦 보관함 (${archived.length})</div>
      ${archived.map(s=>{
        const checked = selectedSessions.has(s.id) ? 'checked' : '';
        return `<div class="session-item" style="opacity:.7" onclick="loadSession('${s.id}')">
          <input type="checkbox" class="session-cb" ${checked} onclick="event.stopPropagation();toggleSessionCheck('${s.id}',this)">
          <span class="session-name">${s.name||'세션'}</span>
        </div>`;
      }).join('')}
    </div>`;
  }

  el.innerHTML = html;

  // 체크된 게 있으면 액션 버튼 표시
  updateSessionActions();
}

function toggleSessionCheck(id, cb){
  if(cb.checked) selectedSessions.add(id);
  else selectedSessions.delete(id);
  updateSessionActions();
}

function updateSessionActions(){
  const el = document.getElementById('sessionActions');
  if(selectedSessions.size === 0){ el.style.display='none'; return; }
  // 체크된 것 중 보관/일반 구분
  let hasArchived=false, hasNormal=false;
  selectedSessions.forEach(id=>{
    if(sessions[id]?.archived) hasArchived=true;
    else hasNormal=true;
  });
  let btns = '';
  if(hasNormal) btns += `<button onclick="archiveSelectedSessions()">📦 보관</button>`;
  if(hasArchived) btns += `<button onclick="restoreSelectedSessions()">↩ 복원</button>`;
  btns += `<button class="danger" onclick="deleteSelectedSessions()">🗑️ 삭제</button>`;
  btns += `<button onclick="clearSessionSelection()">취소</button>`;
  el.innerHTML = btns;
  el.style.display = 'flex';
}

function clearSessionSelection(){
  selectedSessions.clear();
  renderSessionList();
}

function deleteSelectedSessions(){
  if(selectedSessions.size===0) return;
  const count = selectedSessions.size;
  if(!confirm(count + '개 세션을 삭제할까요?')) return;
  let deletedCurrent = false;
  selectedSessions.forEach(id=>{
    if(id===currentSessionId) deletedCurrent = true;
    delete sessions[id];
  });
  selectedSessions.clear();
  saveSessions();
  if(deletedCurrent){
    // createNewSession() 호출 시 saveCurrentSession()이 삭제된 세션을 복원하므로
    // currentSessionId를 먼저 초기화한 후 새 세션 생성
    currentSessionId = null;
    createNewSession();
  } else renderSessionList();
}

function archiveSelectedSessions(){
  if(selectedSessions.size===0) return;
  let archivedCurrent = false;
  selectedSessions.forEach(id=>{
    if(sessions[id]){
      sessions[id].archived = true;
      if(id===currentSessionId) archivedCurrent = true;
    }
  });
  selectedSessions.clear();
  saveSessions();
  if(archivedCurrent){
    currentSessionId = null;
    createNewSession();
  } else renderSessionList();
}

// ===== 사이드바 전체 접기/펼치기 =====
function toggleSidebar(){
  const sb = document.getElementById('sidebar');
  const btn = document.getElementById('sidebarToggle');
  const chatBox = document.querySelector('.chat-box-fixed');
  sb.classList.toggle('collapsed');
  document.body.classList.toggle('sb-collapsed');
  if(sb.classList.contains('collapsed')){
    btn.textContent = '▶';
    btn.title = '사이드바 펼치기';
    if(chatBox) chatBox.style.left = '48px';
  } else {
    btn.textContent = '◀';
    btn.title = '사이드바 접기';
    if(chatBox) chatBox.style.left = '250px';
  }
  // localStorage에 상태 저장
  localStorage.setItem('domos_sidebar_collapsed', sb.classList.contains('collapsed'));
}
// 저장된 상태 복원
(function(){
  const saved = localStorage.getItem('domos_sidebar_collapsed');
  if(saved === 'true'){
    const sb = document.getElementById('sidebar');
    const btn = document.getElementById('sidebarToggle');
    const chatBox = document.querySelector('.chat-box-fixed');
    if(sb){ sb.classList.add('collapsed'); }
    if(btn){ btn.textContent='▶'; btn.title='사이드바 펼치기'; }
    if(chatBox){ chatBox.style.left='48px'; }
    document.body.classList.add('sb-collapsed');
  }
})();
function createNewSession(){
  // Save current before switching
  saveCurrentSession();
  currentSessionId = 'sess_'+Date.now();
  history=[];
  selSkills=[];
  autoLoadedSkills=[];
  dismissedAutoSkills.clear();
  document.getElementById('msgs').innerHTML='';
  document.getElementById('writingStyle').value='';
  document.getElementById('systemPromptInput').value='';
  activeStyleId=null;
  currentPromptId=null;
  selFormat='auto';
  formatManualOverride=false;
  styleManualOverride=false;
  effort=2;
  document.getElementById('effortSlider').value=2;
  document.querySelectorAll('.fmt-btn').forEach(b=>{b.classList.toggle('selected',b.dataset.f==='auto');});
  // 업로드된 파일/CSV 초기화 (이전 세션 데이터 잔류 방지)
  fetch('/api/clear_files', {method:'POST'});
  fetch('/api/clear_csv', {method:'POST'});
  uploadedFilesList = [];
  renderFileList();
  csvLoaded = false;
  csvFilename = '';
  document.getElementById('csvInfoPanel').style.display = 'none';
  chatPendingFiles = [];
  const chatFileWrap = document.getElementById('chatAttachPreview');
  if(chatFileWrap){ chatFileWrap.innerHTML = ''; chatFileWrap.style.display='none'; }
  renderSkills();
  updateLoaded();
  renderStyleChips();
  renderPromptChips();
  updateSpBadge();
  updateEffort();
  sessions[currentSessionId] = { id:currentSessionId, name:'새 세션', history:[], msgsHtml:'', updatedAt:Date.now() };
  saveSessions();
  renderSessionList();
}
function loadSession(id){
  if(id===currentSessionId) return;
  saveCurrentSession();
  const s=sessions[id];
  if(!s) return;
  currentSessionId=id;
  history=s.history||[];
  document.getElementById('msgs').innerHTML=s.msgsHtml||'';
  if(s.writingStyle){ document.getElementById('writingStyle').value=s.writingStyle; syncStyleDropToSidebar(); }
  if(s.writingStyleId){ activeStyleId=s.writingStyleId; renderStyleChips(); }
  styleManualOverride = s.styleManualOverride !== undefined ? s.styleManualOverride : !!s.writingStyle;
  if(s.systemPrompt) document.getElementById('systemPromptInput').value=s.systemPrompt;
  if(s.thinkMode!==undefined) document.getElementById('thinkToggle').checked=s.thinkMode;
  if(s.selFormat){ selFormat=s.selFormat; formatManualOverride=(s.formatManualOverride!==undefined ? s.formatManualOverride : selFormat!=='auto'); document.querySelectorAll('.fmt-btn').forEach(b=>{b.classList.toggle('selected',b.dataset.f===selFormat);}); syncFormatDropToChat(); }
  if(s.effort!==undefined){ effort=s.effort; document.getElementById('effortSlider').value=effort; updateEffort(); }
  // 저장된 수동 스킬 복원
  if(s.selSkills && s.selSkills.length > 0){ selSkills = [...s.selSkills]; }
  else { selSkills = []; }
  // 대화 히스토리 기반 자동 스킬 프리로드
  autoLoadedSkills = [];
  if(autoSkillMode && history.length > 0){
    const lastUser = [...history].reverse().find(m=>m.role==='user');
    if(lastUser){
      fetch('/api/auto-skills',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:lastUser.content,max_skills:5,history:history})})
      .then(r=>r.json()).then(d=>{
        if(d.skills && d.skills.length > 0){
          const manualSet = new Set(selSkills);
          autoLoadedSkills = d.skills.map(s=>s.id).filter(id=>!manualSet.has(id));
          renderSkills();
          updateLoaded();
        }
      }).catch(()=>{});
    }
  }
  renderSkills();
  updateLoaded();
  renderSessionList();
  // 세션 복원 후 차트 재렌더링
  document.querySelectorAll('#msgs canvas[data-chart-json]').forEach(cv=>{
    try{renderChartBlock(cv.id, b2u(cv.dataset.chartJson));}
    catch(e){cv.parentElement.innerHTML='<p style="color:red;padding:12px;">차트 렌더링 실패: '+e.message+'</p>';}
  });
}
function deleteSession(id){
  delete sessions[id];
  saveSessions();
  if(id===currentSessionId) createNewSession();
  else renderSessionList();
}
function restoreArchivedSession(id){
  if(sessions[id]){ sessions[id].archived=false; saveSessions(); renderSessionList(); }
}
function restoreSelectedSessions(){
  if(selectedSessions.size===0) return;
  selectedSessions.forEach(id=>{
    if(sessions[id]) sessions[id].archived=false;
  });
  selectedSessions.clear();
  saveSessions();
  renderSessionList();
}
function newSession(){ createNewSession(); }

// ===== System Prompt =====
function toggleSysPrompt(){
  const body = document.getElementById('spBody');
  const arrow = document.getElementById('spArrow');
  body.classList.toggle('show');
  arrow.classList.toggle('open');
  if(body.classList.contains('show')){
    arrow.textContent='▼';
  } else {
    arrow.textContent='▶';
    updateSpBadge();
  }
}
function updateSpBadge(){
  const badge = document.getElementById('spPreviewBadge');
  const val = document.getElementById('systemPromptInput').value.trim();
  if(val){
    badge.textContent = val.length + '자 커스텀';
    badge.style.background='#fef3c7';
    badge.style.color='#d97706';
  } else {
    badge.textContent = '기본값';
    badge.style.background='#eef2ff';
    badge.style.color='#6366f1';
  }
}
function resetSysPrompt(){
  document.getElementById('systemPromptInput').value='';
  currentPromptId = null;
  updateSpBadge();
  renderPromptChips();
}
document.getElementById('systemPromptInput').addEventListener('input', function(){
  updateSpBadge();
  // 수동 편집하면 active 해제
  if(currentPromptId){
    currentPromptId = null;
    renderPromptChips();
  }
});

// ===== 시스템 프롬프트 프리셋/저장 관리 =====
let currentPromptId = null;
let promptList = [];

async function loadPromptList(){
  try {
    const r = await fetch('/api/prompts');
    const d = await r.json();
    promptList = d.prompts || [];
    renderPromptChips();
  } catch(e){ console.error('프롬프트 목록 로드 실패:', e); }
}

function renderPromptChips(){
  const container = document.getElementById('promptPresets');
  if(!container) return;
  container.innerHTML = '';
  promptList.forEach(p => {
    const chip = document.createElement('span');
    chip.className = 'prompt-chip' + (p.type==='saved'?' saved':'') + (currentPromptId===p.id?' active':'');
    chip.innerHTML = p.icon + ' ' + p.name;
    chip.title = p.preview;
    chip.onclick = () => selectPrompt(p.id);
    container.appendChild(chip);
  });
  // 삭제 버튼 표시/숨김
  const delBtn = document.getElementById('btnDeletePrompt');
  if(delBtn) delBtn.style.display = (currentPromptId && currentPromptId.startsWith('saved:')) ? '' : 'none';
}

async function selectPrompt(pid){
  try {
    const r = await fetch('/api/prompts/' + encodeURIComponent(pid));
    const d = await r.json();
    if(d.content){
      document.getElementById('systemPromptInput').value = d.content;
      currentPromptId = pid;
      updateSpBadge();
      renderPromptChips();
    }
  } catch(e){ console.error('프롬프트 로드 실패:', e); }
}

async function saveCurrentPrompt(){
  const content = document.getElementById('systemPromptInput').value.trim();
  if(!content){ alert('저장할 내용이 없습니다.'); return; }
  const name = prompt('프롬프트 이름을 입력하세요:', currentPromptId?.startsWith('saved:') ? currentPromptId.slice(6) : '');
  if(!name) return;
  try {
    const r = await fetch('/api/prompts', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name, content})
    });
    const d = await r.json();
    if(d.success){
      currentPromptId = d.id;
      const status = document.getElementById('promptSaveStatus');
      status.style.display='inline'; setTimeout(()=>status.style.display='none', 2000);
      await loadPromptList();
    } else { alert(d.error || '저장 실패'); }
  } catch(e){ alert('저장 오류: ' + e); }
}

async function deleteCurrentPrompt(){
  if(!currentPromptId || !currentPromptId.startsWith('saved:')) return;
  const name = currentPromptId.slice(6);
  if(!confirm('"' + name + '" 프롬프트를 삭제하시겠습니까?')) return;
  try {
    const r = await fetch('/api/prompts/' + encodeURIComponent(currentPromptId), {method:'DELETE'});
    const d = await r.json();
    if(d.success){
      document.getElementById('systemPromptInput').value = '';
      currentPromptId = null;
      updateSpBadge();
      await loadPromptList();
    }
  } catch(e){ alert('삭제 오류: ' + e); }
}

// 페이지 로드시 프롬프트 목록 불러오기
loadPromptList();

// ===== 작성 스타일 프리셋 =====
const STYLE_PRESETS = [
  {id:'concise',    icon:'⚡', name:'간결', value:'간결하고 핵심만. 불필요한 설명 생략.'},
  {id:'detailed',   icon:'📖', name:'상세', value:'상세하고 친절하게. 원리, 배경, 예시 포함.'},
  {id:'practical',  icon:'🔨', name:'실용적', value:'실전에서 바로 활용 가능하게. 핵심 요점과 구체적 방법 위주. 선택된 출력형식에 맞춰 답변하세요.'},
  {id:'academic',   icon:'🎓', name:'학술', value:'학술적 톤. 정확한 용어, 인용, 근거 제시.'},
  {id:'comment-kr', icon:'💬', name:'한글주석', value:'코드 주석을 상세한 한국어로. 변수명은 영어 유지.'},
  {id:'senior-dev', icon:'👨‍💻', name:'시니어', value:'시니어 개발자 관점. 왜 이렇게 하는지, 대안은 뭔지, 주의점 포함.'},
  {id:'debug',      icon:'🐛', name:'디버그', value:'에러 원인 분석 중심. traceback 해석, 재현 조건, 해결책 순서.'},
  {id:'data-story', icon:'📊', name:'데이터', value:'데이터 스토리텔링. 숫자→의미→액션 순서로 해석.'},
];
let activeStyleId = null;

function renderStyleChips(){
  const container = document.getElementById('styleChips');
  if(!container) return;
  container.innerHTML = '';
  // 🤖 자동 칩 (맨 앞)
  const autoChip = document.createElement('span');
  autoChip.className = 'prompt-chip' + (!styleManualOverride ? ' active' : '');
  autoChip.innerHTML = '🤖 자동';
  autoChip.title = '채팅 내용에 따라 자동 선택';
  autoChip.onclick = () => {
    activeStyleId = null;
    styleManualOverride = false;
    document.getElementById('writingStyle').value = '';
    renderStyleChips();
  };
  container.appendChild(autoChip);
  STYLE_PRESETS.forEach(s => {
    const chip = document.createElement('span');
    chip.className = 'prompt-chip' + (activeStyleId===s.id ? ' active' : '');
    chip.innerHTML = s.icon + ' ' + s.name;
    chip.title = s.value;
    chip.onclick = () => {
      if(activeStyleId === s.id){
        activeStyleId = null;
        styleManualOverride = false;
        document.getElementById('writingStyle').value = '';
      } else {
        activeStyleId = s.id;
        styleManualOverride = true;
        document.getElementById('writingStyle').value = s.value;
      }
      renderStyleChips();
    };
    container.appendChild(chip);
  });
}
renderStyleChips();

// ===== 채팅창 드롭다운 ↔ 사이드바 동기화 =====
function applyChatStyle(val){
  document.getElementById('writingStyle').value = val;
  const match = STYLE_PRESETS.find(s => s.value === val);
  activeStyleId = match ? match.id : null;
  styleManualOverride = !!val;
  renderStyleChips();
  if(val) syncStyleDropToSidebar();
}
function applyChatFormat(val){
  selFormat = val;
  formatManualOverride = (val !== 'auto');
  // 사이드바 fmt-btn도 동기화
  document.querySelectorAll('.fmt-btn').forEach(b=>{b.classList.toggle('selected',b.dataset.f===val);});
}
function syncStyleDropToSidebar(){
  // 사이드바 변경 → 채팅창 드롭다운 동기화
  const drop = document.getElementById('chatStyleDrop');
  if(!drop) return;
  const val = document.getElementById('writingStyle').value.trim();
  const opts = drop.options;
  let found = false;
  for(let i=0;i<opts.length;i++){
    if(opts[i].value === val){ drop.selectedIndex = i; found = true; break; }
  }
  if(!found) drop.selectedIndex = 0;
}
function syncFormatDropToChat(){
  const drop = document.getElementById('chatFormatDrop');
  if(!drop) return;
  const opts = drop.options;
  for(let i=0;i<opts.length;i++){
    if(opts[i].value === selFormat){ drop.selectedIndex = i; break; }
  }
}

document.getElementById('writingStyle').addEventListener('input', function(){
  // 수동 편집하면 active 해제
  const val = this.value.trim();
  const match = STYLE_PRESETS.find(s => s.value === val);
  activeStyleId = match ? match.id : null;
  renderStyleChips();
});

// ===== Skill Detail Panel =====
let currentSkillDetail = null;

async function openSkillDetail(skillId){
  const overlay = document.getElementById('skillDetailOverlay');
  overlay.style.display = 'flex';
  document.getElementById('sdpTitle').textContent = skillId;
  document.getElementById('sdpBody').innerHTML = '⏳ 로딩중...';

  try{
    const resp = await fetch(`/api/skill/${skillId}`);
    currentSkillDetail = await resp.json();
    renderSdpTabs('scripts');
  }catch(e){
    document.getElementById('sdpBody').innerHTML = `❌ 로딩 실패: ${e.message}`;
  }
}

function closeSkillDetail(){
  document.getElementById('skillDetailOverlay').style.display = 'none';
  currentSkillDetail = null;
}

function renderSdpTabs(active){
  if(!currentSkillDetail) return;
  const d = currentSkillDetail;
  const tabsEl = document.getElementById('sdpTabs');
  const tabs = [
    {id:'scripts', label:`🐍 스크립트 (${(d.scripts||[]).length})`, show: true},
    {id:'references', label:`📄 레퍼런스 (${(d.references||[]).length})`, show: true},
    {id:'assets', label:`📦 에셋 (${(d.assets||[]).length})`, show: (d.assets||[]).length > 0},
    {id:'skillmd', label:'📋 SKILL.md', show: true},
  ];
  tabsEl.innerHTML = '';
  tabs.forEach(t=>{
    if(!t.show) return;
    const el = document.createElement('div');
    el.className = 'sdp-tab' + (active===t.id?' active':'');
    el.textContent = t.label;
    el.onclick = ()=> renderSdpTabs(t.id);
    tabsEl.appendChild(el);
  });
  renderSdpContent(active);
}

function renderSdpContent(tab){
  const body = document.getElementById('sdpBody');
  const d = currentSkillDetail;

  if(tab === 'skillmd'){
    body.innerHTML = `<div class="sdp-code">${esc(d.content || '(내용 없음)')}</div>
      <button class="sdp-btn inject" style="margin-top:10px" onclick="injectToChat('skillmd','')">💬 채팅에 SKILL.md 주입</button>`;
    return;
  }

  const files = d[tab] || [];
  if(files.length === 0){
    body.innerHTML = '<div style="text-align:center;color:#999;padding:40px">파일 없음</div>';
    return;
  }

  const isPy = tab === 'scripts';
  let html = '<ul class="sdp-file-list">';
  files.forEach(f=>{
    const icon = f.name.endsWith('.py') ? '🐍' : f.name.endsWith('.md') ? '📄' : '📎';
    const sizeKb = (f.size / 1024).toFixed(1);
    html += `<li class="sdp-file-item" onclick="viewSkillFile('${d.name}','${f.path}')">
      <span class="sdp-file-icon">${icon}</span>
      <span class="sdp-file-name">${esc(f.name)}</span>
      <span class="sdp-file-size">${sizeKb} KB</span>
      <div class="sdp-file-actions">
        <button class="sdp-btn" onclick="event.stopPropagation();viewSkillFile('${d.name}','${f.path}')">👁️ 보기</button>
        <button class="sdp-btn inject" onclick="event.stopPropagation();injectToChat('${tab}','${f.path}')">💬 주입</button>
        ${isPy && f.name.endsWith('.py') ? `<button class="sdp-btn run" onclick="event.stopPropagation();runSkillScript('${d.name}','${f.path}')">▶ 실행</button>` : ''}
      </div>
    </li>`;
  });
  html += '</ul><div id="sdpFileContent"></div>';
  body.innerHTML = html;
}

async function viewSkillFile(skillId, filePath){
  const container = document.getElementById('sdpFileContent');
  if(!container) return;
  container.innerHTML = '⏳ 로딩중...';
  try{
    const resp = await fetch(`/api/skill/${skillId}/file?path=${encodeURIComponent(filePath)}`);
    const data = await resp.json();
    if(data.error){
      container.innerHTML = `<div class="sdp-result err">${esc(data.error)}</div>`;
    } else {
      container.innerHTML = `<div style="font-size:12px;font-weight:700;margin:10px 0 6px">${esc(data.name)} (${(data.size/1024).toFixed(1)} KB)</div>
        <div class="sdp-code">${esc(data.content)}</div>`;
    }
  }catch(e){
    container.innerHTML = `<div class="sdp-result err">오류: ${e.message}</div>`;
  }
}

async function runSkillScript(skillId, scriptPath){
  const container = document.getElementById('sdpFileContent');
  if(!container) return;
  container.innerHTML = '⏳ 스크립트 실행 중... (최대 60초)';
  try{
    const resp = await fetch(`/api/skill/${skillId}/run`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({script: scriptPath})
    });
    const data = await resp.json();
    if(data.error){
      container.innerHTML = `<div class="sdp-result err">❌ ${esc(data.error)}</div>`;
    } else {
      const cls = data.returncode === 0 ? 'ok' : 'err';
      let output = '';
      if(data.stdout) output += `=== stdout ===\n${data.stdout}\n`;
      if(data.stderr) output += `=== stderr ===\n${data.stderr}\n`;
      output += `\n[종료 코드: ${data.returncode}]`;
      container.innerHTML = `<div style="font-size:12px;font-weight:700;margin:10px 0 6px">▶ ${esc(scriptPath)} 실행 결과</div>
        <div class="sdp-result ${cls}">${esc(output)}</div>`;
    }
  }catch(e){
    container.innerHTML = `<div class="sdp-result err">실행 오류: ${e.message}</div>`;
  }
}

async function injectToChat(type, filePath){
  // 파일 내용을 채팅 입력에 주입
  const input = document.getElementById('input');
  const skillId = currentSkillDetail ? currentSkillDetail.name : '';

  if(type === 'skillmd'){
    input.value += `[${skillId} SKILL.md 참조 중]\n`;
    closeSkillDetail();
    return;
  }

  try{
    const resp = await fetch(`/api/skill/${skillId}/file?path=${encodeURIComponent(filePath)}`);
    const data = await resp.json();
    if(data.content){
      const preview = data.content.length > 500 ? data.content.substring(0,500) + '...' : data.content;
      input.value += `\n--- ${data.name} ---\n${preview}\n---\n`;
    }
  }catch(e){ console.warn('파일 주입 실패:', e); }
  closeSkillDetail();
  input.focus();
}

// ===== CSV Upload =====
let csvLoaded = false;
let csvFilename = '';

// handleUnifiedDrop/Select → 채팅 입력 📎 첨부로 대체됨
async function uploadCsvFile(file){
  const formData = new FormData();
  formData.append('file', file);

  try{
    const resp = await fetch('/api/upload_csv', {method:'POST', body:formData});
    const data = await resp.json();
    if(data.error) return;

    csvLoaded = true;
    csvFilename = data.filename;

    // 미리보기 테이블 생성
    let tableHtml = '<table><tr>' + data.headers.map(h=>`<th>${esc(h)}</th>`).join('') + '</tr>';
    if(data.sample_rows){
      data.sample_rows.forEach(row=>{
        tableHtml += '<tr>' + data.headers.map(h=>`<td>${esc(row[h]||'')}</td>`).join('') + '</tr>';
      });
    }
    tableHtml += '</table>';
    if(data.rows > 3) tableHtml += `<div style="text-align:center;color:#999;font-size:10px;margin-top:4px">... 총 ${data.rows}행</div>`;

    const panel = document.getElementById('csvInfoPanel');
    panel.style.display = 'block';
    panel.innerHTML = `
      <div class="csv-info">
        <div class="fname">📊 ${esc(data.filename)}</div>
        <div class="fstats">${data.rows}행 × ${data.cols}열</div>
        <button class="fremove" onclick="removeCsv()">✕ 제거</button>
        <div class="csv-preview">${tableHtml}</div>
      </div>`;

  }catch(e){}
}
function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ===== Unicode-safe Base64 유틸 (btoa/atob는 Latin1만 지원) =====
function u2b(str){
  var bytes=new TextEncoder().encode(str);
  var bin='';
  for(var i=0;i<bytes.length;i++) bin+=String.fromCharCode(bytes[i]);
  return btoa(bin);
}
function b2u(b64){
  var bin=atob(b64);
  var bytes=new Uint8Array(bin.length);
  for(var i=0;i<bin.length;i++) bytes[i]=bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

// ===== Draw.io 기능 =====
function decodeDrawioXml(btn){
  return b2u(btn.dataset.xml);
}

// Draw.io XML 자동 수정 (LLM이 생성한 XML의 흔한 오류 교정)
function sanitizeDrawioXml(xml){
  // 허용되는 HTML 태그 목록 (Draw.io value 속성 안에서 허용)
  const allowedHtmlTags = ['br','br/','br /','font','b','i','u','sub','sup','hr','span','div','p','strong','em'];
  const allowedPattern = allowedHtmlTags.map(t => t.replace('/','\\/').replace(' ','')).join('|');

  // value/label 속성값 안의 이스케이프 수정 함수
  function fixAttrValue(val){
    // Step 1: & 이스케이프 (이미 된 건 건드리지 않음)
    let s = val.replace(/&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)/g, '&amp;');
    // Step 2: 모든 < > 를 임시 치환 후, 허용된 HTML 태그만 복원
    s = s.replace(/</g, '\x00LT\x00').replace(/>/g, '\x00GT\x00');
    // 허용된 여는 태그 복원: <font ...>, <br>, <br/>, <b>, etc.
    for(const tag of allowedHtmlTags){
      const base = tag.replace(/[\s\/]/g,'');
      // 여는 태그 (속성 포함): <font style="...">
      const openRe = new RegExp('\x00LT\x00(' + base + ')(\\s[^\\x00]*?)?\x00GT\x00', 'gi');
      s = s.replace(openRe, '<$1$2>');
      // 닫는 태그: </font>
      const closeRe = new RegExp('\x00LT\x00/' + base + '\x00GT\x00', 'gi');
      s = s.replace(closeRe, '</' + base + '>');
    }
    // self-closing: <br/>, <br />, <hr/>
    s = s.replace(/\x00LT\x00(br|hr)\s*\/?\x00GT\x00/gi, '<$1/>');
    // 남은 \x00LT\x00, \x00GT\x00 는 진짜 이스케이프 대상
    s = s.replace(/\x00LT\x00/g, '&lt;').replace(/\x00GT\x00/g, '&gt;');
    return s;
  }

  // 1) value="..." 속성 수정
  xml = xml.replace(/(value|label)="([^"]*)"/g, (m, attr, val) => {
    const fixed = fixAttrValue(val);
    return fixed !== val ? `${attr}="${fixed}"` : m;
  });

  // 2) 중복 속성 제거: style="..." style="..." → 첫 번째만
  xml = xml.replace(/(\w+="[^"]*")\s+\1/g, '$1');

  // 3) 중첩된 mxCell 분리 (mxCell 안에 mxCell이 있으면 평탄화)
  //    주의: mxCell > mxGeometry 관계는 정상이므로 건드리지 않음
  //    (기존 regex가 자식 mxGeometry 있는 mxCell을 self-closing으로 바꾸는 버그 수정)

  // 4) mxfile 래퍼가 없으면 추가
  if (!xml.includes('<mxfile') && xml.includes('<mxGraphModel')) {
    xml = `<mxfile host="app.diagrams.net" type="device">\n<diagram id="d1" name="Page-1">\n${xml}\n</diagram>\n</mxfile>`;
  }
  if (!xml.includes('<mxfile') && xml.includes('<mxCell')) {
    xml = `<mxfile host="app.diagrams.net" type="device">\n<diagram id="d1" name="Page-1">\n<mxGraphModel dx="1422" dy="762" grid="1" pageWidth="1169" pageHeight="827">\n<root>\n<mxCell id="0"/>\n<mxCell id="1" parent="0"/>\n${xml}\n</root>\n</mxGraphModel>\n</diagram>\n</mxfile>`;
  }

  // 5) 최종 검증: DOMParser로 파싱 시도, 실패하면 단계별 수정
  try {
    let parser = new DOMParser();
    let doc = parser.parseFromString(xml, 'application/xml');
    if (doc.querySelector('parsererror')) {
      // 5a) 속성값 안의 이스케이프 안 된 < > 수정
      xml = xml.replace(/="([^"]*)"/g, (m, val) => {
        if (val.includes('<') && !val.match(/^[^<]*<(mxGeometry|mxPoint|Array)/)) {
          const forced = val.replace(/</g, '&lt;').replace(/>/g, '&gt;');
          return `="${forced}"`;
        }
        return m;
      });

      // 5b) 재검증 - 여전히 실패하면 구조 수정 시도
      doc = parser.parseFromString(xml, 'application/xml');
      if (doc.querySelector('parsererror')) {
        // mxCell 구조 재구성: 모든 mxCell/mxGeometry를 추출해서 올바른 구조로 재조립
        const cellRe = /<mxCell\s[^>]*?\/>/g;
        const cellWithChildRe = /<mxCell\s[^>]*?>[\s\S]*?<\/mxCell>/g;
        const geoRe = /<mxGeometry\s[^>]*?\/>/g;
        const geoWithChildRe = /<mxGeometry\s[^>]*?>[\s\S]*?<\/mxGeometry>/g;

        // mxfile/diagram/mxGraphModel 래퍼 추출
        const wrapMatch = xml.match(/([\s\S]*?<root>)([\s\S]*?)(<\/root>[\s\S]*)/);
        if (wrapMatch) {
          const [, header, body, footer] = wrapMatch;
          // body에서 모든 완전한 요소 추출 (self-closing + with-children 모두)
          const elements = [];
          const allRe = /<(mxCell|mxGeometry|UserObject|Array|mxPoint)\s[^>]*?(?:\/>|>[\s\S]*?<\/\1>)/g;
          // 먼저 mxCell+children 패턴 추출
          const cellFullRe = /<mxCell\s[^>]*?>(?:\s*<mxGeometry[\s\S]*?(?:\/>|<\/mxGeometry>))*\s*<\/mxCell>/g;
          const cellSelfRe = /<mxCell\s[^>]*?\/>/g;
          let match;
          while ((match = cellFullRe.exec(body)) !== null) elements.push(match[0]);
          while ((match = cellSelfRe.exec(body)) !== null) {
            if (!elements.some(e => e.includes(match[0]))) elements.push(match[0]);
          }
          // UserObject 패턴도 포함
          const uoRe = /<UserObject[\s\S]*?<\/UserObject>/g;
          while ((match = uoRe.exec(body)) !== null) elements.push(match[0]);

          if (elements.length > 0) {
            xml = header + '\n' + elements.join('\n') + '\n' + footer;
          }
        }
      }
    }
  } catch(e) {}

  return xml;
}

function copyDrawioXml(btn){
  const xml = sanitizeDrawioXml(decodeDrawioXml(btn));
  navigator.clipboard.writeText(xml).then(()=>{
    const orig = btn.textContent;
    btn.textContent = '✓ 복사됨';
    setTimeout(()=> btn.textContent = orig, 1500);
  });
}
function downloadDrawio(btn){
  const xml = sanitizeDrawioXml(decodeDrawioXml(btn));
  const blob = new Blob([xml], {type:'application/xml'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'diagram.drawio';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
function openInDrawio(btn){
  const xml = sanitizeDrawioXml(decodeDrawioXml(btn));
  const encoded = encodeURIComponent(xml);
  window.open('https://app.diagrams.net/#R' + encoded, '_blank');
}

// ===== Chart.js 인터랙티브 차트 =====
const _chartInstances = {};

function renderChartBlock(canvasId, jsonStr) {
  // jsonStr은 이미 b2u()로 유니코드 디코딩된 상태
  let raw = jsonStr;
  // 한 줄 주석 제거
  raw = raw.replace(/\/\/[^\n]*/g, '');
  let config;
  try {
    config = JSON.parse(raw);
  } catch(e) {
    // JSON5 스타일 허용: trailing comma, 작은따옴표
    raw = raw.replace(/,\s*([\]}])/g, '$1').replace(/'/g, '"');
    config = JSON.parse(raw);
  }

  // 기본값 보강
  if (!config.type) config.type = 'bar';
  if (!config.options) config.options = {};
  if (!config.options.responsive) config.options.responsive = true;
  if (!config.options.maintainAspectRatio) config.options.maintainAspectRatio = true;
  if (!config.options.plugins) config.options.plugins = {};

  // 한글 폰트 기본 설정
  if (!config.options.plugins.legend) config.options.plugins.legend = {};

  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // 이전 인스턴스 정리
  if (_chartInstances[canvasId]) {
    _chartInstances[canvasId].destroy();
  }
  _chartInstances[canvasId] = new Chart(ctx, config);
}

function downloadChartPng(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const url = canvas.toDataURL('image/png');
  const a = document.createElement('a');
  a.href = url;
  a.download = 'chart.png';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function copyChartJson(btn) {
  const b64 = btn.getAttribute('data-chart');
  const json = b2u(b64);
  navigator.clipboard.writeText(json).then(() => {
    const orig = btn.textContent;
    btn.textContent = '✓ 복사됨';
    setTimeout(() => btn.textContent = orig, 1500);
  });
}

// ===== MD 다운로드/복사/미리보기 =====
function decodeMdB64(btn){
  const b64 = btn.getAttribute('data-md');
  return b2u(b64);
}
function downloadMarkdown(btn){
  const md = decodeMdB64(btn);
  // 제목에서 파일명 추출
  const titleMatch = md.match(/^#\s+(.+)$/m);
  let filename = 'report.md';
  if(titleMatch){
    filename = titleMatch[1].replace(/[^\w가-힣\s-]/g,'').trim().replace(/\s+/g,'_').substring(0,50) + '.md';
  }
  const blob = new Blob([md], {type:'text/markdown;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
function copyMdContent(btn){
  const md = decodeMdB64(btn);
  navigator.clipboard.writeText(md).then(()=>{
    const orig = btn.textContent;
    btn.textContent = '✅ 복사됨';
    setTimeout(()=>btn.textContent=orig, 1500);
  });
}
function renderMdPreview(escapedMd){
  // HTML escaped 마크다운을 간단히 렌더링 (미리보기용)
  let s = escapedMd;
  // 헤딩
  s = s.replace(/^######\s+(.+)$/gm, '<h6>$1</h6>');
  s = s.replace(/^#####\s+(.+)$/gm, '<h5>$1</h5>');
  s = s.replace(/^####\s+(.+)$/gm, '<h4>$1</h4>');
  s = s.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
  s = s.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>');
  s = s.replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');
  // bold / italic
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // 체크리스트
  s = s.replace(/^- \[x\]\s+(.+)$/gm, '<div>☑️ $1</div>');
  s = s.replace(/^- \[ \]\s+(.+)$/gm, '<div>⬜ $1</div>');
  // 리스트
  s = s.replace(/^- (.+)$/gm, '<li>$1</li>');
  s = s.replace(/^(\d+)\.\s+(.+)$/gm, '<li>$2</li>');
  // 수평선
  s = s.replace(/^---+$/gm, '<hr>');
  // 블록쿼트
  s = s.replace(/^&gt;\s+(.+)$/gm, '<blockquote>$1</blockquote>');
  // 인라인 코드
  s = s.replace(/`([^`]+)`/g, '<code style="background:#f1f5f9;padding:1px 4px;border-radius:3px;font-size:12px">$1</code>');
  // 테이블 (간단 처리)
  s = s.replace(/((?:^\|.+\|[ ]*\n){2,})/gm, function(tbl){
    const rows = tbl.trim().split('\n').filter(r=>r.trim());
    if(rows.length < 2) return tbl;
    const parseRow = r => r.replace(/^\|/,'').replace(/\|$/,'').split('|').map(c=>c.trim());
    const hdr = parseRow(rows[0]);
    let startIdx = 1;
    if(rows[1] && /^[\s|:-]+$/.test(rows[1])) startIdx = 2;
    let html = '<table><thead><tr>' + hdr.map(h=>`<th>${h}</th>`).join('') + '</tr></thead><tbody>';
    for(let i=startIdx;i<rows.length;i++){
      const cells = parseRow(rows[i]);
      html += '<tr>' + cells.map(c=>`<td>${c}</td>`).join('') + '</tr>';
    }
    return html + '</tbody></table>';
  });
  // 줄바꿈
  s = s.replace(/\n/g, '<br>');
  return s;
}

// ===== 응답 전체를 MD 다운로드 가능하게 =====
function appendMdDownloadBar(rawContent){
  // 마지막 assistant 메시지에 MD 다운로드 바 추가
  const msgs = document.querySelectorAll('.msg.assistant');
  if(msgs.length === 0) return;
  const lastMsg = msgs[msgs.length - 1];
  const mdB64 = u2b(rawContent);
  const bar = document.createElement('div');
  bar.className = 'md-report-header';
  bar.style.marginTop = '10px';
  bar.style.borderRadius = '8px';
  bar.innerHTML = `<span>📋 마크다운 문서로 저장</span>
    <div class="md-report-actions">
      <button class="md-report-btn" onclick="copyMdContent(this)" data-md="${mdB64}">📋 복사</button>
      <button class="md-report-btn md-download" onclick="downloadMarkdown(this)" data-md="${mdB64}">📥 MD 다운로드</button>
    </div>`;
  lastMsg.appendChild(bar);
}

async function removeCsv(){
  await fetch('/api/clear_csv', {method:'POST'});
  csvLoaded = false;
  csvFilename = '';
  document.getElementById('csvInfoPanel').style.display = 'none';
}

// ===== 파일 관리 =====
let uploadedFilesList = [];

function renderFileList(){
  const panel = document.getElementById('fileListPanel');
  if(uploadedFilesList.length === 0){
    panel.style.display = 'none';
    return;
  }
  panel.style.display = 'block';
  let html = '<div class="uploaded-files-header"><span>📎 첨부 파일 ('+uploadedFilesList.length+')</span><button onclick="clearAllFiles()" style="background:none;border:none;color:#e74c3c;cursor:pointer;font-size:10px;padding:0">전체 제거</button></div>';
  uploadedFilesList.forEach((f,i)=>{
    const sizeStr = f.size > 1048576 ? (f.size/1048576).toFixed(1)+'MB' : (f.size/1024).toFixed(1)+'KB';
    const pendingCls = f.pending ? ' ufi-pending' : (f.drm ? ' ufi-drm' : '');
    const statusLabel = f.pending ? ' ⏳' : (f.drm ? ' 🔒' : '');
    const removeAction = f.pending
      ? `removeChatAttachByName('${esc(f.filename)}')`
      : `removeFile('${esc(f.filename)}',${i})`;
    const titleText = f.drm ? 'DRM 보호 파일 - 내용 분석 불가' : esc(f.preview||'');
    html += `<div class="uploaded-file-item${pendingCls}">
      <span class="ufi-icon">${f.icon}</span>
      <span class="ufi-name" title="${titleText}">${esc(f.filename)}${statusLabel}</span>
      <span class="ufi-size">${sizeStr}</span>
      <button class="ufi-remove" onclick="${removeAction}">✕</button>
    </div>`;
  });
  panel.innerHTML = html;
}

async function removeFile(filename, idx){
  await fetch('/api/remove_file', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filename})});
  uploadedFilesList.splice(idx, 1);
  renderFileList();
}

async function clearAllFiles(){
  await fetch('/api/clear_files', {method:'POST'});
  uploadedFilesList = [];
  renderFileList();
}

// ===== 채팅 입력 첨부파일 =====
let chatPendingFiles = [];  // [{file:File, name:string}]

// 파일 확장자 → 추천 스킬 매핑 (프론트엔드용)
const fileExtSkillHints = {
  'py':['agent-python-pro'], 'js':['agent-nextjs-developer'], 'ts':['agent-nextjs-developer'], 'tsx':['agent-nextjs-developer'],
  'csv':['exploratory-data-analysis','statistical-analysis'],
  'tsv':['exploratory-data-analysis'], 'xlsx':['exploratory-data-analysis'],
  'docx':['docx'], 'pdf':['pdf'], 'pptx':['pptx'],
  'drawio':['drawio-diagram'], 'xml':['drawio-diagram'],
  'md':['markdown-mermaid-writing'], 'html':['web-artifacts-builder'],
};

function handleChatFileSelect(files){
  let hintSkills = [];
  const iconMap = {py:'🐍',js:'📜',ts:'📘',tsx:'📘',jsx:'📜',html:'🌐',css:'🎨',json:'📋',
    yaml:'📋',yml:'📋',md:'📄',java:'☕',c:'⚙️',cpp:'⚙️',go:'🐹',rs:'🦀',sh:'💻',
    png:'🖼️',jpg:'🖼️',jpeg:'🖼️',gif:'🖼️',bmp:'🖼️',webp:'🖼️',svg:'🖼️',
    docx:'📄',xlsx:'📊',pdf:'📕',pptx:'📽️',csv:'📊',txt:'📝',sql:'🗃️',rb:'💎',php:'🐘',vue:'💚',r:'📈'};
  for(const f of files){
    chatPendingFiles.push({file:f, name:f.name});
    // 파일 확장자로 추천 스킬 수집
    const ext = f.name.split('.').pop().toLowerCase();
    if(fileExtSkillHints[ext]){
      hintSkills.push(...fileExtSkillHints[ext]);
    }
    // 즉시 사이드바에 pending 상태로 표시
    uploadedFilesList.push({
      filename: f.name, type: ext, ext: ext,
      size: f.size, icon: iconMap[ext]||'📎', preview: '', pending: true
    });
  }
  renderChatAttach();
  renderFileList();
  document.getElementById('chatFileInput').value = '';

  // 파일 기반 추천 스킬 프리로드 (사이드바 표시 + 전송 시 자동 포함)
  if(hintSkills.length > 0){
    const unique = [...new Set(hintSkills)].filter(id => !selSkills.includes(id));
    autoLoadedSkills = [...new Set([...autoLoadedSkills, ...unique])];
    renderSkills();
    updateLoaded();
  }
}

function removeChatAttach(idx){
  const removed = chatPendingFiles.splice(idx, 1);
  // 사이드바 pending 항목도 함께 제거
  if(removed.length > 0){
    const pi = uploadedFilesList.findIndex(f => f.pending && f.filename === removed[0].name);
    if(pi >= 0) uploadedFilesList.splice(pi, 1);
    renderFileList();
  }
  renderChatAttach();
}
function removeChatAttachByName(name){
  const ci = chatPendingFiles.findIndex(f => f.name === name);
  if(ci >= 0) chatPendingFiles.splice(ci, 1);
  const pi = uploadedFilesList.findIndex(f => f.pending && f.filename === name);
  if(pi >= 0) uploadedFilesList.splice(pi, 1);
  renderChatAttach();
  renderFileList();
}

// 입력창 드래그앤드롭 파일 첨부
(function(){
  const box = document.querySelector('.chat-box-fixed');
  if(!box) return;
  box.addEventListener('dragover', e=>{e.preventDefault();e.stopPropagation();box.style.borderTopColor='#6366f1';});
  box.addEventListener('dragleave', e=>{e.preventDefault();box.style.borderTopColor='';});
  box.addEventListener('drop', e=>{
    e.preventDefault();e.stopPropagation();box.style.borderTopColor='';
    if(e.dataTransfer.files.length > 0) handleChatFileSelect(e.dataTransfer.files);
  });
})();

// Ctrl+V / Cmd+V 이미지 붙여넣기 (스크린샷, 클립보드 이미지)
(function(){
  const input = document.getElementById('input');
  if(!input) return;
  input.addEventListener('paste', function(e){
    const items = e.clipboardData && e.clipboardData.items;
    if(!items) return;
    const imageFiles = [];
    for(let i=0; i<items.length; i++){
      if(items[i].type.startsWith('image/')){
        const blob = items[i].getAsFile();
        if(blob){
          // 파일명 생성: paste_날짜시간.확장자
          const ext = items[i].type.split('/')[1] || 'png';
          const now = new Date();
          const ts = now.getFullYear()+String(now.getMonth()+1).padStart(2,'0')+String(now.getDate()).padStart(2,'0')+'_'+String(now.getHours()).padStart(2,'0')+String(now.getMinutes()).padStart(2,'0')+String(now.getSeconds()).padStart(2,'0');
          const fname = 'paste_'+ts+'.'+ext.replace('jpeg','jpg');
          const file = new File([blob], fname, {type: items[i].type});
          imageFiles.push(file);
        }
      }
    }
    if(imageFiles.length > 0){
      e.preventDefault();  // 텍스트로 붙여넣기 방지
      handleChatFileSelect(imageFiles);
    }
  });
})();

function renderChatAttach(){
  const wrap = document.getElementById('chatAttachPreview');
  if(chatPendingFiles.length === 0){
    wrap.style.display = 'none';
    wrap.innerHTML = '';
    return;
  }
  wrap.style.display = 'flex';
  wrap.innerHTML = chatPendingFiles.map((f,i)=>
    `<span class="chat-attach-badge"><span class="fname">${esc(f.name)}</span><span class="remove-attach" onclick="removeChatAttach(${i})">✕</span></span>`
  ).join('');
}

async function uploadChatPendingFiles(){
  // 첨부된 파일들을 서버로 업로드 (CSV는 csv 엔드포인트, 나머지는 범용)
  // pending 항목을 실제 업로드 결과로 교체
  const results = [];
  for(const pf of chatPendingFiles){
    const ext = pf.name.split('.').pop().toLowerCase();
    const formData = new FormData();
    formData.append('file', pf.file);

    // pending 항목 찾기
    const pendingIdx = uploadedFilesList.findIndex(f => f.pending && f.filename === pf.name);

    // CSV/TSV → CSV 전용 엔드포인트 (데이터 분석용)
    if(['csv','tsv'].includes(ext) && !csvLoaded){
      try{
        await uploadCsvFile(pf.file);
        results.push(pf.name);
        // pending → 완료로 전환
        if(pendingIdx >= 0) uploadedFilesList[pendingIdx].pending = false;
      }catch(e){
        // 실패 시 pending 제거
        if(pendingIdx >= 0) uploadedFilesList.splice(pendingIdx, 1);
      }
      continue;
    }

    // 나머지 → 범용 파일 엔드포인트
    try{
      const resp = await fetch('/api/upload_file', {method:'POST', body:formData});
      const data = await resp.json();
      if(!data.error){
        // pending 항목을 실제 데이터로 교체
        const updated = {
          filename: data.filename, type: data.type, ext: data.ext,
          size: data.size, icon: data.icon, preview: data.preview,
          drm: !!data.drm_warning,
        };
        if(pendingIdx >= 0) uploadedFilesList[pendingIdx] = updated;
        else uploadedFilesList.push(updated);
        results.push(data.filename);
        // DRM 보호 파일이면 채팅에 안내 메시지 표시
        if(data.drm_warning){
          addMsg('assistant', `🔒 **DRM 보호 파일 감지**: ${data.filename}\n${data.drm_warning}\n\n파일은 첨부되었지만, DRM 때문에 내용 분석이 불가능합니다.`);
        }
      } else {
        // 업로드 실패 시 pending 제거 + 사용자에게 알림
        if(pendingIdx >= 0) uploadedFilesList.splice(pendingIdx, 1);
        addMsg('assistant', data.error);
      }
    }catch(e){
      if(pendingIdx >= 0) uploadedFilesList.splice(pendingIdx, 1);
    }
  }
  chatPendingFiles = [];
  renderChatAttach();
  renderFileList();
  return results;
}

// ===== 오른쪽 코드 어시스턴트 패널 =====
let rpFiles = [];  // [{name, path, content, ext, size}]
let rpSelectedMode = null;
let rpTreeData = null;  // 트리 구조 데이터
let rpSelectedFile = null;  // 현재 선택된 파일 인덱스
const rpModeLabels = {
  explain:'코드 설명', find_bugs:'버그 탐지', improve:'개선 제안',
  tests:'테스트 생성', docstring:'문서화', refactor:'리팩토링'
};
const rpModePrompts = {
  explain:'[EXPLAIN] 아래 코드를 분석해주세요.\n⚠️ 먼저 코드의 프로그래밍 언어와 프레임워크를 판별한 후, 해당 언어의 관점에서 분석하세요.\n1. 감지된 언어/프레임워크 명시\n2. 전체 목적과 아키텍처\n3. 주요 함수/클래스 구조\n4. 실행 흐름과 핵심 알고리즘\n5. 의존성 분석\n한국어로 설명해주세요.',
  find_bugs:'[FIND_BUGS] 아래 코드에서 버그를 찾아주세요.\n⚠️ 먼저 코드의 프로그래밍 언어를 판별하고, 해당 언어 특유의 버그 패턴을 중점 검토하세요.\n1. 감지된 언어 명시\n2. 언어 특화 버그 패턴 (Python: 인덴트/타입, Java: NPE/동시성, C#: async/dispose, JS: 타입강제/this바인딩)\n3. 로직 오류, 보안 취약점\n4. 심각도 🔴🟡🟢 표시 + 수정 코드 제시\n한국어로 분석해주세요.',
  improve:'[IMPROVE] 아래 코드의 개선점을 제안해주세요.\n⚠️ 먼저 코드의 프로그래밍 언어를 판별하고, 해당 언어의 베스트 프랙티스 기준으로 개선하세요.\n1. 감지된 언어 명시\n2. 언어별 관용적 표현 적용 (Python: PEP8/컴프리헨션, Java: Stream API, C#: LINQ, JS: ES6+)\n3. 가독성, 성능, 유지보수성 개선\n4. 개선 전/후 코드 비교\n한국어로 설명해주세요.',
  tests:'[TESTS] 아래 코드에 대한 테스트 코드를 생성해주세요.\n⚠️ 먼저 코드의 프로그래밍 언어를 판별하고, 해당 언어의 표준 테스트 프레임워크를 사용하세요.\n(Python→pytest, Java→JUnit5, C#→xUnit, JS/TS→Jest, Go→go test, Rust→cargo test)\n1. 감지된 언어/프레임워크 명시\n2. 정상/엣지/에러 케이스\n3. 즉시 실행 가능한 코드\n한국어 주석으로 설명해주세요.',
  docstring:'[DOCSTRING] 아래 코드에 문서화를 추가해주세요.\n⚠️ 먼저 코드의 프로그래밍 언어를 판별하고, 해당 언어의 표준 문서화 스타일을 적용하세요.\n(Python→Google/NumPy docstring, Java→Javadoc, C#→XML 주석, JS/TS→JSDoc)\n1. 감지된 언어 명시\n2. 함수/클래스별 문서화 (파라미터, 반환값, 예외, 예시)\n3. 핵심 로직 인라인 주석 (한국어)\n⚠️ 중요: 전체 파일 내용을 반복하지 말고, 문서화가 추가된 핵심 함수/클래스만 제시하세요.',
  refactor:'[REFACTOR] 아래 코드를 리팩토링해주세요.\n⚠️ 먼저 코드의 프로그래밍 언어를 판별하고, 해당 언어의 디자인 패턴과 관례에 맞게 리팩토링하세요.\n1. 감지된 언어 명시\n2. SOLID 원칙 적용\n3. 언어별 관용 패턴 (Python: 데코레이터/컨텍스트매니저, Java: Builder/DI, C#: async패턴)\n4. 리팩토링 전/후 비교\n한국어로 설명해주세요.'
};

// 코드 어시스턴트 전용 언어/도구 감지 매핑
const rpExtLangMap = {
  'py':'Python','js':'JavaScript','ts':'TypeScript','tsx':'TypeScript(React)','jsx':'JavaScript(React)',
  'java':'Java','c':'C','cpp':'C++','h':'C/C++ Header','cs':'C#',
  'go':'Go','rs':'Rust','rb':'Ruby','php':'PHP','swift':'Swift','kt':'Kotlin',
  'html':'HTML','css':'CSS','json':'JSON','xml':'XML','yaml':'YAML','yml':'YAML',
  'sh':'Shell','bat':'Batch','sql':'SQL','r':'R','scala':'Scala','dart':'Dart',
  'md':'Markdown','dockerfile':'Docker','makefile':'Make','toml':'TOML','cfg':'Config','ini':'Config'
};
const rpExtSkillMap = {
  'py':['agent-python-pro'], 'js':['agent-javascript-pro','agent-nextjs-developer'],
  'ts':['agent-javascript-pro','agent-nextjs-developer'], 'tsx':['agent-javascript-pro','agent-nextjs-developer'],
  'jsx':['agent-javascript-pro','agent-nextjs-developer'],
  'java':['code-assistant'], 'c':['code-assistant'], 'cpp':['code-assistant'], 'cs':['code-assistant'],
  'go':['code-assistant'], 'rs':['code-assistant'], 'rb':['code-assistant'], 'php':['code-assistant'],
  'swift':['code-assistant'], 'kt':['code-assistant'], 'html':['web-artifacts-builder'],
  'css':['web-artifacts-builder'], 'sql':['code-assistant'], 'r':['code-assistant'],
  'dart':['code-assistant'], 'scala':['code-assistant']
};
const rpLangColors = {
  'Python':'#3776ab','JavaScript':'#f7df1e','TypeScript':'#3178c6','TypeScript(React)':'#3178c6',
  'JavaScript(React)':'#61dafb','Java':'#b07219','C':'#555555','C++':'#f34b7d','C#':'#178600',
  'Go':'#00add8','Rust':'#dea584','Ruby':'#701516','PHP':'#4f5d95','Swift':'#f05138','Kotlin':'#a97bff',
  'HTML':'#e34c26','CSS':'#563d7c','Shell':'#89e051','SQL':'#e38c00','R':'#198ce7','Dart':'#00b4ab'
};

// 코드 요약: 큰 파일은 시그니처+핵심부분만 추출
function summarizeCode(content, ext, maxLines){
  maxLines = maxLines || 150;
  const lines = content.split('\n');
  if(lines.length <= maxLines) return content; // 작은 파일은 전체 전달

  const result = [];
  const sigPatterns = {
    'py': /^\s*(class |def |async def |import |from .+ import|@)/,
    'java': /^\s*(public |private |protected |class |interface |import |@|package )/,
    'cs': /^\s*(public |private |protected |internal |class |interface |using |namespace |\[)/,
    'js': /^\s*(function |const |let |var |class |export |import |async |module\.)/,
    'ts': /^\s*(function |const |let |var |class |export |import |async |interface |type |module\.)/,
    'tsx': /^\s*(function |const |let |var |class |export |import |async |interface |type )/,
    'jsx': /^\s*(function |const |let |var |class |export |import |async )/,
    'go': /^\s*(func |type |package |import |var |const )/,
    'rs': /^\s*(fn |pub |struct |enum |impl |use |mod |trait )/,
    'rb': /^\s*(class |def |module |require |include |attr_)/,
    'php': /^\s*(function |class |namespace |use |public |private |protected )/,
    'swift': /^\s*(func |class |struct |enum |protocol |import |var |let )/,
    'kt': /^\s*(fun |class |object |interface |import |val |var |data class)/,
    'c': /^\s*(#include |#define |typedef |struct |int |void |char |float |double |static )/,
    'cpp': /^\s*(#include |#define |typedef |struct |class |template |namespace |int |void )/,
    'h': /^\s*(#include |#define |#ifndef |typedef |struct |class |extern )/,
  };
  const pat = sigPatterns[ext] || /^\s*(function |class |def |import |export |public |private )/;

  // 1) 첫 30줄 (imports, 설정)
  result.push('// === 파일 시작 (첫 30줄) ===');
  result.push(...lines.slice(0, 30));

  // 2) 시그니처 라인 추출 (함수/클래스 정의 + 전후 2줄)
  result.push('');
  result.push('// === 주요 시그니처 & 구조 ===');
  const added = new Set();
  lines.forEach((line, i)=>{
    if(i < 30) return; // 이미 포함됨
    if(pat.test(line)){
      for(let j=Math.max(i-1,30); j<=Math.min(lines.length-1,i+3); j++){
        if(!added.has(j)){
          added.add(j);
          result.push(`[L${j+1}] ${lines[j]}`);
        }
      }
      result.push('  ...');
    }
  });

  // 3) 마지막 10줄
  result.push('');
  result.push('// === 파일 끝 (마지막 10줄) ===');
  result.push(...lines.slice(-10));

  result.push(`\n// [총 ${lines.length}줄 중 시그니처 요약 - 전체 코드 필요시 요청하세요]`);
  return result.join('\n');
}

// catalog에 실제 존재하는 스킬인지 확인
function isSkillInCatalog(skillId){
  for(const did of Object.keys(catalog)){
    const d = catalog[did];
    if(d && d.skills && d.skills.some(s=>s.id===skillId && s.available)) return true;
  }
  return false;
}

function detectRpLanguages(){
  const langCount = {};
  const detectedSkills = new Set();

  rpFiles.forEach(f=>{
    const lang = rpExtLangMap[f.ext];
    if(lang){
      langCount[lang] = (langCount[lang]||0) + 1;
    }
    const skills = rpExtSkillMap[f.ext];
    if(skills) skills.forEach(s=>detectedSkills.add(s));
  });

  const panel = document.getElementById('rpLangDetect');
  const tagsEl = document.getElementById('rpLangTags');
  const autoEl = document.getElementById('rpAutoSkills');

  if(rpFiles.length === 0 || Object.keys(langCount).length === 0){
    panel.style.display = 'none';
    return;
  }
  panel.style.display = 'block';

  // 감지된 언어 태그 렌더링 (파일 수 기준 내림차순)
  const sorted = Object.entries(langCount).sort((a,b)=>b[1]-a[1]);
  const primaryLang = sorted[0][0];
  tagsEl.innerHTML = sorted.map(([lang, cnt])=>{
    const color = rpLangColors[lang] || '#6b7280';
    const isPrimary = lang === primaryLang;
    return `<span style="display:inline-flex;align-items:center;gap:3px;padding:2px 7px;border-radius:10px;background:${color}22;border:${isPrimary?'2':'1'}px solid ${color}${isPrimary?'':'55'};color:${color};font-weight:600;font-size:10px;">${isPrimary?'⭐ ':''}${lang} <span style="font-weight:400;color:#888">${cnt}</span></span>`;
  }).join('');

  // 자동 스킬 로드 - catalog에 실제 있는 것만
  const newSkills = [];
  const loadedNames = [];
  // code-assistant 항상 먼저
  if(isSkillInCatalog('code-assistant') && !selSkills.includes('code-assistant')){
    selSkills.push('code-assistant');
    newSkills.push('code-assistant');
  }
  if(selSkills.includes('code-assistant')) loadedNames.push('code-assistant');

  detectedSkills.forEach(skill=>{
    if(skill === 'code-assistant') return;
    if(isSkillInCatalog(skill)){
      if(!selSkills.includes(skill)){
        selSkills.push(skill);
        newSkills.push(skill);
      }
      loadedNames.push(skill);
    }
  });

  // autoLoadedSkills 업데이트 (UI에서 🧠자동 배지 표시용)
  autoLoadedSkills = [...new Set([...autoLoadedSkills, ...newSkills])];

  if(newSkills.length > 0){
    renderSkills();
    updateLoaded();
  }

  // 상태 표시
  const availableLoaded = loadedNames.filter(s=>s!=='code-assistant');
  if(availableLoaded.length > 0){
    autoEl.innerHTML = '✅ <b>자동 로드:</b> ' + availableLoaded.join(', ') + ' + code-assistant';
  } else {
    autoEl.innerHTML = '✅ <b>주 언어:</b> ' + primaryLang + ' → LLM이 자동 판별하여 분석합니다';
  }
}

function toggleRightPanel(){
  const rp = document.getElementById('rightPanel');
  const btn = document.getElementById('rightPanelToggle');
  rp.classList.toggle('collapsed');
  document.body.classList.toggle('rp-collapsed');
  if(rp.classList.contains('collapsed')){
    btn.textContent = '◀';
    document.querySelector('.chat-box-fixed').style.right = '0';
  } else {
    btn.textContent = '▶';
    document.querySelector('.chat-box-fixed').style.right = '340px';
  }
  localStorage.setItem('demos_rp_collapsed', rp.classList.contains('collapsed'));
}

// 초기 상태 복원
(function(){
  const saved = localStorage.getItem('demos_rp_collapsed');
  if(saved === 'true'){
    const rp = document.getElementById('rightPanel');
    const btn = document.getElementById('rightPanelToggle');
    if(rp){ rp.classList.add('collapsed'); document.body.classList.add('rp-collapsed'); btn.textContent='◀'; }
  }
})();

// 드래그앤드롭
(function(){
  const drop = document.getElementById('rpFileDrop');
  if(!drop) return;
  drop.addEventListener('dragover', e=>{e.preventDefault();e.stopPropagation();drop.classList.add('dragover');});
  drop.addEventListener('dragleave', e=>{e.preventDefault();drop.classList.remove('dragover');});
  drop.addEventListener('drop', e=>{
    e.preventDefault();e.stopPropagation();drop.classList.remove('dragover');
    if(e.dataTransfer.files.length > 0) handleRpFileSelect(e.dataTransfer.files);
  });
})();

function switchUploadMode(mode, btn){
  document.querySelectorAll('.rp-tab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('rpFolderMode').style.display = mode==='folder' ? 'block' : 'none';
  document.getElementById('rpFileMode').style.display = mode==='files' ? 'block' : 'none';
}

function handleRpFolderSelect(files){
  const codeExts = ['py','js','html','css','json','xml','yaml','yml','c','cpp','h','java','go','rs','sh','bat','md','txt','ts','tsx','jsx','rb','php','swift','kt','toml','cfg','ini','env','gitignore','dockerfile','makefile'];
  const skipDirs = ['node_modules','.git','__pycache__','.venv','venv','.next','dist','build','.idea','.vscode'];
  let pending = 0;
  let totalSize = 0;

  for(const f of files){
    const path = f.webkitRelativePath || f.name;
    // 불필요한 디렉토리 스킵
    const parts = path.split('/');
    if(parts.some(p => skipDirs.includes(p))) continue;

    const ext = f.name.split('.').pop().toLowerCase();
    if(!codeExts.includes(ext) && f.name.indexOf('.') !== -1) continue;
    if(f.size > 500000) continue; // 500KB 이상 파일 스킵

    pending++;
    totalSize += f.size;
    const reader = new FileReader();
    reader.onload = function(e){
      rpFiles.push({name:f.name, path:path, content:e.target.result, ext:ext, size:f.size});
      pending--;
      if(pending === 0){
        buildRpTree();
        renderRpTree();
        updateRpRunBtn();
        detectRpLanguages();
      }
    };
    reader.readAsText(f);
  }
  document.getElementById('rpFolderInput').value = '';
}

function buildRpTree(){
  rpTreeData = {name:'root', children:{}, files:[]};
  rpFiles.forEach((f,idx)=>{
    const parts = (f.path || f.name).split('/');
    let node = rpTreeData;
    // 첫 번째 파트는 루트 폴더명
    for(let i=0; i<parts.length-1; i++){
      if(!node.children[parts[i]]){
        node.children[parts[i]] = {name:parts[i], children:{}, files:[]};
      }
      node = node.children[parts[i]];
    }
    node.files.push({name:parts[parts.length-1], idx:idx, ext:f.ext, size:f.size});
  });
}

let rpCheckedFiles = new Set();  // 체크된 파일 인덱스

function renderRpTree(){
  const container = document.getElementById('rpTree');
  const stats = document.getElementById('rpTreeStats');
  const selectBar = document.getElementById('rpSelectBar');
  if(rpFiles.length === 0){
    container.style.display = 'none';
    stats.style.display = 'none';
    selectBar.style.display = 'none';
    return;
  }
  container.style.display = 'block';
  stats.style.display = 'flex';
  selectBar.style.display = 'flex';

  // 기본: 전체 체크
  if(rpCheckedFiles.size === 0){
    rpFiles.forEach((_,i)=>rpCheckedFiles.add(i));
  }

  // 통계
  const folders = new Set();
  rpFiles.forEach(f=>{
    const parts = (f.path||f.name).split('/');
    for(let i=1;i<parts.length;i++) folders.add(parts.slice(0,i).join('/'));
  });
  const totalSize = rpFiles.reduce((s,f)=>s+f.size,0);
  const sizeStr = totalSize > 1048576 ? (totalSize/1048576).toFixed(1)+'MB' : (totalSize/1024).toFixed(1)+'KB';
  document.getElementById('rpStatsInfo').textContent = rpFiles.length+' 파일 / '+folders.size+' 폴더';
  document.getElementById('rpStatsSize').textContent = sizeStr;

  container.innerHTML = renderTreeNode(rpTreeData, true);
  updateCheckedCount();
  updateParentFolderChecks();
}

function renderTreeNode(node, isRoot){
  let html = '';
  // 폴더들
  const childKeys = Object.keys(node.children).sort();
  childKeys.forEach(key=>{
    const child = node.children[key];
    const fileCount = countFiles(child);
    const folderIdxs = collectFileIdxs(child);
    const allChecked = folderIdxs.every(i=>rpCheckedFiles.has(i));
    const someChecked = folderIdxs.some(i=>rpCheckedFiles.has(i));
    html += `<div class="rp-tree-node rp-tree-folder">
      <input type="checkbox" class="rp-tree-cb" data-folder-idxs="${folderIdxs.join(',')}" ${allChecked?'checked':''} onclick="event.stopPropagation();toggleFolderCheck(this)" onchange="event.stopPropagation()">
      <span class="rp-tree-node-content" onclick="toggleTreeFolder(this.parentElement)"><span class="rp-tree-toggle">▶</span><span class="rp-tree-icon">📁</span>${esc(key)} <span style="color:#bbb;font-weight:400">(${fileCount})</span></span>
    </div><div class="rp-tree-children" style="display:none">${renderTreeNode(child, false)}</div>`;
  });
  // 파일들
  node.files.sort((a,b)=>a.name.localeCompare(b.name)).forEach(f=>{
    const langIcon = {py:'🐍',js:'📜',html:'🌐',css:'🎨',java:'☕',c:'⚙️',cpp:'⚙️',go:'🐹',rs:'🦀',sh:'💻',json:'📋',yaml:'📋',yml:'📋',md:'📄',ts:'📘',tsx:'📘'}[f.ext]||'📄';
    const checked = rpCheckedFiles.has(f.idx);
    html += `<div class="rp-tree-node rp-tree-file" data-idx="${f.idx}">
      <input type="checkbox" class="rp-tree-cb" data-idx="${f.idx}" ${checked?'checked':''} onclick="event.stopPropagation();toggleFileCheck(${f.idx},this)">
      <span class="rp-tree-node-content" onclick="selectTreeFile(${f.idx},this.parentElement)"><span class="rp-tree-toggle"> </span><span class="rp-tree-icon">${langIcon}</span>${esc(f.name)}</span>
    </div>`;
  });
  return html;
}

function collectFileIdxs(node){
  let idxs = node.files.map(f=>f.idx);
  Object.values(node.children).forEach(ch=>{ idxs = idxs.concat(collectFileIdxs(ch)); });
  return idxs;
}

function toggleFileCheck(idx, cb){
  if(cb.checked) rpCheckedFiles.add(idx);
  else rpCheckedFiles.delete(idx);
  updateCheckedCount();
  updateParentFolderChecks();
}

function toggleFolderCheck(cb){
  const idxs = cb.dataset.folderIdxs.split(',').filter(Boolean).map(Number).filter(n=>!isNaN(n));
  if(cb.checked) idxs.forEach(i=>rpCheckedFiles.add(i));
  else idxs.forEach(i=>rpCheckedFiles.delete(i));
  // 하위 파일 + 하위 폴더 체크박스 모두 동기화
  const parent = cb.closest('.rp-tree-folder');
  const childContainer = parent ? parent.nextElementSibling : null;
  if(childContainer && childContainer.classList.contains('rp-tree-children')){
    childContainer.querySelectorAll('input.rp-tree-cb').forEach(childCb=>{
      childCb.checked = cb.checked;
      childCb.indeterminate = false;
    });
  }
  updateCheckedCount();
  updateParentFolderChecks();
}

function updateParentFolderChecks(){
  document.querySelectorAll('.rp-tree-cb[data-folder-idxs]').forEach(cb=>{
    const idxs = cb.dataset.folderIdxs.split(',').filter(Boolean).map(Number).filter(n=>!isNaN(n));
    const checkedCount = idxs.filter(i=>rpCheckedFiles.has(i)).length;
    cb.checked = checkedCount === idxs.length;
    cb.indeterminate = checkedCount > 0 && checkedCount < idxs.length;
  });
}

function rpCheckAll(checked){
  if(checked) rpFiles.forEach((_,i)=>rpCheckedFiles.add(i));
  else rpCheckedFiles.clear();
  document.querySelectorAll('.rp-tree-cb').forEach(cb=>{ cb.checked=checked; cb.indeterminate=false; });
  updateCheckedCount();
}

function updateCheckedCount(){
  const count = rpCheckedFiles.size;
  document.getElementById('rpCheckedCount').textContent = count+'개 선택';
  updateRpRunBtn();
}

function countFiles(node){
  let c = node.files.length;
  Object.values(node.children).forEach(ch=>{ c += countFiles(ch); });
  return c;
}

function toggleTreeFolder(el){
  const children = el.nextElementSibling;
  const toggle = el.querySelector('.rp-tree-toggle');
  if(children.style.display === 'none'){
    children.style.display = 'block';
    toggle.textContent = '▼';
  } else {
    children.style.display = 'none';
    toggle.textContent = '▶';
  }
}

function selectTreeFile(idx, el){
  document.querySelectorAll('.rp-tree-file').forEach(e=>e.classList.remove('active'));
  if(el) el.classList.add('active');
  rpSelectedFile = idx;
}

function clearRpProject(){
  rpFiles = [];
  rpTreeData = null;
  rpSelectedFile = null;
  rpCheckedFiles.clear();
  document.getElementById('rpTree').style.display = 'none';
  document.getElementById('rpTreeStats').style.display = 'none';
  document.getElementById('rpLangDetect').style.display = 'none';
  document.getElementById('rpSelectBar').style.display = 'none';
  document.getElementById('rpFileList').innerHTML = '';
  updateRpRunBtn();
}

function handleRpFileSelect(files){
  const codeExts = ['py','js','html','css','json','xml','yaml','yml','c','cpp','h','java','go','rs','sh','bat','md','txt','ts','tsx','jsx','rb','php','swift','kt'];
  for(const f of files){
    const ext = f.name.split('.').pop().toLowerCase();
    if(!codeExts.includes(ext)){ alert(f.name+': 지원하지 않는 파일 형식입니다.'); continue; }
    const reader = new FileReader();
    reader.onload = function(e){
      rpFiles.push({name:f.name, content:e.target.result, ext:ext, size:f.size});
      renderRpFiles();
      updateRpRunBtn();
      detectRpLanguages();
    };
    reader.readAsText(f);
  }
  document.getElementById('rpFileInput').value = '';
}

function renderRpFiles(){
  const list = document.getElementById('rpFileList');
  if(rpFiles.length === 0){ list.innerHTML = ''; return; }
  list.innerHTML = rpFiles.map((f,i)=>{
    const sizeStr = f.size > 1048576 ? (f.size/1048576).toFixed(1)+'MB' : (f.size/1024).toFixed(1)+'KB';
    const langIcon = {py:'🐍',js:'📜',html:'🌐',css:'🎨',java:'☕',c:'⚙️',cpp:'⚙️',go:'🐹',rs:'🦀',sh:'💻',json:'📋',yaml:'📋',yml:'📋',md:'📄',ts:'📘',tsx:'📘'}[f.ext] || '📄';
    return `<div class="rp-file-item">
      <span>${langIcon}</span><span class="fname">${esc(f.name)}</span><span class="fsize">${sizeStr}</span>
      <button class="fremove" onclick="event.stopPropagation();removeRpFile(${i})">✕</button>
    </div>`;
  }).join('');
}

function removeRpFile(idx){
  rpFiles.splice(idx, 1);
  renderRpFiles();
  updateRpRunBtn();
}


function selectRpMode(btn){
  document.querySelectorAll('.rp-action-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  rpSelectedMode = btn.dataset.mode;
  updateRpRunBtn();
}

function updateRpRunBtn(){
  const btn = document.getElementById('rpRunBtn');
  const checkedCount = rpCheckedFiles.size;
  if(rpFiles.length > 0 && rpSelectedMode && checkedCount > 0){
    btn.disabled = false;
    btn.textContent = '▶ ' + rpModeLabels[rpSelectedMode] + ' (' + checkedCount + '개 파일)';
  } else if(rpFiles.length > 0 && checkedCount === 0){
    btn.disabled = true;
    btn.textContent = '▶ 분석할 파일을 선택하세요 (체크박스)';
  } else {
    btn.disabled = true;
    btn.textContent = rpFiles.length === 0 ? '▶ 코드 파일을 업로드하세요' : '▶ 분석 모드를 선택하세요';
  }
}

async function runCodeAssistant(){
  if(rpFiles.length === 0 || !rpSelectedMode) return;
  if(!selEnv){ alert('먼저 LLM 환경을 선택해주세요.'); return; }

  // code-assistant 스킬 자동 로드
  if(!selSkills.includes('code-assistant')){
    selSkills.push('code-assistant');
    renderSkills();
    updateLoaded();
  }

  // 체크된 파일만 필터링
  const selectedFiles = rpFiles.filter((_,i)=>rpCheckedFiles.has(i));
  if(selectedFiles.length === 0){ alert('분석할 파일을 선택해주세요 (체크박스).'); return; }

  // 프로젝트 트리 구조 생성
  let treeText = '';
  if(rpTreeData){
    treeText = '=== 프로젝트 구조 ===\n' + buildTreeText(rpTreeData, '') + '\n';
    treeText += `\n(전체 ${rpFiles.length}개 중 ${selectedFiles.length}개 파일 선택됨)\n`;
  }

  // 선택된 파일 내용을 스마트하게 추려서 구성
  let codeContent = treeText;
  const maxLinesPerFile = selectedFiles.length <= 3 ? 300 : selectedFiles.length <= 8 ? 150 : 80;

  selectedFiles.forEach(f=>{
    const summarized = summarizeCode(f.content, f.ext, maxLinesPerFile);
    codeContent += `\n--- 파일: ${f.path||f.name} (${f.ext}) ---\n${summarized}\n`;
  });

  const prompt = rpModePrompts[rpSelectedMode] + '\n\n' + codeContent;
  const modeLabel = rpModeLabels[rpSelectedMode] || rpSelectedMode;

  // 모든 분석 모드 → 채팅으로 전송 (대화 이어가기 위해)
  {
    const fileNames = selectedFiles.map(f=>f.name).join(', ');
    const shortLabel = '\u{1f4ca} [' + modeLabel + '] ' + fileNames;
    addMsg('user', shortLabel);
    history.push({role:'user', content:prompt});

    const typing = addTyping();
    isSending = true;
    const btn = document.getElementById('sendBtn');
    btn.textContent = '\u23f9';
    btn.classList.add('stop-mode');
    btn.disabled = false;
    chatAbort = new AbortController();

    // 버튼 비활성화
    const runBtn = document.getElementById('rpRunBtn');
    runBtn.disabled = true;
    runBtn.textContent = '\u23f3 분석 중...';

    try{
      const resp = await fetch('/api/chat',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        signal: chatAbort.signal,
        body:JSON.stringify({
          env: selEnv,
          messages: history,
          skills: [...selSkills],
          effort,
          format: formatManualOverride ? selFormat : 'auto',
          writing_style: styleManualOverride ? document.getElementById('writingStyle').value.trim() : 'auto',
          system_prompt: document.getElementById('systemPromptInput').value.trim(),
          max_tokens: maxTokens,
          think_mode: document.getElementById('thinkToggle').checked,
        })
      });
      const data = await resp.json();
      typing.remove();
      if(data.error){
        addMsg('assistant','\u274c '+data.error);
      } else {
        let info = '';
        if(data.loaded_skills && data.loaded_skills.length > 0){
          let extra = data.tokens_budget ? ' ['+data.tokens_budget+']' : '';
          let mName = data.model_used || (envs[selEnv] ? envs[selEnv].name : selEnv);
          info = '\n[\u2705 '+data.loaded_skills.join(', ')+'] ['+mName+'] ('+data.system_prompt_length+'\uc790)'+extra;
        }
        if(data.auto_routed){ info += ' [\uD83E\uDD16 자동: '+data.model_used+' ('+data.route_reason+')]'; }
        if(data.fallback_used){ info += ' [\u26A0\uFE0F 대체: '+data.fallback_from+' \u2192 '+data.model_used+']'; }
        let truncWarn = data.truncated ? '\n\n⚠️ **응답이 토큰 한도('+maxTokens+')에 도달하여 잘렸습니다.** "계속 이어서 작성해줘"라고 입력하면 이어서 받을 수 있습니다.' : '';
        const assistantDisplayText = data.content + truncWarn + info;
        const assistantRawForDetect = data.content + truncWarn;
        addMsg('assistant', assistantDisplayText, assistantRawForDetect);
        history.push({role:'assistant', content:data.content});
      }
    }catch(e){
      typing.remove();
      if(e.name !== 'AbortError'){
        addMsg('assistant','\u274c 서버 연결 실패: '+e.message);
      }
    }
    isSending = false;
    chatAbort = null;
    btn.textContent = '\u25b6';
    btn.classList.remove('stop-mode');
    btn.disabled = false;
    document.getElementById('input').focus();
    runBtn.disabled = false;
    updateRpRunBtn();
    saveCurrentSession();
  }
}

let rpLastAnalysisContent = '';

function toggleRpResultExpand(){
  document.getElementById('rpResult').classList.toggle('expanded');
}
function copyRpResult(){
  const body = document.querySelector('.rp-result-body');
  if(body) navigator.clipboard.writeText(body.innerText).then(()=>alert('복사 완료!'));
}
function sendRpResultToChat(){
  if(!rpLastAnalysisContent) return;
  addMsg('assistant', rpLastAnalysisContent);
  history.push({role:'assistant', content: rpLastAnalysisContent});
  saveCurrentSession();
}

function buildTreeText(node, prefix){
  let text = '';
  const childKeys = Object.keys(node.children).sort();
  const allItems = [...childKeys.map(k=>({type:'dir',name:k,node:node.children[k]})), ...node.files.map(f=>({type:'file',name:f.name}))];
  allItems.forEach((item,i)=>{
    const isLast = i === allItems.length - 1;
    const connector = isLast ? '└── ' : '├── ';
    const nextPrefix = prefix + (isLast ? '    ' : '│   ');
    if(item.type === 'dir'){
      text += prefix + connector + '📁 ' + item.name + '/\n';
      text += buildTreeText(item.node, nextPrefix);
    } else {
      text += prefix + connector + item.name + '\n';
    }
  });
  return text;
}

// 폴더 드래그앤드롭
(function(){
  const drop = document.getElementById('rpFolderDrop');
  if(!drop) return;
  drop.addEventListener('dragover', e=>{e.preventDefault();e.stopPropagation();drop.classList.add('dragover');});
  drop.addEventListener('dragleave', e=>{e.preventDefault();drop.classList.remove('dragover');});
  drop.addEventListener('drop', e=>{
    e.preventDefault();e.stopPropagation();drop.classList.remove('dragover');
    // 폴더 드롭은 DataTransferItem으로 처리
    if(e.dataTransfer.items){
      const items = [...e.dataTransfer.items];
      const entries = items.map(i=>i.webkitGetAsEntry&&i.webkitGetAsEntry()).filter(Boolean);
      if(entries.length > 0){
        readEntriesRecursive(entries);
        return;
      }
    }
    if(e.dataTransfer.files.length > 0) handleRpFolderSelect(e.dataTransfer.files);
  });
})();

function readEntriesRecursive(entries){
  const codeExts = ['py','js','html','css','json','xml','yaml','yml','c','cpp','h','java','go','rs','sh','bat','md','txt','ts','tsx','jsx','rb','php','swift','kt','toml','cfg','ini'];
  const skipDirs = ['node_modules','.git','__pycache__','.venv','venv','.next','dist','build','.idea','.vscode'];
  let pending = 0;

  function readEntry(entry, path){
    if(entry.isFile){
      const ext = entry.name.split('.').pop().toLowerCase();
      if(!codeExts.includes(ext)) return;
      pending++;
      entry.file(f=>{
        if(f.size > 500000){ pending--; return; }
        const reader = new FileReader();
        reader.onload = function(e){
          rpFiles.push({name:f.name, path:path+f.name, content:e.target.result, ext:ext, size:f.size});
          pending--;
          if(pending===0){ buildRpTree(); renderRpTree(); updateRpRunBtn(); detectRpLanguages(); }
        };
        reader.readAsText(f);
      });
    } else if(entry.isDirectory){
      if(skipDirs.includes(entry.name)) return;
      const dirReader = entry.createReader();
      dirReader.readEntries(entries=>{
        entries.forEach(e=>readEntry(e, path+entry.name+'/'));
      });
    }
  }
  entries.forEach(e=>readEntry(e, ''));
}

// 초기 실행 버튼 상태
updateRpRunBtn();

// ==================== 인트로 → 모드 선택 시스템 ====================
function dismissIntroAndShowMode(){
  var ov = document.getElementById('introOverlay');
  if(!ov || ov._gone) return;
  ov._gone = true;
  var vi = document.getElementById('introVideo');
  if(vi) vi.pause();
  ov.classList.add('fade-out');
  setTimeout(function(){
    if(ov.parentNode) ov.parentNode.removeChild(ov);
    showModeSelector();
  }, 700);
}

function showModeSelector(){
  var ms = document.getElementById('modeSelector');
  if(ms){ ms.style.display='flex'; ms.style.opacity=''; ms.classList.remove('visible'); setTimeout(function(){ ms.classList.add('visible'); }, 30); }
}

function selectMode(mode){
  var ms = document.getElementById('modeSelector');
  if(ms){ ms.classList.remove('visible'); setTimeout(function(){ ms.style.display='none'; ms.style.opacity=''; }, 500); }
  if(mode === 'uio'){
    document.getElementById('uioContainer').style.display='block';
    document.getElementById('uioFrame').src='/uio/';
    // 기본 UI 숨김
    document.getElementById('sidebar').style.display='none';
    var mainEl = document.querySelector('.main');
    if(mainEl) mainEl.style.display='none';
    var chatBox = document.querySelector('.chat-box-fixed');
    if(chatBox) chatBox.style.display='none';
    var sideToggle = document.querySelector('.sidebar-toggle');
    if(sideToggle) sideToggle.style.display='none';
  }
  // mode === 'default' → 기본 UI 그대로 표시 (아무것도 안 함)
}

function exitUioMode(){
  document.getElementById('uioContainer').style.display='none';
  document.getElementById('uioFrame').src='about:blank';
  // 기본 UI 복원
  document.getElementById('sidebar').style.display='';
  var mainEl = document.querySelector('.main');
  if(mainEl) mainEl.style.display='';
  var chatBox = document.querySelector('.chat-box-fixed');
  if(chatBox) chatBox.style.display='';
  var sideToggle = document.querySelector('.sidebar-toggle');
  if(sideToggle) sideToggle.style.display='';
  // 모드 선택 다시 표시
  showModeSelector();
}

(function(){
  var ov = document.getElementById('introOverlay');
  var vi = document.getElementById('introVideo');
  if(!ov) return;
  if(vi){
    vi.addEventListener('ended', dismissIntroAndShowMode);
    vi.addEventListener('error', dismissIntroAndShowMode);
    var src = vi.querySelector('source');
    if(src) src.addEventListener('error', dismissIntroAndShowMode);
    // 진행률 바
    var prog = document.getElementById('introProgress');
    if(prog) vi.addEventListener('timeupdate', function(){
      if(vi.duration) prog.style.width = (vi.currentTime/vi.duration*100)+'%';
    });
  }
  // 안전장치: 4초 안에 재생 안 되면 자동 제거
  setTimeout(function(){ if(vi && vi.readyState < 2) dismissIntroAndShowMode(); }, 4000);
  // ESC 키
  document.addEventListener('keydown', function(e){ if(e.key==='Escape') dismissIntroAndShowMode(); });
})();

// ==================== PPT 코드 감지 & 생성 ====================
function detectAndAddPptxButtons(msgEl, rawText){
  // python-pptx 코드 블록 감지 (from pptx import ... 또는 python-pptx 패턴)
  const codeBlockRegex = /```(?:python|py)?\s*\n([\s\S]*?)```/gi;
  let match;
  const pptxCodes = [];
  while((match = codeBlockRegex.exec(rawText)) !== null){
    const code = match[1];
    if(code.includes('from pptx') || code.includes('import pptx') || code.includes('Presentation(') || code.includes('add_slide')){
      pptxCodes.push(code.trim());
    }
  }
  if(pptxCodes.length === 0) return;

  // 각 PPT 코드 블록 아래에 생성 버튼 삽입
  const allPre = msgEl.querySelectorAll('pre');
  let codeIdx = 0;
  allPre.forEach(pre => {
    const preText = pre.textContent;
    if((preText.includes('from pptx') || preText.includes('import pptx') || preText.includes('Presentation(') || preText.includes('add_slide')) && codeIdx < pptxCodes.length){
      const code = pptxCodes[codeIdx];
      codeIdx++;
      const block = document.createElement('div');
      block.className = 'pptx-gen-block';
      block.innerHTML = `
        <div class="pptx-gen-header">
          <span>📽️ PowerPoint 생성</span>
          <div class="pptx-gen-actions">
            <span class="pptx-gen-status"></span>
            <button class="pptx-gen-btn secondary" onclick="copyPptxCode(this)">📋 실행용 코드 복사</button>
            <button class="pptx-gen-btn" onclick="generatePptx(this)">📽️ PPT 생성 & 다운로드</button>
          </div>
        </div>
        <div class="pptx-remake-bar">
          <span class="pptx-remake-label">🔄 다시 만들기:</span>
          <button class="pptx-remake-btn" onclick="_remakePpt('그래프/차트 없이 텍스트와 표 위주로 PPT 다시 만들어줘')">📝 그래프 없이</button>
          <button class="pptx-remake-btn" onclick="_remakePpt('그래프와 차트를 포함해서 PPT 다시 만들어줘')">📊 그래프 포함</button>
          <button class="pptx-remake-btn" onclick="_remakePpt('디자인을 더 심플하게 PPT 다시 만들어줘')">🎨 심플하게</button>
          <button class="pptx-remake-btn" onclick="_remakePpt('슬라이드 수를 줄여서 핵심만 PPT 다시 만들어줘')">✂️ 핵심만</button>
        </div>`;
      block.dataset.pptxCode = u2b(code);
      pre.parentNode.insertBefore(block, pre.nextSibling);
    }
  });
}

async function copyPptxCode(btn){
  const block = btn.closest('.pptx-gen-block');
  const rawCode = b2u(block.dataset.pptxCode);
  let codeToCopy = rawCode;

  try{
    const resp = await fetch('/api/prepare_pptx_code', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({code: rawCode})
    });
    if(resp.ok){
      const data = await resp.json();
      if(data && data.code) codeToCopy = data.code;
    }
  }catch(e){}

  navigator.clipboard.writeText(codeToCopy);
  btn.textContent = '✓ 복사됨';
  setTimeout(()=>{ btn.textContent = '📋 실행용 코드 복사'; }, 1500);
}

async function generatePptx(btn){
  const block = btn.closest('.pptx-gen-block');
  const code = b2u(block.dataset.pptxCode);
  const status = block.querySelector('.pptx-gen-status');

  btn.disabled = true;
  btn.textContent = '⏳ 생성 중...';
  status.textContent = 'python-pptx 코드 실행 중...';

  try{
    const resp = await fetch('/api/generate_pptx', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({code, filename:'presentation.pptx'})
    });

    if(!resp.ok){
      const err = await resp.json();
      status.textContent = '❌ ' + (err.error || '생성 실패');
      btn.textContent = '📽️ 재시도';
      btn.disabled = false;
      return;
    }

    const json = await resp.json();
    if(json.download_url){
        window.location.href = json.download_url;
        status.textContent = '✅ 다운로드 완료!';
    } else if(json.message) {
        status.textContent = '✅ ' + json.message;
    } else {
        status.textContent = '❌ 파일 응답 형식 오류';
    }
    btn.textContent = '📽️ 다시 생성';
    btn.disabled = false;
  }catch(e){
    status.textContent = '❌ 네트워크 오류';
    btn.textContent = '📽️ 재시도';
    btn.disabled = false;
  }
}

function _remakePpt(text){
  const input = document.getElementById('input');
  if(input){ input.value = text; input.focus(); input.style.height='auto'; input.style.height=input.scrollHeight+'px'; }
  // 자동 전송
  setTimeout(()=>send(), 100);
}
</script>
</body>
</html>
"""

# ============================================
# 실행
# ============================================
if __name__ == "__main__":
    # Windows 콘솔 인코딩 사전 보호 (colorama/click 충돌 방지)
    if sys.platform == "win32":
        # UTF-8 콘솔 모드 강제
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        os.environ.setdefault("PYTHONUTF8", "1")
        try:
            os.system("chcp 65001 >nul 2>&1")
        except Exception:
            pass
        # colorama 초기화 문제 방지
        os.environ.setdefault("NO_COLOR", "1")
        os.environ.setdefault("TERM", "dumb")

    print("=" * 50)
    print("  Demos(민중) Alpha 0.8")
    print("=" * 50)

    # 스킬 폴더 확인
    if os.path.isdir(SKILLS_DIR):
        skills = scan_skills()
        print(f"  📂 스킬 폴더: {SKILLS_DIR}")
        print(f"  ✅ 발견된 스킬: {len(skills)}개")
        for s in sorted(skills.keys())[:10]:
            print(f"     - {s}")
        if len(skills) > 10:
            print(f"     ... 외 {len(skills)-10}개")
    else:
        print(f"  ⚠️  스킬 폴더 없음: {SKILLS_DIR}")
        print(f"     scientific-skills 폴더를 app.py와 같은 위치에 복사하세요.")

    # 토큰 확인
    if API_TOKEN:
        print(f"  🔑 TOKEN.TXT: 로드됨 ({len(API_TOKEN)}자)")
    else:
        print(f"  ⚠️  TOKEN.TXT: 없음 또는 비어있음")
        print(f"     → {TOKEN_FILE} 에 API 키를 넣어주세요")

    # ============================================
    # GGUF 자동 감지 & Python으로 직접 로드 (다중 모델 지원)
    # ============================================
    gguf_files = find_gguf_files()

    if gguf_files:
        # 크기 내림차순 정렬
        gguf_files.sort(key=lambda x: x["size_gb"], reverse=True)
        print(f"\n  💻 GGUF 자동 감지! ({len(gguf_files)}개 모델)")

        for idx, gf in enumerate(gguf_files):
            env_key = f"gguf-{idx}"
            model_name = gf["name"].replace(".gguf", "")
            ENV_CONFIG[env_key] = {
                "url": "python://llama-cpp-python",
                "model": model_name,
                "name": f"LOCAL ({model_name})",
                "_gguf_path": gf["path"],  # 내부용: 모델 파일 경로
            }
            print(f"     [{env_key}] {gf['name']} ({gf['size_gb']} GB)")

        # 첫 번째(가장 큰) 모델을 기본 로드
        first_gguf = gguf_files[0]
        if load_gguf_model(first_gguf["path"]):
            print(f"     ✅ 기본 모델 로드 완료: {first_gguf['name']}")
        else:
            print(f"     ⚠️  기본 모델 로드 실패 (환경 전환 시 재시도)")
    else:
        print(f"\n  ℹ️  GGUF 파일 없음 → LOCAL GGUF 비활성")

    # 환경 목록
    print()
    print("  🖥️  사용 가능한 LLM 환경:")
    for eid, ecfg in ENV_CONFIG.items():
        print(f"     [{eid}] {ecfg['name']} → {ecfg['url']}")

    print()
    print(f"  🌐 http://localhost:10009 에서 접속하세요")
    print("=" * 50)

    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Windows: GGUF 모델 로드 후 콘솔 핸들이 깨질 수 있으므로 복원
    if sys.platform == "win32":
        import io as _io

        # 1) colorama/click 배너 출력 시 OSError: Windows error 6 방지
        #    → 환경변수로 Flask CLI 배너를 비활성화
        os.environ.setdefault("WERKZEUG_RUN_MAIN", "false")

        # 2) stdout/stderr 핸들 복원 시도
        for _stream_name in ("stdout", "stderr"):
            try:
                getattr(sys, _stream_name).write("")
            except (OSError, AttributeError, ValueError):
                try:
                    _orig = getattr(sys, f"__{_stream_name}__")
                    _fd = _orig.fileno()
                    setattr(sys, _stream_name, _io.TextIOWrapper(
                        open(_fd, "wb", closefd=False),
                        encoding="utf-8", errors="replace"
                    ))
                except (OSError, AttributeError, ValueError):
                    # fileno()도 죽은 경우 → devnull로 대체 (서버는 정상 동작)
                    setattr(sys, _stream_name, open(os.devnull, "w", encoding="utf-8"))

        # 3) click의 echo가 colorama를 통해 쓸 때 터지는 것 방지
        try:
            import click
            click.echo = lambda message=None, file=None, nl=True, err=False, color=None: None
        except ImportError:
            pass

    app.run(host="0.0.0.0", port=10009, debug=False)
    #app.run(host="127.0.0.1", port=18080, debug=False, use_reloader=False)
