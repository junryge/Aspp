# Scientific-Assistant 스킬 목록

> CEO 멀티에이전트 시스템(scientific-assistant)의 전체 스킬 정리
> 총 **355개 스킬** | **24개 카테고리** | **7개 실행 그룹**
>
> 정의 위치: `SKILL/scientific-assistant/app.py` → `DOMAIN_SKILLS` (line 621)

---

## 1. 카테고리별 스킬 목록 (DOMAIN_SKILLS — UI 표시용 24개)

### 🧬 생물정보학 (bioinformatics) — 22개
biopython, scanpy, pydeseq2, bioservices, anndata, arboreto, cellxgene-census, deeptools, gget, geniml, gtars, pysam, scikit-bio, scvelo, scvi-tools, tiledbvcf, flowio, phylogenetics, etetoolkit, cobrapy, glycoengineering, esm

### 🗄️ 생물 DB (bio-databases) — 21개
gene-database, ensembl-database, uniprot-database, geo-database, clinvar-database, gnomad-database, gtex-database, gwas-database, ena-database, biorxiv-database, string-database, reactome-database, kegg-database, interpro-database, jaspar-database, monarch-database, alphafold-database, pdb-database, cosmic-database, cbioportal-database, depmap, opentargets-database

### ⚗️ 화학/신약 (cheminformatics) — 21개
rdkit, datamol, deepchem, molfeat, matchms, medchem, diffdock, molecular-dynamics, torchdrug, chembl-database, drugbank-database, pubchem-database, bindingdb-database, zinc-database, hmdb-database, clinpgx-database, brenda-database, metabolomics-workbench-database, primekg, pytdc, rowan, pyopenms

### ⚛️ 재료/물리/양자 (materials-physics) — 8개
pymatgen, astropy, fluidsim, sympy, qiskit, cirq, pennylane, qutip

### 📊 데이터/ML (data-ml) — 26개
matplotlib, seaborn, plotly, scikit-learn, pytorch-lightning, polars, dask, vaex, networkx, shap, umap-learn, statsmodels, statistical-analysis, exploratory-data-analysis, torch-geometric, stable-baselines3, pufferlib, transformers, simpy, pymoo, pymc, aeon, timesfm-forecasting, geopandas, geomaster, scikit-survival

### 💰 금융/경제 (finance) — 8개
alpha-vantage, edgartools, hedgefundmonitor, fred-economic-data, usfiscaldata, datacommons-client, market-research-reports, uspto-database

### 🏥 임상/의학 (clinical) — 13개
clinical-decision-support, clinical-reports, clinicaltrials-database, fda-database, treatment-plans, pydicom, pyhealth, pathml, histolab, imaging-data-commons, iso-13485-certification, neurokit2, neuropixels-analysis

### 📝 논문/연구 (writing-comm) — 20개
scientific-writing, literature-review, citation-management, peer-review, research-grants, scientific-brainstorming, scientific-critical-thinking, hypothesis-generation, scholar-evaluation, scientific-visualization, scientific-schematics, scientific-slides, venue-templates, latex-posters, pptx-posters, infographics, markdown-mermaid-writing, paper-2-web, pubmed-database, openalex-database

### 🤖 랩 자동화 (lab-automation) — 12개
adaptyv, benchling-integration, ginkgo-cloud-lab, opentrons-integration, pylabrobot, labarchive-integration, lamindb, latchbio-integration, dnanexus-integration, omero-integration, protocolsio-integration, pyzotero

### 🔧 유틸리티 (utilities) — 21개
drawio-diagram, docx, xlsx, pdf, pptx, markitdown, matlab, modal, generate-image, get-available-resources, bgpt-paper-search, research-lookup, perplexity-search, parallel-web, open-notebook, consciousness-council, what-if-oracle, hypogenic, dhdna-profiler, denario, offer-k-dense-web, zarr-python

### 📚 도메인 지식 (domain-knowledge) — 1개
knowledge-search *(수동 선택 전용 — 자동 추천 안 됨)*

### 🛠️ 개발 도구 (dev-tools) — 38개
aesthetic, artifacts-builder, backend-development, better-auth, brand-guidelines, canvas-design, changelog-generator, chrome-devtools, claude-code, code-review, databases, devops, docs-seeker, domain-name-brainstormer, engineer-skill-creator, file-organizer, frontend-design, frontend-development, github-ecosystem, google-adk-python, mcp-builder, python-deprecation-fixer, python-project-skel, repomix, sequential-thinking, skill-creator, skill-share, template-skill, theme-factory, ui-styling, web-artifacts-builder, web-frameworks, webapp-testing, react-best-practices, web-design-guidelines, owasp-security, logpresso-query, logpresso-search, common, debugging, problem-solving

### 🤖 AI/프롬프트 (ai-prompt) — 12개
anthropic-architect, anthropic-prompt-engineer, openai-prompt-engineer, notebooklm-skill, ai-multimodal, content-research-writer, lead-research-assistant, image-enhancer, media-processing, meeting-insights-analyzer, video-downloader, slack-gif-creator

### 💼 비즈니스 (business-ops) — 7개
competitive-ads-extractor, datadog-entity-generator, developer-growth-analysis, internal-comms, invoice-organizer, raffle-winner-picker, shopify

### ⚙️ 코어 개발 에이전트 (agent-core-dev) — 11개
agent-api-designer, agent-backend-developer, agent-electron-pro, agent-frontend-developer, agent-fullstack-developer, agent-graphql-architect, agent-microservices-architect, agent-mobile-developer, agent-ui-designer, agent-websocket-engineer, agent-wordpress-master

### 📜 언어 전문 에이전트 (agent-lang-spec) — 23개
agent-angular-architect, agent-cpp-pro, agent-csharp-developer, agent-django-developer, agent-dotnet-core-expert, agent-dotnet-framework-4.8-expert, agent-flutter-expert, agent-golang-pro, agent-java-architect, agent-javascript-pro, agent-kotlin-specialist, agent-laravel-specialist, agent-nextjs-developer, agent-php-pro, agent-python-pro, agent-rails-expert, agent-react-specialist, agent-rust-engineer, agent-spring-boot-engineer, agent-sql-pro, agent-swift-expert, agent-typescript-pro, agent-vue-expert

### ☁️ 인프라 에이전트 (agent-infra) — 12개
agent-cloud-architect, agent-database-administrator, agent-deployment-engineer, agent-devops-engineer, agent-devops-incident-responder, agent-incident-responder, agent-kubernetes-specialist, agent-network-engineer, agent-platform-engineer, agent-security-engineer, agent-sre-engineer, agent-terraform-engineer

### 🛡️ 품질/보안 에이전트 (agent-quality) — 12개
agent-accessibility-tester, agent-architect-reviewer, agent-chaos-engineer, agent-code-reviewer, agent-compliance-auditor, agent-debugger, agent-error-detective, agent-penetration-tester, agent-performance-engineer, agent-qa-expert, agent-security-auditor, agent-test-automator

### 🧠 데이터/AI 에이전트 (agent-data-ai) — 12개
agent-ai-engineer, agent-data-analyst, agent-data-engineer, agent-data-scientist, agent-database-optimizer, agent-llm-architect, agent-machine-learning-engineer, agent-ml-engineer, agent-mlops-engineer, agent-nlp-engineer, agent-postgres-pro, agent-prompt-engineer

### 🔧 개발자경험 에이전트 (agent-dx) — 10개
agent-build-engineer, agent-cli-developer, agent-dependency-manager, agent-documentation-engineer, agent-dx-optimizer, agent-git-workflow-manager, agent-legacy-modernizer, agent-mcp-developer, agent-refactoring-specialist, agent-tooling-engineer

### 🎯 특수 도메인 에이전트 (agent-domain) — 11개
agent-api-documenter, agent-blockchain-developer, agent-embedded-systems, agent-fintech-engineer, agent-game-developer, agent-iot-engineer, agent-mobile-app-developer, agent-payment-integration, agent-quant-analyst, agent-risk-manager, agent-seo-specialist

### 📊 비즈니스 에이전트 (agent-business) — 10개
agent-business-analyst, agent-content-marketer, agent-customer-success-manager, agent-legal-advisor, agent-product-manager, agent-project-manager, agent-sales-engineer, agent-scrum-master, agent-technical-writer, agent-ux-researcher

### 🎼 오케스트레이션 에이전트 (agent-meta) — 8개
agent-agent-organizer, agent-context-manager, agent-error-coordinator, agent-knowledge-synthesizer, agent-multi-agent-coordinator, agent-performance-monitor, agent-task-distributor, agent-workflow-orchestrator

### 🔍 리서치 에이전트 (agent-research) — 8개
agent-competitive-analyst, agent-data-researcher, agent-market-researcher, agent-research-analyst, agent-search-specialist, agent-trend-analyst, agent-datadog-api-expert, agent-datadog-pro

### 📖 개발 가이드 (guides) — 15개
guide-documentation, guide-git, guide-github-actions, guide-golang, guide-hmhco, guide-mcp-reference, guide-opus-4-5-agent, guide-opus-4-5, guide-python, guide-react, guide-testing, guide-version-discovery, guide-opus-migration, guide-hooks, guide-claude-md

### ⌨️ CLI 커맨드 (commands) — 11개
cmd-cr-fx, cmd-cr, cmd-deep-research, cmd-explore, cmd-git-cm, cmd-git-cp, cmd-git-ff, cmd-git-fr, cmd-git-pr, cmd-git-prune, cmd-git-sync

---

## 2. 병렬 실행 그룹 (SKILL_GROUPS — 7개)

> 정의 위치: `app.py` → `SKILL_GROUPS` (line 936)
> 같은 그룹 내 스킬은 1개 에이전트가 처리, 다른 그룹은 별도 에이전트로 병렬 실행

| 그룹 | 모델 크기 | 포함 카테고리 | 비고 |
|------|----------|-------------|------|
| **scientific** | large | bioinformatics, bio-databases, cheminformatics, materials-physics, clinical, lab-automation | 대형 모델 필요 |
| **data-analysis** | medium | data-ml, finance, writing-comm 일부 | |
| **code** | medium | agent-core-dev, agent-lang-spec, agent-quality, agent-dx, dev-tools | 코드 관련 전체 |
| **infrastructure** | small | agent-infra (cloud, devops, k8s, network, sre) | |
| **writing-docs** | medium | writing-comm, utilities 일부 (drawio, docx, xlsx, pdf, pptx) | 문서 생성 |
| **ai-business** | small | ai-prompt, agent-business, agent-domain, agent-meta, agent-research, guides, commands | |
| **search** | small | logpresso-search, knowledge-search, logpresso-query, perplexity-search, parallel-web, bgpt-paper-search, research-lookup | **pre_process: True** (선처리) |

---

## 3. SKILL.md 파일 보유 스킬 (11개)

> 경로: `SKILL/scientific-assistant/scientific-skills/*/SKILL.md`
> 나머지 344개 스킬은 SKILL.md 없이 `DOMAIN_SKILLS`에만 등록됨

| 스킬 ID | 제목 | 언어 | 비고 |
|---------|------|------|------|
| code-assistant | 코드 분석 전문 스킬 | KR | 코드 업로드 시 자동 로드 |
| drawio-diagram | Draw.io 다이어그램 생성 | KR | 아키텍처/플로우 XML 생성 |
| frontend-design | Frontend Design | EN | 프로덕션 UI 생성 |
| knowledge-search | 도메인 지식 검색 | KR | **수동 전용** (MANUAL_ONLY) |
| logpresso-query | LPQL 쿼리 작성 | KR | 자연어 → LPQL 변환 |
| logpresso-search | 로그프레소 조회 | KR | 서버 직접 쿼리 실행 |
| owasp-security | OWASP Security Review | EN | OWASP Top 10 2025 체크 |
| react-best-practices | React Best Practices | EN | Vercel 성능 가이드 |
| skill-creator | Skill Creator | EN | 스킬 개발 라이프사이클 |
| web-design-guidelines | Web Interface Guidelines | EN | UI 접근성/UX 감사 |

---

## 4. CEO 멀티에이전트 리서치 모드별 스킬 배정

> 정의 위치: `UIO/index.html` → `RESEARCH_MODES`

### 🔬 연구하기

| 에이전트 | 역할 | 스킬 |
|---------|------|------|
| SEO | 데이터 분석 | agent-data-analyst, statistical-analysis |
| YOON | PPT 작성 | pptx |
| LEE | 구조도 작성 | drawio-diagram |
| PARK | 보고서 작성 | scientific-writing |
| KIM | 차트 생성 | matplotlib |

### 💻 프로그램 만들기

| 에이전트 | 역할 | 스킬 |
|---------|------|------|
| SEO | 설계 | agent-api-designer |
| KIM | 코드 작성 | agent-python-pro |
| LEE | 아키텍처 | drawio-diagram |
| PARK | 문서 작성 | markdown-mermaid-writing |
| CHOI | 보안 검토 | agent-security-engineer |

### 📊 데이터 분석하기

| 에이전트 | 역할 | 스킬 |
|---------|------|------|
| SEO | 통계 분석 | statistical-analysis, exploratory-data-analysis |
| KIM | 시각화 | matplotlib |
| PARK | 보고서 | scientific-writing |

### 🔍 소스코드 분석/추가

| 에이전트 | 역할 | 스킬 |
|---------|------|------|
| SEO | 코드 분석 | agent-code-reviewer, debugging |
| KIM | 코드 작성/추가 | agent-python-pro |
| LEE | 아키텍처 분석 | drawio-diagram |
| PARK | 문서화 | markdown-mermaid-writing |
| CHOI | 보안 검토 | agent-security-engineer |

---

## 5. 에이전트 팀 구성

> 정의 위치: `UIO/index.html` → `agentBaseData`

| 에이전트 | 색상 | 전문 분야 |
|---------|------|----------|
| **CEO** | Gold (#f59e0b) | 총괄 지휘, 보고서 합성 |
| **SEO** | Blue (#00b4d8) | 데이터 분석 |
| **YOON** | Purple (#c084fc) | PPT/프레젠테이션 |
| **LEE** | Green (#4ade80) | DrawIO 다이어그램 |
| **PARK** | Gold (#e2b714) | 마크다운 문서 |
| **KIM** | Red (#ff6b6b) | 차트/시각화 |
| **CHOI** | Cyan (#00e5ff) | 로그프레소/보안 |
| **JUNG** | Indigo (#a78bfa) | 지식 검색 |
