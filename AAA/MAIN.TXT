"""
Domos(민중) 프로젝트 베타 V0.2 - Flask 웹앱
=======================================
사용법:
  1. scientific-skills 폴더를 이 파일과 같은 위치에 복사
  2. TOKEN.TXT 파일에 API 키를 넣어두기 (같은 폴더)
  3. pip install flask requests
  4. python app.py
  5. 브라우저에서 http://localhost:10009 접속

폴더 구조:
  app.py
  TOKEN.TXT              ← API 키 (한 줄)
  scientific-skills/
    ├── biopython/
    │   └── SKILL.md
    ├── rdkit/
    │   └── SKILL.md
    └── ... (나머지 스킬 폴더들)
"""

import os
import sys
import io
import csv
import json
import glob
import requests as req
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 제한

# GGUF 모델 (llama-cpp-python)
gguf_model = None  # Llama 인스턴스

# 업로드된 CSV 데이터 (세션별 - 단순 메모리 저장)
uploaded_csv_data = {
    "filename": "",
    "headers": [],
    "rows": [],
    "summary": "",
    "raw_preview": "",
}

# ============================================
# 설정
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(BASE_DIR, "scientific-skills")
TOKEN_FILE = os.path.join(BASE_DIR, "TOKEN.TXT")

# 회사 LLM API 환경 설정
ENV_CONFIG = {
    "dev": {
        "url": "http://dev.hcp.llm.skhynix.com/v1/chat/completions",
        "model": "GLM-4.7",
        "name": "4.7"
    },
    "prod": {
        "url": "http://dev.hcp.llm.skhynix.com/v1/chat/completions",
        "model": "Qwen3.5-397B-A17B",
        "name": "PROD (397B)"
    },
    "common": {
        "url": "http://dev.hcp.llm.skhynix.com/v1/chat/completions",
        "model": "gpt-oss-120b",
        "name": "COMMON (120B)"
    },
    # gguf-local은 앱 시작 시 자동 감지되면 추가됨
}


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

# 174개 스킬 한글 설명
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
    "zarr-python":"청크 N-D 배열 저장",
    "pyopenms":"질량분석 데이터 처리",
    "scikit-survival":"생존 분석 ML",
}

# 분야별 스킬 매핑 (174개 전체 분류)
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
            "docx","xlsx","pdf","pptx","markitdown","matlab",
            "modal","generate-image","get-available-resources",
            "bgpt-paper-search","research-lookup","perplexity-search",
            "parallel-web","open-notebook","consciousness-council",
            "what-if-oracle","hypogenic","dhdna-profiler","denario",
            "offer-k-dense-web","zarr-python",
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


def load_gguf_model(model_path, n_ctx=8192, n_gpu_layers=99):
    """llama-cpp-python으로 GGUF 모델 로드"""
    global gguf_model
    try:
        from llama_cpp import Llama
        print(f"     모델 로딩 중: {os.path.basename(model_path)}...")

        # Windows: llama.cpp C 라이브러리가 stdout/stderr 핸들을 건드려서
        # Flask(click/colorama) 콘솔 출력이 깨지는 문제 방지
        saved_stdout = sys.stdout
        saved_stderr = sys.stderr
        try:
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

        return True
    except ImportError:
        print(f"     ❌ llama-cpp-python 패키지 없음")
        print(f"        → pip install llama-cpp-python")
        return False
    except Exception as e:
        print(f"     ❌ 모델 로드 실패: {e}")
        return False


def gguf_chat(messages, temperature=0.5, max_tokens=4096):
    """로드된 GGUF 모델로 채팅"""
    global gguf_model
    if gguf_model is None:
        return None, "GGUF 모델이 로드되지 않았습니다."
    try:
        resp = gguf_model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if resp and "choices" in resp and len(resp["choices"]) > 0:
            return resp["choices"][0]["message"]["content"], None
        return None, f"예상치 못한 응답: {resp}"
    except Exception as e:
        return None, f"GGUF 추론 오류: {str(e)}"


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
            "stdout": result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout,
            "stderr": result.stderr[-4000:] if len(result.stderr) > 4000 else result.stderr,
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


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """LLM API 프록시 - 스킬을 시스템 프롬프트에 넣어서 회사 API로 전달"""
    data = request.json
    # 환경 선택 시 서버에서 URL/모델 결정, 토큰은 TOKEN.TXT에서 읽음
    env_id = data.get("env", "")
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
    max_tokens = data.get("max_tokens", 4096)

    if not api_url or not model:
        return jsonify({"error": "API URL과 모델 이름을 설정해주세요."}), 400

    # 시스템 프롬프트 구성
    default_prompt = "당신은 Domos(민중) - 과학 연구를 돕는 전문 AI 어시스턴트입니다.\n반드시 한국어(한글)로 답변하세요. 코드 주석도 한글로 작성하세요.\n\n"

    if custom_system_prompt:
        system_prompt = custom_system_prompt + "\n\n" + default_prompt
    else:
        system_prompt = default_prompt

    # 스킬 로드 (SKILL.md + references 요약 + scripts 목록 포함)
    loaded = []
    available = scan_skills()
    for sid in skill_ids:
        content = load_skill_content(sid)
        if content:
            system_prompt += f"=== SKILL: {sid} ===\n{content}\n\n"
            loaded.append(sid)

            # scripts 목록 추가
            info = available.get(sid, {})
            scripts = info.get("scripts", [])
            if scripts:
                script_list = ", ".join([s["name"] for s in scripts])
                system_prompt += f"[{sid} 실행 가능 스크립트: {script_list}]\n"

            # references 요약 (파일명만)
            refs = info.get("references", [])
            if refs:
                ref_list = ", ".join([r["name"] for r in refs])
                system_prompt += f"[{sid} 참고 문서: {ref_list}]\n"

            system_prompt += "\n"

    if loaded:
        system_prompt += f"[로드된 스킬: {', '.join(loaded)}]\n\n"

    # CSV 데이터가 업로드되어 있으면 시스템 프롬프트에 포함
    include_csv = data.get("include_csv", True)
    if include_csv and uploaded_csv_data["filename"]:
        csv_info = uploaded_csv_data["summary"]
        # 데이터 미리보기 (최대 50행)
        preview_limit = min(50, len(uploaded_csv_data["rows"]))
        csv_rows_text = ",".join(uploaded_csv_data["headers"]) + "\n"
        for row in uploaded_csv_data["rows"][:preview_limit]:
            csv_rows_text += ",".join(row) + "\n"
        if len(uploaded_csv_data["rows"]) > preview_limit:
            csv_rows_text += f"... (총 {len(uploaded_csv_data['rows'])}행 중 {preview_limit}행만 표시)\n"

        system_prompt += f"=== 업로드된 CSV 데이터 ===\n{csv_info}\n\n데이터 미리보기:\n{csv_rows_text}\n\n"
        system_prompt += "사용자가 이 데이터에 대해 질문하면 위 CSV 데이터를 기반으로 분석해주세요.\n\n"

    # 응답 수준
    effort_map = [
        "매우 간결하게 핵심만 답변하세요.",
        "간결하게 답변하세요.",
        "표준적인 깊이로 설명하세요.",
        "매우 상세하고 전문적으로 분석하세요. 코드에 주석을 자세히 달고, 원리를 설명하세요.",
    ]
    format_map = {
        "code": "답변을 Python 코드 중심으로 작성하세요.",
        "report": "답변을 보고서 형식으로 작성하세요.",
        "table": "답변에 표를 적극 활용하세요.",
        "step-by-step": "답변을 단계별로 작성하세요.",
    }

    system_prompt += effort_map[min(effort, 3)] + "\n"
    system_prompt += format_map.get(output_format, "") + "\n"
    if writing_style:
        system_prompt += f"작성 스타일: {writing_style}\n"

    # API 요청 구성
    api_messages = [{"role": "system", "content": system_prompt}] + messages
    temperature_map = [0.1, 0.3, 0.5, 0.7]

    # ===== GGUF 로컬 모델: Python에서 직접 추론 =====
    if env_id == "gguf-local":
        if gguf_model is None:
            return jsonify({"error": "GGUF 모델이 로드되지 않았습니다. .gguf 파일과 llama-cpp-python이 필요합니다."}), 400

        answer, err = gguf_chat(
            api_messages,
            temperature=temperature_map[min(effort, 3)],
            max_tokens=max_tokens,
        )
        if err:
            return jsonify({"error": err}), 500

        return jsonify({
            "content": answer,
            "loaded_skills": loaded,
            "system_prompt_length": len(system_prompt),
        })

    # ===== 회사 API: HTTP 요청 =====
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

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
            verify=False,  # 폐쇄망 인증서 문제 대응
        )
        resp.raise_for_status()
        result = resp.json()

        # 응답 추출
        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0]["message"]["content"]
        elif "error" in result:
            answer = f"API 에러: {result['error']}"
        else:
            answer = f"예상치 못한 응답: {json.dumps(result, ensure_ascii=False, indent=2)}"

        return jsonify({
            "content": answer,
            "loaded_skills": loaded,
            "system_prompt_length": len(system_prompt),
        })

    except req.exceptions.Timeout:
        return jsonify({"error": "API 응답 시간 초과 (120초). max_tokens를 줄이거나 API 서버 상태를 확인하세요."}), 504
    except req.exceptions.ConnectionError as e:
        return jsonify({"error": f"API 연결 실패: {str(e)}. URL을 확인하세요."}), 502
    except req.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        if code == 401 or code == 403:
            return jsonify({"error": f"인증 실패 ({code}): TOKEN.TXT의 API 키를 확인하세요."}), code
        return jsonify({"error": f"API HTTP 에러 ({code}): {str(e)}"}), code or 500
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
<title>Domos(민중) 베타 V0.2</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f8f7f4;color:#1a1a1a;display:flex;height:100vh}
.sidebar{width:250px;background:#fff;border-right:1px solid #e5e3de;padding:20px 16px;display:flex;flex-direction:column;overflow-y:auto}
.sidebar-logo{font-size:22px;font-weight:700;margin-bottom:24px}.sidebar-logo span{color:#6366f1}
.sidebar-btn{width:100%;padding:10px 14px;border-radius:8px;border:none;cursor:pointer;font-size:14px;text-align:left;margin-bottom:4px;background:transparent;color:#555;transition:all .15s}
.sidebar-btn:hover{background:#f3f2ef}.sidebar-btn.active{background:#6366f1;color:#fff}
.session-item{display:flex;align-items:center;padding:6px 10px;border-radius:6px;cursor:pointer;font-size:12px;color:#555;transition:all .12s;gap:6px;margin:2px 4px}
.session-item:hover{background:#f3f2ef}
.session-item.current{background:#eef2ff;color:#6366f1;font-weight:600}
.session-item .session-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.session-item .session-del{opacity:0;border:none;background:none;cursor:pointer;font-size:11px;color:#999;padding:0 2px}
.session-item:hover .session-del{opacity:1}
.session-item .session-del:hover{color:#ef4444}
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
.chat-box-fixed{position:fixed;bottom:0;left:250px;right:0;background:#fff;border-top:2px solid #e5e3de;padding:12px 32px;z-index:100;box-shadow:0 -2px 10px rgba(0,0,0,.05)}
.chat-box-fixed:focus-within{border-top-color:#6366f1}
.chat-box-fixed-inner{max-width:800px;margin:0 auto}
.chat-input{width:100%;border:none;outline:none;font-size:15px;resize:none;min-height:40px;max-height:200px;font-family:inherit;line-height:1.5}
.chat-input::placeholder{color:#aaa}
.chat-footer{display:flex;justify-content:space-between;align-items:center;margin-top:4px}
.send-btn{width:40px;height:40px;border-radius:50%;border:none;background:#6366f1;color:#fff;cursor:pointer;font-size:18px;transition:background .15s}
.send-btn:hover{background:#4f46e5}
.send-btn:disabled{background:#ccc;cursor:not-allowed}
.content{padding-bottom:100px!important}
/* Tags */
.tag-row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:24px}
.tag{padding:8px 18px;border-radius:20px;border:2px solid #e5e3de;font-size:13px;font-weight:500;cursor:pointer;transition:all .15s;background:#fff;user-select:none}
.tag:hover{border-color:#6366f1}.tag.selected{border-color:var(--c,#6366f1);background:var(--bg,#eef2ff);color:var(--fg,#4f46e5)}
/* Skills */
.skill-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px;margin-bottom:24px;max-height:220px;overflow-y:auto;padding:4px}
.skill-card{padding:10px 14px;border-radius:10px;border:2px solid #e5e3de;background:#fff;cursor:pointer;transition:all .15s;font-size:13px;position:relative}
.skill-card:hover{border-color:#6366f1;transform:translateY(-1px)}
.skill-card.selected{border-color:#6366f1;background:#eef2ff}
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
.sysprompt-preview{font-size:11px;color:#6366f1;background:#eef2ff;padding:6px 10px;border-radius:6px;margin-top:6px;max-height:60px;overflow:hidden;cursor:pointer}
.sysprompt-preview:hover{max-height:200px;overflow-y:auto}
/* CSV Upload */
.csv-section{margin-bottom:24px}
.csv-upload-area{border:2px dashed #d1d5db;border-radius:12px;padding:20px;text-align:center;cursor:pointer;transition:all .2s;background:#fafaf8}
.csv-upload-area:hover{border-color:#6366f1;background:#eef2ff}
.csv-upload-area.dragover{border-color:#6366f1;background:#eef2ff;border-style:solid}
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
pre{position:relative}
@media(max-width:768px){.sidebar{display:none}.chat-box-fixed{left:0}.style-row{flex-direction:column}.msg.user{margin-left:16px}.msg.assistant{margin-right:16px}.msg{font-size:12px}}
</style>
</head>
<body>
<div class="sidebar">
  <div class="sidebar-logo"><span>D</span>emos <span style="font-size:12px;color:#999">베타 V0.2</span></div>
  <button class="sidebar-btn active" onclick="createNewSession()">✨ 새 세션</button>
  <div class="sidebar-section">세션 목록</div>
  <div id="sessionList" style="max-height:200px;overflow-y:auto;margin-bottom:8px;"></div>
  <div class="sidebar-section">불러온 스킬 <span class="skill-count" id="loadedCount">0</span></div>
  <div id="loadedSkillsList" style="font-size:12px;color:#666;padding:0 8px;"></div>
  <div class="sidebar-section" style="margin-top:16px;">안내</div>
  <div style="font-size:12px;color:#888;padding:0 8px;line-height:1.5;">
    1. ⚙️ API 설정<br>
    2. 분야/스킬 선택<br>
    3. 질문 입력 후 전송<br><br>
    <strong>SKILL.md 파일이 있는 스킬</strong>만 ✅ 표시됩니다.
  </div>
  <div class="sidebar-footer">
    <div class="credits">🔬 Domos(민중) 베타 V0.2</div>
  </div>
</div>

<div class="main">
  <div class="header">
    <div class="project-title">📁 Domos(민중) 프로젝트</div>
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

      <!-- CSV 업로드 -->
      <div class="csv-section">
        <div class="section-label">📎 데이터 업로드</div>
        <div id="csvUploadArea" class="csv-upload-area" onclick="document.getElementById('csvFileInput').click()"
             ondragover="event.preventDefault();this.classList.add('dragover')"
             ondragleave="this.classList.remove('dragover')"
             ondrop="event.preventDefault();this.classList.remove('dragover');handleCsvDrop(event)">
          <div class="icon">📂</div>
          <div class="label">CSV / TSV 파일을 끌어놓거나 클릭하여 업로드</div>
          <div class="sub">최대 50MB · UTF-8/CP949 자동 인식</div>
          <input type="file" id="csvFileInput" accept=".csv,.tsv,.txt" style="display:none" onchange="handleCsvSelect(event)">
        </div>
        <div id="csvInfoPanel" style="display:none"></div>
      </div>

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
          <textarea class="sysprompt-textarea" id="systemPromptInput" placeholder="시스템 프롬프트를 입력하세요. 비워두면 기본값 사용됩니다.&#10;&#10;예: 당신은 반도체 공정 전문가입니다. DRAM/NAND 관련 질문에 답변하세요."></textarea>
          <div class="sysprompt-info">
            💡 여기 입력한 내용이 스킬 프롬프트 <strong>앞에</strong> 추가됩니다. 비워두면 기본 프롬프트 사용.
            <button style="background:#f3f2ef;border:1px solid #e5e3de;border-radius:4px;padding:2px 8px;font-size:11px;cursor:pointer;margin-left:8px" onclick="resetSysPrompt()">기본값 복원</button>
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

      <div class="style-row">
        <div class="style-group">
          <label>작성 스타일</label>
          <textarea id="writingStyle" placeholder="예: 간결하고 데이터 중심, 학술 논문 톤..."></textarea>
        </div>
        <div class="style-group">
          <label>출력 형식</label>
          <div class="fmt-btns">
            <div class="fmt-btn selected" data-f="code" onclick="selFmt(this)">💻 코드</div>
            <div class="fmt-btn" data-f="report" onclick="selFmt(this)">📄 보고서</div>
            <div class="fmt-btn" data-f="table" onclick="selFmt(this)">📊 표</div>
            <div class="fmt-btn" data-f="step-by-step" onclick="selFmt(this)">📝 단계별</div>
          </div>
        </div>
      </div>

      <div class="quick-row">
        <button class="quick-btn" onclick="qp('이 데이터를 분석해줘')">📊 데이터 분석</button>
        <button class="quick-btn" onclick="qp('업로드된 CSV 데이터의 통계 요약을 보여줘')">📋 CSV 요약</button>
        <button class="quick-btn" onclick="qp('이 데이터로 시각화 코드를 작성해줘')">📈 시각화</button>
        <button class="quick-btn" onclick="qp('Python 코드를 작성해줘')">🐍 코드 작성</button>
        <button class="quick-btn" onclick="qp('이 데이터에서 이상치를 찾아줘')">🔍 이상치 탐색</button>
        <button class="quick-btn" onclick="qp('논문 스타일로 정리해줘')">📝 논문 정리</button>
      </div>

      <div class="messages" id="msgs"></div>
    </div>
  </div>

  <!-- 질문 입력 - 하단 고정 -->
  <div class="chat-box-fixed">
    <div class="chat-box-fixed-inner">
      <textarea class="chat-input" id="input" placeholder="질문을 하거나 수행하려는 분석을 설명하세요..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}" oninput="this.style.height='auto';this.style.height=this.scrollHeight+'px'"></textarea>
      <div class="chat-footer">
        <span style="font-size:12px;color:#bbb">Enter 전송 / Shift+Enter 줄바꿈</span>
        <button class="send-btn" onclick="send()" id="sendBtn">▶</button>
      </div>
    </div>
  </div>
</div>

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
let selEnv = '';
let catalog = {};
let selDomains = ['general-science'];
let selSkills = [];
let selFormat = 'code';
let effort = 2;
let history = [];
let maxTokens = 4096;
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
    if(s.writingStyle) document.getElementById('writingStyle').value = s.writingStyle;
    if(s.systemPrompt) document.getElementById('systemPromptInput').value = s.systemPrompt;
    if(s.selFormat){ selFormat=s.selFormat; document.querySelectorAll('.fmt-btn').forEach(b=>{b.classList.toggle('selected',b.dataset.f===selFormat);}); }
    if(s.effort!==undefined){ effort=s.effort; document.getElementById('effortSlider').value=effort; }
    if(s.selEnv){ selEnv=s.selEnv; renderEnvs(); updateStatus(); }
    renderSessionList();
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
  const icons = {'dev':'🧪','prod':'🚀','common':'🌐','gguf-local':'💻'};
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
  if(selEnv && envs[selEnv]){
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
function renderSkills(){
  const grid = document.getElementById('skillGrid');
  grid.innerHTML = '';
  selDomains.forEach(did=>{
    const d = catalog[did];
    if(!d) return;
    d.skills.forEach(s=>{
      const c = document.createElement('div');
      c.className = 'skill-card' + (selSkills.includes(s.id)?' selected':'') + (!s.available?' unavailable':'');
      const desc = s.desc ? s.desc : '';
      const avail = s.available ? '✅' : '❌';
      // extras: 스크립트/레퍼런스 개수
      let extras = [];
      if(s.scripts > 0) extras.push(`🐍${s.scripts}`);
      if(s.references > 0) extras.push(`📄${s.references}`);
      if(s.assets > 0) extras.push(`📦${s.assets}`);
      const extrasHtml = extras.length ? `<div class="extras">${extras.join(' ')}</div>` : '';
      c.innerHTML = `<div class="sn">${avail} ${s.name}</div><div class="sd">${desc}</div>${extrasHtml}`;
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
  document.getElementById('loadedCount').textContent = selSkills.length;
  document.getElementById('loadedSkillsList').innerHTML = selSkills.map(s=>`<div style="padding:2px 0">✅ ${s}</div>`).join('') || '<div style="color:#bbb">선택된 스킬 없음</div>';
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
}

// ===== Quick prompt =====
function qp(t){ document.getElementById('input').value=t; }

// ===== Chat =====
async function send(){
  const el=document.getElementById('input');
  const text=el.value.trim();
  if(!text) return;
  if(!selEnv){alert('먼저 위에서 LLM 환경을 선택해주세요.');return;}

  addMsg('user',text);
  history.push({role:'user',content:text});
  el.value=''; el.style.height='auto';

  const typing=addTyping();
  document.getElementById('sendBtn').disabled=true;

  try{
    const resp=await fetch('/api/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        env: selEnv,
        messages:history, skills:selSkills, effort,
        format:selFormat,
        writing_style:document.getElementById('writingStyle').value.trim(),
        system_prompt:document.getElementById('systemPromptInput').value.trim(),
        max_tokens:maxTokens,
      })
    });
    const data=await resp.json();
    typing.remove();

    if(data.error){
      addMsg('assistant','❌ '+data.error);
    } else {
      let info = '';
      if(data.loaded_skills && data.loaded_skills.length > 0){
        info = `\n[적용된 스킬: ${data.loaded_skills.join(', ')}] [모델: ${envs[selEnv].name}] (프롬프트 ${data.system_prompt_length}자)`;
      }
      addMsg('assistant', data.content + info);
      history.push({role:'assistant',content:data.content});
    }
  }catch(e){
    typing.remove();
    addMsg('assistant','❌ 서버 연결 실패: '+e.message);
  }
  document.getElementById('sendBtn').disabled=false;
  const inp=document.getElementById('input');
  inp.focus();
  inp.style.height='auto';
  saveCurrentSession();
}

function renderMd(text){
  // 1) escape HTML
  let s=text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  // 2) code blocks (```lang\n...``` 또는 ```lang코드...``` 모두 지원)
  s=s.replace(/```(\w*)\s*([\s\S]*?)```/g,(_,lang,code)=>{
    const cls=lang?` class="language-${lang}"`:'';
    return `<pre><code${cls}>${code.trim()}</code></pre>`;
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
    if(/^<(pre|h[1-4]|ul|ol|table|hr|blockquote)/.test(t)) return t;
    return '<p>'+t.replace(/\n/g,'<br>')+'</p>';
  }).join('\n');
  // single newlines inside remaining text
  return s;
}
function addMsg(role,text){
  const c=document.getElementById('msgs');
  const d=document.createElement('div');
  d.className='msg '+role;
  let html = role==='user' ? text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') : renderMd(text);
  d.innerHTML=`<div class="msg-label">${role==='user'?'나':'Domos'}</div>${html}`;
  c.appendChild(d);
  d.querySelectorAll('pre').forEach(pre=>{
    const btn=document.createElement('button');
    btn.className='copy-btn';btn.textContent='복사';
    btn.onclick=()=>{navigator.clipboard.writeText(pre.textContent.replace('복사',''));btn.textContent='✓';setTimeout(()=>btn.textContent='복사',1500);};
    pre.appendChild(btn);
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
  localStorage.setItem('domos_sessions', JSON.stringify(sessions));
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
    effort: effort,
    writingStyle: document.getElementById('writingStyle')?.value || '',
    systemPrompt: document.getElementById('systemPromptInput')?.value || '',
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
function renderSessionList(){
  const el=document.getElementById('sessionList');
  if(!el) return;
  const sorted = Object.values(sessions).sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0));
  if(sorted.length===0){ el.innerHTML='<div style="font-size:11px;color:#bbb;padding:4px 8px;">저장된 세션 없음</div>'; return; }
  el.innerHTML = sorted.map(s=>`<div class="session-item${s.id===currentSessionId?' current':''}" onclick="loadSession('${s.id}')">
    <span class="session-name">${s.name||'새 세션'}</span>
    <button class="session-del" onclick="event.stopPropagation();deleteSession('${s.id}')" title="삭제">✕</button>
  </div>`).join('');
}
function createNewSession(){
  // Save current before switching
  saveCurrentSession();
  currentSessionId = 'sess_'+Date.now();
  history=[];
  document.getElementById('msgs').innerHTML='';
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
  if(s.writingStyle) document.getElementById('writingStyle').value=s.writingStyle;
  if(s.systemPrompt) document.getElementById('systemPromptInput').value=s.systemPrompt;
  if(s.selFormat){ selFormat=s.selFormat; document.querySelectorAll('.fmt-btn').forEach(b=>{b.classList.toggle('selected',b.dataset.f===selFormat);}); }
  if(s.effort!==undefined){ effort=s.effort; document.getElementById('effortSlider').value=effort; updateEffort(); }
  renderSessionList();
}
function deleteSession(id){
  if(!confirm('이 세션을 삭제할까요?')) return;
  delete sessions[id];
  saveSessions();
  if(id===currentSessionId) createNewSession();
  else renderSessionList();
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
  updateSpBadge();
}
document.getElementById('systemPromptInput').addEventListener('input', updateSpBadge);

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
    {id:'scripts', label:`🐍 스크립트 (${d.scripts.length})`, show: true},
    {id:'references', label:`📄 레퍼런스 (${d.references.length})`, show: true},
    {id:'assets', label:`📦 에셋 (${d.assets.length})`, show: d.assets.length > 0},
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
  }catch(e){}
  closeSkillDetail();
  input.focus();
}

// ===== CSV Upload =====
let csvLoaded = false;
let csvFilename = '';

function handleCsvDrop(e){
  const files = e.dataTransfer.files;
  if(files.length > 0) uploadCsvFile(files[0]);
}
function handleCsvSelect(e){
  const files = e.target.files;
  if(files.length > 0) uploadCsvFile(files[0]);
  e.target.value = ''; // 같은 파일 재업로드 가능
}
async function uploadCsvFile(file){
  const area = document.getElementById('csvUploadArea');
  area.innerHTML = '<div class="icon">⏳</div><div class="label">업로드 중...</div>';

  const formData = new FormData();
  formData.append('file', file);

  try{
    const resp = await fetch('/api/upload_csv', {method:'POST', body:formData});
    const data = await resp.json();

    if(data.error){
      area.innerHTML = `<div class="icon">❌</div><div class="label">${data.error}</div><div class="sub">다시 시도하세요</div>`;
      setTimeout(()=>resetCsvArea(), 3000);
      return;
    }

    csvLoaded = true;
    csvFilename = data.filename;
    area.style.display = 'none';

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

  }catch(e){
    area.innerHTML = `<div class="icon">❌</div><div class="label">업로드 실패: ${e.message}</div>`;
    setTimeout(()=>resetCsvArea(), 3000);
  }
}
function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function removeCsv(){
  await fetch('/api/clear_csv', {method:'POST'});
  csvLoaded = false;
  csvFilename = '';
  resetCsvArea();
  document.getElementById('csvInfoPanel').style.display = 'none';
}
function resetCsvArea(){
  const area = document.getElementById('csvUploadArea');
  area.style.display = '';
  area.innerHTML = `<div class="icon">📂</div><div class="label">CSV / TSV 파일을 끌어놓거나 클릭하여 업로드</div><div class="sub">최대 50MB · UTF-8/CP949 자동 인식</div><input type="file" id="csvFileInput" accept=".csv,.tsv,.txt" style="display:none" onchange="handleCsvSelect(event)">`;
}
</script>
</body>
</html>
"""

# ============================================
# 실행
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("  Domos(민중) 프로젝트 베타 V0.2")
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
    # GGUF 자동 감지 & Python으로 직접 로드
    # ============================================
    gguf_files = find_gguf_files()

    if gguf_files:
        # GGUF 파일 중 가장 큰 것 선택
        best_gguf = max(gguf_files, key=lambda x: x["size_gb"])
        print(f"\n  💻 GGUF 자동 감지!")
        print(f"     모델: {best_gguf['name']} ({best_gguf['size_gb']} GB)")

        if load_gguf_model(best_gguf["path"]):
            ENV_CONFIG["gguf-local"] = {
                "url": "python://llama-cpp-python",
                "model": best_gguf["name"].replace(".gguf", ""),
                "name": f"LOCAL GGUF ({best_gguf['name']})"
            }
            print(f"     ✅ GGUF 모델 로드 완료!")
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
        import io
        try:
            # 기존 stdout/stderr가 죽었는지 테스트
            sys.stdout.write("")
        except OSError:
            sys.stdout = io.TextIOWrapper(
                open(sys.__stdout__.fileno(), "wb", closefd=False),
                encoding="utf-8", errors="replace"
            )
            sys.stderr = io.TextIOWrapper(
                open(sys.__stderr__.fileno(), "wb", closefd=False),
                encoding="utf-8", errors="replace"
            )

    app.run(host="0.0.0.0", port=10009, debug=False)
