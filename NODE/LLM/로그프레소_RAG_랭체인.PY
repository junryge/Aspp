#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로그프레소 RAG - 랭체인(LangChain) 버전
"""

import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

print("="*60)
print("로그프레소 RAG - 랭체인 버전")
print("="*60)

# 1. MD 파일 로드
print("\n📂 MD 파일 로드 중...")
md_file = "로그프레소_매뉴얼.md"

with open(md_file, 'r', encoding='utf-8') as f:
    markdown_text = f.read()

print(f"✅ 파일 로드 완료: {len(markdown_text):,}자")

# 2. 마크다운 헤더 기준 분할
print("\n✂️ 마크다운 청크 분할 중...")

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

md_header_splits = markdown_splitter.split_text(markdown_text)

print(f"✅ 청크 생성 완료: {len(md_header_splits)}개")

# 3. 임베딩 모델 설정
print("\n🔄 임베딩 모델 로드 중...")

embeddings = HuggingFaceEmbeddings(
    model_name="./embeddings/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

print("✅ 임베딩 모델 로드 완료")

# 4. 벡터 저장소 생성
print("\n💾 벡터 저장소 생성 중...")

vectorstore = Chroma.from_documents(
    documents=md_header_splits,
    embedding=embeddings,
    persist_directory="./logpresso_chroma_db"
)

vectorstore.persist()

print("✅ 벡터 저장소 생성 완료")

# 5. LLM 설정
print("\n🤖 LLM 로드 중...")

llm = LlamaCpp(
    model_path="Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf",
    n_ctx=2048,
    n_batch=256,
    n_gpu_layers=30,
    temperature=0.3,
    max_tokens=400,
    verbose=False
)

print("✅ LLM 로드 완료")

# 6. 프롬프트 템플릿
prompt_template = """당신은 로그프레소(Logpresso) 전문가입니다.

참고 문서:
{context}

질문: {question}

위 참고 문서를 바탕으로 질문에 답변하세요.
명령어 사용법이나 예제가 있으면 포함해서 설명하세요.
문서에 없는 내용은 "문서에 없습니다"라고 답변하세요.

답변:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 7. QA 체인 생성
print("\n🔗 QA 체인 생성 중...")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

print("✅ QA 체인 생성 완료")

# 8. 질의응답 루프
print("\n" + "="*60)
print("로그프레소 매뉴얼 질의응답 시스템 (랭체인)")
print("="*60)
print("\n사용 가능합니다! 로그프레소에 대해 질문하세요.")
print("예: set 명령어 사용법 알려줘")
print("예: 프로시저란 무엇인가요?\n")

while True:
    question = input("\n질문 (종료: quit): ").strip()
    
    if question.lower() in ['quit', 'exit', '종료']:
        print("종료합니다.")
        break
    
    if question:
        print("\n🔍 검색 및 답변 생성 중...")
        
        result = qa_chain({"query": question})
        
        print(f"\n📝 답변:\n{result['result']}")
        
        # 참고한 문서 출력 (선택)
        print(f"\n📚 참고 문서: {len(result['source_documents'])}개")