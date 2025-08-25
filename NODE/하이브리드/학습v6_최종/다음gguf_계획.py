"""
폐쇄망 CSV RAG 시스템 - 단순 버전
phi-4-mini-instruct-q4_k_m.gguf 사용
"""

import os
import glob
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

def main():
    print("=" * 60)
    print("📊 CSV RAG 시스템 (Phi-4 사용)")
    print("=" * 60)
    
    # 1. CSV 파일들 로드
    folder_path = "./output_by_date"
    print(f"\n📁 폴더 검색 중: {folder_path}")
    
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print("❌ CSV 파일이 없습니다!")
        return
    
    print(f"✅ 발견된 파일: {len(csv_files)}개")
    
    # 모든 CSV 읽기
    all_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            all_data.append(df)
        except:
            pass
    
    # 데이터 합치기
    df = pd.concat(all_data, ignore_index=True)
    print(f"✅ 전체 데이터: {len(df)}행")
    
    # 2. 간단한 텍스트 문서 생성
    documents = []
    
    # 기본 정보
    doc_text = f"전체 데이터 수: {len(df)}개\n"
    doc_text += f"컬럼: {', '.join(df.columns)}\n\n"
    
    # 각 행을 텍스트로 변환 (처음 1000행만)
    for idx, row in df.head(1000).iterrows():
        row_text = f"데이터 {idx}: "
        for col in df.columns:
            row_text += f"{col}={row[col]}, "
        row_text += "\n"
        
        # 500자씩 묶어서 문서 생성
        if len(doc_text) > 500:
            documents.append(Document(page_content=doc_text))
            doc_text = ""
        doc_text += row_text
    
    if doc_text:
        documents.append(Document(page_content=doc_text))
    
    print(f"✅ 문서 생성: {len(documents)}개")
    
    # 3. 임베딩 생성
    print("\n🔄 임베딩 생성 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 4. 벡터 스토어
    print("🔄 벡터 스토어 생성 중...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # 5. Phi-4 모델 로드
    print("\n🤖 Phi-4 모델 로딩 중...")
    model_path = "./phi-4-mini-instruct-q4_k_m.gguf"
    
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        max_tokens=256,
        temperature=0.1,
        verbose=False
    )
    
    # 6. QA 체인
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    print("\n✅ 준비 완료! 질문을 입력하세요.")
    print("=" * 60)
    
    # 7. 질의응답
    while True:
        question = input("\n💬 질문: ")
        if question.lower() in ['quit', 'exit']:
            break
            
        try:
            result = qa_chain({"query": question})
            print("\n🤖 답변:", result['result'])
        except Exception as e:
            print(f"❌ 오류: {e}")

if __name__ == "__main__":
    main()