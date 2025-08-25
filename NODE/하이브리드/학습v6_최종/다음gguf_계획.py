"""
오프라인 GGUF CSV RAG 시스템
미리 다운로드한 임베딩 모델 사용
📋 사전 준비사항

임베딩 모델 다운로드 (인터넷 있는 PC에서)

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main
모든 파일을 ./offline_models/all-MiniLM-L6-v2/ 폴더에 저장


폴더 구조
프로젝트/
├── output_by_date/
│   ├── 20240201_data.csv
│   └── ...
├── offline_models/
│   └── all-MiniLM-L6-v2/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
├── phi-4-mini-instruct-q4_k_m.gguf
└── offline_gguf_rag.py

필요 패키지 (오프라인 설치)
bashpip install pandas langchain langchain-community faiss-cpu sentence-transformers llama-cpp-python --no-deps


이제 인터넷 연결 없이 실행 가능합니다!
"""

import os
import glob
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def main():
    print("=" * 60)
    print("📊 오프라인 GGUF CSV RAG 시스템")
    print("=" * 60)
    
    # 1. 오프라인 임베딩 모델 경로 설정
    local_model_path = "./offline_models/all-MiniLM-L6-v2"
    
    # 모델 경로 확인
    if not os.path.exists(local_model_path):
        print(f"❌ 임베딩 모델이 없습니다: {local_model_path}")
        print("\n다음 방법으로 다운로드하세요:")
        print("1. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main")
        print("2. 모든 파일을 다운로드해서 위 경로에 저장")
        return
    
    # 2. CSV 파일들 로드
    folder_path = "./output_by_date"
    print(f"\n📁 CSV 폴더: {folder_path}")
    
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print("❌ CSV 파일이 없습니다!")
        return
    
    print(f"✅ 발견된 파일: {len(csv_files)}개")
    
    # 데이터 로드 (메모리 절약을 위해 일부만)
    all_data = []
    for i, file in enumerate(csv_files[:30]):  # 처음 30개만
        try:
            df = pd.read_csv(file)
            all_data.append(df)
        except:
            pass
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"✅ 로드된 데이터: {len(df)}행")
    
    # 3. 문서 생성
    print("\n📄 문서 생성 중...")
    documents = []
    
    # 전체 통계
    doc_text = f"""전체 데이터 정보:
총 행 수: {len(df)}
컬럼: {', '.join(df.columns)}
"""
    
    # TOTALCNT 통계
    if 'TOTALCNT' in df.columns:
        doc_text += f"""
TOTALCNT 통계:
- 평균: {df['TOTALCNT'].mean():.0f}
- 최대: {df['TOTALCNT'].max()}
- 최소: {df['TOTALCNT'].min()}
- 1400 이상: {(df['TOTALCNT'] >= 1400).sum()}건
- 1500 이상: {(df['TOTALCNT'] >= 1500).sum()}건
"""
    
    documents.append(Document(page_content=doc_text))
    
    # 각 행을 문서로 (샘플)
    for idx, row in df.head(500).iterrows():
        row_text = f"데이터 {idx}: "
        for col in ['CURRTIME', 'TOTALCNT', 'M14AM14B', 'M14AM10A']:
            if col in df.columns:
                row_text += f"{col}={row[col]}, "
        documents.append(Document(page_content=row_text))
    
    print(f"✅ 생성된 문서: {len(documents)}개")
    
    # 4. 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    
    # 5. 오프라인 임베딩 생성
    print("\n🔄 임베딩 생성 중 (오프라인)...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=local_model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ 임베딩 모델 로드 성공")
    except Exception as e:
        print(f"❌ 임베딩 모델 로드 실패: {e}")
        return
    
    # 6. 벡터 스토어
    print("🔄 벡터 스토어 생성 중...")
    vectorstore = FAISS.from_documents(texts, embeddings)
    print("✅ 벡터 스토어 생성 완료")
    
    # 7. GGUF 모델 로드
    print("\n🤖 Phi-4 GGUF 모델 로딩 중...")
    model_path = "./phi-4-mini-instruct-q4_k_m.gguf"
    
    if not os.path.exists(model_path):
        print(f"❌ GGUF 모델이 없습니다: {model_path}")
        return
    
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        max_tokens=256,
        temperature=0.1,
        n_threads=8,
        verbose=False
    )
    print("✅ GGUF 모델 로드 완료")
    
    # 8. QA 체인
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    
    print("\n" + "=" * 60)
    print("✅ 시스템 준비 완료!")
    print("질문 예시:")
    print("- 전체 데이터는 몇 개야?")
    print("- TOTALCNT 평균은?")
    print("- 1500 이상인 데이터는 몇 개?")
    print("=" * 60)
    
    # 9. 질의응답
    while True:
        question = input("\n💬 질문: ")
        if question.lower() in ['quit', 'exit', '종료']:
            print("\n👋 종료합니다.")
            break
            
        try:
            print("\n🤔 생각 중...")
            result = qa_chain({"query": question})
            print("\n🤖 답변:")
            print(result['result'])
        except Exception as e:
            print(f"❌ 오류: {e}")

if __name__ == "__main__":
    main()