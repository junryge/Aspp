# -*- coding: utf-8 -*-
"""
수정된 GGUF CSV RAG 시스템
답변 정확도 개선 버전
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
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def main():
    print("=" * 60)
    print("📊 오프라인 GGUF CSV RAG 시스템 (개선 버전)")
    print("=" * 60)
   
    # 1. 임베딩 모델 경로
    local_model_path = "./offline_models/all-MiniLM-L6-v2"
   
    if not os.path.exists(local_model_path):
        print(f"❌ 임베딩 모델이 없습니다: {local_model_path}")
        print("\n온라인 모델 사용을 시도합니다...")
        local_model_path = "sentence-transformers/all-MiniLM-L6-v2"
   
    # 2. CSV 파일 로드
    folder_path = "./output_by_date"
    print(f"\n📁 CSV 폴더: {folder_path}")
   
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print("❌ CSV 파일이 없습니다!")
        return
   
    print(f"✅ 발견된 파일: {len(csv_files)}개")
   
    # 데이터 로드
    all_data = []
    for i, file in enumerate(csv_files[:30]):
        try:
            df = pd.read_csv(file)
            all_data.append(df)
            print(f"  - {os.path.basename(file)}: {len(df)}행")
        except Exception as e:
            print(f"  - 오류: {file} - {e}")
   
    if not all_data:
        print("❌ 로드된 데이터가 없습니다!")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ 총 로드된 데이터: {len(df)}행")
    print(f"✅ 컬럼: {list(df.columns)}")
   
    # 3. 문서 생성 (중요: 더 명확한 문서 생성)
    print("\n📄 문서 생성 중...")
    documents = []
   
    # === 핵심 개선: 정확한 통계 정보를 명시적으로 문서화 ===
    
    # 전체 요약 문서
    summary_text = f"""
데이터 요약 정보:
- 전체 데이터 개수: {len(df)}개
- 전체 행 수: {len(df)}행
- 컬럼 목록: {', '.join(df.columns)}
"""
    documents.append(Document(page_content=summary_text))
   
    # TOTALCNT 통계 (있는 경우)
    if 'TOTALCNT' in df.columns:
        # 정확한 통계 계산
        avg_val = df['TOTALCNT'].mean()
        max_val = df['TOTALCNT'].max()
        min_val = df['TOTALCNT'].min()
        count_1400 = (df['TOTALCNT'] >= 1400).sum()
        count_1500 = (df['TOTALCNT'] >= 1500).sum()
        
        stats_text = f"""
TOTALCNT 통계:
- TOTALCNT 평균: {avg_val:.0f}
- TOTALCNT 최대값: {max_val}
- TOTALCNT 최소값: {min_val}
- 1400 이상인 데이터 개수: {count_1400}개
- 1500 이상인 데이터 개수: {count_1500}개
"""
        documents.append(Document(page_content=stats_text))
        
        # 질문 답변용 명시적 문서
        qa_text = f"""
질문: 전체 데이터는 몇 개야?
답변: {len(df)}개

질문: TOTALCNT 평균은?
답변: {avg_val:.0f}

질문: 1500 이상인 데이터는 몇 개?
답변: {count_1500}개

질문: 1400 이상인 데이터는 몇 개?
답변: {count_1400}개
"""
        documents.append(Document(page_content=qa_text))
   
    # 샘플 데이터 추가 (상위 100개)
    for idx, row in df.head(100).iterrows():
        row_text = f"데이터 인덱스 {idx}: "
        for col in df.columns:
            if pd.notna(row[col]):
                row_text += f"{col}={row[col]}, "
        documents.append(Document(page_content=row_text[:500]))  # 너무 길면 자르기
   
    print(f"✅ 생성된 문서: {len(documents)}개")
   
    # 4. 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,  # 오버랩 증가
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(f"✅ 분할된 텍스트: {len(texts)}개")
   
    # 5. 임베딩 생성
    print("\n🔄 임베딩 생성 중...")
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
    
    # 벡터 스토어 저장 (재사용 가능)
    vectorstore.save_local("./faiss_index")
    print("✅ 벡터 스토어 저장 완료")
   
    # 7. GGUF 모델 로드
    print("\n🤖 GGUF 모델 로딩 중...")
    model_path = "./KoSOLAR-10.7B-v0.2.Q3_K_M.gguf"
   
    if not os.path.exists(model_path):
        print(f"❌ GGUF 모델이 없습니다: {model_path}")
        return
   
    # 콜백 매니저 (스트리밍 출력)
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
   
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        max_tokens=256,
        temperature=0.1,  # 낮은 온도로 정확도 향상
        top_p=0.9,
        n_threads=8,
        callback_manager=callback_manager,
        verbose=False
    )
    print("✅ GGUF 모델 로드 완료")
   
    # 8. 개선된 프롬프트 템플릿
    prompt_template = """아래 문맥을 참고하여 질문에 정확히 답변해주세요.
만약 문맥에 답이 없다면 "정보를 찾을 수 없습니다"라고 답하세요.

문맥:
{context}

질문: {question}

답변: """

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
   
    # 9. QA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 5}  # 더 많은 문서 검색
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True  # 소스 문서도 반환
    )
   
    print("\n" + "=" * 60)
    print("✅ 시스템 준비 완료!")
    print("\n📊 데이터 정보:")
    print(f"  - 총 데이터: {len(df)}개")
    if 'TOTALCNT' in df.columns:
        print(f"  - TOTALCNT 평균: {df['TOTALCNT'].mean():.0f}")
        print(f"  - 1500 이상: {(df['TOTALCNT'] >= 1500).sum()}개")
    print("\n💡 테스트 질문:")
    print("  - 전체 데이터는 몇 개야?")
    print("  - TOTALCNT 평균은?")
    print("  - 1500 이상인 데이터는 몇 개?")
    print("=" * 60)
   
    # 10. 자동 테스트
    print("\n🧪 자동 테스트 실행:")
    test_questions = [
        "전체 데이터는 몇 개야?",
        "TOTALCNT 평균은?",
        "1500 이상인 데이터는 몇 개?"
    ]
    
    for q in test_questions:
        print(f"\n❓ {q}")
        try:
            result = qa_chain({"query": q})
            print(f"✅ {result['result']}")
            # 검색된 문서 확인
            if result.get('source_documents'):
                print(f"   (검색된 문서 {len(result['source_documents'])}개)")
        except Exception as e:
            print(f"❌ 오류: {e}")
    
    # 11. 대화형 질의응답
    print("\n" + "=" * 60)
    print("💬 대화 모드 (종료: quit/exit)")
    print("=" * 60)
    
    while True:
        question = input("\n💬 질문: ")
        if question.lower() in ['quit', 'exit', '종료']:
            print("\n👋 종료합니다.")
            break
           
        try:
            print("\n🤔 검색 중...")
            result = qa_chain({"query": question})
            print("\n🤖 답변:")
            print(result['result'])
            
            # 디버그 정보 (선택적)
            show_debug = input("\n📋 검색된 문서 보기? (y/n): ")
            if show_debug.lower() == 'y':
                print("\n📄 검색된 문서:")
                for i, doc in enumerate(result.get('source_documents', [])[:3]):
                    print(f"\n[문서 {i+1}]")
                    print(doc.page_content[:200])
                    
        except Exception as e:
            print(f"❌ 오류: {e}")

if __name__ == "__main__":
    main()