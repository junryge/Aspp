#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로그프레소 매뉴얼 질의응답 시스템
"""

import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogpressoRAG:
    """로그프레소 매뉴얼 RAG 시스템"""
    
    def __init__(self,
                 model_path,
                 db_path="./logpresso_db",
                 embedding_model_path="./embeddings/all-MiniLM-L6-v2"):
        
        self.model_path = model_path
        self.db_path = db_path
        self.embedding_model_path = embedding_model_path
        
        self._initialize()
    
    def _initialize(self):
        """초기화"""
        logger.info("로그프레소 RAG 초기화 중...")
        
        # 1. 벡터DB 로드
        self._load_vectorstore()
        
        # 2. 임베딩 모델 로드
        self._load_embedding_model()
        
        # 3. LLM 설정
        self._setup_llm()
        
        logger.info("초기화 완료!")
    
    def _load_vectorstore(self):
        """벡터DB 로드"""
        db_file = os.path.join(self.db_path, 'vectordb.pkl')
        
        if not os.path.exists(db_file):
            raise FileNotFoundError(
                f"벡터DB 없음. 먼저 로그프레소_벡터DB생성.py 실행\n"
                f"경로: {db_file}"
            )
        
        logger.info(f"벡터DB 로드: {db_file}")
        
        with open(db_file, 'rb') as f:
            db_data = pickle.load(f)
        
        self.documents = db_data['documents']
        self.embeddings = db_data['embeddings']
        
        logger.info(f"문서 로드 완료: {len(self.documents)}개")
    
    def _load_embedding_model(self):
        """임베딩 모델 로드"""
        logger.info(f"임베딩 모델 로드: {self.embedding_model_path}")
        
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except:
            device = 'cpu'
        
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(self.embedding_model_path)
        if device == 'cuda':
            self.embedding_model = self.embedding_model.to('cuda')
        
        logger.info("임베딩 모델 로드 완료")
    
    def _setup_llm(self):
        """LLM 설정"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일 없음: {self.model_path}")
        
        logger.info(f"LLM 로드: {self.model_path}")
        
        from llama_cpp import Llama
        
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,       # MD 문서는 길 수 있음
            n_batch=256,
            n_gpu_layers=30,
            n_threads=2,
            verbose=False
        )
        
        logger.info("LLM 로드 완료")
    
    def _cosine_similarity(self, a, b):
        """코사인 유사도"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _search_similar(self, query, k=3):
        """유사 문서 검색"""
        query_embedding = self.embedding_model.encode([query])[0]
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, sim, self.documents[i]['content']))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 k개 반환
        results = []
        for idx, score, content in similarities[:k]:
            # 너무 긴 문서는 앞부분만
            truncated = content[:1000] if len(content) > 1000 else content
            results.append(truncated)
        
        return results
    
    def _call_llm(self, prompt, max_tokens=300):
        """LLM 호출"""
        print(f"💬 AI 답변 생성 중... (최대 {max_tokens} 토큰)")
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,  # 매뉴얼은 정확해야 하므로 낮게
            top_p=0.9,
            repeat_penalty=1.2,
            stop=["질문:", "###", "\n\n질문"]
        )
        result = response['choices'][0]['text'].strip()
        print(f"✅ 생성 완료 ({len(result)}자)")
        return result
    
    def ask(self, question):
        """질문하기"""
        print(f"\n🔍 관련 문서 검색 중...")
        
        # 유사 문서 검색
        similar_docs = self._search_similar(question, k=3)
        context = "\n\n---\n\n".join(similar_docs)
        
        print(f"✅ {len(similar_docs)}개 문서 검색 완료")
        
        # LLM 프롬프트
        prompt = f"""당신은 로그프레소(Logpresso) 전문가입니다.

참고 문서:
{context}

질문: {question}

위 참고 문서를 바탕으로 질문에 답변하세요. 
명령어 사용법이나 예제가 있으면 포함해서 설명하세요.
문서에 없는 내용은 "문서에 없습니다"라고 답변하세요.

답변:"""
        
        # LLM 답변 생성
        answer = self._call_llm(prompt, max_tokens=400)
        
        return answer

def main():
    """메인 실행"""
    MODEL_PATH = "Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf"
    
    print("="*60)
    print("로그프레소 매뉴얼 질의응답 시스템")
    print("="*60)
    
    rag = LogpressoRAG(model_path=MODEL_PATH)
    
    print("\n사용 가능합니다! 로그프레소에 대해 질문하세요.")
    print("예: set 명령어 사용법 알려줘")
    print("예: 프로시저란 무엇인가요?")
    print("예: 쿼리 매개변수 설명해줘\n")
    
    # 대화 루프
    while True:
        question = input("\n질문 (종료: quit): ").strip()
        
        if question.lower() in ['quit', 'exit', '종료']:
            print("종료합니다.")
            break
        
        if question:
            answer = rag.ask(question)
            print(f"\n📝 답변:\n{answer}")

if __name__ == "__main__":
    main()