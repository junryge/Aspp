#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
벡터 저장소 사전 생성 스크립트
LLM 없이 임베딩만 생성하여 저장
날짜/시간 파싱 및 검색 최적화 버전
"""

import os
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
import time
import json

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class VectorStoreBuilder:
    """벡터 저장소만 생성하는 클래스"""
    
    def __init__(self, embedding_model_path: str = "./models/paraphrase-multilingual-MiniLM-L12-v2"):
        print("임베딩 모델 로딩 중...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'trust_remote_code': True
            },
            encode_kwargs={'normalize_embeddings': True}
        )
        print("임베딩 모델 로딩 완료!")
        
        self.columns = [
            "CURRTIME", "TOTALCNT", "M14AM10A", "M10AM14A", "M14AM10ASUM",
            "M14AM14B", "M14BM14A", "M14AM14BSUM", "M14AM16", "M16M14A", 
            "M14AM16SUM", "TIME"
        ]
    
    def parse_datetime(self, time_str: str) -> Dict[str, str]:
        """
        TIME 필드 파싱: 202508071558 -> 날짜와 시간 분리
        """
        try:
            time_str = str(time_str)
            if len(time_str) == 12 and time_str.isdigit():
                year = time_str[:4]
                month = time_str[4:6]
                day = time_str[6:8]
                hour = time_str[8:10]
                minute = time_str[10:12]
                
                return {
                    'date': f"{year}-{month}-{day}",
                    'time': f"{hour}:{minute}",
                    'year': year,
                    'month': month,
                    'day': day,
                    'hour': hour,
                    'minute': minute,
                    'formatted': f"{year}년 {month}월 {day}일 {hour}시 {minute}분"
                }
        except Exception:
            pass
        
        return {
            'date': 'N/A',
            'time': 'N/A',
            'formatted': time_str
        }
    
    def parse_filename_date(self, filename: str) -> str:
        """
        파일명에서 날짜 추출: 20250807.csv -> 2025-08-07
        """
        try:
            # .csv 제거하고 숫자만 추출
            date_str = filename.replace('.csv', '').replace('.CSV', '')
            if len(date_str) == 8 and date_str.isdigit():
                year = date_str[:4]
                month = date_str[4:6]
                day = date_str[6:8]
                return f"{year}-{month}-{day}"
        except Exception:
            pass
        return filename
    
    def process_csv_batch(self, csv_files: List[str], data_folder: str) -> List[Document]:
        """CSV 파일 배치 처리 - 날짜/시간 파싱 포함"""
        documents = []
        
        for filename in csv_files:
            filepath = os.path.join(data_folder, filename)
            try:
                df = pd.read_csv(filepath)
                
                # 파일명에서 날짜 추출
                file_date = self.parse_filename_date(filename)
                
                # 최대값 미리 계산 (검색 성능 향상)
                max_totalcnt = df['TOTALCNT'].max() if 'TOTALCNT' in df.columns else 0
                max_totalcnt_row = df[df['TOTALCNT'] == max_totalcnt].iloc[0] if max_totalcnt > 0 else None
                
                for idx, row in df.iterrows():
                    # TIME 필드 파싱
                    time_info = self.parse_datetime(row.get("TIME", ""))
                    
                    # 콘텐츠 생성 (한글 설명 제거, 원본 컬럼명만 사용)
                    content_parts = [
                        f"파일명: {filename}",
                        f"파일날짜: {file_date}",
                        f"날짜: {time_info['date']}",
                        f"시간: {time_info['time']}",
                        f"년: {time_info['year']}",
                        f"월: {time_info['month']}",
                        f"일: {time_info['day']}",
                        f"시: {time_info['hour']}",
                        f"분: {time_info['minute']}"
                    ]
                    
                    # 데이터 필드 추가 (원본 컬럼명만 사용)
                    for col in self.columns:
                        if col in df.columns:
                            value = row.get(col, "N/A")
                            content_parts.append(f"{col}: {value}")
                    
                    # 이 행이 최대값인지 표시
                    if max_totalcnt_row is not None and idx == max_totalcnt_row.name:
                        content_parts.append(f"최대TOTALCNT: True")
                        content_parts.append(f"최대값: {max_totalcnt}")
                    
                    content = "\n".join(content_parts)
                    
                    # 메타데이터 강화
                    metadata = {
                        "filename": filename,
                        "file_date": file_date,
                        "row_index": idx,
                        "currtime": row.get("CURRTIME", ""),
                        "totalcnt": int(row.get("TOTALCNT", 0)),
                        "date": time_info['date'],
                        "time": time_info['time'],
                        "year": time_info['year'],
                        "month": time_info['month'],
                        "day": time_info['day'],
                        "hour": time_info['hour'],
                        "minute": time_info['minute'],
                        "is_max_totalcnt": idx == max_totalcnt_row.name if max_totalcnt_row is not None else False
                    }
                    
                    documents.append(Document(page_content=content, metadata=metadata))
                
                print(f"✓ {filename}: {len(df)}개 행 처리 (최대 TOTALCNT: {max_totalcnt})")
                
            except Exception as e:
                print(f"✗ {filename} 오류: {e}")
        
        return documents
    
    def build_vector_stores(self, data_folder: str, batch_size: int = 10):
        """배치별로 벡터 저장소 생성"""
        csv_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])
        total_files = len(csv_files)
        
        print(f"\n총 {total_files}개 CSV 파일 발견")
        print(f"배치 크기: {batch_size}개 파일\n")
        
        # 텍스트 분할기
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len
        )
        
        # 전체 통계 수집
        all_stats = {
            'total_records': 0,
            'max_totalcnt': 0,
            'max_totalcnt_info': {},
            'date_range': {'min': None, 'max': None}
        }
        
        # 배치별 처리
        for i in range(0, total_files, batch_size):
            batch_num = i // batch_size + 1
            batch_files = csv_files[i:i+batch_size]
            
            print(f"=== 배치 {batch_num} 처리 시작 ({len(batch_files)}개 파일) ===")
            start_time = time.time()
            
            # 문서 생성
            documents = self.process_csv_batch(batch_files, data_folder)
            print(f"문서 생성 완료: {len(documents)}개")
            
            # 통계 업데이트
            all_stats['total_records'] += len(documents)
            for doc in documents:
                totalcnt = doc.metadata.get('totalcnt', 0)
                if totalcnt > all_stats['max_totalcnt']:
                    all_stats['max_totalcnt'] = totalcnt
                    all_stats['max_totalcnt_info'] = {
                        'filename': doc.metadata.get('filename'),
                        'date': doc.metadata.get('date'),
                        'time': doc.metadata.get('time'),
                        'value': totalcnt
                    }
            
            # 문서 분할
            split_docs = text_splitter.split_documents(documents)
            print(f"문서 분할 완료: {len(split_docs)}개 청크")
            
            # 벡터 저장소 생성
            print("벡터 임베딩 생성 중...")
            vector_store = FAISS.from_documents(split_docs, self.embeddings)
            
            # 저장
            save_path = f"./vector_stores/batch_{batch_num:03d}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            vector_store.save_local(save_path)
            
            elapsed = time.time() - start_time
            print(f"✓ 배치 {batch_num} 완료! ({elapsed:.1f}초)")
            print(f"  현재까지 최대 TOTALCNT: {all_stats['max_totalcnt']}\n")
        
        # 메타데이터 저장
        self.save_metadata(csv_files, batch_size, all_stats)
        print("모든 배치 처리 완료!")
        print(f"전체 최대 TOTALCNT: {all_stats['max_totalcnt']} ({all_stats['max_totalcnt_info']})")
    
    def save_metadata(self, csv_files: List[str], batch_size: int, stats: Dict[str, Any]):
        """처리 정보 저장"""
        metadata = {
            "total_files": len(csv_files),
            "batch_size": batch_size,
            "total_batches": (len(csv_files) + batch_size - 1) // batch_size,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_list": csv_files,
            "statistics": stats,
            "column_info": {
                "columns": self.columns,
                "descriptions": {
                    "CURRTIME": "현재시간 (YYYYMMDDHHMM)",
                    "TOTALCNT": "전체 카운트",
                    "M14AM10A": "14A에서 10A로 이동한 수",
                    "M10AM14A": "10A에서 14A로 이동한 수",
                    "M14AM10ASUM": "14A-10A 이동 합계",
                    "M14AM14B": "14A에서 14B로 이동한 수",
                    "M14BM14A": "14B에서 14A로 이동한 수",
                    "M14AM14BSUM": "14A-14B 이동 합계",
                    "M14AM16": "14A에서 16으로 이동한 수",
                    "M16M14A": "16에서 14A로 이동한 수",
                    "M14AM16SUM": "14A-16 이동 합계",
                    "TIME": "시간 (YYYYMMDDHHMM)"
                }
            }
        }
        
        with open("./vector_stores/metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"메타데이터 저장 완료: ./vector_stores/metadata.json")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="벡터 저장소 생성")
    parser.add_argument("--data-dir", default="./output_by_date", help="CSV 데이터 폴더")
    parser.add_argument("--batch-size", type=int, default=10, help="배치 크기")
    parser.add_argument("--embedding-model", default="./models/paraphrase-multilingual-MiniLM-L12-v2", 
                       help="임베딩 모델 경로")
    
    args = parser.parse_args()
    
    # 벡터 저장소 빌더 생성
    builder = VectorStoreBuilder(embedding_model_path=args.embedding_model)
    
    # 벡터 저장소 생성
    builder.build_vector_stores(args.data_dir, args.batch_size)
    
    print("\n사용법:")
    print("1. 모든 배치를 합치려면: python merge_vectors.py")
    print("2. LLM과 함께 사용: python search_with_llm.py")

if __name__ == "__main__":
    import torch  # GPU 체크용
    main()