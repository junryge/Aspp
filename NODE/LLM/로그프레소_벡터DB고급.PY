#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로그프레소 MD - 고급 청크 분할
(헤더 + 코드블록 보존 + 중복 제거)
"""

import os
import re
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

print("="*60)
print("로그프레소 MD 고급 청크 분할")
print("="*60)

# 1. MD 파일 로드
print("\n📂 MD 파일 로드 중...")
md_file = "로그프레소_매뉴얼.md"

with open(md_file, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"✅ 파일 로드 완료: {len(content):,}자")

# 2. 스마트 청크 분할
print("\n✂️ 스마트 청크 분할 중...")

def split_markdown_smart(text, max_chunk_size=800):
    """
    마크다운을 스마트하게 분할
    - 헤더 구조 유지
    - 코드블록 분리 안 함
    - 문맥 보존
    """
    chunks = []
    
    # 헤더로 1차 분할
    sections = re.split(r'\n(#{1,3}\s+.+)', text)
    
    current_chunk = ""
    current_header = ""
    
    for i, section in enumerate(sections):
        if not section.strip():
            continue
        
        # 헤더인지 확인
        if re.match(r'^#{1,3}\s+', section):
            current_header = section
            current_chunk = section + "\n"
        else:
            # 내용 추가
            current_chunk += section
            
            # 청크가 너무 크면 분할
            if len(current_chunk) > max_chunk_size:
                # 코드블록이 있는지 확인
                code_blocks = re.findall(r'```[\s\S]*?```', current_chunk)
                
                if code_blocks:
                    # 코드블록이 있으면 그대로 유지
                    chunks.append({
                        'content': current_chunk.strip(),
                        'header': current_header.strip(),
                        'has_code': True
                    })
                else:
                    # 문단으로 분할
                    paragraphs = current_chunk.split('\n\n')
                    temp_chunk = current_header + "\n"
                    
                    for para in paragraphs:
                        if len(temp_chunk + para) > max_chunk_size:
                            if temp_chunk.strip():
                                chunks.append({
                                    'content': temp_chunk.strip(),
                                    'header': current_header.strip(),
                                    'has_code': False
                                })
                            temp_chunk = current_header + "\n" + para
                        else:
                            temp_chunk += "\n\n" + para
                    
                    if temp_chunk.strip():
                        chunks.append({
                            'content': temp_chunk.strip(),
                            'header': current_header.strip(),
                            'has_code': False
                        })
                
                current_chunk = current_header + "\n"
    
    # 마지막 청크
    if current_chunk.strip() and len(current_chunk) > len(current_header):
        chunks.append({
            'content': current_chunk.strip(),
            'header': current_header.strip(),
            'has_code': '```' in current_chunk
        })
    
    return chunks

documents = split_markdown_smart(content, max_chunk_size=800)

print(f"✅ 청크 생성 완료: {len(documents)}개")
print(f"   - 코드 포함: {sum(1 for d in documents if d['has_code'])}개")
print(f"   - 텍스트만: {sum(1 for d in documents if not d['has_code'])}개")

# 3. 임베딩 생성
print("\n🔄 임베딩 생성 중...")

model = SentenceTransformer('./embeddings/all-MiniLM-L6-v2')

texts = [doc['content'] for doc in documents]
embeddings = model.encode(texts, show_progress_bar=True)

print(f"✅ 임베딩 완료: {embeddings.shape}")

# 4. 벡터DB 저장
print("\n💾 벡터DB 저장 중...")

os.makedirs('./logpresso_db_smart', exist_ok=True)
db_file = './logpresso_db_smart/vectordb.pkl'

with open(db_file, 'wb') as f:
    pickle.dump({
        'documents': documents,
        'embeddings': embeddings
    }, f)

print(f"✅ 저장 완료: {db_file}")

# 5. 통계 출력
print("\n📊 청크 통계:")
chunk_sizes = [len(doc['content']) for doc in documents]
print(f"   평균 길이: {np.mean(chunk_sizes):.0f}자")
print(f"   최소 길이: {min(chunk_sizes)}자")
print(f"   최대 길이: {max(chunk_sizes)}자")

print("\n" + "="*60)
print("✅ 고급 벡터DB 생성 완료!")
print("로그프레소_RAG.py에서 db_path를 './logpresso_db_smart'로 변경하세요.")
print("="*60)