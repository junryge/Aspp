#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
로그프레소 쿼리 사용법 전용 벡터 저장 시스템
"""

import os
from typing import List
from datetime import datetime
import time
import json

# LangChain 관련
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class LogpressoQueryVectorBuilder:
    """로그프레소 쿼리 사용법 전용 벡터 저장소"""
    
    def __init__(self, embedding_model_path: str = "./models/paraphrase-multilingual-MiniLM-L12-v2"):
        print("🔄 로그프레소 쿼리 벡터 저장소 초기화...")
        
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🖥️  디바이스: {device}")
        except ImportError:
            device = 'cpu'
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={'device': device, 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ 임베딩 모델 로딩 완료!")
        
    def get_logpresso_query_docs(self) -> str:
        """로그프레소 쿼리 사용법 문서"""
        return """# 로그프레소 쿼리 사용법

## 기본 쿼리 구조
로그프레소 쿼리는 파이프(|)로 연결된 명령어들로 구성됩니다.
데이터 소스 → 변환 → 필터링 → 출력 순서로 진행됩니다.

## 데이터 소스 명령어

### table 명령어
테이블에 저장된 데이터를 조회합니다.
```
table 테이블명
table logs
table access_log
```

### csvfile 명령어
CSV 파일을 읽어옵니다.
```
csvfile /path/to/file.csv
csvfile tab=t /path/to/file.tsv  // TSV 파일용 (탭 구분자)
csvfile encoding=utf-8 /path/to/korean.csv  // 인코딩 지정
```

### jsonfile 명령어
JSON 파일을 읽어옵니다.
```
jsonfile /path/to/file.json
jsonfile overlay=t /path/to/file.json  // 원본 라인도 함께 출력
```

### logger 명령어
실시간 로그 수집기 데이터를 조회합니다.
```
logger 로거명
logger window=1h syslog_logger  // 1시간 동안의 데이터
logger window=1d web_logger     // 1일 동안의 데이터
```

## 데이터 변환 명령어

### parse 명령어
로그를 파싱하여 구조화된 필드로 변환합니다.
```
table raw_log | parse apache_log
table syslog | parse syslog
table iis_log | parse iis
```

### eval 명령어
새로운 필드를 생성하거나 기존 필드를 변경합니다.
```
table logs | eval status_group = if(status < 400, "success", "error")
table logs | eval timestamp = dateformat(_time, "yyyy-MM-dd HH:mm:ss")
table logs | eval size_mb = bytes / 1024 / 1024
```

### rename 명령어
필드명을 변경합니다.
```
table logs | rename src_ip as source_ip, dst_ip as dest_ip
table logs | rename "old name" as new_name
```

### fields 명령어
특정 필드만 선택하거나 제외합니다.
```
table logs | fields _time, src_ip, dst_ip, status
table logs | fields - _raw, _id  // _raw, _id 필드 제외
```

## 필터링 명령어

### where 명령어
조건에 맞는 데이터만 필터링합니다.
```
table logs | where status >= 400
table logs | where src_ip == "192.168.1.100"
table logs | where method == "POST" and status != 200
table logs | where len(url) > 100
```

### search 명령어
텍스트 검색을 수행합니다.
```
table logs | search "error"
table logs | search src_ip="192.168.1.*"
table logs | search NOT status=200
```

### limit 명령어
결과 개수를 제한합니다.
```
table logs | limit 100
table logs | sort _time desc | limit 10  // 최신 10건
```

### head / tail 명령어
처음 또는 마지막 N개 레코드만 출력합니다.
```
table logs | head 50
table logs | tail 20
```

## 통계 및 집계 명령어

### stats 명령어
통계 정보를 계산합니다.
```
table logs | stats count
table logs | stats count by status
table logs | stats avg(response_time), max(bytes) by method
table logs | stats dc(src_ip) as unique_ips  // distinct count
table logs | stats sum(bytes) as total_bytes by date_trunc("1h", _time)
```

#### 통계 함수들
- count: 개수
- sum: 합계
- avg: 평균
- min: 최솟값
- max: 최댓값
- dc: 유니크 개수 (distinct count)
- stdev: 표준편차
- var: 분산
- first: 첫 번째 값
- last: 마지막 값

### timechart 명령어
시간대별 통계 차트를 생성합니다.
```
table logs | timechart span=1h count by status
table logs | timechart span=5m avg(response_time)
table logs | timechart span=1d sum(bytes) as daily_traffic
```

### sort 명령어
데이터를 정렬합니다.
```
table logs | sort _time desc
table logs | sort status asc, _time desc
table logs | stats count by src_ip | sort count desc
```

## 시간 관련 함수

### 시간 필터링
```
table logs | where _time >= "2025-01-01 00:00:00"
table logs | where _time between "2025-01-01" and "2025-01-31"
```

### 날짜 함수들
```
// 현재 시간
table logs | eval now_time = now()

// 날짜 포맷 변경
table logs | eval formatted_time = dateformat(_time, "yyyy-MM-dd")

// 날짜 파싱
table logs | eval parsed_time = dateparse("dd/MMM/yyyy:HH:mm:ss", log_time)

// 날짜 자르기 (시간 단위로 그룹핑용)
table logs | eval hour_group = date_trunc("1h", _time)
```

## 문자열 처리 함수

### 문자열 함수들
```
// 길이
table logs | eval url_length = len(url)

// 부분 문자열
table logs | eval domain = substr(url, 8, 20)

// 대소문자 변환
table logs | eval upper_method = upper(method)
table logs | eval lower_url = lower(url)

// 문자열 분할
table logs | eval url_parts = split(url, "/")

// 정규식 매칭
table logs | eval ip_match = match(src_ip, "192\.168\.\d+\.\d+")

// 문자열 치환
table logs | eval clean_url = replace(url, "%20", " ")
```

## 조건문과 논리 연산

### if 함수
```
table logs | eval status_category = if(status < 300, "success", if(status < 400, "redirect", "error"))
```

### case 함수
```
table logs | eval status_type = case(
    status < 300, "Success",
    status < 400, "Redirect", 
    status < 500, "Client Error",
    "Server Error"
)
```

### 논리 연산자
```
table logs | where status == 200 and method == "GET"
table logs | where status == 404 or status == 500
table logs | where not (method == "OPTIONS")
```

## 데이터 출력 및 저장

### import 명령어
데이터를 테이블에 저장합니다.
```
table source_logs | parse apache_log | import processed_logs
csvfile /path/data.csv | import csv_table
```

### outputcsv 명령어
CSV 파일로 출력합니다.
```
table logs | stats count by status | outputcsv /path/result.csv
```

## 고급 쿼리 예시

### 웹 로그 분석
```sql
// 시간대별 HTTP 상태코드 분포
table web_logs 
| parse apache_log 
| timechart span=1h count by status

// 상위 10개 IP별 요청 수
table web_logs 
| parse apache_log 
| stats count by src_ip 
| sort count desc 
| limit 10

// 404 에러 페이지 분석
table web_logs 
| parse apache_log 
| where status == 404 
| stats count by url 
| sort count desc
```

### 시스템 로그 분석
```sql
// 에러 로그 추출
table system_logs 
| search "ERROR" or "FATAL" 
| fields _time, host, message

// 시간대별 로그 레벨 분포
table system_logs 
| parse syslog 
| timechart span=1h count by level
```

### 보안 로그 분석
```sql
// 실패한 로그인 시도 분석
table auth_logs 
| where event_type == "login_failed" 
| stats count by src_ip, user 
| where count > 10 
| sort count desc

// 비정상 접근 패턴 탐지
table access_logs 
| parse apache_log 
| where status == 401 or status == 403 
| stats count by src_ip, date_trunc("1h", _time) as hour 
| where count > 100
```

## 쿼리 최적화 팁

### 성능 향상 방법
1. where 조건을 가능한 앞쪽에 배치
2. 시간 범위를 명시하여 검색 범위 축소
3. 필요한 필드만 fields 명령으로 선택
4. 인덱스가 있는 필드 활용

### 좋은 쿼리 예시
```sql
// 좋은 예: 조건을 먼저 적용
table logs 
| where _time >= "2025-01-01" and status >= 400 
| parse apache_log 
| fields _time, src_ip, url, status 
| stats count by src_ip

// 나쁜 예: 모든 데이터 파싱 후 필터링
table logs 
| parse apache_log 
| stats count by src_ip 
| where _time >= "2025-01-01" and status >= 400
```

## 자주 사용하는 쿼리 패턴

### 기본 로그 조회
```sql
table logs | limit 100
table logs | where _time >= "2025-01-01" | limit 100
```

### 에러 로그 찾기
```sql
table logs | search "error" or "exception" or "fail"
table web_logs | parse apache_log | where status >= 400
```

### 통계 분석
```sql
table logs | stats count by host, level
table logs | timechart span=1h count
```

### 상위 N개 분석
```sql
table logs | stats count by src_ip | sort count desc | limit 10
```

이 문서는 로그프레소 쿼리의 핵심 사용법을 다루며, 
실무에서 자주 사용되는 패턴들을 포함하고 있습니다."""

    def build_vector_store(self, output_dir: str = "./vector_stores"):
        """로그프레소 쿼리 문서 벡터 저장소 생성"""
        print("\n🔄 로그프레소 쿼리 벡터 저장소 생성 시작...")
        
        # 1. 문서 내용 생성
        docs_content = self.get_logpresso_query_docs()
        
        # 2. 섹션별로 분할하여 Document 객체 생성
        sections = docs_content.split('\n## ')
        documents = []
        
        for i, section in enumerate(sections):
            if i == 0:
                title = "로그프레소 쿼리 사용법"
                content = section
            else:
                lines = section.split('\n', 1)
                title = lines[0].strip()
                content = f"## {section}"
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": "로그프레소 쿼리 문서",
                    "section": title,
                    "section_index": i
                }
            )
            documents.append(doc)
        
        print(f"📄 총 {len(documents)}개 섹션 생성")
        
        # 3. 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n### ", "\n```", "\n", "```", ". ", " "],
            length_function=len
        )
        
        split_docs = text_splitter.split_documents(documents)
        print(f"✂️  총 {len(split_docs)}개 청크 생성")
        
        # 4. 벡터 저장소 생성
        print("🔍 벡터 임베딩 생성 중...")
        vector_store = FAISS.from_documents(split_docs, self.embeddings)
        
        # 5. 저장
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "logpresso_query")
        vector_store.save_local(save_path)
        
        # 6. 메타데이터 저장
        metadata = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_documents": len(documents),
            "total_chunks": len(split_docs),
            "doc_type": "logpresso_query_docs"
        }
        
        with open(os.path.join(output_dir, "logpresso_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 벡터 저장소 생성 완료: {save_path}")
        print(f"📊 메타데이터 저장: {os.path.join(output_dir, 'logpresso_metadata.json')}")
        return save_path

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="로그프레소 쿼리 벡터 저장소 생성")
    parser.add_argument("--output-dir", default="./vector_stores", help="벡터 저장소 출력 디렉토리")
    parser.add_argument("--embedding-model", default="./models/paraphrase-multilingual-MiniLM-L12-v2", help="임베딩 모델 경로")
    
    args = parser.parse_args()
    
    # 벡터 저장소 빌더 생성
    builder = LogpressoQueryVectorBuilder(embedding_model_path=args.embedding_model)
    
    # 벡터 저장소 생성
    save_path = builder.build_vector_store(args.output_dir)
    
    print(f"\n🎉 완료! 벡터 저장소 위치: {save_path}")
    print("\n사용법:")
    print("1. LLM 서비스에서 이 벡터 저장소를 로드하여 사용")
    print("2. 로그프레소 쿼리 관련 질문에 대한 답변 제공")

if __name__ == "__main__":
    main()