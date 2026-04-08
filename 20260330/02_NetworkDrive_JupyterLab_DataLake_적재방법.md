# Network Drive → JupyterLab → DataLake 적재 방법

## 전체 흐름

```
Network Drive (Testlog 원본)
    ↓  마운트 or pysmb
JupyterLab (Python Parsing)
    ↓  DataFrame 변환
DataLake (DB 적재)
```

---

## Step 1: Network Drive 연결

JupyterLab 서버에서 네트워크 드라이브에 접근하는 방법은 3가지입니다.

### 방법 A: CIFS 마운트 (관리자 권한 필요)

서버에서 직접 네트워크 드라이브를 마운트합니다.

```bash
# 마운트 포인트 생성
sudo mkdir -p /mnt/testlog

# 마운트 실행
sudo mount -t cifs //네트워크드라이브IP/공유폴더 /mnt/testlog \
  -o username=YOUR_ID,password=YOUR_PW,domain=YOUR_DOMAIN,vers=3.0,iocharset=utf8

# 확인
ls /mnt/testlog/
```

영구 마운트 (서버 재부팅 후에도 유지):

```bash
# 인증 정보 파일 생성
sudo vi /etc/samba/credentials
# username=YOUR_ID
# password=YOUR_PW
# domain=YOUR_DOMAIN

sudo chmod 600 /etc/samba/credentials

# fstab 등록
echo '//IP/공유폴더 /mnt/testlog cifs credentials=/etc/samba/credentials,vers=3.0,iocharset=utf8 0 0' \
  | sudo tee -a /etc/fstab
```

### 방법 B: pysmb (관리자 권한 없이 Python으로 직접 접근)

```bash
pip install pysmb
```

```python
from smb.SMBConnection import SMBConnection
import os

# 연결 설정
conn = SMBConnection(
    username='YOUR_ID',
    password='YOUR_PW',
    my_name='jupyter-client',
    remote_name='파일서버명',
    domain='YOUR_DOMAIN',
    use_ntlm_v2=True
)
conn.connect('서버IP', 445)

# 공유폴더 내 파일 목록 조회
share_name = '공유폴더명'
files = conn.listPath(share_name, '/Testlog/')

for f in files:
    if not f.isDirectory:
        print(f.filename)

# 파일 다운로드
local_dir = '/tmp/testlog/'
os.makedirs(local_dir, exist_ok=True)

for f in files:
    if f.filename.endswith('.log'):
        local_path = os.path.join(local_dir, f.filename)
        with open(local_path, 'wb') as fp:
            conn.retrieveFile(share_name, f'/Testlog/{f.filename}', fp)
        print(f'Downloaded: {f.filename}')

conn.close()
```

### 방법 C: smbclient (CLI 기반)

```bash
# 파일 목록 확인
smbclient //서버IP/공유폴더 -U 'DOMAIN\USER' -c 'ls Testlog/'

# 파일 다운로드
smbclient //서버IP/공유폴더 -U 'DOMAIN\USER' \
  -c 'cd Testlog; mget *.log'
```

### 사전 확인 사항

- JupyterLab 서버 → 네트워크 드라이브 IP 간 방화벽 오픈 여부 (445 포트)
- 네트워크 드라이브 접근 권한 (AD 계정/공유 폴더 권한)

```bash
# 방화벽 확인
ping 네트워크드라이브IP
telnet 네트워크드라이브IP 445
```

---

## Step 2: Testlog Parsing (Python)

### 기본 구조

```python
import pandas as pd
import os
import glob

# 마운트 방식일 때
log_dir = '/mnt/testlog/'
# pysmb 다운로드 방식일 때
# log_dir = '/tmp/testlog/'

log_files = glob.glob(os.path.join(log_dir, '*.log'))
print(f'총 {len(log_files)}개 파일 발견')
```

### Parsing 함수 (로그 포맷에 맞게 수정 필요)

```python
def parse_testlog(filepath):
    """
    Testlog 파일 1개를 파싱하여 DataFrame으로 변환
    실제 로그 포맷에 맞게 수정 필요
    """
    records = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # === 예시 1: 구분자 기반 (CSV/TSV 유사) ===
            # parts = line.split('\t')  # 또는 split(',')
            # record = {
            #     'timestamp': parts[0],
            #     'test_item': parts[1],
            #     'value': parts[2],
            #     'result': parts[3],
            # }

            # === 예시 2: Key=Value 형태 ===
            # pairs = dict(item.split('=') for item in line.split(';') if '=' in item)
            # record = pairs

            # === 예시 3: 고정폭 포맷 ===
            # record = {
            #     'timestamp': line[0:19],
            #     'code': line[20:30].strip(),
            #     'value': line[31:45].strip(),
            # }

            records.append(record)

    df = pd.DataFrame(records)
    df['source_file'] = os.path.basename(filepath)
    return df
```

### 전체 파일 통합

```python
dfs = []
for f in log_files:
    try:
        df = parse_testlog(f)
        dfs.append(df)
    except Exception as e:
        print(f'Error parsing {f}: {e}')

df_all = pd.concat(dfs, ignore_index=True)
print(f'총 {len(df_all)}건 파싱 완료')
print(df_all.head())
print(df_all.dtypes)
```

### 데이터 정제

```python
# 타입 변환
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
df_all['value'] = pd.to_numeric(df_all['value'], errors='coerce')

# 중복 제거
df_all.drop_duplicates(inplace=True)

# NULL 확인
print(df_all.isnull().sum())
```

---

## Step 3: DataLake 적재

### 방법 A: RDBMS 직접 적재 (Oracle / PostgreSQL / MySQL)

```bash
# 필요 패키지
pip install sqlalchemy cx_Oracle  # Oracle
# pip install sqlalchemy psycopg2  # PostgreSQL
# pip install sqlalchemy pymysql   # MySQL
```

```python
from sqlalchemy import create_engine

# Oracle 예시
engine = create_engine('oracle+cx_oracle://user:password@host:1521/service_name')

# PostgreSQL 예시
# engine = create_engine('postgresql://user:password@host:5432/dbname')

# 적재
df_all.to_sql(
    name='TESTLOG_PARSED',      # 테이블명
    con=engine,
    schema='YOUR_SCHEMA',       # 스키마 (없으면 생략)
    if_exists='append',         # append: 기존에 추가 / replace: 덮어쓰기
    index=False,
    chunksize=5000,             # 대량 데이터 시 chunk 단위 커밋
    method='multi'              # 다중 INSERT (성능 향상)
)
print('DB 적재 완료')
```

### 방법 B: Hive / Spark (Hadoop 기반 DataLake)

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("testlog_ingestion") \
    .enableHiveSupport() \
    .getOrCreate()

sdf = spark.createDataFrame(df_all)

# Hive 테이블로 저장
sdf.write \
    .mode("append") \
    .partitionBy("test_date") \    # 파티션 키 (선택)
    .saveAsTable("datalake_db.testlog_parsed")
```

### 방법 C: 파일 기반 적재 (Parquet / CSV)

```python
# Parquet (권장: 압축 + 컬럼 기반 → 조회 성능 우수)
output_path = '/datalake/경로/testlog_parsed.parquet'
df_all.to_parquet(output_path, index=False, engine='pyarrow')

# CSV
output_path = '/datalake/경로/testlog_parsed.csv'
df_all.to_csv(output_path, index=False, encoding='utf-8-sig')
```

---

## 자동화 (선택)

주기적으로 Testlog를 수집/적재하려면 스케줄러를 활용합니다.

### crontab 예시

```bash
# 매일 새벽 2시 실행
0 2 * * * /usr/bin/python3 /home/user/testlog_pipeline.py >> /var/log/testlog.log 2>&1
```

### Airflow DAG (인프라에 Airflow가 있는 경우)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def run_pipeline():
    # 위 Step 1~3 코드를 함수로 묶어 실행
    pass

dag = DAG('testlog_pipeline', schedule_interval='@daily', start_date=datetime(2026, 1, 1))

task = PythonOperator(
    task_id='ingest_testlog',
    python_callable=run_pipeline,
    dag=dag
)
```

---

## 사전 확인 체크리스트

| 항목 | 확인 내용 |
|------|-----------|
| 네트워크 | JupyterLab → 네트워크 드라이브 IP 방화벽 (445 포트) |
| 권한 | 네트워크 드라이브 읽기 권한 (AD 계정) |
| 로그 포맷 | Testlog 파일 포맷 샘플 (파싱 로직 작성용) |
| DataLake | 적재 대상 DB 접속 정보 (호스트/포트/계정/스키마) |
| DataLake | 테이블 생성 또는 쓰기 권한 |
| Python 패키지 | JupyterLab 서버에 필요 패키지 설치 가능 여부 |

---

## 추가 참고

Testlog 파일 샘플을 공유해주시면 정확한 파싱 코드를 작성해드릴 수 있습니다.
