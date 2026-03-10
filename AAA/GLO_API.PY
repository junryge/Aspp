import requests
import urllib.parse
import pandas as pd
from io import StringIO
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HOST    = "10.40.42.27"
PORT    = 8888
API_KEY = "db1d2335-49cf-e859-3519-1ca132922e38"


def query_carrier(from_dt: str, to_dt: str, carrier: str = "", text: str = "") -> pd.DataFrame:
    """
    Logpresso Carrier/TEXT 조회 함수

    Parameters
    ----------
    from_dt : str  ex) "20260308000000"
    to_dt   : str  ex) "20260308235959"
    carrier : str  ex) "4PDK2966"   (없으면 빈 문자열)
    text    : str  ex) "4PDK2966"   (없으면 빈 문자열)

    Returns
    -------
    pd.DataFrame
    """
    c = carrier.strip()
    t = text.strip()

    if not c and not t:
        raise ValueError("carrier 또는 text 중 하나는 반드시 입력해야 합니다.")

    if c and t:
        cond = f'((CARRIER=="{c}") or (TEXT=="{t}"))'
    elif c:
        cond = f'((CARRIER=="{c}") or (TEXT=="{c}"))'
    else:
        cond = f'((TEXT=="{t}"))'

    q = (
        f'fulltext from={from_dt} to={to_dt} '
        f'(LEVEL=="WELL" or LEVEL=="WARN" or LEVEL=="ERROR" or LEVEL=="FATAL") '
        f'and {cond} '
        f'from ts_data_view_m14a, ts_data_view_m14b, ts_data_view_m16, ts_data_view_m16b '
        f'| fields _time, TIME_EX, MACHINENAME, MACHINETYPE, UNITNAME, CARRIER, COMMANDID, COMMAND, OPERATION_NAME, MESSAGENAME, PROCESS, TRANSACTIONID, TEXT, THREAD, LEVEL, XML, SECSII, RESULTCODE '
        f'| sort _time '
        f'| limit 0 1000 '
        f'| eval No = seq() + 0'
    )

    clean_q = " ".join(q.split())
    encoded = urllib.parse.quote(clean_q, safe="")
    url = f"http://{HOST}:{PORT}/logpresso/httpexport/query.csv?_apikey={API_KEY}&_q={encoded}"

    print("[QUERY]", clean_q)

    resp = requests.get(url, verify=False, timeout=120)
    if resp.status_code == 200:
        df = pd.read_csv(StringIO(resp.text))
        print(f"[OK] {len(df)}건 조회됨")
        return df

    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")


# ─────────────────────────────────────────────
# 직접 실행 시 여기서 값 바꿔서 테스트
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = query_carrier(
        from_dt  = "20260308000000",
        to_dt    = "20260308235959",
        carrier  = "4PDK2966",
        text     = "",           # TEXT 검색어 (선택)
    )
    print(df)