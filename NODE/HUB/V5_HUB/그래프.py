import pandas as pd
import plotly.graph_objects as go

try:
    # 1. CSV 파일 불러오기
    # 이 코드 파일과 HH8.CSV 파일을 같은 폴더에 위치시켜 주세요.
    df = pd.read_csv('HH8.CSV')

    # 2. 날짜 형식 변환
    # '날짜'와 '예측날짜' 컬럼을 컴퓨터가 이해할 수 있는 날짜 형식으로 바꿉니다.
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['예측날짜'] = pd.to_datetime(df['예측날짜'])

    # 3. 정보창에 표시할 날짜/시간 데이터 준비
    # 그래프 정보창에 'YYYY-MM-DD' 형식의 날짜와 'HH:MM:SS' 형식의 시간을 따로 표시하기 위해 새로운 컬럼을 만듭니다.
    df['표시_날짜'] = df['날짜'].dt.strftime('%Y-%m-%d')
    df['표시_시간'] = df['날짜'].dt.strftime('%H:%M:%S')
    df['표시_예측날짜'] = df['예측날짜'].dt.strftime('%Y-%m-%d')
    df['표시_예측시간'] = df['예측날짜'].dt.strftime('%H:%M:%S')

    # 4. 그래프 객체 생성
    fig = go.Figure()

    # 5. '실제값' 라인 추가 (정보창에 '점프예측' O/X 표시)
    fig.add_trace(go.Scatter(
        x=df['날짜'],
        y=df['실제값'],
        mode='lines',
        name='실제값 (Actual)',
        # customdata를 이용해 정보창으로 전달할 데이터를 지정합니다.
        customdata=df[['표시_날짜', '표시_시간', '점프예측']],
        # hovertemplate를 사용해 정보창의 내용을 원하는 형식으로 구성합니다.
        hovertemplate=(
            '<b>날짜</b>: %{customdata[0]}<br>'
            '<b>시간</b>: %{customdata[1]}<br>'
            '<b>실제값</b>: %{y}<br>'
            '<b>점프예측</b>: %{customdata[2]}'
            '<extra></extra>' # 추가 정보(trace 이름) 숨기기
        )
    ))

    # 6. '예측값' 라인 추가
    fig.add_trace(go.Scatter(
        x=df['예측날짜'],
        y=df['예측값'],
        mode='lines',
        name='예측값 (Predicted)',
        line=dict(dash='dot'),
        customdata=df[['표시_예측날짜', '표시_예측시간']],
        hovertemplate=(
            '<b>예측날짜</b>: %{customdata[0]}<br>'
            '<b>예측시간</b>: %{customdata[1]}<br>'
            '<b>예측값</b>: %{y}'
            '<extra></extra>'
        )
    ))

    # 7. 그래프 레이아웃 설정
    fig.update_layout(
        title='실제값 vs 예측값 시계열 분석',
        xaxis_title='타임라인',
        yaxis_title='값',
        hovermode='x unified' # 마우스를 올렸을 때 x축 기준 모든 정보 표시
    )

    # 8. 그래프를 HTML 파일로 저장
    fig.write_html("final_interactive_graph.html")
    print("성공! 'final_interactive_graph.html' 파일을 열어 최종 그래프를 확인하세요.")

except FileNotFoundError:
    print("오류: 'HH8.CSV' 파일을 찾을 수 없습니다. 코드와 CSV 파일이 같은 폴더에 있는지 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")