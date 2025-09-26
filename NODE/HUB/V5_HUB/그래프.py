import pandas as pd
import plotly.graph_objects as go

try:
    # 1. CSV 파일 불러오기
    df = pd.read_csv('HH8.CSV')

    # 2. 날짜 형식 변환
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['예측날짜'] = pd.to_datetime(df['예측날짜'])

    # 3. 정보창 표시용 데이터 준비 (✨ '초' 제거)
    # strftime 포맷에서 '%S' (초) 부분을 제거하여 'HH:MM'까지만 표시합니다.
    df['표시_날짜'] = df['날짜'].dt.strftime('%Y-%m-%d')
    df['표시_시간'] = df['날짜'].dt.strftime('%H:%M') # 초 제거
    df['표시_예측날짜'] = df['예측날짜'].dt.strftime('%Y-%m-%d')
    df['표시_예측시간'] = df['예측날짜'].dt.strftime('%H:%M') # 초 제거

    # 4. 그래프 객체 생성
    fig = go.Figure()

    # 5. '실제값' 라인 추가
    fig.add_trace(go.Scatter(
        x=df['날짜'],
        y=df['실제값'],
        mode='lines',
        name='실제값 (Actual)',
        customdata=df[['표시_날짜', '표시_시간', '점프예측']],
        hovertemplate=(
            '<b>날짜</b>: %{customdata[0]}<br>'
            '<b>시간</b>: %{customdata[1]}<br>'
            '<b>실제값</b>: %{y}<br>'
            '<b>점프예측</b>: %{customdata[2]}'
            '<extra></extra>'
        )
    ))

    # 6. '예측값' 라인 추가
    fig.add_trace(go.Scatter(
        x=df['예측날짜'],
        y=df['예측값'],
        mode='lines',
        name='예측값 (Predicted)',
        line=dict(dash='dot'),
        customdata=df[['표시_날짜', '표시_시간', '표시_예측날짜', '표시_예측시간']],
        hovertemplate=(
            '<b>측정 시간 (From)</b>: %{customdata[0]} %{customdata[1]}<br>'
            '<b>예측 시간 (To)</b>: %{customdata[2]} %{customdata[3]}<br>'
            '<b>예측값</b>: %{y}'
            '<extra></extra>'
        )
    ))

    # 7. 그래프 레이아웃 설정 (✨ 통합 정보창으로 변경)
    fig.update_layout(
        title='실제값 vs 예측값 시계열 분석',
        xaxis_title='타임라인',
        yaxis_title='값',
        hovermode='x unified' # 마우스를 올린 x축 위치의 모든 정보를 하나의 창에 통합하여 표시
    )

    # 8. 그래프를 HTML 파일로 저장
    fig.write_html("final_unified_graph.html")
    print("성공! 'final_unified_graph.html' 파일을 열어 최종 그래프를 확인하세요.")

except FileNotFoundError:
    print("오류: 'HH8.CSV' 파일을 찾을 수 없습니다. 코드와 CSV 파일이 같은 폴더에 있는지 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")