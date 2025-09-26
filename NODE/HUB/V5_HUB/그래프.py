import pandas as pd
import plotly.graph_objects as go

try:
    # 1. CSV 파일 불러오기
    # 코드 파일과 HH8.CSV 파일을 같은 폴더에 위치시켜 주세요.
    df = pd.read_csv('HH8.CSV')

    # 2. 날짜 형식 변환
    # '날짜'와 '예측날짜' 컬럼을 컴퓨터가 이해할 수 있는 날짜 형식으로 바꿉니다.
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['예측날짜'] = pd.to_datetime(df['예측날짜'])

    # 3. 그래프 객체 생성
    fig = go.Figure()

    # 4. '실제값' 라인 추가
    fig.add_trace(go.Scatter(
        x=df['날짜'],
        y=df['실제값'],
        mode='lines',
        name='실제값 (Actual)',
        hovertemplate='<b>날짜</b>: %{x}<br><b>실제값</b>: %{y}<extra></extra>'
    ))

    # 5. '예측값' 라인 추가 (점선으로 표시)
    fig.add_trace(go.Scatter(
        x=df['예측날짜'],
        y=df['예측값'],
        mode='lines',
        name='예측값 (Predicted)',
        line=dict(dash='dot'),
        hovertemplate='<b>예측날짜</b>: %{x}<br><b>예측값</b>: %{y}<extra></extra>'
    ))

    # 6. '점프예측' 지점 추가 (빨간 별 모양)
    # '점프예측' 컬럼의 값이 'X'가 아닌 경우만 필터링하여 표시합니다.
    jump_df = df[df['점프예측'] != 'X']
    fig.add_trace(go.Scatter(
        x=jump_df['날짜'],
        y=jump_df['실제값'], # 점프 지점을 실제값 위치에 표시
        mode='markers',
        name='점프예측 (Jump)',
        marker=dict(color='red', size=10, symbol='star'),
        hovertemplate='<b>점프예측!</b><br><b>날짜</b>: %{x}<br><b>값</b>: %{y}<extra></extra>'
    ))

    # 7. 그래프 레이아웃 설정
    fig.update_layout(
        title='실제값 vs 예측값 시계열 분석',
        xaxis_title='날짜',
        yaxis_title='값',
        hovermode='x unified' # 마우스를 올렸을 때 x축 기준 모든 정보 표시
    )

    # 8. 그래프를 HTML 파일로 저장
    # 이 코드를 실행하면 'interactive_graph.html' 파일이 생성됩니다.
    fig.write_html("interactive_graph.html")
    print("성공! 'interactive_graph.html' 파일을 열어 대화형 그래프를 확인하세요.")

except FileNotFoundError:
    print("오류: 'HH8.CSV' 파일을 찾을 수 없습니다. 코드와 CSV 파일이 같은 폴더에 있는지 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")