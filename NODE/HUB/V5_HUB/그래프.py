import pandas as pd
import plotly.graph_objects as go

try:
    # 1. CSV 파일 불러오기
    df = pd.read_csv('HH8.CSV')

    # 2. 날짜 형식 변환
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['예측날짜'] = pd.to_datetime(df['예측날짜'])

    # 3. 정보창 표시용 데이터 준비 ('초' 제외)
    df['표시_날짜'] = df['날짜'].dt.strftime('%Y-%m-%d')
    df['표시_시간'] = df['날짜'].dt.strftime('%H:%M')
    df['표시_예측날짜'] = df['예측날짜'].dt.strftime('%Y-%m-%d')
    df['표시_예측시간'] = df['예측날짜'].dt.strftime('%H:%M')

    # 4. 그래프 객체 생성
    fig = go.Figure()

    # 5. '실제값' 라인 추가 (✨ 핵심: 모든 정보를 이 곳 정보창에 집중)
    fig.add_trace(go.Scatter(
        x=df['날짜'],
        y=df['실제값'],
        mode='lines',
        name='실제값 (Actual)',
        # 정보창에 표시할 '같은 열'의 모든 데이터를 customdata로 전달
        customdata=df[['표시_날짜', '표시_시간', '예측값', '표시_예측날짜', '표시_예측시간', '점프예측']],
        # '같은 열'의 모든 데이터를 보여주도록 정보창 형식 지정
        hovertemplate=(
            '<b>--- 측정 시점 (기준) ---</b><br>'
            '<b>시간:</b> %{customdata[0]} %{customdata[1]}<br>'
            '<b>실제값:</b> %{y}<br>'
            '<b>점프예측:</b> %{customdata[5]}<br>'
            '<br><b>--- 해당 시점의 예측 정보 ---</b><br>'
            '<b>예측 대상 시간:</b> %{customdata[3]} %{customdata[4]}<br>'
            '<b>예측값:</b> %{customdata[2]}'
            '<extra></extra>'
        )
    ))

    # 6. '예측값' 라인 추가 (보조 역할)
    fig.add_trace(go.Scatter(
        x=df['예측날짜'],
        y=df['예측값'],
        mode='lines',
        name='예측값 (Predicted)',
        line=dict(dash='dot'),
        # 예측값 라인의 정보창은 단순하게 표시하여 중복을 피함
        hovertemplate='<b>예측값:</b> %{y}<extra></extra>'
    ))

    # 7. 그래프 레이아웃 설정
    fig.update_layout(
        title='실제값 vs 예측값 시계열 분석',
        xaxis_title='타임라인',
        yaxis_title='값',
        hovermode='x unified' # 통합 정보창 사용
    )

    # 8. 그래프를 HTML 파일로 저장
    fig.write_html("final_row_data_graph.html")
    print("성공! 'final_row_data_graph.html' 파일을 열어 최종 그래프를 확인하세요.")

except FileNotFoundError:
    print("오류: 'HH8.CSV' 파일을 찾을 수 없습니다. 코드와 CSV 파일이 같은 폴더에 있는지 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")