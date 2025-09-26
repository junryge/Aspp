import pandas as pd
import plotly.graph_objects as go

try:
    # 1. CSV 파일 불러오기
    df = pd.read_csv('HH8.CSV')

    # 2. 날짜 형식 변환
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['예측날짜'] = pd.to_datetime(df['예측날짜'])

    # --- ✨ 개선된 부분: 정보창에 표시할 날짜/시간 데이터 준비 ---
    # 그래프 정보창에 'YYYY-MM-DD' 형식의 날짜와 'HH:MM:SS' 형식의 시간을 따로 표시하기 위해 새로운 컬럼을 만듭니다.
    df['표시_날짜'] = df['날짜'].dt.strftime('%Y-%m-%d')
    df['표시_시간'] = df['날짜'].dt.strftime('%H:%M:%S')
    df['표시_예측날짜'] = df['예측날짜'].dt.strftime('%Y-%m-%d')
    df['표시_예측시간'] = df['예측날짜'].dt.strftime('%H:%M:%S')
    # ---------------------------------------------------------

    # 3. 그래프 객체 생성
    fig = go.Figure()

    # 4. '실제값' 라인 추가 (✨ 정보창 형식 수정)
    fig.add_trace(go.Scatter(
        x=df['날짜'],
        y=df['실제값'],
        mode='lines',
        name='실제값 (Actual)',
        # customdata를 이용해 새로 만든 날짜/시간 데이터를 정보창으로 전달
        customdata=df[['표시_날짜', '표시_시간']],
        # hovertemplate에서 customdata를 사용해 원하는 형식으로 정보를 표시
        hovertemplate=(
            '<b>날짜</b>: %{customdata[0]}<br>'
            '<b>시간</b>: %{customdata[1]}<br>'
            '<b>실제값</b>: %{y}'
            '<extra></extra>' # 추가 정보(trace 이름) 숨기기
        )
    ))

    # 5. '예측값' 라인 추가 (✨ 정보창 형식 수정)
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

    # 6. '점프예측' 지점 추가
    jump_df = df[df['점프예측'] != 'X'].copy()
    jump_df['표시_날짜'] = jump_df['날짜'].dt.strftime('%Y-%m-%d')
    jump_df['표시_시간'] = jump_df['날짜'].dt.strftime('%H:%M:%S')
    
    fig.add_trace(go.Scatter(
        x=jump_df['날짜'],
        y=jump_df['실제값'],
        mode='markers',
        name='점프예측 (Jump)',
        marker=dict(color='red', size=10, symbol='star'),
        customdata=jump_df[['표시_날짜', '표시_시간']],
        hovertemplate=(
            '<b>점프 예측!</b><br>'
            '<b>날짜</b>: %{customdata[0]}<br>'
            '<b>시간</b>: %{customdata[1]}<br>'
            '<b>값</b>: %{y}'
            '<extra></extra>'
        )
    ))

    # 7. 그래프 레이아웃 설정
    fig.update_layout(
        title='실제값 vs 예측값 시계열 분석',
        xaxis_title='타임라인',
        yaxis_title='값',
        hovermode='x unified'
    )

    # 8. 그래프를 HTML 파일로 저장
    fig.write_html("interactive_graph_v2.html")
    print("성공! 'interactive_graph_v2.html' 파일을 열어 개선된 그래프를 확인하세요.")

except FileNotFoundError:
    print("오류: 'HH8.CSV' 파일을 찾을 수 없습니다. 코드와 CSV 파일이 같은 폴더에 있는지 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")