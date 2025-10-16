import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, scrolledtext, ttk
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

# --------------------------------------------------------------------------
# 인코딩 자동 감지 함수 추가
# --------------------------------------------------------------------------
def read_csv_safe(filepath):
    """여러 인코딩을 시도하여 CSV 파일을 안전하게 읽습니다."""
    encodings = ['utf-8', 'cp949', 'euc-kr', 'ms949', 'latin-1']
    
    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except UnicodeDecodeError:
            continue
    
    return pd.read_csv(filepath, encoding='utf-8', errors='ignore')

# --------------------------------------------------------------------------
# 성능 정보 창 클래스
# --------------------------------------------------------------------------
class PerformanceWindow:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("평가 결과 분석 입력")
        self.window.geometry("600x700")
        
        self.values = None
        
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        top_frame = tk.LabelFrame(main_frame, text="📊 전체 성능 지표 입력", padx=10, pady=10)
        top_frame.pack(fill='x', pady=(0, 10))
        
        input_frame = tk.Frame(top_frame)
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="MAE:", font=('Arial', 10)).grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.mae_var = tk.StringVar(value="30.23")
        tk.Entry(input_frame, textvariable=self.mae_var, width=15).grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(input_frame, text="RMSE:", font=('Arial', 10)).grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.rmse_var = tk.StringVar(value="36.02")
        tk.Entry(input_frame, textvariable=self.rmse_var, width=15).grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(input_frame, text="R²:", font=('Arial', 10)).grid(row=2, column=0, sticky='e', padx=5, pady=5)
        self.r2_var = tk.StringVar(value="0.72")
        tk.Entry(input_frame, textvariable=self.r2_var, width=15).grid(row=2, column=1, padx=5, pady=5)
        
        bottom_frame = tk.LabelFrame(main_frame, text="📝 기타 정보 (붙여넣기 가능)", padx=10, pady=10)
        bottom_frame.pack(fill='both', expand=True)
        
        self.text_widget = scrolledtext.ScrolledText(bottom_frame, height=20, width=70, wrap=tk.WORD)
        self.text_widget.pack(fill='both', expand=True)
        
        self.text_widget.insert('1.0', """📊 평가 결과 분석""")
        
        tk.Button(main_frame, text="확인", 
                 command=self.save_and_close,
                 bg='#3498DB', fg='white',
                 font=('Arial', 10, 'bold')).pack(pady=10)
        
    def save_and_close(self):
        self.values = {
            'mae': self.mae_var.get(),
            'rmse': self.rmse_var.get(),
            'r2': self.r2_var.get(),
            'other_info': self.text_widget.get('1.0', 'end-1c')
        }
        self.window.destroy()

# --------------------------------------------------------------------------
# 그래프 생성 로직
# --------------------------------------------------------------------------
def create_graph(params, perf_values=None):
    try:
        df = read_csv_safe(params['file_path'])
        
        actual_x_col = params['actual_x']
        actual_y_col = params['actual_y']
        hubroom_x_col = params['hubroom_x']
        hubroom_y_col = params['hubroom_y']
        predicted1_x_col = params['predicted1_x']
        predicted1_y_col = params['predicted1_y']
        predicted2_x_col = params['predicted2_x']
        predicted2_y_col = params['predicted2_y']
        predicted3_x_col = params['predicted3_x']
        predicted3_y_col = params['predicted3_y']
        
        # Shift 값 가져오기
        predicted1_shift = int(params.get('predicted1_shift', 0))
        predicted2_shift = int(params.get('predicted2_shift', 0))
        predicted3_shift = int(params.get('predicted3_shift', 0))
        
        # 선택된 라인 체크 (params에서 enable 플래그 사용)
        selected_lines = {
            'actual': params.get('enable_actual', True),
            'hubroom': params.get('enable_hubroom', True),
            'predicted1': params.get('enable_predicted1', True),
            'predicted2': params.get('enable_predicted2', True),
            'predicted3': params.get('enable_predicted3', True)
        }
        
        # 선택된 라인의 Y값을 숫자로 변환
        if selected_lines['actual'] and actual_y_col and actual_y_col in df.columns:
            df[actual_y_col] = pd.to_numeric(df[actual_y_col], errors='coerce')
        else:
            selected_lines['actual'] = False
            
        if selected_lines['hubroom'] and hubroom_y_col and hubroom_y_col in df.columns:
            df[hubroom_y_col] = pd.to_numeric(df[hubroom_y_col], errors='coerce')
        else:
            selected_lines['hubroom'] = False
            
        if selected_lines['predicted1'] and predicted1_y_col and predicted1_y_col in df.columns:
            df[predicted1_y_col] = pd.to_numeric(df[predicted1_y_col], errors='coerce')
        else:
            selected_lines['predicted1'] = False
            
        if selected_lines['predicted2'] and predicted2_y_col and predicted2_y_col in df.columns:
            df[predicted2_y_col] = pd.to_numeric(df[predicted2_y_col], errors='coerce')
        else:
            selected_lines['predicted2'] = False
            
        if selected_lines['predicted3'] and predicted3_y_col and predicted3_y_col in df.columns:
            df[predicted3_y_col] = pd.to_numeric(df[predicted3_y_col], errors='coerce')
        else:
            selected_lines['predicted3'] = False
        
        # 최소 1개 라인은 있어야 함
        if not any(selected_lines.values()):
            messagebox.showerror("오류", "최소 1개 이상의 데이터 라인을 선택해야 합니다.")
            return
        
        # 트렌드 데이터 처리
        trend_data = params.get('trend_data', [])
        for trend_info in trend_data:
            x_col = trend_info['x_col']
            y_col = trend_info['y_col']
            if x_col in df.columns and y_col in df.columns:
                df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
                try:
                    df[x_col] = pd.to_datetime(df[x_col])
                except:
                    pass
        
        # NaN 제거 (선택된 컬럼만)
        dropna_cols = []
        if selected_lines['actual']:
            dropna_cols.append(actual_y_col)
        if selected_lines['hubroom']:
            dropna_cols.append(hubroom_y_col)
        if selected_lines['predicted1']:
            dropna_cols.append(predicted1_y_col)
        if selected_lines['predicted2']:
            dropna_cols.append(predicted2_y_col)
        if selected_lines['predicted3']:
            dropna_cols.append(predicted3_y_col)
            
        if dropna_cols:
            df = df.dropna(subset=dropna_cols)
        
        # 점프예측 컬럼 확인
        jump_col = None
        for col in df.columns:
            if '점프예측' in col or 'jump' in col.lower():
                jump_col = col
                break
        
        # 패턴예측 컬럼 확인
        pattern_col = None
        for col in df.columns:
            if '패턴예측' in col or 'pattern' in col.lower():
                pattern_col = col
                break
        
        # 날짜 변환 및 포맷팅 (선택된 라인만)
        if selected_lines['actual']:
            try:
                df[actual_x_col] = pd.to_datetime(df[actual_x_col])
                df['actual_time_str'] = df[actual_x_col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                df['actual_time_str'] = df[actual_x_col].astype(str)
        else:
            df['actual_time_str'] = ''
                
        if selected_lines['hubroom']:
            try:
                df[hubroom_x_col] = pd.to_datetime(df[hubroom_x_col])
                df['hubroom_time_str'] = df[hubroom_x_col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                df['hubroom_time_str'] = df[hubroom_x_col].astype(str)
        else:
            df['hubroom_time_str'] = ''
                
        if selected_lines['predicted1']:
            try:
                df[predicted1_x_col] = pd.to_datetime(df[predicted1_x_col])
                # 원본 시간 문자열 저장 (INFO용 - 변경 안 됨!)
                df['predicted1_time_str_original'] = df[predicted1_x_col].dt.strftime('%Y-%m-%d %H:%M:%S')
                # Shift된 시간 계산 (그래프 표시용) - 분 단위! 오른쪽으로 이동!
                df['predicted1_shifted_time'] = df[predicted1_x_col] + pd.Timedelta(minutes=predicted1_shift)
            except:
                df['predicted1_time_str_original'] = df[predicted1_x_col].astype(str)
                df['predicted1_shifted_time'] = df[predicted1_x_col]
        else:
            df['predicted1_time_str_original'] = ''
                
        if selected_lines['predicted2']:
            try:
                df[predicted2_x_col] = pd.to_datetime(df[predicted2_x_col])
                # 원본 시간 문자열 저장 (INFO용 - 변경 안 됨!)
                df['predicted2_time_str_original'] = df[predicted2_x_col].dt.strftime('%Y-%m-%d %H:%M:%S')
                # Shift된 시간 계산 (그래프 표시용) - 분 단위! 오른쪽으로 이동!
                df['predicted2_shifted_time'] = df[predicted2_x_col] + pd.Timedelta(minutes=predicted2_shift)
            except:
                df['predicted2_time_str_original'] = df[predicted2_x_col].astype(str)
                df['predicted2_shifted_time'] = df[predicted2_x_col]
        else:
            df['predicted2_time_str_original'] = ''
                
        if selected_lines['predicted3']:
            try:
                df[predicted3_x_col] = pd.to_datetime(df[predicted3_x_col])
                # 원본 시간 문자열 저장 (INFO용 - 변경 안 됨!)
                df['predicted3_time_str_original'] = df[predicted3_x_col].dt.strftime('%Y-%m-%d %H:%M:%S')
                # Shift된 시간 계산 (그래프 표시용) - 분 단위! 오른쪽으로 이동!
                df['predicted3_shifted_time'] = df[predicted3_x_col] + pd.Timedelta(minutes=predicted3_shift)
            except:
                df['predicted3_time_str_original'] = df[predicted3_x_col].astype(str)
                df['predicted3_shifted_time'] = df[predicted3_x_col]
        else:
            df['predicted3_time_str_original'] = ''
        
        # 공통 시간축 설정 (첫 번째 선택된 라인의 X축 사용)
        if selected_lines['actual']:
            df['common_time'] = df[actual_x_col]
        elif selected_lines['hubroom']:
            df['common_time'] = df[hubroom_x_col]
        elif selected_lines['predicted1']:
            df['common_time'] = df[predicted1_x_col]
        elif selected_lines['predicted2']:
            df['common_time'] = df[predicted2_x_col]
        elif selected_lines['predicted3']:
            df['common_time'] = df[predicted3_x_col]
        
        # 점프예측 값 준비 및 색상 결정
        if jump_col:
            df['jump_value'] = df[jump_col].astype(str)
            df['jump_color'] = df[jump_col].apply(
                lambda x: '#27AE60' if str(x).upper() == 'O' else '#E74C3C' if str(x).upper() == 'X' else '#666'
            )
        else:
            df['jump_value'] = 'N/A'
            df['jump_color'] = '#666'
        
        # 패턴예측 값 준비 및 색상 결정
        if pattern_col:
            df['pattern_value'] = df[pattern_col].astype(str)
            def get_pattern_color(val):
                val_str = str(val).strip()
                if '상승' in val_str or '증가' in val_str:
                    return '#27AE60'
                elif '하락' in val_str or '감소' in val_str:
                    return '#E74C3C'
                elif '안정' in val_str or '유지' in val_str:
                    return '#3498DB'
                else:
                    return '#8E44AD'
            df['pattern_color'] = df[pattern_col].apply(get_pattern_color)
        else:
            df['pattern_value'] = 'N/A'
            df['pattern_color'] = '#95A5A6'
        
        # 트렌드 데이터 Y값을 숫자로 변환
        for trend_info in trend_data:
            y_col = trend_info.get('y_col', '')
            if y_col and y_col in df.columns:
                try:
                    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
                except:
                    pass

        fig = go.Figure()

        # 실제값 라인 (선택된 경우에만)
        if selected_lines['actual']:
            actual_name = params.get('actual_name', '실제값')
            hubroom_name = params.get('hubroom_name', 'HUBROOM')
            predicted1_name = params.get('predicted1_name', '예측값1')
            predicted2_name = params.get('predicted2_name', '예측값2')
            predicted3_name = params.get('predicted3_name', '예측값3')
            
            # customdata에 모든 라인의 정보 포함 (순서: 실제값, 예측값1, 예측값2, 예측값3, HUBROOM)
            customdata_cols = ['actual_time_str']
            
            if selected_lines['predicted1']:
                customdata_cols.extend(['predicted1_time_str_original', predicted1_y_col])
            else:
                df['_dummy_pred1_time'] = ''
                df['_dummy_pred1_val'] = 0
                customdata_cols.extend(['_dummy_pred1_time', '_dummy_pred1_val'])
                
            if selected_lines['predicted2']:
                customdata_cols.extend(['predicted2_time_str_original', predicted2_y_col])
            else:
                df['_dummy_pred2_time'] = ''
                df['_dummy_pred2_val'] = 0
                customdata_cols.extend(['_dummy_pred2_time', '_dummy_pred2_val'])
                
            if selected_lines['predicted3']:
                customdata_cols.extend(['predicted3_time_str_original', predicted3_y_col])
            else:
                df['_dummy_pred3_time'] = ''
                df['_dummy_pred3_val'] = 0
                customdata_cols.extend(['_dummy_pred3_time', '_dummy_pred3_val'])
            
            if selected_lines['hubroom']:
                customdata_cols.extend(['hubroom_time_str', hubroom_y_col])
            else:
                df['_dummy_hubroom_time'] = ''
                df['_dummy_hubroom_val'] = 0
                customdata_cols.extend(['_dummy_hubroom_time', '_dummy_hubroom_val'])
            
            # hovertemplate 생성 (순서: 실제값 → 예측값1 → 예측값2 → 예측값3 → HUBROOM)
            hover_text = f'<b style="color: #2E86C1; font-size: 14px;">📊 INFO 정보</b><br>'
            hover_text += '<span style="color: #85C1E2;">═══════════════════</span><br>'
            hover_text += f'<b style="color: #1f77b4;">🔵 {actual_name}</b><br>'
            hover_text += '<span style="color: #666;">시간:</span> %{customdata[0]}<br>'
            hover_text += '<span style="color: #666;">값:</span> <b>%{y:.2f}</b><br>'
            
            if selected_lines['predicted1']:
                hover_text += '<br>'
                hover_text += f'<b style="color: #ff7f0e;">🔶 {predicted1_name}</b><br>'
                hover_text += '<span style="color: #666;">시간:</span> %{customdata[1]}<br>'
                hover_text += '<span style="color: #666;">값:</span> <b>%{customdata[2]:.2f}</b><br>'
                
            if selected_lines['predicted2']:
                idx = 3 if selected_lines['predicted1'] else 1
                hover_text += '<br>'
                hover_text += f'<b style="color: #2ca02c;">🔷 {predicted2_name}</b><br>'
                hover_text += f'<span style="color: #666;">시간:</span> %{{customdata[{idx}]}}<br>'
                hover_text += f'<span style="color: #666;">값:</span> <b>%{{customdata[{idx+1}]:.2f}}</b><br>'
                
            if selected_lines['predicted3']:
                idx = 1
                if selected_lines['predicted1']:
                    idx += 2
                if selected_lines['predicted2']:
                    idx += 2
                hover_text += '<br>'
                hover_text += f'<b style="color: #d62728;">🔸 {predicted3_name}</b><br>'
                hover_text += f'<span style="color: #666;">시간:</span> %{{customdata[{idx}]}}<br>'
                hover_text += f'<span style="color: #666;">값:</span> <b>%{{customdata[{idx+1}]:.2f}}</b><br>'
            
            if selected_lines['hubroom']:
                idx = 1
                if selected_lines['predicted1']:
                    idx += 2
                if selected_lines['predicted2']:
                    idx += 2
                if selected_lines['predicted3']:
                    idx += 2
                hover_text += '<br>'
                hover_text += f'<b style="color: #9467bd;">🟣 {hubroom_name}</b><br>'
                hover_text += f'<span style="color: #666;">시간:</span> %{{customdata[{idx}]}}<br>'
                hover_text += f'<span style="color: #666;">값:</span> <b>%{{customdata[{idx+1}]:.2f}}</b><br>'
            
            hover_text += '<extra></extra>'
            
            fig.add_trace(go.Scattergl(
                x=df['common_time'],
                y=df[actual_y_col], 
                mode='lines+markers',
                name=f'{actual_name} (Actual)',
                line=dict(color=params['actual_color'], 
                         dash=None if params['actual_style'] == 'Solid' else params['actual_style'].lower(), 
                         width=3),
                marker=dict(size=6),
                customdata=df[customdata_cols].values,
                hovertemplate=hover_text
            ))
        
        # HUBROOM 라인 (선택된 경우에만)
        if selected_lines['hubroom']:
            hubroom_name = params.get('hubroom_name', 'HUBROOM')
            fig.add_trace(go.Scattergl(
                x=df['common_time'],
                y=df[hubroom_y_col], 
                mode='lines+markers',
                name=hubroom_name,
                line=dict(color=params['hubroom_color'], 
                         dash=None if params['hubroom_style'] == 'Solid' else params['hubroom_style'].lower(), 
                         width=3),
                marker=dict(size=6),
                hoverinfo='skip'  # 호버 정보 표시 안 함
            ))
        
        # 예측값1 라인 (선택된 경우에만) - Shift 적용!
        if selected_lines['predicted1']:
            predicted1_name = params.get('predicted1_name', '예측값1')
            fig.add_trace(go.Scattergl(
                x=df['predicted1_shifted_time'],  # ← Shift된 시간 사용 (그래프 위치)
                y=df[predicted1_y_col], 
                mode='lines+markers',
                name=f'{predicted1_name} (Predicted1)',
                line=dict(color=params['predicted1_color'], 
                         dash=None if params['predicted1_style'] == 'Solid' else params['predicted1_style'].lower(), 
                         width=3),
                marker=dict(size=6),
                hoverinfo='skip'  # 호버 정보 표시 안 함
            ))
        
        # 예측값2 라인 (선택된 경우에만) - Shift 적용!
        if selected_lines['predicted2']:
            predicted2_name = params.get('predicted2_name', '예측값2')
            fig.add_trace(go.Scattergl(
                x=df['predicted2_shifted_time'],  # ← Shift된 시간 사용
                y=df[predicted2_y_col], 
                mode='lines+markers',
                name=f'{predicted2_name} (Predicted2)',
                line=dict(color=params['predicted2_color'], 
                         dash=None if params['predicted2_style'] == 'Solid' else params['predicted2_style'].lower(), 
                         width=3),
                marker=dict(size=6),
                hoverinfo='skip'  # 호버 정보 표시 안 함
            ))
        
        # 예측값3 라인 (선택된 경우에만) - Shift 적용!
        if selected_lines['predicted3']:
            predicted3_name = params.get('predicted3_name', '예측값3')
            fig.add_trace(go.Scattergl(
                x=df['predicted3_shifted_time'],  # ← Shift된 시간 사용
                y=df[predicted3_y_col], 
                mode='lines+markers',
                name=f'{predicted3_name} (Predicted3)',
                line=dict(color=params['predicted3_color'], 
                         dash=None if params['predicted3_style'] == 'Solid' else params['predicted3_style'].lower(), 
                         width=3),
                marker=dict(size=6),
                hoverinfo='skip'  # 호버 정보 표시 안 함
            ))
        
        # 트렌드 데이터 라인들 - 간단하게만 표시
        trend_colors = ['#808080', '#A9A9A9', '#696969', '#778899', '#708090', 
                       '#2F4F4F', '#556B2F', '#8B4513', '#A0522D', '#B8860B',
                       '#BDB76B', '#8B008B', '#9932CC', '#8B0000', '#DC143C',
                       '#FF6347', '#FF4500']
        
        for idx, trend_info in enumerate(trend_data):
            x_col = trend_info.get('x_col', '')
            y_col = trend_info.get('y_col', '')
            name = trend_info.get('name', '')
            
            # 빈 문자열이거나 컬럼이 없으면 스킵
            if not x_col or not y_col or not name:
                continue
            
            if x_col in df.columns and y_col in df.columns:
                color = trend_colors[idx % len(trend_colors)]
                
                fig.add_trace(go.Scattergl(
                    x=df[x_col],
                    y=df[y_col],
                    mode='lines',
                    name=f'{name}',
                    line=dict(color=color, dash='dot', width=1.5),
                    opacity=0.7,
                    hovertemplate=f'<b>{name}:</b> %{{y:.2f}}<extra></extra>'
                ))
        
        # 리미트선
        try:
            limit_value = float(params.get('limit_value', 300))
            all_x = df['common_time'].sort_values()
            limit_x = pd.date_range(start=all_x.iloc[0], end=all_x.iloc[-1], periods=100)
            limit_y = [limit_value] * 100
            
            fig.add_trace(go.Scatter(
                x=limit_x,
                y=limit_y,
                mode='lines',
                name=f'리미트선 ({limit_value})',
                line=dict(color='red', width=2.5),
                hovertemplate=f'<b style="color: red;">⚠️ 리미트선: {limit_value:.2f}</b><extra></extra>'
            ))
        except Exception as e:
            print(f"리미트선 생성 중 오류: {e}")
            pass
        
        # 레이아웃
        fig.update_layout(
            title=params['title'], 
            xaxis_title='시간', 
            yaxis_title='값', 
            hovermode='x unified',
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial",
                bordercolor="#2E86C1"
            )
        )

        output_filename = "final_guided_graph.html"
        html_content = fig.to_html(include_plotlyjs='cdn')

        # HTML 하단에 평가 결과 분석 섹션 추가
        if perf_values:
            other_info = perf_values.get('other_info', '정보가 입력되지 않았습니다.')
            other_info_escaped = other_info.replace('<', '&lt;').replace('>', '&gt;')
            
            performance_section = f"""
            <div style="margin: 30px auto; max-width: 1200px; padding: 0 20px;">
                <div style="background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <h2 style="color: #2E86C1; text-align: center; margin-bottom: 30px; border-bottom: 3px solid #2E86C1; padding-bottom: 15px;">
                        📊 평가 결과 분석
                    </h2>
                    <div style="white-space: pre-wrap; line-height: 1.8; font-size: 14px; color: #333; background-color: #f8f9fa; padding: 25px; border-radius: 8px; border: 2px solid #dee2e6; font-family: 'Courier New', monospace;">
{other_info_escaped}
                    </div>
                </div>
            </div>
            """
        else:
            performance_section = ""

        html_content = html_content.replace('</body>', performance_section + '</body>')

        style_section = """
        <style>
            body {
                background-color: #f8f9fa;
                margin: 0;
                padding: 20px 0;
            }
            .plotly-graph-div {
                margin: 0 auto;
                max-width: 1400px;
            }
        </style>
        """
        html_content = html_content.replace('</head>', style_section + '</head>')

        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        webbrowser.open('file://' + os.path.realpath(output_filename))

        trend_count = len(trend_data)
        shift_info = []
        if predicted1_shift != 0:
            shift_info.append(f"예측값1: {predicted1_shift}분")
        if predicted2_shift != 0:
            shift_info.append(f"예측값2: {predicted2_shift}분")
        if predicted3_shift != 0:
            shift_info.append(f"예측값3: {predicted3_shift}분")
        
        shift_msg = f"\n✅ 시간 이동: {', '.join(shift_info)}" if shift_info else ""
        
        messagebox.showinfo("성공", 
            f"'{output_filename}' 파일이 생성되었습니다.\n\n"
            f"✅ 메인 5개 라인 + 트렌드 {trend_count}개\n"
            f"✅ 실제값 라인에 전체 INFO 정보 포함\n"
            f"✅ 범례 클릭으로 표시/숨김 가능{shift_msg}")
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        messagebox.showerror("오류 발생", f"그래프 생성 중 오류:\n{e}\n\n상세:\n{error_detail}")

# --------------------------------------------------------------------------
# GUI 애플리케이션
# --------------------------------------------------------------------------
class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("그래프 생성기 v4.3 (시간 이동 기능)")
        self.root.geometry("1200x900")
        
        self.file_path = ""
        self.df_columns = []
        self.perf_values = None
        self.trend_widgets = {}
        
        # 예측값 시간 shift 변수 추가
        self.predicted1_shift_var = tk.StringVar(value="0")
        self.predicted2_shift_var = tk.StringVar(value="0")
        self.predicted3_shift_var = tk.StringVar(value="0")

        # 전체 스크롤 가능한 메인 컨테이너
        main_canvas = tk.Canvas(root)
        scrollbar = tk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
        self.scrollable_frame = tk.Frame(main_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # 마우스 휠 스크롤 지원
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 상단 버튼 프레임
        top_button_frame = tk.Frame(self.scrollable_frame)
        top_button_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(top_button_frame, text="📊 내용정보 입력", 
                 command=self.open_performance_window,
                 font=('Arial', 10, 'bold'),
                 bg='#2ECC71', fg='white',
                 width=15, height=2).pack(side='right', padx=5)

        # 1단계: 파일 선택
        self.step1_frame = tk.LabelFrame(self.scrollable_frame, text="✅ 1단계: 파일 선택", padx=10, pady=10)
        self.step1_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(self.step1_frame, text="CSV 파일 열기", command=self.select_file, 
                 font=('Helvetica', 10, 'bold'), bg='#e8f4f8').pack(side='left')
        self.file_label = tk.Label(self.step1_frame, text="선택된 파일이 없습니다.", fg="blue")
        self.file_label.pack(side='left', padx=10)
        
        # 2단계: 데이터 컬럼 선택
        self.step2_frame = tk.LabelFrame(self.scrollable_frame, text="🔒 2단계: 메인 데이터 컬럼 선택", padx=10, pady=10)
        self.step2_frame.pack(fill='x', padx=10, pady=5)
        self.create_step2_widgets()
        
        # 2.5단계: 트렌드 데이터 선택 (그룹별 정리)
        self.step2_5_frame = tk.LabelFrame(self.scrollable_frame, text="🔒 2.5단계: 트렌드 데이터 선택 (그룹별)", padx=10, pady=10)
        self.step2_5_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.create_step2_5_widgets()

        # 3단계: 그래프 스타일 설정
        self.step3_frame = tk.LabelFrame(self.scrollable_frame, text="🔒 3단계: 그래프 스타일 설정", padx=10, pady=10)
        self.step3_frame.pack(fill='x', padx=10, pady=5)
        self.create_step3_widgets()
        
        # 4단계: 그래프 생성
        self.step4_frame = tk.LabelFrame(self.scrollable_frame, text="🔒 4단계: 그래프 생성", padx=10, pady=10)
        self.step4_frame.pack(fill='x', padx=10, pady=5)
        self.create_step4_widgets()
        
        # 초기 비활성화
        self.toggle_widgets_state(self.step2_frame, 'disabled')
        self.toggle_widgets_state(self.step2_5_frame, 'disabled')
        self.toggle_widgets_state(self.step3_frame, 'disabled')
        self.toggle_widgets_state(self.step4_frame, 'disabled')
    
    def open_performance_window(self):
        perf_window = PerformanceWindow(self.root)
        self.root.wait_window(perf_window.window)
        self.perf_values = perf_window.values
    
    def toggle_widgets_state(self, frame, state):
        for child in frame.winfo_children():
            try:
                child.config(state=state)
            except tk.TclError:
                pass

    def create_step2_widgets(self):
        labels = [
            "실제값 X축 (시간):", 
            "실제값 Y축 (값):", 
            "예측값1 X축 (시간):", 
            "예측값1 Y축 (값):",
            "예측값2 X축 (시간):",
            "예측값2 Y축 (값):",
            "예측값3 X축 (시간):",
            "예측값3 Y축 (값):",
            "HUBROOM X축 (시간):",
            "HUBROOM Y축 (값):"
        ]
        
        # 각 데이터 라인별 체크박스 변수 (5개 그룹)
        self.enable_actual = tk.BooleanVar(value=True)
        self.enable_hubroom = tk.BooleanVar(value=True)
        self.enable_predicted1 = tk.BooleanVar(value=True)
        self.enable_predicted2 = tk.BooleanVar(value=True)
        self.enable_predicted3 = tk.BooleanVar(value=True)
        
        # 모든 라인의 이름 변수 추가
        self.actual_name_var = tk.StringVar(value="실제값")
        self.hubroom_name_var = tk.StringVar(value="HUBROOM")
        self.predicted1_name_var = tk.StringVar(value="예측값1")
        self.predicted2_name_var = tk.StringVar(value="예측값2")
        self.predicted3_name_var = tk.StringVar(value="예측값3")
        
        self.column_vars = [tk.StringVar() for _ in labels]
        self.column_menus = []

        # 각 그룹마다 체크박스 + shift 입력 필드 추가
        enable_vars = [
            (self.enable_actual, "✅ 실제값 사용", None),
            (self.enable_actual, "", None),
            (self.enable_predicted1, "✅ 예측값1 사용", self.predicted1_shift_var),
            (self.enable_predicted1, "", None),
            (self.enable_predicted2, "✅ 예측값2 사용", self.predicted2_shift_var),
            (self.enable_predicted2, "", None),
            (self.enable_predicted3, "✅ 예측값3 사용", self.predicted3_shift_var),
            (self.enable_predicted3, "", None),
            (self.enable_hubroom, "✅ HUBROOM 사용", None),
            (self.enable_hubroom, "", None)
        ]

        for i, label_text in enumerate(labels):
            # X축 줄 - 체크박스 + shift 입력만
            if i % 2 == 0:  # X축 줄
                enable_var, checkbox_text, shift_var = enable_vars[i]
                
                # 체크박스
                tk.Checkbutton(self.step2_frame, text=checkbox_text, variable=enable_var, 
                              font=('Arial', 9, 'bold')).grid(row=i, column=0, sticky='w', padx=5, pady=(8, 2))
                
                # 예측값의 경우 shift 입력 필드 추가 (X축 줄)
                if shift_var is not None:
                    shift_frame = tk.Frame(self.step2_frame)
                    shift_frame.grid(row=i, column=0, sticky='e', padx=(200, 5))
                    
                    tk.Label(shift_frame, text="⏭️", font=('Arial', 10)).pack(side='left')
                    tk.Entry(shift_frame, textvariable=shift_var, width=5, 
                            font=('Arial', 9)).pack(side='left', padx=2)
                    tk.Label(shift_frame, text="분", font=('Arial', 9)).pack(side='left')
            
            # Y축 줄 - 이름 입력 필드
            else:  # Y축 줄 (i=1,3,5,7,9)
                # 이름 입력 필드를 Y축 줄에 추가
                name_frame = tk.Frame(self.step2_frame)
                name_frame.grid(row=i, column=0, sticky='e', padx=(200, 5))
                
                tk.Label(name_frame, text="이름:", font=('Arial', 8)).pack(side='left')
                
                # 각 라인별 이름 변수 선택 (새로운 순서)
                if i == 1:  # 실제값 Y축
                    name_var = self.actual_name_var
                elif i == 3:  # 예측값1 Y축
                    name_var = self.predicted1_name_var
                elif i == 5:  # 예측값2 Y축
                    name_var = self.predicted2_name_var
                elif i == 7:  # 예측값3 Y축
                    name_var = self.predicted3_name_var
                elif i == 9:  # HUBROOM Y축
                    name_var = self.hubroom_name_var
                else:
                    name_var = None
                
                if name_var:
                    tk.Entry(name_frame, textvariable=name_var, width=12, 
                            font=('Arial', 9)).pack(side='left', padx=2)
            
            # 라벨과 메뉴는 한 칸 오른쪽으로
            tk.Label(self.step2_frame, text=label_text, font=('Arial', 9)).grid(row=i, column=1, sticky='w', padx=5, pady=2)
            menu = tk.OptionMenu(self.step2_frame, self.column_vars[i], "")
            menu.config(width=30)
            menu.grid(row=i, column=2, sticky='ew', padx=5, pady=2)
            self.column_menus.append(menu)
    
    def create_step2_5_widgets(self):
        """트렌드 데이터 그룹별로 정리"""
        # 트렌드 데이터 그룹
        trend_groups = {
            '📦 MAXCAP 그룹': [
                'M14A_3F_CNV_MAXCAP',
                'M14B_7F_LFT_MAXCAP',
                'M16A_2F_LFT_MAXCAP',
                'M16A_3F_CNV_MAXCAP',
                'M16A_3F_LFT_MAXCAP',
                'M16A_3F_M14BLFT_MAXCAP',
                'M16A_6F_LFT_MAXCAP'
            ],
            '📊 UTIL 그룹': [
                'M16A_3F_STORAGE_UTIL'
            ],
            '🔄 JOB 그룹': [
                'M14A_3F_TO_HUB_JOB2',
                'M14A_3F_TO_HUB_JOB_ALT',
                'M14B_7F_TO_HUB_JOB2',
                'M14B_7F_TO_HUB_JOB_ALT',
                'M16A_2F_TO_HUB_JOB2',
                'M16A_2F_TO_HUB_JOB_ALT',
                'M16A_6F_TO_HUB_JOB',
                'M16A_6F_TO_HUB_JOB_ALT',
                'M16B_10F_TO_HUB_JOB'
            ]
        }
        
        # 스크롤 가능한 프레임 (가로/세로 스크롤)
        canvas = tk.Canvas(self.step2_5_frame, height=450)
        scrollbar_trend_v = tk.Scrollbar(self.step2_5_frame, orient="vertical", command=canvas.yview)
        scrollbar_trend_h = tk.Scrollbar(self.step2_5_frame, orient="horizontal", command=canvas.xview)
        scrollable_trend_frame = tk.Frame(canvas)
        
        scrollable_trend_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_trend_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_trend_v.set, xscrollcommand=scrollbar_trend_h.set)
        
        # step2_5_frame의 grid weight 설정
        self.step2_5_frame.grid_rowconfigure(0, weight=1)
        self.step2_5_frame.grid_columnconfigure(0, weight=1)
        
        current_row = 0
        
        # 그룹별로 표시
        for group_name, trend_list in trend_groups.items():
            # 그룹 헤더
            group_frame = tk.LabelFrame(scrollable_trend_frame, text=group_name, 
                                       font=('Arial', 10, 'bold'), 
                                       bg='#E8F4F8', padx=10, pady=5)
            group_frame.grid(row=current_row, column=0, columnspan=4, sticky='ew', padx=5, pady=10)
            current_row += 1
            
            # 그룹 전체 선택/해제 버튼
            btn_frame = tk.Frame(group_frame)
            btn_frame.pack(fill='x', pady=5)
            
            group_vars = []
            
            def make_toggle_func(vars_list, value):
                return lambda: [v.set(value) for v in vars_list]
            
            # 헤더
            inner_frame = tk.Frame(group_frame)
            inner_frame.pack(fill='both', expand=True)
            
            tk.Label(inner_frame, text="사용", font=('Arial', 9, 'bold'), width=6).grid(row=0, column=0, padx=5, sticky='w')
            tk.Label(inner_frame, text="트렌드 이름", font=('Arial', 9, 'bold'), width=32).grid(row=0, column=1, sticky='w', padx=5)
            tk.Label(inner_frame, text="X축 (시간)", font=('Arial', 9, 'bold'), width=32).grid(row=0, column=2, padx=5, sticky='w')
            tk.Label(inner_frame, text="Y축 (값)", font=('Arial', 9, 'bold'), width=32).grid(row=0, column=3, padx=5, sticky='w')
            
            # 각 트렌드 데이터
            for idx, trend_name in enumerate(trend_list):
                row = idx + 1
                
                # 체크박스
                var = tk.BooleanVar(value=False)
                cb = tk.Checkbutton(inner_frame, variable=var)
                cb.grid(row=row, column=0, padx=5, pady=3, sticky='w')
                group_vars.append(var)
                
                # 이름
                tk.Label(inner_frame, text=trend_name, font=('Arial', 9), anchor='w', width=32).grid(row=row, column=1, sticky='w', padx=5, pady=3)
                
                # X축 선택
                x_var = tk.StringVar()
                x_menu = tk.OptionMenu(inner_frame, x_var, "")
                x_menu.config(width=30, font=('Arial', 9))
                x_menu.grid(row=row, column=2, padx=5, pady=3, sticky='ew')
                
                # Y축 선택
                y_var = tk.StringVar()
                y_menu = tk.OptionMenu(inner_frame, y_var, "")
                y_menu.config(width=30, font=('Arial', 9))
                y_menu.grid(row=row, column=3, padx=5, pady=3, sticky='ew')
                
                # trend_widgets에 저장
                self.trend_widgets[trend_name] = {
                    'name': trend_name,
                    'enabled': var,
                    'x_var': x_var,
                    'x_menu': x_menu,
                    'y_var': y_var,
                    'y_menu': y_menu
                }
            
            # 컬럼 너비 설정 - 확실하게 크게!
            inner_frame.grid_columnconfigure(0, minsize=60, weight=0)
            inner_frame.grid_columnconfigure(1, minsize=280, weight=0)
            inner_frame.grid_columnconfigure(2, minsize=300, weight=1)
            inner_frame.grid_columnconfigure(3, minsize=300, weight=1)
            
            # 그룹 전체 선택/해제 버튼 추가
            tk.Button(btn_frame, text=f"전체선택", 
                     command=make_toggle_func(group_vars, True),
                     bg='#3498DB', fg='white', font=('Arial', 8)).pack(side='left', padx=3)
            tk.Button(btn_frame, text=f"전체해제", 
                     command=make_toggle_func(group_vars, False),
                     bg='#E74C3C', fg='white', font=('Arial', 8)).pack(side='left', padx=3)
            
            # 그룹 자동 매칭 버튼
            def make_auto_match_func(trend_names, group_title):
                def auto_match_group():
                    if not self.df_columns:
                        messagebox.showwarning("경고", "먼저 CSV 파일을 선택해주세요!")
                        return
                    
                    # X축 선택 창 띄우기
                    select_window = tk.Toplevel(self.root)
                    select_window.title(f"{group_title} X축 선택")
                    select_window.geometry("400x200")
                    select_window.transient(self.root)
                    select_window.grab_set()
                    
                    # 중앙 배치
                    select_window.update_idletasks()
                    x = (select_window.winfo_screenwidth() // 2) - (400 // 2)
                    y = (select_window.winfo_screenheight() // 2) - (200 // 2)
                    select_window.geometry(f'400x200+{x}+{y}')
                    
                    selected_col = tk.StringVar()
                    
                    frame = tk.Frame(select_window)
                    frame.pack(fill='both', expand=True, padx=20, pady=20)
                    
                    tk.Label(frame, text=f"{group_title}의 X축(시간) 컬럼을 선택하세요:", 
                            font=('Arial', 10, 'bold')).pack(pady=10)
                    
                    # 드롭다운
                    col_menu = ttk.Combobox(frame, textvariable=selected_col, 
                                           values=self.df_columns, 
                                           font=('Arial', 10),
                                           width=40,
                                           state='readonly')
                    col_menu.pack(pady=10)
                    
                    # 날짜/시간 컬럼 자동 추천
                    time_keywords = ['날짜', 'date', 'time', '시간', 'datetime', 'timestamp']
                    for col in self.df_columns:
                        col_lower = col.lower()
                        if any(keyword in col_lower for keyword in time_keywords):
                            selected_col.set(col)
                            break
                    
                    def confirm_selection():
                        time_col = selected_col.get()
                        if not time_col:
                            messagebox.showwarning("경고", "X축 컬럼을 선택해주세요!")
                            return
                        
                        select_window.destroy()
                        
                        matched = 0
                        for trend_name in trend_names:
                            widget_dict = self.trend_widgets[trend_name]
                            # Y축 찾기
                            y_col_found = None
                            for col in self.df_columns:
                                if trend_name in col or col in trend_name:
                                    y_col_found = col
                                    break
                            
                            if y_col_found:
                                widget_dict['x_var'].set(time_col)
                                widget_dict['y_var'].set(y_col_found)
                                widget_dict['enabled'].set(True)
                                matched += 1
                        
                        messagebox.showinfo("자동 매칭", f"{group_title}: {matched}개 매칭 완료\nX축: {time_col}")
                    
                    btn_frame = tk.Frame(frame)
                    btn_frame.pack(pady=20)
                    
                    tk.Button(btn_frame, text="확인", command=confirm_selection,
                             bg='#3498DB', fg='white', font=('Arial', 10, 'bold'),
                             width=10).pack(side='left', padx=5)
                    tk.Button(btn_frame, text="취소", command=select_window.destroy,
                             bg='#95A5A6', fg='white', font=('Arial', 10, 'bold'),
                             width=10).pack(side='left', padx=5)
                
                return auto_match_group
            
            tk.Button(btn_frame, text=f"🔄 자동매칭", 
                     command=make_auto_match_func(trend_list, group_name),
                     bg='#9B59B6', fg='white', font=('Arial', 8)).pack(side='left', padx=3)
        
        # 캔버스와 스크롤바 배치 (가로/세로 스크롤)
        canvas.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=(10, 0))
        scrollbar_trend_v.grid(row=0, column=1, sticky="ns", pady=(10, 0))
        scrollbar_trend_h.grid(row=1, column=0, sticky="ew", padx=(10, 0))
        
        # 자동 매칭 버튼 프레임
        auto_match_frame = tk.Frame(self.step2_5_frame)
        auto_match_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=10, padx=10)
        
        tk.Button(auto_match_frame, text="🔄 전체 자동 매칭", 
                 command=self.auto_match_all_trends,
                 bg='#9B59B6', fg='white', 
                 font=('Arial', 10, 'bold'),
                 width=20, height=2).pack(pady=5)
        
        tk.Label(auto_match_frame, 
                text="※ 자동 매칭: 트렌드 이름과 일치하는 Y축 컬럼 + 날짜/시간 X축을 자동 선택",
                font=('Arial', 8), fg='#666').pack()
            
    def create_step3_widgets(self):
        tk.Label(self.step3_frame, text="그래프 제목:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.title_var = tk.StringVar(value="사용자 정의 그래프")
        tk.Entry(self.step3_frame, textvariable=self.title_var, width=50).grid(row=0, column=1, columnspan=3, sticky='ew', padx=5, pady=2)
        
        # 실제값
        tk.Label(self.step3_frame, text="[실제값]", font=('Arial', 9, 'bold')).grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.actual_color_var = tk.StringVar(value='#1f77b4')
        self.actual_color_btn = tk.Button(self.step3_frame, text="색상", 
                                         command=lambda: self.choose_color(self.actual_color_var, self.actual_color_btn), 
                                         bg=self.actual_color_var.get())
        self.actual_color_btn.grid(row=1, column=1, padx=5)
        self.actual_style_var = tk.StringVar(value='Solid')
        tk.OptionMenu(self.step3_frame, self.actual_style_var, 'Solid', 'Dash', 'Dot').grid(row=1, column=2, padx=5)

        # HUBROOM
        tk.Label(self.step3_frame, text="[HUBROOM]", font=('Arial', 9, 'bold')).grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.hubroom_color_var = tk.StringVar(value='#9467bd')
        self.hubroom_color_btn = tk.Button(self.step3_frame, text="색상", 
                                         command=lambda: self.choose_color(self.hubroom_color_var, self.hubroom_color_btn), 
                                         bg=self.hubroom_color_var.get())
        self.hubroom_color_btn.grid(row=2, column=1, padx=5)
        self.hubroom_style_var = tk.StringVar(value='Solid')
        tk.OptionMenu(self.step3_frame, self.hubroom_style_var, 'Solid', 'Dash', 'Dot').grid(row=2, column=2, padx=5)

        # 예측값1
        tk.Label(self.step3_frame, text="[예측값1]", font=('Arial', 9, 'bold')).grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.predicted1_color_var = tk.StringVar(value='#ff7f0e')
        self.predicted1_color_btn = tk.Button(self.step3_frame, text="색상", 
                                           command=lambda: self.choose_color(self.predicted1_color_var, self.predicted1_color_btn), 
                                           bg=self.predicted1_color_var.get())
        self.predicted1_color_btn.grid(row=3, column=1, padx=5)
        self.predicted1_style_var = tk.StringVar(value='Dash')
        tk.OptionMenu(self.step3_frame, self.predicted1_style_var, 'Solid', 'Dash', 'Dot').grid(row=3, column=2, padx=5)
        
        # 예측값2
        tk.Label(self.step3_frame, text="[예측값2]", font=('Arial', 9, 'bold')).grid(row=4, column=0, sticky='w', padx=5, pady=5)
        self.predicted2_color_var = tk.StringVar(value='#2ca02c')
        self.predicted2_color_btn = tk.Button(self.step3_frame, text="색상", 
                                           command=lambda: self.choose_color(self.predicted2_color_var, self.predicted2_color_btn), 
                                           bg=self.predicted2_color_var.get())
        self.predicted2_color_btn.grid(row=4, column=1, padx=5)
        self.predicted2_style_var = tk.StringVar(value='Dot')
        tk.OptionMenu(self.step3_frame, self.predicted2_style_var, 'Solid', 'Dash', 'Dot').grid(row=4, column=2, padx=5)
        
        # 예측값3
        tk.Label(self.step3_frame, text="[예측값3]", font=('Arial', 9, 'bold')).grid(row=5, column=0, sticky='w', padx=5, pady=5)
        self.predicted3_color_var = tk.StringVar(value='#d62728')
        self.predicted3_color_btn = tk.Button(self.step3_frame, text="색상", 
                                           command=lambda: self.choose_color(self.predicted3_color_var, self.predicted3_color_btn), 
                                           bg=self.predicted3_color_var.get())
        self.predicted3_color_btn.grid(row=5, column=1, padx=5)
        self.predicted3_style_var = tk.StringVar(value='Dash')
        tk.OptionMenu(self.step3_frame, self.predicted3_style_var, 'Solid', 'Dash', 'Dot').grid(row=5, column=2, padx=5)
        
        # 리미트선
        tk.Label(self.step3_frame, text="[🔴 리미트선]", font=('Arial', 9, 'bold')).grid(row=6, column=0, sticky='w', padx=5, pady=5)
        tk.Label(self.step3_frame, text="값:").grid(row=6, column=1, sticky='e', padx=(0, 5))
        self.limit_value_var = tk.StringVar(value="300")
        tk.Entry(self.step3_frame, textvariable=self.limit_value_var, width=10).grid(row=6, column=2, sticky='w', padx=5)

    def create_step4_widgets(self):
        self.generate_button = tk.Button(self.step4_frame, text="🎨 그래프 생성 실행", 
                                        command=self.generate_graph, 
                                        font=('Helvetica', 12, 'bold'), 
                                        bg='#d3ffd3',
                                        height=2)
        self.generate_button.pack(fill='x')

    def choose_color(self, color_var, button):
        color_code = colorchooser.askcolor(title="색상 선택")[1]
        if color_code:
            color_var.set(color_code)
            button.config(bg=color_code)
    
    def auto_match_all_trends(self):
        """모든 트렌드 데이터의 X축/Y축을 자동으로 매칭 - X축은 사용자 선택"""
        if not self.df_columns:
            messagebox.showwarning("경고", "먼저 CSV 파일을 선택해주세요!")
            return
        
        # X축 선택 창 띄우기
        select_window = tk.Toplevel(self.root)
        select_window.title("X축(시간) 컬럼 선택")
        select_window.geometry("400x200")
        select_window.transient(self.root)
        select_window.grab_set()
        
        # 중앙 배치
        select_window.update_idletasks()
        x = (select_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (select_window.winfo_screenheight() // 2) - (200 // 2)
        select_window.geometry(f'400x200+{x}+{y}')
        
        selected_col = tk.StringVar()
        
        frame = tk.Frame(select_window)
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(frame, text="모든 트렌드 데이터의 X축(시간)으로 사용할 컬럼을 선택하세요:", 
                font=('Arial', 10, 'bold')).pack(pady=10)
        
        # 드롭다운
        col_menu = ttk.Combobox(frame, textvariable=selected_col, 
                               values=self.df_columns, 
                               font=('Arial', 10),
                               width=40,
                               state='readonly')
        col_menu.pack(pady=10)
        
        # 날짜/시간 컬럼 자동 추천
        time_keywords = ['날짜', 'date', 'time', '시간', 'datetime', 'timestamp']
        for col in self.df_columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in time_keywords):
                selected_col.set(col)
                break
        
        def confirm_selection():
            time_col = selected_col.get()
            if not time_col:
                messagebox.showwarning("경고", "X축 컬럼을 선택해주세요!")
                return
            
            select_window.destroy()
            
            # 각 트렌드에 대해 자동 매칭
            matched_count = 0
            for trend_name, widget_dict in self.trend_widgets.items():
                # Y축: 트렌드 이름과 일치하는 컬럼 찾기
                y_col_found = None
                for col in self.df_columns:
                    if trend_name in col or col in trend_name:
                        y_col_found = col
                        break
                
                if y_col_found:
                    # X축 설정
                    widget_dict['x_var'].set(time_col)
                    # Y축 설정
                    widget_dict['y_var'].set(y_col_found)
                    # 체크박스 자동 선택
                    widget_dict['enabled'].set(True)
                    matched_count += 1
            
            messagebox.showinfo("자동 매칭 완료", 
                               f"총 {matched_count}개의 트렌드 데이터가 자동 매칭되었습니다.\n\n"
                               f"X축: {time_col}\n"
                               f"Y축: 각 트렌드 이름과 일치하는 컬럼")
        
        btn_frame = tk.Frame(frame)
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="확인", command=confirm_selection,
                 bg='#3498DB', fg='white', font=('Arial', 10, 'bold'),
                 width=10).pack(side='left', padx=5)
        tk.Button(btn_frame, text="취소", command=select_window.destroy,
                 bg='#95A5A6', fg='white', font=('Arial', 10, 'bold'),
                 width=10).pack(side='left', padx=5)

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path: return
        
        self.file_path = path
        filename = os.path.basename(path)
        self.file_label.config(text=f"선택됨: {filename}")
        
        try:
            df = read_csv_safe(self.file_path)
            self.df_columns = df.columns.tolist()
            
            # 메인 데이터 컬럼 메뉴 업데이트
            for i, menu in enumerate(self.column_menus):
                menu['menu'].delete(0, 'end')
                for col in self.df_columns:
                    menu['menu'].add_command(label=col, command=tk._setit(self.column_vars[i], col))
            
            # 트렌드 데이터 컬럼 메뉴 업데이트
            for widget_dict in self.trend_widgets.values():
                # X축
                widget_dict['x_menu']['menu'].delete(0, 'end')
                for col in self.df_columns:
                    widget_dict['x_menu']['menu'].add_command(label=col, command=tk._setit(widget_dict['x_var'], col))
                
                # Y축
                widget_dict['y_menu']['menu'].delete(0, 'end')
                for col in self.df_columns:
                    widget_dict['y_menu']['menu'].add_command(label=col, command=tk._setit(widget_dict['y_var'], col))
            
            # 자동 추천
            patterns = [
                ['날짜', 'date', 'time'],
                ['실제값', 'actual', 'real'],
                ['예측날짜', 'pred_date', 'forecast'],
                ['예측값', 'predicted', 'pred'],
                ['예측날짜', 'pred_date', 'forecast', '2'],
                ['예측값', 'predicted', 'pred', '2'],
                ['예측날짜', 'pred_date', 'forecast', '3'],
                ['예측값', 'predicted', 'pred', '3'],
                ['hubroom', 'hub'],
                ['hubroom', 'hub']
            ]
            
            for idx, pattern_list in enumerate(patterns):
                for col in self.df_columns:
                    col_lower = col.lower()
                    if any(p in col_lower for p in pattern_list):
                        self.column_vars[idx].set(col)
                        break

            self.step2_frame.config(text="✅ 2단계: 메인 데이터 컬럼 선택")
            self.toggle_widgets_state(self.step2_frame, 'normal')
            self.step2_5_frame.config(text="✅ 2.5단계: 트렌드 데이터 선택 (그룹별)")
            self.toggle_widgets_state(self.step2_5_frame, 'normal')
            self.step3_frame.config(text="✅ 3단계: 그래프 스타일 설정")
            self.toggle_widgets_state(self.step3_frame, 'normal')
            self.step4_frame.config(text="✅ 4단계: 그래프 생성")
            self.toggle_widgets_state(self.step4_frame, 'normal')

        except Exception as e:
            messagebox.showerror("오류", f"CSV 파일을 읽는 데 실패했습니다:\n{e}")

    def generate_graph(self):
        # 선택된 트렌드 데이터 수집
        selected_trends = []
        for widget_dict in self.trend_widgets.values():
            if widget_dict['enabled'].get():
                x_col = widget_dict['x_var'].get()
                y_col = widget_dict['y_var'].get()
                if x_col and y_col:
                    selected_trends.append({
                        'name': widget_dict['name'],
                        'x_col': x_col,
                        'y_col': y_col
                    })
        
        # Shift 값 가져오기
        try:
            pred1_shift = int(self.predicted1_shift_var.get() or 0)
        except:
            pred1_shift = 0
        
        try:
            pred2_shift = int(self.predicted2_shift_var.get() or 0)
        except:
            pred2_shift = 0
            
        try:
            pred3_shift = int(self.predicted3_shift_var.get() or 0)
        except:
            pred3_shift = 0
        
        params = {
            'file_path': self.file_path, 
            'title': self.title_var.get(),
            'actual_x': self.column_vars[0].get() if self.enable_actual.get() else '', 
            'actual_y': self.column_vars[1].get() if self.enable_actual.get() else '',
            'actual_name': self.actual_name_var.get(),
            'hubroom_x': self.column_vars[8].get() if self.enable_hubroom.get() else '',
            'hubroom_y': self.column_vars[9].get() if self.enable_hubroom.get() else '',
            'hubroom_name': self.hubroom_name_var.get(),
            'predicted1_x': self.column_vars[2].get() if self.enable_predicted1.get() else '', 
            'predicted1_y': self.column_vars[3].get() if self.enable_predicted1.get() else '',
            'predicted1_name': self.predicted1_name_var.get(),
            'predicted2_x': self.column_vars[4].get() if self.enable_predicted2.get() else '',
            'predicted2_y': self.column_vars[5].get() if self.enable_predicted2.get() else '',
            'predicted2_name': self.predicted2_name_var.get(),
            'predicted3_x': self.column_vars[6].get() if self.enable_predicted3.get() else '',
            'predicted3_y': self.column_vars[7].get() if self.enable_predicted3.get() else '',
            'predicted3_name': self.predicted3_name_var.get(),
            'predicted1_shift': pred1_shift,
            'predicted2_shift': pred2_shift,
            'predicted3_shift': pred3_shift,
            'actual_color': self.actual_color_var.get(), 
            'actual_style': self.actual_style_var.get(),
            'hubroom_color': self.hubroom_color_var.get(),
            'hubroom_style': self.hubroom_style_var.get(),
            'predicted1_color': self.predicted1_color_var.get(), 
            'predicted1_style': self.predicted1_style_var.get(),
            'predicted2_color': self.predicted2_color_var.get(),
            'predicted2_style': self.predicted2_style_var.get(),
            'predicted3_color': self.predicted3_color_var.get(),
            'predicted3_style': self.predicted3_style_var.get(),
            'limit_value': self.limit_value_var.get(),
            'trend_data': selected_trends,
            'enable_actual': self.enable_actual.get(),
            'enable_hubroom': self.enable_hubroom.get(),
            'enable_predicted1': self.enable_predicted1.get(),
            'enable_predicted2': self.enable_predicted2.get(),
            'enable_predicted3': self.enable_predicted3.get()
        }
        
        # 최소 1개 이상의 메인 데이터가 활성화되어 있는지 확인
        enabled_count = sum([
            self.enable_actual.get(),
            self.enable_hubroom.get(),
            self.enable_predicted1.get(),
            self.enable_predicted2.get(),
            self.enable_predicted3.get()
        ])
        
        if enabled_count == 0:
            messagebox.showwarning("경고", "최소 1개 이상의 메인 데이터를 체크해주세요.")
            return
        
        # 체크된 것만 컬럼 선택 확인
        missing = []
        if self.enable_actual.get() and (not params['actual_x'] or not params['actual_y']):
            missing.append("실제값")
        if self.enable_hubroom.get() and (not params['hubroom_x'] or not params['hubroom_y']):
            missing.append("HUBROOM")
        if self.enable_predicted1.get() and (not params['predicted1_x'] or not params['predicted1_y']):
            missing.append("예측값1")
        if self.enable_predicted2.get() and (not params['predicted2_x'] or not params['predicted2_y']):
            missing.append("예측값2")
        if self.enable_predicted3.get() and (not params['predicted3_x'] or not params['predicted3_y']):
            missing.append("예측값3")
        
        if missing:
            messagebox.showwarning("경고", f"체크된 항목의 컬럼을 선택해주세요:\n{', '.join(missing)}")
            return
        
        if not self.perf_values:
            self.perf_values = {
                'mae': '30.23',
                'rmse': '36.02',
                'r2': '0.72',
                'other_info': '기타 정보가 입력되지 않았습니다.'
            }
            
        create_graph(params, self.perf_values)

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()