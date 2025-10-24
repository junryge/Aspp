# 변경 전:
"PREDICTION_TIME": f"{prediction_start.strftime('%H:%M')}~{prediction_end.strftime('%H:%M')}"

# 변경 후:
"PREDICTION_TIME": f"{prediction_start.strftime('%Y-%m-%d %H:%M')} ~ {prediction_end.strftime('%Y-%m-%d %H:%M')}"



# 변경 전:
<div style="color: #718096; font-size: 14px;">시퀀스 범위: {result['SEQUENCE_START_TIME'].split(' ')[1]} ~ {result['CURRENT_TIME'].split(' ')[1]}</div>

# 변경 후:
<div style="color: #718096; font-size: 14px;">시퀀스 범위: {result['SEQUENCE_START_TIME']} ~ {result['CURRENT_TIME']}</div>