# 변경 전:
"PREDICTION_TIME": f"{prediction_start.strftime('%H:%M')}~{prediction_end.strftime('%H:%M')}"

# 변경 후:
"PREDICTION_TIME": f"{prediction_start.strftime('%Y-%m-%d %H:%M')} ~ {prediction_end.strftime('%Y-%m-%d %H:%M')}"