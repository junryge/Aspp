"""
V7 급증 예측 실시간 시뮬레이션 애니메이션
- 30분 시퀀스 → 10분 후 예측
- 실시간 데이터 흐름 시각화
- 급증 감지 및 경고 표시
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# 윈도우 한글 폰트 설정
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class SurgePredictionSimulator:
    def __init__(self):
        # 데이터 버퍼 (30분 = 30개 포인트, 1분 간격)
        self.window_size = 30
        self.predict_ahead = 10
        
        # 실시간 데이터
        self.times = deque(maxlen=60)
        self.actual_values = deque(maxlen=60)
        self.predicted_values = deque(maxlen=60)
        self.hub_values = deque(maxlen=60)
        self.risk_scores = deque(maxlen=60)
        
        # 통계
        self.surge_detected = 0
        self.surge_total = 0
        self.current_time = 0
        
        # 임계값
        self.surge_threshold = 300
        self.warning_threshold = 280
        
        # 초기 데이터 생성
        self._init_data()
    
    def _init_data(self):
        """초기 30분 데이터 생성"""
        for i in range(30):
            self.current_time = i
            value = self._generate_value(i)
            hub = self._generate_hub(i, value)
            
            self.times.append(i)
            self.actual_values.append(value)
            self.hub_values.append(hub)
            self.predicted_values.append(None)
            self.risk_scores.append(0)
    
    def _generate_value(self, t):
        """CURRENT_M16A_3F_JOB_2 시뮬레이션"""
        base = 220 + np.sin(t * 0.1) * 30
        noise = np.random.normal(0, 8)
        
        # 주기적 급증 패턴 (120분 주기)
        surge_phase = t % 120
        if 75 < surge_phase < 95:
            progress = (surge_phase - 75) / 20
            surge = progress ** 2 * 180  # 급격한 상승
            return min(400, base + surge + noise)
        elif 95 <= surge_phase < 105:
            # 급증 후 하락
            progress = (surge_phase - 95) / 10
            return max(200, 350 - progress * 100 + noise)
        
        return max(150, base + noise)
    
    def _generate_hub(self, t, current_value):
        """HUBROOMTOTAL 시뮬레이션 (급증 전 감소)"""
        base = 625
        noise = np.random.normal(0, 5)
        
        surge_phase = t % 120
        if 65 < surge_phase < 95:
            # 급증 전 Hub 감소
            progress = (surge_phase - 65) / 30
            return max(570, base - progress * 50 + noise)
        
        return base + noise
    
    def _calculate_risk_score(self, hub, storage_util=7):
        """surge_risk_score 계산"""
        score = 0
        if hub < 590:
            score += 3  # hub_critical
        elif hub < 610:
            score += 2  # hub_high
        
        if storage_util >= 10:
            score += 2  # storage_util_critical
        elif storage_util >= 7:
            score += 1  # storage_util_high
        
        return score
    
    def _predict_next(self):
        """10분 후 값 예측 (간단한 시뮬레이션)"""
        if len(self.actual_values) < 10:
            return None
        
        recent = list(self.actual_values)[-10:]
        
        # Momentum 계산
        last_5_mean = np.mean(recent[-5:])
        prev_5_mean = np.mean(recent[:5])
        acceleration = (last_5_mean - prev_5_mean) / 5
        
        # 현재 트렌드 기반 예측
        current = recent[-1]
        slope = (recent[-1] - recent[0]) / len(recent)
        
        # 급증 징후 감지
        hub = list(self.hub_values)[-1]
        if hub < 610 and acceleration > 0.5:
            # 급증 가능성 높음
            prediction = current + acceleration * 10 + 30
        else:
            prediction = current + slope * 10
        
        return max(150, min(400, prediction))
    
    def update(self):
        """한 스텝 업데이트"""
        self.current_time += 1
        t = self.current_time
        
        # 새 데이터 생성
        actual = self._generate_value(t)
        hub = self._generate_hub(t, actual)
        predicted = self._predict_next()
        risk = self._calculate_risk_score(hub)
        
        self.times.append(t)
        self.actual_values.append(actual)
        self.hub_values.append(hub)
        self.predicted_values.append(predicted)
        self.risk_scores.append(risk)
        
        # 급증 감지 통계
        if actual >= self.surge_threshold:
            self.surge_total += 1
            if predicted and predicted >= self.warning_threshold:
                self.surge_detected += 1
        
        return actual, predicted, hub, risk


def create_animation():
    """애니메이션 생성"""
    sim = SurgePredictionSimulator()
    
    # Figure 설정
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a2e')
    
    # 서브플롯
    ax1 = fig.add_subplot(3, 1, 1)  # 메인 예측 차트
    ax2 = fig.add_subplot(3, 1, 2)  # Hub Room
    ax3 = fig.add_subplot(3, 1, 3)  # Risk Score
    
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#4a4a6a')
        ax.spines['top'].set_color('#4a4a6a')
        ax.spines['left'].set_color('#4a4a6a')
        ax.spines['right'].set_color('#4a4a6a')
    
    # 라인 객체
    line_actual, = ax1.plot([], [], 'cyan', linewidth=2, label='Actual')
    line_pred, = ax1.plot([], [], 'yellow', linewidth=2, linestyle='--', label='Predicted')
    line_hub, = ax2.plot([], [], '#00ff88', linewidth=2, label='Hub Room')
    bars = None
    
    # 상태 텍스트
    status_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14, 
                           color='white', weight='bold')
    alert_text = fig.text(0.85, 0.95, '', ha='center', fontsize=12, color='red')
    
    def init():
        ax1.set_xlim(0, 60)
        ax1.set_ylim(150, 420)
        ax1.axhline(y=300, color='red', linestyle='--', alpha=0.7, label='Surge (300)')
        ax1.axhline(y=280, color='orange', linestyle='--', alpha=0.5, label='Warning (280)')
        ax1.set_ylabel('CURRENT_M16A_3F_JOB_2', color='white')
        ax1.legend(loc='upper left', facecolor='#16213e', labelcolor='white')
        ax1.set_title('V7 Surge Prediction - Real-time Simulation', color='white', fontsize=12)
        
        ax2.set_xlim(0, 60)
        ax2.set_ylim(560, 650)
        ax2.axhline(y=590, color='red', linestyle='--', alpha=0.7, label='Critical (590)')
        ax2.axhline(y=610, color='orange', linestyle='--', alpha=0.5, label='High (610)')
        ax2.set_ylabel('HUBROOMTOTAL', color='white')
        ax2.legend(loc='upper left', facecolor='#16213e', labelcolor='white')
        
        ax3.set_xlim(0, 60)
        ax3.set_ylim(0, 6)
        ax3.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Critical (5)')
        ax3.axhline(y=3, color='orange', linestyle='--', alpha=0.5, label='High (3)')
        ax3.set_ylabel('Risk Score', color='white')
        ax3.set_xlabel('Time (minutes)', color='white')
        ax3.legend(loc='upper left', facecolor='#16213e', labelcolor='white')
        
        return line_actual, line_pred, line_hub
    
    def animate(frame):
        nonlocal bars
        
        # 데이터 업데이트
        actual, predicted, hub, risk = sim.update()
        
        times = list(sim.times)
        actuals = list(sim.actual_values)
        preds = [p if p else np.nan for p in sim.predicted_values]
        hubs = list(sim.hub_values)
        risks = list(sim.risk_scores)
        
        # X축 범위 조정 (슬라이딩 윈도우)
        if len(times) > 0:
            x_min = max(0, times[-1] - 55)
            x_max = times[-1] + 5
            ax1.set_xlim(x_min, x_max)
            ax2.set_xlim(x_min, x_max)
            ax3.set_xlim(x_min, x_max)
        
        # 라인 업데이트
        line_actual.set_data(times, actuals)
        line_pred.set_data(times, preds)
        line_hub.set_data(times, hubs)
        
        # Risk 바 차트
        if bars:
            for bar in bars:
                bar.remove()
        
        colors = []
        for r in risks:
            if r >= 5:
                colors.append('#ff0000')
            elif r >= 3:
                colors.append('#ff8800')
            elif r >= 1:
                colors.append('#ffff00')
            else:
                colors.append('#00ff00')
        
        bars = ax3.bar(times, risks, color=colors, alpha=0.7, width=0.8)
        
        # 상태 텍스트 업데이트
        detect_rate = (sim.surge_detected / sim.surge_total * 100) if sim.surge_total > 0 else 0
        pred_str = f'{predicted:.0f}' if predicted else 'N/A'
        status = f'Time: {sim.current_time}min | Current: {actual:.0f} | Predicted: {pred_str} | Detection Rate: {detect_rate:.1f}%'
        status_text.set_text(status)
        
        # 경고 표시
        if risk >= 5:
            alert_text.set_text('⚠ CRITICAL ALERT!')
            alert_text.set_color('#ff0000')
        elif risk >= 3:
            alert_text.set_text('⚠ HIGH RISK')
            alert_text.set_color('#ff8800')
        elif predicted and predicted >= 280:
            alert_text.set_text('! Warning')
            alert_text.set_color('#ffff00')
        else:
            alert_text.set_text('Normal')
            alert_text.set_color('#00ff00')
        
        return line_actual, line_pred, line_hub
    
    # 애니메이션 생성
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=200, interval=200, blit=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    return fig, anim


if __name__ == '__main__':
    print("V7 급증 예측 시뮬레이션 시작...")
    print("- 30분 시퀀스 → 10분 후 예측")
    print("- 임계값: 급증=300, 경고=280")
    print("- Hub<610 & acceleration>0.5 → 급증 징후")
    print("\n창을 닫으면 종료됩니다.")
    
    fig, anim = create_animation()
    
    # GIF 저장 여부 선택
    save_gif = input("\nGIF로 저장하시겠습니까? (y/n): ").strip().lower()
    if save_gif == 'y':
        import os
        save_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'surge_prediction.gif')
        print(f"GIF 저장 중... {save_path}")
        anim.save(save_path, writer='pillow', fps=5, dpi=100)
        print("저장 완료!")
    
    plt.show()