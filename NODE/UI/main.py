"""
M14 반송 큐 실시간 모니터링 UI
- 1분 간격 데이터 업데이트
- 실시간 그래프 표시
- main.py에서 기동
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import threading
import time
import warnings

warnings.filterwarnings('ignore')

# M14 데이터 모듈 import
from m14_data import M14DataManager, get_realtime_data

# 한글 폰트 설정
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class M14RealtimeMonitor:
    """M14 실시간 모니터링 클래스"""
    
    def __init__(self, window_minutes=60):
        """
        Args:
            window_minutes: 화면에 표시할 데이터 윈도우 (분)
        """
        self.window_minutes = window_minutes
        self.data_mgr = M14DataManager(window_minutes=window_minutes)
        
        # 업데이트 주기 (초)
        self.update_interval = 60  # 1분
        
        # 상태
        self.is_running = False
        self.last_update_time = None
        self.update_thread = None
        
        # 임계값 설정
        self.threshold_warning = 500
        self.threshold_critical = 700
    
    def _background_update(self):
        """백그라운드 데이터 업데이트 스레드"""
        while self.is_running:
            try:
                success = self.data_mgr.update()
                if success:
                    self.last_update_time = datetime.now()
                    print(f"[{self.last_update_time.strftime('%H:%M:%S')}] 데이터 업데이트 완료")
            except Exception as e:
                print(f"[ERROR] 업데이트 실패: {e}")
            
            # 다음 업데이트까지 대기
            time.sleep(self.update_interval)
    
    def start(self):
        """모니터링 시작"""
        print("=" * 60)
        print("M14 반송 큐 실시간 모니터링")
        print(f"업데이트 주기: {self.update_interval}초")
        print(f"데이터 윈도우: {self.window_minutes}분")
        print("=" * 60)
        
        # 초기 데이터 로드
        print("\n[초기화] 데이터 로드 중...")
        if not self.data_mgr.initialize():
            print("[ERROR] 초기 데이터 로드 실패!")
            print("데모 모드로 시작합니다...")
            self._start_demo_mode()
            return
        
        # 백그라운드 업데이트 시작
        self.is_running = True
        self.update_thread = threading.Thread(target=self._background_update, daemon=True)
        self.update_thread.start()
        
        # UI 시작
        self._create_ui()
    
    def _start_demo_mode(self):
        """데모 모드 (API 연결 안될 때)"""
        print("\n[데모 모드] 시뮬레이션 데이터 사용")
        self._create_demo_ui()
    
    def _create_ui(self):
        """실시간 UI 생성"""
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor('#1a1a2e')
        fig.suptitle('M14 반송 큐 실시간 모니터링', color='white', fontsize=14, fontweight='bold')
        
        # 서브플롯 생성
        ax1 = fig.add_subplot(3, 1, 1)  # TOTALCNT
        ax2 = fig.add_subplot(3, 1, 2)  # M14A-M10A / M14A-M14B / M14A-M16
        ax3 = fig.add_subplot(3, 1, 3)  # OHT Util / Queue Count
        
        axes = [ax1, ax2, ax3]
        for ax in axes:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#4a4a6a')
        
        # 라인 객체 초기화
        line_total, = ax1.plot([], [], 'cyan', linewidth=2, label='TOTALCNT')
        line_m10a, = ax2.plot([], [], '#00ff88', linewidth=1.5, label='M14A-M10A')
        line_m14b, = ax2.plot([], [], '#ff8800', linewidth=1.5, label='M14A-M14B')
        line_m16, = ax2.plot([], [], '#ff00ff', linewidth=1.5, label='M14A-M16')
        line_oht, = ax3.plot([], [], '#ffff00', linewidth=2, label='OHT Util')
        
        # 상태 텍스트
        status_text = fig.text(0.02, 0.97, '', fontsize=10, color='white')
        alert_text = fig.text(0.85, 0.97, '', fontsize=12, color='#00ff00', fontweight='bold')
        
        def init():
            # ax1: TOTALCNT
            ax1.set_xlim(0, self.window_minutes)
            ax1.set_ylim(0, 1000)
            ax1.axhline(y=self.threshold_warning, color='orange', linestyle='--', alpha=0.5, label=f'Warning ({self.threshold_warning})')
            ax1.axhline(y=self.threshold_critical, color='red', linestyle='--', alpha=0.5, label=f'Critical ({self.threshold_critical})')
            ax1.set_ylabel('TOTALCNT', color='white')
            ax1.legend(loc='upper left', facecolor='#16213e', labelcolor='white', fontsize=8)
            ax1.set_title('Total Transport Count', color='white', fontsize=10)
            
            # ax2: FAB 간 이동
            ax2.set_xlim(0, self.window_minutes)
            ax2.set_ylim(0, 200)
            ax2.set_ylabel('Count', color='white')
            ax2.legend(loc='upper left', facecolor='#16213e', labelcolor='white', fontsize=8)
            ax2.set_title('Inter-FAB Transport', color='white', fontsize=10)
            
            # ax3: OHT
            ax3.set_xlim(0, self.window_minutes)
            ax3.set_ylim(0, 100)
            ax3.set_ylabel('OHT Util (%)', color='white')
            ax3.set_xlabel('Time (minutes)', color='white')
            ax3.legend(loc='upper left', facecolor='#16213e', labelcolor='white', fontsize=8)
            ax3.set_title('OHT Utilization', color='white', fontsize=10)
            
            return line_total, line_m10a, line_m14b, line_m16, line_oht
        
        def animate(frame):
            data = self.data_mgr.get_data()
            
            if data is None or len(data) == 0:
                return line_total, line_m10a, line_m14b, line_m16, line_oht
            
            n = len(data)
            x = np.arange(n)
            
            # X축 범위 조정
            x_min = max(0, n - self.window_minutes)
            x_max = n + 5
            for ax in axes:
                ax.set_xlim(x_min, x_max)
            
            # 데이터 업데이트
            line_total.set_data(x, data['TOTALCNT'].values)
            line_m10a.set_data(x, data['M14AM10ASUM'].values)
            line_m14b.set_data(x, data['M14AM14BSUM'].values)
            line_m16.set_data(x, data['M14AM16SUM'].values)
            
            # OHT Util
            oht = data['M14.QUE.OHT.OHTUTIL'].values
            line_oht.set_data(x, oht)
            
            # Y축 자동 조정
            total_max = data['TOTALCNT'].max()
            ax1.set_ylim(0, max(1000, total_max * 1.2))
            
            fab_max = max(data['M14AM10ASUM'].max(), data['M14AM14BSUM'].max(), data['M14AM16SUM'].max())
            ax2.set_ylim(0, max(200, fab_max * 1.2))
            
            # 상태 텍스트
            latest = data.iloc[-1]
            currtime = latest.get('CURRTIME', 'N/A')
            total = latest.get('TOTALCNT', 0)
            
            update_str = self.last_update_time.strftime('%H:%M:%S') if self.last_update_time else 'N/A'
            status_text.set_text(f"Last Update: {update_str} | CURRTIME: {currtime} | TOTALCNT: {total:.0f} | Data Points: {n}")
            
            # 경고 표시
            if total >= self.threshold_critical:
                alert_text.set_text('⚠ CRITICAL!')
                alert_text.set_color('#ff0000')
            elif total >= self.threshold_warning:
                alert_text.set_text('⚠ WARNING')
                alert_text.set_color('#ff8800')
            else:
                alert_text.set_text('● NORMAL')
                alert_text.set_color('#00ff00')
            
            return line_total, line_m10a, line_m14b, line_m16, line_oht
        
        # 애니메이션 (5초마다 화면 갱신)
        anim = FuncAnimation(fig, animate, init_func=init, 
                            frames=None, interval=5000, blit=False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
        # 종료 시 정리
        self.is_running = False
    
    def _create_demo_ui(self):
        """데모 UI (시뮬레이션 데이터)"""
        from collections import deque
        
        # 시뮬레이션 데이터
        times = deque(maxlen=self.window_minutes)
        totals = deque(maxlen=self.window_minutes)
        m10a = deque(maxlen=self.window_minutes)
        m14b = deque(maxlen=self.window_minutes)
        m16 = deque(maxlen=self.window_minutes)
        oht = deque(maxlen=self.window_minutes)
        
        # 초기 데이터 생성
        for i in range(30):
            times.append(i)
            totals.append(400 + np.random.normal(0, 50))
            m10a.append(30 + np.random.normal(0, 10))
            m14b.append(50 + np.random.normal(0, 15))
            m16.append(20 + np.random.normal(0, 8))
            oht.append(60 + np.random.normal(0, 10))
        
        current_time = [30]
        
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor('#1a1a2e')
        fig.suptitle('M14 반송 큐 모니터링 [DEMO MODE]', color='yellow', fontsize=14, fontweight='bold')
        
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)
        
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#4a4a6a')
        
        line_total, = ax1.plot([], [], 'cyan', linewidth=2, label='TOTALCNT')
        line_m10a, = ax2.plot([], [], '#00ff88', linewidth=1.5, label='M14A-M10A')
        line_m14b, = ax2.plot([], [], '#ff8800', linewidth=1.5, label='M14A-M14B')
        line_m16, = ax2.plot([], [], '#ff00ff', linewidth=1.5, label='M14A-M16')
        line_oht, = ax3.plot([], [], '#ffff00', linewidth=2, label='OHT Util')
        
        status_text = fig.text(0.02, 0.97, '', fontsize=10, color='white')
        alert_text = fig.text(0.85, 0.97, '', fontsize=12, color='#00ff00', fontweight='bold')
        
        def init():
            ax1.set_xlim(0, 60)
            ax1.set_ylim(0, 800)
            ax1.axhline(y=500, color='orange', linestyle='--', alpha=0.5)
            ax1.axhline(y=700, color='red', linestyle='--', alpha=0.5)
            ax1.set_ylabel('TOTALCNT', color='white')
            ax1.legend(loc='upper left', facecolor='#16213e', labelcolor='white', fontsize=8)
            
            ax2.set_xlim(0, 60)
            ax2.set_ylim(0, 150)
            ax2.set_ylabel('Count', color='white')
            ax2.legend(loc='upper left', facecolor='#16213e', labelcolor='white', fontsize=8)
            
            ax3.set_xlim(0, 60)
            ax3.set_ylim(0, 100)
            ax3.set_ylabel('OHT Util (%)', color='white')
            ax3.set_xlabel('Time (minutes)', color='white')
            ax3.legend(loc='upper left', facecolor='#16213e', labelcolor='white', fontsize=8)
            
            return line_total, line_m10a, line_m14b, line_m16, line_oht
        
        def animate(frame):
            current_time[0] += 1
            t = current_time[0]
            
            # 시뮬레이션 데이터 생성
            surge = 200 * np.sin(t * 0.05) ** 2 if 50 < t % 120 < 80 else 0
            
            times.append(t)
            totals.append(400 + surge + np.random.normal(0, 30))
            m10a.append(30 + surge * 0.1 + np.random.normal(0, 5))
            m14b.append(50 + surge * 0.15 + np.random.normal(0, 8))
            m16.append(20 + surge * 0.05 + np.random.normal(0, 4))
            oht.append(min(95, 60 + surge * 0.1 + np.random.normal(0, 5)))
            
            x = list(times)
            x_min = max(0, t - 55)
            x_max = t + 5
            
            ax1.set_xlim(x_min, x_max)
            ax2.set_xlim(x_min, x_max)
            ax3.set_xlim(x_min, x_max)
            
            line_total.set_data(x, list(totals))
            line_m10a.set_data(x, list(m10a))
            line_m14b.set_data(x, list(m14b))
            line_m16.set_data(x, list(m16))
            line_oht.set_data(x, list(oht))
            
            current_total = list(totals)[-1]
            now_str = datetime.now().strftime('%H:%M:%S')
            status_text.set_text(f"[DEMO] Time: {now_str} | TOTALCNT: {current_total:.0f}")
            
            if current_total >= 700:
                alert_text.set_text('⚠ CRITICAL!')
                alert_text.set_color('#ff0000')
            elif current_total >= 500:
                alert_text.set_text('⚠ WARNING')
                alert_text.set_color('#ff8800')
            else:
                alert_text.set_text('● NORMAL')
                alert_text.set_color('#00ff00')
            
            return line_total, line_m10a, line_m14b, line_m16, line_oht
        
        anim = FuncAnimation(fig, animate, init_func=init, 
                            frames=None, interval=1000, blit=False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("M14 반송 큐 실시간 모니터링 시스템")
    print("=" * 60)
    print("\n옵션:")
    print("  1. 실시간 모니터링 (API 연결)")
    print("  2. 데모 모드 (시뮬레이션)")
    print("  3. 종료")
    
    choice = input("\n선택 (1/2/3): ").strip()
    
    if choice == '1':
        monitor = M14RealtimeMonitor(window_minutes=60)
        monitor.start()
    elif choice == '2':
        monitor = M14RealtimeMonitor(window_minutes=60)
        monitor._start_demo_mode()
    else:
        print("종료합니다.")


if __name__ == '__main__':
    main()