"""
================================================================================
AMHS 예측 모델 패치노트 관리 프로그램 (GUI)
================================================================================
기능:
  - 패치노트 작성/수정/삭제
  - JSON 파일로 저장/로드
  - 탭으로 버전별 분리
  - MD 내보내기
================================================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class PatchNote:
    """패치노트 데이터 클래스"""
    
    def __init__(self, version: str, date: str = None, note_type: str = "Major"):
        self.version = version
        self.date = date or datetime.now().strftime("%Y-%m-%d")
        self.type = note_type
        self.new: List[str] = []
        self.improved: List[str] = []
        self.fixed: List[str] = []
        self.breaking: List[str] = []
        self.metrics: List[str] = []
        self.files: List[str] = []
        self.next: List[str] = []
        self.important: List[str] = []
        # 버전 관리 필드
        self.vm_date: str = ""
        self.vm_operation: str = ""
        self.vm_development: str = ""
        self.vm_fab: str = ""
        self.vm_model_name: str = ""
        self.vm_version: str = ""
        self.vm_total_predict: str = ""
        self.vm_early_detect: str = ""
        self.vm_model_desc: List[str] = []
    
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "date": self.date,
            "type": self.type,
            "new": self.new,
            "improved": self.improved,
            "fixed": self.fixed,
            "breaking": self.breaking,
            "metrics": self.metrics,
            "files": self.files,
            "next": self.next,
            "important": self.important,
            "vm_date": self.vm_date,
            "vm_operation": self.vm_operation,
            "vm_development": self.vm_development,
            "vm_fab": self.vm_fab,
            "vm_model_name": self.vm_model_name,
            "vm_version": self.vm_version,
            "vm_total_predict": self.vm_total_predict,
            "vm_early_detect": self.vm_early_detect,
            "vm_model_desc": self.vm_model_desc,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PatchNote':
        note = cls(data["version"], data["date"], data.get("type", "Major"))
        note.new = data.get("new", [])
        note.fixed = data.get("fixed", [])
        note.breaking = data.get("breaking", [])
        note.files = data.get("files", [])
        note.next = data.get("next", [])
        note.important = data.get("important", [])
        # 버전 관리 필드
        note.vm_date = data.get("vm_date", "")
        note.vm_operation = data.get("vm_operation", "")
        note.vm_development = data.get("vm_development", "")
        note.vm_fab = data.get("vm_fab", "")
        note.vm_model_name = data.get("vm_model_name", "")
        note.vm_version = data.get("vm_version", "")
        note.vm_total_predict = data.get("vm_total_predict", "")
        note.vm_early_detect = data.get("vm_early_detect", "")
        note.vm_model_desc = data.get("vm_model_desc", [])
        
        # improved 호환성 처리
        improved_data = data.get("improved", [])
        if improved_data and isinstance(improved_data[0], dict):
            note.improved = [f"{item.get('item', '')}: {item.get('before', '')} → {item.get('after', '')}" for item in improved_data]
        else:
            note.improved = improved_data
        
        # metrics 호환성 처리
        metrics_data = data.get("metrics", [])
        if isinstance(metrics_data, dict):
            note.metrics = []
            for name, info in metrics_data.items():
                if isinstance(info, dict):
                    note.metrics.append(f"{name}: {info.get('before', '-')} → {info.get('after', '-')} ({info.get('change', '')})")
                else:
                    note.metrics.append(f"{name}: {info}")
        else:
            note.metrics = metrics_data
        
        return note


class PatchNoteApp:
    """패치노트 GUI 애플리케이션"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("AMHS 예측 모델 패치노트 관리")
        self.root.geometry("900x700")
        self.root.configure(bg="#2b2b2b")
        
        self.notes: Dict[str, PatchNote] = {}
        self.current_version: Optional[str] = None
        self.save_file = "patch_notes.json"
        
        # 스타일 설정
        self.setup_styles()
        
        # UI 구성
        self.create_widgets()
        
        # 샘플 데이터 로드
        self.load_or_create_sample()
    
    def setup_styles(self):
        """스타일 설정"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 다크 테마 색상
        self.colors = {
            'bg': '#2b2b2b',
            'fg': '#ffffff',
            'accent': '#4a9eff',
            'accent2': '#ff6b6b',
            'card': '#3c3c3c',
            'input': '#404040',
            'border': '#555555',
        }
        
        style.configure('TFrame', background=self.colors['bg'])
        style.configure('Card.TFrame', background=self.colors['card'])
        style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['fg'], font=('맑은 고딕', 10))
        style.configure('Title.TLabel', font=('맑은 고딕', 14, 'bold'))
        style.configure('Header.TLabel', font=('맑은 고딕', 12, 'bold'))
        style.configure('TButton', font=('맑은 고딕', 10))
        style.configure('Accent.TButton', font=('맑은 고딕', 10, 'bold'))
        style.configure('TNotebook', background=self.colors['bg'])
        style.configure('TNotebook.Tab', font=('맑은 고딕', 10), padding=[10, 5])
    
    def create_widgets(self):
        """위젯 생성"""
        
        # ===== 상단 헤더 =====
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(header_frame, text="📋 AMHS 예측 모델 패치노트", style='Title.TLabel').pack(side=tk.LEFT)
        
        # 버튼들
        btn_frame = ttk.Frame(header_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(btn_frame, text="💾 저장", command=self.save_notes).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📂 로드", command=self.load_notes).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📄 MD 내보내기", command=self.export_md).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="➕ 새 버전", command=self.new_version_dialog).pack(side=tk.LEFT, padx=2)
        
        # ===== 메인 영역 =====
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 좌측: 버전 목록
        left_frame = ttk.Frame(main_frame, width=150)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        ttk.Label(left_frame, text="📦 버전 목록", style='Header.TLabel').pack(pady=5)
        
        self.version_listbox = tk.Listbox(
            left_frame, 
            font=('맑은 고딕', 11),
            bg=self.colors['card'],
            fg=self.colors['fg'],
            selectbackground=self.colors['accent'],
            selectforeground='white',
            borderwidth=0,
            highlightthickness=1,
            highlightcolor=self.colors['border']
        )
        self.version_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.version_listbox.bind('<<ListboxSelect>>', self.on_version_select)
        
        ttk.Button(left_frame, text="🗑️ 삭제", command=self.delete_version).pack(fill=tk.X, pady=5)
        
        # 우측: 패치노트 상세
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 탭 노트북
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 탭 1: 기본 정보
        self.tab_info = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_info, text="📝 기본 정보")
        self.create_info_tab()
        
        # 탭 2: 신규/개선
        self.tab_changes = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_changes, text="✨ 변경사항")
        self.create_changes_tab()
        
        # 탭 3: 성능/파일
        self.tab_metrics = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_metrics, text="📊 성능/파일")
        self.create_metrics_tab()
        
        # 탭 4: 버전 관리
        self.tab_version = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_version, text="🔖 버전 관리")
        self.create_version_tab()
        
        # 탭 5: 미리보기
        self.tab_preview = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_preview, text="👁️ 미리보기")
        self.create_preview_tab()
    
    def create_info_tab(self):
        """기본 정보 탭"""
        frame = ttk.Frame(self.tab_info, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 버전
        ttk.Label(frame, text="버전:").grid(row=0, column=0, sticky='w', pady=5)
        self.entry_version = ttk.Entry(frame, font=('맑은 고딕', 11), width=20)
        self.entry_version.grid(row=0, column=1, sticky='w', pady=5, padx=5)
        
        # 날짜
        ttk.Label(frame, text="날짜:").grid(row=1, column=0, sticky='w', pady=5)
        self.entry_date = ttk.Entry(frame, font=('맑은 고딕', 11), width=20)
        self.entry_date.grid(row=1, column=1, sticky='w', pady=5, padx=5)
        
        # 유형
        ttk.Label(frame, text="유형:").grid(row=2, column=0, sticky='w', pady=5)
        self.combo_type = ttk.Combobox(frame, values=["Major", "Minor", "Patch"], font=('맑은 고딕', 11), width=17)
        self.combo_type.grid(row=2, column=1, sticky='w', pady=5, padx=5)
        self.combo_type.set("Major")
        
        # 중요 정보
        ttk.Label(frame, text="⚠️ 중요 정보:").grid(row=3, column=0, sticky='nw', pady=5)
        self.text_important = scrolledtext.ScrolledText(frame, height=4, width=50, font=('맑은 고딕', 10),
                                                        bg=self.colors['input'], fg=self.colors['fg'])
        self.text_important.grid(row=3, column=1, sticky='w', pady=5, padx=5)
        
        # Breaking Changes
        ttk.Label(frame, text="Breaking Changes:").grid(row=4, column=0, sticky='nw', pady=5)
        self.text_breaking = scrolledtext.ScrolledText(frame, height=4, width=50, font=('맑은 고딕', 10),
                                                        bg=self.colors['input'], fg=self.colors['fg'])
        self.text_breaking.grid(row=4, column=1, sticky='w', pady=5, padx=5)
        
        # 다음 버전 예고
        ttk.Label(frame, text="다음 버전 예고:").grid(row=5, column=0, sticky='nw', pady=5)
        self.text_next = scrolledtext.ScrolledText(frame, height=4, width=50, font=('맑은 고딕', 10),
                                                    bg=self.colors['input'], fg=self.colors['fg'])
        self.text_next.grid(row=5, column=1, sticky='w', pady=5, padx=5)
    
    def create_changes_tab(self):
        """변경사항 탭"""
        frame = ttk.Frame(self.tab_changes, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 신규 기능
        ttk.Label(frame, text="✨ 신규 기능 (줄바꿈으로 구분):", style='Header.TLabel').pack(anchor='w')
        self.text_new = scrolledtext.ScrolledText(frame, height=6, font=('맑은 고딕', 10),
                                                   bg=self.colors['input'], fg=self.colors['fg'])
        self.text_new.pack(fill=tk.X, pady=(5, 15))
        
        # 개선사항
        ttk.Label(frame, text="⬆️ 개선사항 (줄바꿈으로 구분):", style='Header.TLabel').pack(anchor='w')
        self.text_improved = scrolledtext.ScrolledText(frame, height=6, font=('맑은 고딕', 10),
                                                        bg=self.colors['input'], fg=self.colors['fg'])
        self.text_improved.pack(fill=tk.X, pady=(5, 15))
        
        # 버그 수정
        ttk.Label(frame, text="🔧 버그 수정 (줄바꿈으로 구분):", style='Header.TLabel').pack(anchor='w')
        self.text_fixed = scrolledtext.ScrolledText(frame, height=6, font=('맑은 고딕', 10),
                                                     bg=self.colors['input'], fg=self.colors['fg'])
        self.text_fixed.pack(fill=tk.X, pady=(5, 15))
    
    def create_metrics_tab(self):
        """성능/파일 탭"""
        frame = ttk.Frame(self.tab_metrics, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 성능 지표
        ttk.Label(frame, text="📊 성능 변화 (줄바꿈으로 구분):", style='Header.TLabel').pack(anchor='w')
        self.text_metrics = scrolledtext.ScrolledText(frame, height=6, font=('맑은 고딕', 10),
                                                       bg=self.colors['input'], fg=self.colors['fg'])
        self.text_metrics.pack(fill=tk.X, pady=(5, 15))
        
        # 변경 파일
        ttk.Label(frame, text="📁 변경 파일 (줄바꿈으로 구분):", style='Header.TLabel').pack(anchor='w')
        self.text_files = scrolledtext.ScrolledText(frame, height=8, font=('맑은 고딕', 10),
                                                     bg=self.colors['input'], fg=self.colors['fg'])
        self.text_files.pack(fill=tk.X, pady=(5, 15))
    
    def create_version_tab(self):
        """버전 관리 탭"""
        frame = ttk.Frame(self.tab_version, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 상단 그리드 (날짜, 운영, 개발, FAB)
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 날짜
        ttk.Label(top_frame, text="📅 날짜:").grid(row=0, column=0, sticky='w', pady=5)
        self.entry_vm_date = ttk.Entry(top_frame, font=('맑은 고딕', 11), width=15)
        self.entry_vm_date.grid(row=0, column=1, sticky='w', pady=5, padx=5)
        
        # 운영
        ttk.Label(top_frame, text="🏭 운영:").grid(row=0, column=2, sticky='w', pady=5, padx=(20, 0))
        self.entry_vm_operation = ttk.Entry(top_frame, font=('맑은 고딕', 11), width=15)
        self.entry_vm_operation.grid(row=0, column=3, sticky='w', pady=5, padx=5)
        
        # 개발
        ttk.Label(top_frame, text="💻 개발:").grid(row=0, column=4, sticky='w', pady=5, padx=(20, 0))
        self.entry_vm_development = ttk.Entry(top_frame, font=('맑은 고딕', 11), width=15)
        self.entry_vm_development.grid(row=0, column=5, sticky='w', pady=5, padx=5)
        
        # FAB
        ttk.Label(top_frame, text="🏢 FAB:").grid(row=1, column=0, sticky='w', pady=5)
        self.entry_vm_fab = ttk.Entry(top_frame, font=('맑은 고딕', 11), width=15)
        self.entry_vm_fab.grid(row=1, column=1, sticky='w', pady=5, padx=5)
        
        # 예측모델_NAME
        ttk.Label(top_frame, text="🤖 예측모델_NAME:").grid(row=1, column=2, sticky='w', pady=5, padx=(20, 0))
        self.entry_vm_model_name = ttk.Entry(top_frame, font=('맑은 고딕', 11), width=25)
        self.entry_vm_model_name.grid(row=1, column=3, columnspan=3, sticky='w', pady=5, padx=5)
        
        # 중간 그리드 (버전, 전체예측, 사전감지예측)
        mid_frame = ttk.Frame(frame)
        mid_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 버전
        ttk.Label(mid_frame, text="🔖 버전:").grid(row=0, column=0, sticky='w', pady=5)
        self.entry_vm_version = ttk.Entry(mid_frame, font=('맑은 고딕', 11), width=15)
        self.entry_vm_version.grid(row=0, column=1, sticky='w', pady=5, padx=5)
        
        # 전체예측
        ttk.Label(mid_frame, text="📊 전체예측:").grid(row=0, column=2, sticky='w', pady=5, padx=(20, 0))
        self.entry_vm_total_predict = ttk.Entry(mid_frame, font=('맑은 고딕', 11), width=20)
        self.entry_vm_total_predict.grid(row=0, column=3, sticky='w', pady=5, padx=5)
        
        # 사전감지예측
        ttk.Label(mid_frame, text="🔍 사전감지예측:").grid(row=0, column=4, sticky='w', pady=5, padx=(20, 0))
        self.entry_vm_early_detect = ttk.Entry(mid_frame, font=('맑은 고딕', 11), width=20)
        self.entry_vm_early_detect.grid(row=0, column=5, sticky='w', pady=5, padx=5)
        
        # 모델 내용
        ttk.Label(frame, text="📝 모델 내용 (줄바꿈으로 구분):", style='Header.TLabel').pack(anchor='w')
        self.text_vm_model_desc = scrolledtext.ScrolledText(frame, height=12, font=('맑은 고딕', 10),
                                                            bg=self.colors['input'], fg=self.colors['fg'])
        self.text_vm_model_desc.pack(fill=tk.BOTH, expand=True, pady=(5, 15))
    
    def create_preview_tab(self):
        """미리보기 탭"""
        frame = ttk.Frame(self.tab_preview, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(frame, text="🔄 새로고침", command=self.refresh_preview).pack(anchor='w', pady=5)
        
        self.text_preview = scrolledtext.ScrolledText(
            frame, 
            font=('Consolas', 10),
            bg=self.colors['card'], 
            fg=self.colors['fg'],
            wrap=tk.NONE
        )
        self.text_preview.pack(fill=tk.BOTH, expand=True, pady=5)
    
    # ===== 기능 메서드 =====
    
    def load_or_create_sample(self):
        """파일 로드 또는 샘플 생성"""
        if os.path.exists(self.save_file):
            self._load_from_file(self.save_file)
        else:
            self._create_sample_data()
        self.refresh_version_list()
    
    def _create_sample_data(self):
        """샘플 데이터 생성"""
        # V8
        v8 = PatchNote("V8.0.0", "2025-12-01", "Major")
        v8.important = [
            "V7 모델과 호환되지 않음 - 반드시 V8 코드 사용",
            "기존 patch_notes.json 삭제 후 재실행 필요",
        ]
        v8.new = [
            "범주형 3-Class 분류 모델 추가 (하락/소폭/급증)",
            "M16HUB QUE 컬럼 5개 추가",
            "M16A QUE 컬럼 7개 추가",
            "CURRENT_M16A_3F_JOB 대체 타겟 컬럼 추가",
        ]
        v8.improved = [
            "입력 컬럼: 19개 → 32개",
            "Feature 수: 180개 → 222개",
            "수치형 모델: 50+ 변화 샘플 3배 가중치 적용",
            "범주형 모델: Class2(급증) 5배 가중치 적용",
        ]
        v8.metrics = [
            "MAE: 12.5 → 9.8 (↓21%)",
            "R²: 0.62 → 0.68 (↑9.7%)",
            "범주형 정확도: 72% (신규)",
        ]
        v8.breaking = ["V7 모델 파일(.pkl)과 호환되지 않음", "V8 전용 평가 코드 사용 필요"]
        v8.files = [
            "V8_수치형_학습코드.py", "V8_수치형_평가코드.py",
            "V8_범주형_학습코드.py", "V8_범주형_평가코드.py",
            "xgboost_수치형_V8.pkl", "xgboost_범주형_V8.pkl"
        ]
        v8.next = ["MLflow Tracking 연동", "하이퍼파라미터 자동 탐색 (Optuna)"]
        # 버전 관리 정보
        v8.vm_date = "2025-12-01"
        v8.vm_operation = "M16A"
        v8.vm_development = "AI팀"
        v8.vm_fab = "M16"
        v8.vm_model_name = "AMHS_JOB2_예측모델"
        v8.vm_version = "V8.0.0"
        v8.vm_total_predict = "MAE 9.8, R² 0.68"
        v8.vm_early_detect = "범주형 정확도 72%"
        v8.vm_model_desc = [
            "수치형: XGBoost Regressor (MIN/MAX/AVG 멀티아웃풋)",
            "범주형: XGBoost Classifier (3-Class: 하락/소폭/급증)",
            "입력: 32개 컬럼 → 222개 Feature",
            "시퀀스: 30분 → 10분 후 예측",
            "가중치: 수치형 50+ 3배, 범주형 Class2 5배",
        ]
        self.notes["V8.0.0"] = v8
        
        # V7
        v7 = PatchNote("V7.0.0", "2025-11-15", "Major")
        v7.new = ["M16HUB.QUE.ALL.CURRENTQCNT 컬럼 추가", "QUE 기반 병목 감지 Feature 추가"]
        v7.improved = ["입력 컬럼: 17개 → 19개", "병목 감지 로직 강화"]
        v7.metrics = ["MAE: 14.2 → 12.5 (↓12%)"]
        v7.fixed = ["HUBROOMTOTAL 결측치 처리 오류 수정"]
        v7.breaking = ["V6 모델 파일과 호환되지 않음"]
        v7.next = ["범주형 분류 모델 추가"]
        self.notes["V7.0.0"] = v7
        
        # V6
        v6 = PatchNote("V6.0.0", "2025-10-20", "Major")
        v6.new = ["멀티아웃풋 구조 도입 (MIN/MAX/AVG)", "10분 구간 통계 예측"]
        v6.improved = ["Feature 수: 120개 → 165개", "시퀀스 길이: 20분 → 30분"]
        v6.metrics = ["MAE: 18.5 → 14.2 (↓23%)"]
        v6.fixed = ["큰 변화 예측 누락 문제 해결"]
        v6.breaking = ["단일 출력 → 멀티 출력 구조 변경"]
        self.notes["V6.0.0"] = v6
    
    def refresh_version_list(self):
        """버전 목록 갱신"""
        self.version_listbox.delete(0, tk.END)
        for version in sorted(self.notes.keys(), reverse=True):
            self.version_listbox.insert(tk.END, version)
        
        if self.notes:
            self.version_listbox.selection_set(0)
            self.on_version_select(None)
    
    def on_version_select(self, event):
        """버전 선택 시"""
        selection = self.version_listbox.curselection()
        if not selection:
            return
        
        version = self.version_listbox.get(selection[0])
        self.current_version = version
        self.load_note_to_ui(self.notes[version])
    
    def load_note_to_ui(self, note: PatchNote):
        """패치노트를 UI에 로드"""
        # 기본 정보
        self.entry_version.delete(0, tk.END)
        self.entry_version.insert(0, note.version)
        
        self.entry_date.delete(0, tk.END)
        self.entry_date.insert(0, note.date)
        
        self.combo_type.set(note.type)
        
        self.text_important.delete('1.0', tk.END)
        self.text_important.insert('1.0', '\n'.join(note.important))
        
        self.text_breaking.delete('1.0', tk.END)
        self.text_breaking.insert('1.0', '\n'.join(note.breaking))
        
        self.text_next.delete('1.0', tk.END)
        self.text_next.insert('1.0', '\n'.join(note.next))
        
        # 변경사항
        self.text_new.delete('1.0', tk.END)
        self.text_new.insert('1.0', '\n'.join(note.new))
        
        self.text_improved.delete('1.0', tk.END)
        self.text_improved.insert('1.0', '\n'.join(note.improved))
        
        self.text_fixed.delete('1.0', tk.END)
        self.text_fixed.insert('1.0', '\n'.join(note.fixed))
        
        # 성능/파일
        self.text_metrics.delete('1.0', tk.END)
        self.text_metrics.insert('1.0', '\n'.join(note.metrics))
        
        self.text_files.delete('1.0', tk.END)
        self.text_files.insert('1.0', '\n'.join(note.files))
        
        # 버전 관리
        self.entry_vm_date.delete(0, tk.END)
        self.entry_vm_date.insert(0, note.vm_date)
        
        self.entry_vm_operation.delete(0, tk.END)
        self.entry_vm_operation.insert(0, note.vm_operation)
        
        self.entry_vm_development.delete(0, tk.END)
        self.entry_vm_development.insert(0, note.vm_development)
        
        self.entry_vm_fab.delete(0, tk.END)
        self.entry_vm_fab.insert(0, note.vm_fab)
        
        self.entry_vm_model_name.delete(0, tk.END)
        self.entry_vm_model_name.insert(0, note.vm_model_name)
        
        self.entry_vm_version.delete(0, tk.END)
        self.entry_vm_version.insert(0, note.vm_version)
        
        self.entry_vm_total_predict.delete(0, tk.END)
        self.entry_vm_total_predict.insert(0, note.vm_total_predict)
        
        self.entry_vm_early_detect.delete(0, tk.END)
        self.entry_vm_early_detect.insert(0, note.vm_early_detect)
        
        self.text_vm_model_desc.delete('1.0', tk.END)
        self.text_vm_model_desc.insert('1.0', '\n'.join(note.vm_model_desc))
        
        # 미리보기
        self.refresh_preview()
    
    def save_ui_to_note(self) -> Optional[PatchNote]:
        """UI에서 패치노트로 저장"""
        version = self.entry_version.get().strip()
        if not version:
            messagebox.showerror("오류", "버전을 입력하세요")
            return None
        
        note = PatchNote(
            version,
            self.entry_date.get().strip() or datetime.now().strftime("%Y-%m-%d"),
            self.combo_type.get()
        )
        
        note.new = [x.strip() for x in self.text_new.get('1.0', tk.END).strip().split('\n') if x.strip()]
        note.improved = [x.strip() for x in self.text_improved.get('1.0', tk.END).strip().split('\n') if x.strip()]
        note.fixed = [x.strip() for x in self.text_fixed.get('1.0', tk.END).strip().split('\n') if x.strip()]
        note.breaking = [x.strip() for x in self.text_breaking.get('1.0', tk.END).strip().split('\n') if x.strip()]
        note.next = [x.strip() for x in self.text_next.get('1.0', tk.END).strip().split('\n') if x.strip()]
        note.files = [x.strip() for x in self.text_files.get('1.0', tk.END).strip().split('\n') if x.strip()]
        note.metrics = [x.strip() for x in self.text_metrics.get('1.0', tk.END).strip().split('\n') if x.strip()]
        note.important = [x.strip() for x in self.text_important.get('1.0', tk.END).strip().split('\n') if x.strip()]
        
        # 버전 관리
        note.vm_date = self.entry_vm_date.get().strip()
        note.vm_operation = self.entry_vm_operation.get().strip()
        note.vm_development = self.entry_vm_development.get().strip()
        note.vm_fab = self.entry_vm_fab.get().strip()
        note.vm_model_name = self.entry_vm_model_name.get().strip()
        note.vm_version = self.entry_vm_version.get().strip()
        note.vm_total_predict = self.entry_vm_total_predict.get().strip()
        note.vm_early_detect = self.entry_vm_early_detect.get().strip()
        note.vm_model_desc = [x.strip() for x in self.text_vm_model_desc.get('1.0', tk.END).strip().split('\n') if x.strip()]
        
        return note
    
    def refresh_preview(self):
        """미리보기 갱신"""
        note = self.save_ui_to_note()
        if not note:
            return
        
        preview = self._generate_preview(note)
        self.text_preview.delete('1.0', tk.END)
        self.text_preview.insert('1.0', preview)
    
    def _generate_preview(self, note: PatchNote) -> str:
        """미리보기 텍스트 생성"""
        lines = []
        w = 58
        
        lines.append("┌" + "─" * w + "┐")
        lines.append(f"│  AMHS 예측 모델 {note.version}".ljust(w) + " │")
        lines.append(f"│  {note.date} | {note.type} Update".ljust(w) + " │")
        lines.append("├" + "─" * w + "┤")
        
        if note.important:
            lines.append(f"│  [⚠️ IMPORTANT] 중요 정보".ljust(w) + " │")
            for item in note.important:
                line = f"│    • {item}"
                lines.append(line[:w].ljust(w) + " │")
        
        if note.new:
            lines.append(f"│  [NEW] 신규 기능".ljust(w) + " │")
            for item in note.new:
                line = f"│    • {item}"
                lines.append(line[:w].ljust(w) + " │")
        
        if note.improved:
            lines.append(f"│  [IMPROVED] 개선사항".ljust(w) + " │")
            for item in note.improved:
                line = f"│    • {item}"
                lines.append(line[:w].ljust(w) + " │")
        
        if note.metrics:
            lines.append(f"│  [METRICS] 성능 변화".ljust(w) + " │")
            for item in note.metrics:
                line = f"│    • {item}"
                lines.append(line[:w].ljust(w) + " │")
        
        lines.append(f"│  [FIXED] 버그 수정".ljust(w) + " │")
        if note.fixed:
            for item in note.fixed:
                line = f"│    • {item}"
                lines.append(line[:w].ljust(w) + " │")
        else:
            lines.append(f"│    • 없음".ljust(w) + " │")
        
        if note.breaking:
            lines.append(f"│  [BREAKING] 호환성".ljust(w) + " │")
            for item in note.breaking:
                line = f"│    • {item}"
                lines.append(line[:w].ljust(w) + " │")
        
        if note.files:
            lines.append(f"│  [FILES] 변경 파일".ljust(w) + " │")
            for item in note.files:
                line = f"│    • {item}"
                lines.append(line[:w].ljust(w) + " │")
        
        if note.next:
            lines.append(f"│  [NEXT] 다음 버전 예고".ljust(w) + " │")
            for item in note.next:
                line = f"│    • {item}"
                lines.append(line[:w].ljust(w) + " │")
        
        lines.append("└" + "─" * w + "┘")
        
        return '\n'.join(lines)
    
    def save_notes(self):
        """저장"""
        # 현재 편집 중인 노트 저장
        if self.current_version:
            note = self.save_ui_to_note()
            if note:
                # 버전이 변경된 경우
                if note.version != self.current_version:
                    del self.notes[self.current_version]
                self.notes[note.version] = note
                self.current_version = note.version
        
        # 파일로 저장
        data = {
            "project": "AMHS 예측 모델",
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": {v: n.to_dict() for v, n in self.notes.items()}
        }
        
        with open(self.save_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.refresh_version_list()
        messagebox.showinfo("저장", f"저장 완료: {self.save_file}")
    
    def load_notes(self):
        """로드"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self._load_from_file(filename)
            self.refresh_version_list()
            messagebox.showinfo("로드", f"로드 완료: {filename}")
    
    def _load_from_file(self, filename):
        """파일에서 로드"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.notes = {}
        for version, note_data in data.get("notes", {}).items():
            self.notes[version] = PatchNote.from_dict(note_data)
    
    def export_md(self):
        """마크다운 내보내기"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown files", "*.md"), ("All files", "*.*")],
            initialfile="PATCH_NOTES.md"
        )
        if not filename:
            return
        
        lines = []
        lines.append("# AMHS 예측 모델 패치노트\n")
        lines.append(f"*최종 업데이트: {datetime.now().strftime('%Y-%m-%d')}*\n")
        
        for version in sorted(self.notes.keys(), reverse=True):
            note = self.notes[version]
            lines.append(f"\n---\n")
            lines.append(f"## {note.version} ({note.date})\n")
            lines.append(f"**{note.type} Update**\n")
            
            if note.important:
                lines.append("\n### ⚠️ 중요 정보\n")
                for item in note.important:
                    lines.append(f"- {item}\n")
            
            if note.new:
                lines.append("\n### ✨ 신규 기능\n")
                for item in note.new:
                    lines.append(f"- {item}\n")
            
            if note.improved:
                lines.append("\n### ⬆️ 개선사항\n")
                for item in note.improved:
                    lines.append(f"- {item}\n")
            
            if note.metrics:
                lines.append("\n### 📊 성능 변화\n")
                for item in note.metrics:
                    lines.append(f"- {item}\n")
            
            if note.fixed:
                lines.append("\n### 🔧 버그 수정\n")
                for item in note.fixed:
                    lines.append(f"- {item}\n")
            
            if note.breaking:
                lines.append("\n### ⚠️ Breaking Changes\n")
                for item in note.breaking:
                    lines.append(f"- {item}\n")
            
            if note.files:
                lines.append("\n### 📁 변경 파일\n")
                for item in note.files:
                    lines.append(f"- `{item}`\n")
            
            if note.next:
                lines.append("\n### 🔮 다음 버전 예고\n")
                for item in note.next:
                    lines.append(f"- {item}\n")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        messagebox.showinfo("내보내기", f"MD 저장 완료: {filename}")
    
    def new_version_dialog(self):
        """새 버전 다이얼로그"""
        dialog = tk.Toplevel(self.root)
        dialog.title("새 패치노트")
        dialog.geometry("300x150")
        dialog.configure(bg=self.colors['bg'])
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="새 버전:").pack(pady=10)
        entry = ttk.Entry(dialog, font=('맑은 고딕', 12), width=20)
        entry.pack(pady=5)
        entry.insert(0, "V9.0.0")
        entry.focus()
        
        def create():
            version = entry.get().strip()
            if version:
                if version in self.notes:
                    messagebox.showerror("오류", "이미 존재하는 버전입니다")
                    return
                self.notes[version] = PatchNote(version)
                self.refresh_version_list()
                # 새 버전 선택
                idx = list(sorted(self.notes.keys(), reverse=True)).index(version)
                self.version_listbox.selection_clear(0, tk.END)
                self.version_listbox.selection_set(idx)
                self.on_version_select(None)
                dialog.destroy()
        
        ttk.Button(dialog, text="생성", command=create).pack(pady=15)
    
    def delete_version(self):
        """버전 삭제"""
        if not self.current_version:
            return
        
        if messagebox.askyesno("삭제 확인", f"{self.current_version}을(를) 삭제하시겠습니까?"):
            del self.notes[self.current_version]
            self.current_version = None
            self.refresh_version_list()


def main():
    root = tk.Tk()
    app = PatchNoteApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()