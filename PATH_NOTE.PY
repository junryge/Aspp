"""
================================================================================
AMHS 예측 모델 패치노트 관리 프로그램
================================================================================
기능:
  - 패치노트 작성
  - JSON 파일로 저장/로드
  - 탭으로 버전별 분리
  - 콘솔/파일 출력
================================================================================
"""

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
        self.metrics: Dict[str, Dict] = {}
        self.files: List[str] = []
        self.next: List[str] = []
        self.notes: List[str] = []
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
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
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PatchNote':
        """딕셔너리에서 생성"""
        note = cls(data["version"], data["date"], data.get("type", "Major"))
        note.new = data.get("new", [])
        note.improved = data.get("improved", [])
        note.fixed = data.get("fixed", [])
        note.breaking = data.get("breaking", [])
        note.metrics = data.get("metrics", {})
        note.files = data.get("files", [])
        note.next = data.get("next", [])
        note.notes = data.get("notes", [])
        return note


class PatchNoteManager:
    """패치노트 관리자"""
    
    def __init__(self, project_name: str = "AMHS 예측 모델"):
        self.project_name = project_name
        self.notes: Dict[str, PatchNote] = {}  # version -> PatchNote
        self.save_file = "patch_notes.json"
    
    # ========== 추가/삭제 ==========
    
    def add(self, note: PatchNote):
        """패치노트 추가"""
        self.notes[note.version] = note
        print(f"✅ {note.version} 추가됨")
        return self
    
    def remove(self, version: str):
        """패치노트 삭제"""
        if version in self.notes:
            del self.notes[version]
            print(f"🗑️ {version} 삭제됨")
        else:
            print(f"❌ {version} 없음")
        return self
    
    def get(self, version: str) -> Optional[PatchNote]:
        """버전으로 조회"""
        return self.notes.get(version)
    
    def list_versions(self) -> List[str]:
        """모든 버전 목록"""
        return sorted(self.notes.keys(), reverse=True)
    
    # ========== 저장/로드 ==========
    
    def save(self, filename: str = None):
        """JSON 파일로 저장"""
        filename = filename or self.save_file
        data = {
            "project": self.project_name,
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": {v: n.to_dict() for v, n in self.notes.items()}
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 저장됨: {filename}")
        return self
    
    def load(self, filename: str = None):
        """JSON 파일에서 로드"""
        filename = filename or self.save_file
        if not os.path.exists(filename):
            print(f"❌ 파일 없음: {filename}")
            return self
        
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.project_name = data.get("project", self.project_name)
        self.notes = {}
        for version, note_data in data.get("notes", {}).items():
            self.notes[version] = PatchNote.from_dict(note_data)
        
        print(f"📂 로드됨: {filename} ({len(self.notes)}개 버전)")
        return self
    
    # ========== 출력 ==========
    
    def print_one(self, version: str):
        """단일 버전 출력"""
        note = self.get(version)
        if not note:
            print(f"❌ {version} 없음")
            return
        
        self._print_note(note)
    
    def print_all(self):
        """전체 버전 출력"""
        for version in self.list_versions():
            self._print_note(self.notes[version])
            print()
    
    def print_tabs(self):
        """탭 형식으로 버전 목록 출력"""
        versions = self.list_versions()
        if not versions:
            print("패치노트 없음")
            return
        
        print("\n" + "=" * 60)
        print(f"  {self.project_name} 패치노트")
        print("=" * 60)
        
        # 탭 헤더
        print("\n┌" + "─" * 58 + "┐")
        tab_line = "│ "
        for i, v in enumerate(versions):
            tab_line += f"[{i+1}] {v}  "
        print(tab_line.ljust(59) + "│")
        print("└" + "─" * 58 + "┘")
        
        # 버전 선택
        while True:
            try:
                choice = input("\n버전 번호 선택 (0=종료): ").strip()
                if choice == "0" or choice == "":
                    break
                idx = int(choice) - 1
                if 0 <= idx < len(versions):
                    self._print_note(self.notes[versions[idx]])
                else:
                    print("잘못된 번호")
            except ValueError:
                print("숫자를 입력하세요")
            except KeyboardInterrupt:
                break
    
    def _print_note(self, note: PatchNote):
        """패치노트 포맷 출력"""
        w = 58  # 너비
        
        print()
        print("┌" + "─" * w + "┐")
        print(f"│  {self.project_name} {note.version}".ljust(w) + " │")
        print(f"│  {note.date} | {note.type} Update".ljust(w) + " │")
        print("├" + "─" * w + "┤")
        
        # NEW
        if note.new:
            print(f"│  [NEW] 신규 기능".ljust(w) + " │")
            for item in note.new:
                line = f"│    • {item}"
                if len(line) > w:
                    line = line[:w-3] + "..."
                print(line.ljust(w) + " │")
        
        # IMPROVED
        if note.improved:
            print(f"│  [IMPROVED] 개선사항".ljust(w) + " │")
            for item in note.improved:
                line = f"│    • {item}"
                if len(line) > w:
                    line = line[:w-3] + "..."
                print(line.ljust(w) + " │")
        
        # METRICS
        if note.metrics:
            print(f"│  [METRICS] 성능 변화".ljust(w) + " │")
            for name, data in note.metrics.items():
                before = data.get('before', '-')
                after = data.get('after', '-')
                change = data.get('change', '')
                line = f"│    • {name}: {before} → {after} ({change})"
                if len(line) > w:
                    line = line[:w-3] + "..."
                print(line.ljust(w) + " │")
        
        # FIXED
        print(f"│  [FIXED] 버그 수정".ljust(w) + " │")
        if note.fixed:
            for item in note.fixed:
                line = f"│    • {item}"
                if len(line) > w:
                    line = line[:w-3] + "..."
                print(line.ljust(w) + " │")
        else:
            print(f"│    • 없음".ljust(w) + " │")
        
        # BREAKING
        if note.breaking:
            print(f"│  [BREAKING] 호환성".ljust(w) + " │")
            for item in note.breaking:
                line = f"│    • {item}"
                if len(line) > w:
                    line = line[:w-3] + "..."
                print(line.ljust(w) + " │")
        
        # FILES
        if note.files:
            print(f"│  [FILES] 변경 파일".ljust(w) + " │")
            for item in note.files:
                line = f"│    • {item}"
                if len(line) > w:
                    line = line[:w-3] + "..."
                print(line.ljust(w) + " │")
        
        # NEXT
        if note.next:
            print(f"│  [NEXT] 다음 버전 예고".ljust(w) + " │")
            for item in note.next:
                line = f"│    • {item}"
                if len(line) > w:
                    line = line[:w-3] + "..."
                print(line.ljust(w) + " │")
        
        print("└" + "─" * w + "┘")
    
    # ========== 내보내기 ==========
    
    def export_md(self, filename: str = "PATCH_NOTES.md"):
        """마크다운으로 내보내기"""
        lines = []
        lines.append(f"# {self.project_name} 패치노트\n")
        lines.append(f"*최종 업데이트: {datetime.now().strftime('%Y-%m-%d')}*\n")
        
        for version in self.list_versions():
            note = self.notes[version]
            lines.append(f"\n---\n")
            lines.append(f"## {note.version} ({note.date})\n")
            lines.append(f"**{note.type} Update**\n")
            
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
                lines.append("| 지표 | Before | After | 변화 |\n")
                lines.append("|------|--------|-------|------|\n")
                for name, data in note.metrics.items():
                    before = data.get('before', '-')
                    after = data.get('after', '-')
                    change = data.get('change', '')
                    lines.append(f"| {name} | {before} | {after} | {change} |\n")
            
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
        
        print(f"📄 MD 저장됨: {filename}")
        return self


# ========== 샘플 데이터 ==========

def create_sample_data() -> PatchNoteManager:
    """V6~V8 샘플 패치노트 생성"""
    
    manager = PatchNoteManager("AMHS 예측 모델")
    
    # V8.0.0
    v8 = PatchNote("V8.0.0", "2025-12-01", "Major")
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
    v8.metrics = {
        "MAE": {"before": 12.5, "after": 9.8, "change": "↓21%"},
        "R²": {"before": 0.62, "after": 0.68, "change": "↑9.7%"},
        "범주형정확도": {"before": "-", "after": "72%", "change": "신규"},
    }
    v8.fixed = []
    v8.breaking = [
        "V7 모델 파일(.pkl)과 호환되지 않음",
        "V8 전용 평가 코드 사용 필요",
    ]
    v8.files = [
        "V8_수치형_학습코드.py",
        "V8_수치형_평가코드.py",
        "V8_범주형_학습코드.py",
        "V8_범주형_평가코드.py",
        "xgboost_수치형_V8.pkl",
        "xgboost_범주형_V8.pkl",
    ]
    v8.next = [
        "MLflow Tracking 연동",
        "하이퍼파라미터 자동 탐색 (Optuna)",
    ]
    manager.add(v8)
    
    # V7.0.0
    v7 = PatchNote("V7.0.0", "2025-11-15", "Major")
    v7.new = [
        "M16HUB.QUE.ALL.CURRENTQCNT 컬럼 추가",
        "M16HUB.QUE.TIME.AVGTOTALTIME1MIN 컬럼 추가",
        "QUE 기반 병목 감지 Feature 추가",
    ]
    v7.improved = [
        "입력 컬럼: 17개 → 19개",
        "Feature 수: 165개 → 180개",
        "병목 감지 로직 강화",
    ]
    v7.metrics = {
        "MAE": {"before": 14.2, "after": 12.5, "change": "↓12%"},
        "R²": {"before": 0.58, "after": 0.62, "change": "↑6.9%"},
    }
    v7.fixed = [
        "HUBROOMTOTAL 결측치 처리 오류 수정",
    ]
    v7.breaking = [
        "V6 모델 파일과 호환되지 않음",
    ]
    v7.files = [
        "V7_수치형_학습코드.py",
        "V7_수치형_평가코드.py",
    ]
    v7.next = [
        "범주형 분류 모델 추가",
        "QUE 컬럼 확장",
    ]
    manager.add(v7)
    
    # V6.0.0
    v6 = PatchNote("V6.0.0", "2025-10-20", "Major")
    v6.new = [
        "멀티아웃풋 구조 도입 (MIN/MAX/AVG)",
        "10분 구간 통계 예측",
        "방향 정확도 지표 추가",
    ]
    v6.improved = [
        "Feature 수: 120개 → 165개",
        "시퀀스 길이: 20분 → 30분",
        "예측 안정성 향상",
    ]
    v6.metrics = {
        "MAE": {"before": 18.5, "after": 14.2, "change": "↓23%"},
        "R²": {"before": 0.45, "after": 0.58, "change": "↑29%"},
    }
    v6.fixed = [
        "큰 변화 예측 누락 문제 해결",
    ]
    v6.breaking = [
        "단일 출력 → 멀티 출력 구조 변경",
    ]
    v6.files = [
        "V6_학습코드.py",
        "V6_평가코드.py",
    ]
    v6.next = [
        "QUE 컬럼 추가",
        "실시간 예측 모듈",
    ]
    manager.add(v6)
    
    return manager


# ========== 메인 ==========

def main():
    """메인 메뉴"""
    
    manager = PatchNoteManager("AMHS 예측 모델")
    
    # 기존 파일 있으면 로드
    if os.path.exists("patch_notes.json"):
        manager.load()
    else:
        # 샘플 데이터 생성
        manager = create_sample_data()
        manager.save()
    
    while True:
        print("\n" + "=" * 40)
        print("  패치노트 관리 시스템")
        print("=" * 40)
        print("  [1] 버전별 탭 보기")
        print("  [2] 전체 출력")
        print("  [3] 특정 버전 보기")
        print("  [4] 새 패치노트 추가")
        print("  [5] 패치노트 삭제")
        print("  [6] 저장")
        print("  [7] 로드")
        print("  [8] MD 내보내기")
        print("  [0] 종료")
        print("=" * 40)
        
        choice = input("선택: ").strip()
        
        if choice == "1":
            manager.print_tabs()
        
        elif choice == "2":
            manager.print_all()
        
        elif choice == "3":
            version = input("버전 입력 (예: V8.0.0): ").strip()
            manager.print_one(version)
        
        elif choice == "4":
            print("\n[새 패치노트 추가]")
            version = input("버전 (예: V9.0.0): ").strip()
            date = input("날짜 (YYYY-MM-DD, 엔터=오늘): ").strip()
            note_type = input("유형 (Major/Minor/Patch): ").strip() or "Major"
            
            note = PatchNote(version, date if date else None, note_type)
            
            print("\n신규 기능 (빈 줄로 종료):")
            while True:
                item = input("  • ").strip()
                if not item:
                    break
                note.new.append(item)
            
            print("\n개선사항 (빈 줄로 종료):")
            while True:
                item = input("  • ").strip()
                if not item:
                    break
                note.improved.append(item)
            
            print("\n버그 수정 (빈 줄로 종료):")
            while True:
                item = input("  • ").strip()
                if not item:
                    break
                note.fixed.append(item)
            
            print("\nBreaking Changes (빈 줄로 종료):")
            while True:
                item = input("  • ").strip()
                if not item:
                    break
                note.breaking.append(item)
            
            manager.add(note)
            manager.save()
        
        elif choice == "5":
            version = input("삭제할 버전: ").strip()
            confirm = input(f"{version} 삭제? (y/n): ").strip().lower()
            if confirm == 'y':
                manager.remove(version)
                manager.save()
        
        elif choice == "6":
            manager.save()
        
        elif choice == "7":
            filename = input("파일명 (엔터=patch_notes.json): ").strip()
            manager.load(filename if filename else None)
        
        elif choice == "8":
            filename = input("파일명 (엔터=PATCH_NOTES.md): ").strip()
            manager.export_md(filename if filename else "PATCH_NOTES.md")
        
        elif choice == "0":
            print("종료")
            break
        
        else:
            print("잘못된 선택")


if __name__ == "__main__":
    main()