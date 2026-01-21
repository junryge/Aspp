# main.py
"""
코딩 에이전트 메인
CrewAI + Ralph Loop + SK Hynix API
"""

import json
import os
import re
from datetime import datetime
from typing import Optional

from llm import SKHynixLLM
from agents import BackendRalph, FrontendRalph, TestRalph

# 출력 폴더
OUTPUT_DIR = "output"


class CodingAgentManager:
    """코딩 에이전트 팀 매니저 (CrewAI 역할)"""
    
    def __init__(self, use_interpreter: bool = False):
        print("🚀 코딩 에이전트 초기화 중...")
        
        # output 폴더 생성
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # LLM 초기화
        self.llm = SKHynixLLM()
        
        # Open Interpreter (선택)
        self.interpreter = None
        if use_interpreter:
            try:
                import interpreter
                self.interpreter = interpreter
                self.interpreter.llm.model = "openai/custom"  # 커스텀 설정 필요
                print("✅ Open Interpreter 연결됨")
            except ImportError:
                print("⚠️ Open Interpreter 미설치. subprocess 모드로 실행.")
        
        # Ralph 에이전트 팀 구성
        self.backend_ralph = BackendRalph(self.llm, self.interpreter)
        self.frontend_ralph = FrontendRalph(self.llm, self.interpreter)
        self.test_ralph = TestRalph(self.llm, self.interpreter)
        
        print("✅ 에이전트 팀 준비 완료!")
        print(f"   - {self.backend_ralph.name} ({self.backend_ralph.role})")
        print(f"   - {self.frontend_ralph.name} ({self.frontend_ralph.role})")
        print(f"   - {self.test_ralph.name} ({self.test_ralph.role})")
    
    def run_from_prd(self, prd_path: str = "prd.json") -> dict:
        """
        PRD 파일 기반으로 전체 작업 실행
        Ralph 패턴: 모든 작업이 완료될 때까지 반복
        """
        if not os.path.exists(prd_path):
            print(f"❌ PRD 파일 없음: {prd_path}")
            return {"success": False, "error": "PRD 파일 없음"}
        
        with open(prd_path, 'r', encoding='utf-8') as f:
            prd = json.load(f)
        
        project_name = prd.get("projectName", "Unknown")
        tasks = prd.get("userStories", [])
        
        print(f"\n📋 프로젝트: {project_name}")
        print(f"📝 총 작업 수: {len(tasks)}")
        
        results = []
        
        for i, task in enumerate(tasks, 1):
            if task.get("passes", False):
                print(f"\n⏭️ [{i}/{len(tasks)}] {task['title']} - 이미 완료됨")
                continue
            
            print(f"\n{'='*60}")
            print(f"🎯 [{i}/{len(tasks)}] {task['title']}")
            print(f"{'='*60}")
            
            # 작업 유형에 따라 적절한 에이전트 선택
            result = self._dispatch_task(task)
            results.append(result)
            
            # 성공하면 PRD 업데이트 + 코드 저장
            if result["success"]:
                task["passes"] = True
                self._save_prd(prd_path, prd)
                self._log_progress(task, result)
                
                # 생성된 코드 파일로 저장
                saved_path = self._save_code(task, result)
                if saved_path:
                    print(f"💾 코드 저장됨: {saved_path}")
        
        # 최종 결과
        success_count = sum(1 for r in results if r["success"])
        print(f"\n{'='*60}")
        print(f"🏁 완료! 성공: {success_count}/{len(results)}")
        print(f"{'='*60}")
        
        return {
            "success": success_count == len(results),
            "total": len(results),
            "passed": success_count
        }
    
    def _dispatch_task(self, task: dict) -> dict:
        """작업 유형에 따라 적절한 에이전트에게 할당"""
        task_type = task.get("type", "backend").lower()
        language = task.get("language", "python")
        
        agent_task = {
            "description": task.get("title", "") + "\n" + task.get("description", ""),
            "requirements": "\n".join(task.get("acceptanceCriteria", [])),
            "language": language
        }
        
        if task_type == "frontend":
            return self.frontend_ralph.run(agent_task)
        elif task_type == "test":
            agent_task["target_code"] = task.get("targetCode", "")
            return self.test_ralph.run(agent_task)
        else:  # backend 또는 기본
            return self.backend_ralph.run(agent_task)
    
    def _save_prd(self, path: str, prd: dict):
        """PRD 파일 저장"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(prd, f, ensure_ascii=False, indent=2)
    
    def _log_progress(self, task: dict, result: dict):
        """progress.txt에 진행 상황 기록"""
        with open("progress.txt", 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*40}\n")
            f.write(f"[{datetime.now().isoformat()}]\n")
            f.write(f"작업: {task['title']}\n")
            f.write(f"결과: {'성공' if result['success'] else '실패'}\n")
            f.write(f"시도 횟수: {result.get('attempts', 'N/A')}\n")
            if result.get("errors"):
                f.write(f"에러 히스토리:\n")
                for err in result["errors"]:
                    f.write(f"  - {err[:100]}...\n")
    
    def _save_code(self, task: dict, result: dict) -> str:
        """생성된 코드를 파일로 저장"""
        # 코드 가져오기
        code = result.get("code") or result.get("test_code")
        if not code:
            return ""
        
        # 파일명 생성
        task_id = task.get("id", "unknown")
        title = task.get("title", "code")
        # 파일명에 사용할 수 없는 문자 제거
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:30]
        
        # 언어별 확장자
        language = task.get("language", "python")
        extensions = {
            "python": ".py",
            "java": ".java",
            "csharp": ".cs",
            "javascript": ".js",
            "typescript": ".ts",
            "html": ".html",
            "css": ".css"
        }
        ext = extensions.get(language, ".txt")
        
        # 파일 경로
        filename = f"{task_id}_{safe_title}{ext}"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
        
        return filepath
    
    def run_single_task(self, 
                        description: str, 
                        task_type: str = "backend",
                        language: str = "python") -> dict:
        """단일 작업 실행"""
        task = {
            "id": f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": description[:50],
            "description": description,
            "requirements": "",
            "language": language,
            "type": task_type
        }
        
        if task_type == "frontend":
            result = self.frontend_ralph.run(task)
        elif task_type == "test":
            result = self.test_ralph.run(task)
        else:
            result = self.backend_ralph.run(task)
        
        # 성공하면 파일 저장
        if result["success"]:
            saved_path = self._save_code(task, result)
            result["saved_path"] = saved_path
        
        return result


def main():
    """메인 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="코딩 에이전트")
    parser.add_argument("--prd", default="prd.json", help="PRD 파일 경로")
    parser.add_argument("--task", help="단일 작업 설명")
    parser.add_argument("--type", default="backend", choices=["backend", "frontend", "test"])
    parser.add_argument("--lang", default="python", help="프로그래밍 언어")
    parser.add_argument("--interpreter", action="store_true", help="Open Interpreter 사용")
    
    args = parser.parse_args()
    
    # 매니저 초기화
    manager = CodingAgentManager(use_interpreter=args.interpreter)
    
    if args.task:
        # 단일 작업 모드
        print(f"\n🎯 단일 작업 실행: {args.task}")
        result = manager.run_single_task(args.task, args.type, args.lang)
        
        if result["success"]:
            print("\n✅ 생성된 코드:")
            print("-" * 40)
            print(result.get("code", ""))
            print("-" * 40)
            if result.get("saved_path"):
                print(f"💾 저장됨: {result['saved_path']}")
        else:
            print(f"\n❌ 실패: {result.get('message')}")
    else:
        # PRD 기반 전체 실행
        manager.run_from_prd(args.prd)


if __name__ == "__main__":
    main()