#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Correction 루프 - 자기 교정 기능
"""

import logging
from .prompt_builder import SC_REVIEW_PROMPT

logger = logging.getLogger(__name__)


def run_self_correction(llm_provider, prompt: str, system_prompt: str, max_retries: int = 3) -> dict:
    """Self-Correction 루프 실행"""

    answer = ""
    review = ""
    is_valid = False
    attempt = 0

    for attempt in range(1, max_retries + 1):
        logger.info(f"[SC] 시도 {attempt}/{max_retries}")

        # 1. 답변 생성
        if attempt == 1:
            gen_prompt = prompt
        else:
            gen_prompt = f"{prompt}\n\n[이전 검토 피드백]\n{review}\n\n위 피드백을 반영하여 다시 답변해주세요."

        result = llm_provider.call(gen_prompt, system_prompt)
        if not result["success"]:
            return {"success": False, "error": result["error"], "retry_count": attempt}

        answer = result["content"]

        # 2. 답변 검토
        review_prompt = f"[원본 요청]\n{prompt}\n\n[생성된 답변]\n{answer}"
        review_result = llm_provider.call(review_prompt, SC_REVIEW_PROMPT, max_tokens=500)

        if not review_result["success"]:
            return {
                "success": True,
                "answer": answer,
                "retry_count": attempt,
                "is_valid": False,
                "review": "검토 실패"
            }

        review = review_result["content"]
        is_valid = review.strip().upper().startswith("PASS")

        logger.info(f"  결과: {'PASS' if is_valid else 'FAIL'}")

        if is_valid:
            break

    return {
        "success": True,
        "answer": answer,
        "retry_count": attempt,
        "is_valid": is_valid,
        "review": review
    }
