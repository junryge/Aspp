def analyze_amhs_log(df: pd.DataFrame, question: str = "", filename: str = "") -> dict:
    """AMHS 로그 분석 공통 함수 - 설비별 프롬프트 및 전처리 자동 적용"""
    # 기본 분석
    analysis = analyze_csv_basic(df)
    
    # 설비 유형 감지 (내용물 기반)
    equipment_type, equip_details = detect_equipment_type(df)
    logger.info(f"Detected equipment: {equipment_type}, details: {equip_details}")
    
    # ========================================
    # AMHS 로그 여부 검증 (내용물 기반)
    # ========================================
    if equipment_type == "UNKNOWN":
        logger.warning(f"Not an AMHS log file: {filename}")
        return {
            "success": False,
            "error": "관계없는 데이터입니다. AMHS 로그 파일을 새로 찾아서 업로드 해주세요.",
            "filename": filename,
            "equipment_type": "UNKNOWN",
            "basic_info": {
                "row_count": analysis["row_count"],
                "columns": analysis["columns"][:10],
                "message_types": dict(list(analysis["message_types"].items())[:5]) if analysis["message_types"] else {}
            },
            "analysis": "이 파일은 AMHS 로그 형식이 아닙니다.\n\n"
                       "**필요한 조건:**\n"
                       "- MESSAGENAME 컬럼이 있어야 합니다\n"
                       "- RAIL-*, INTERRAIL-*, STORAGE-*, VM-* 메시지가 포함되어야 합니다\n\n"
                       "OHT, Conveyor, Lifter, FABJOB 관련 CSV 파일을 업로드 해주세요."
        }
    
    # ========================================
    # 설비별 전처리 실행
    # ========================================
    preprocess_text = ""
    preprocess_result = None
    
    if equipment_type == "FABJOB" and FABJOB_PREPROCESSOR_AVAILABLE:
        try:
            logger.info("FABJOB detected - running preprocessor...")
            preprocess_result = analyze_fabjob(df)
            preprocess_text = preprocess_result.get('preprocessed_text', '')
            logger.info(f"FABJOB preprocessing complete. Text length: {len(preprocess_text)}")
        except Exception as e:
            logger.error(f"FABJOB preprocessing failed: {e}")
    
    elif equipment_type == "OHT" and OHT_PREPROCESSOR_AVAILABLE:
        try:
            logger.info("OHT detected - running preprocessor...")
            preprocess_result = analyze_oht(df)
            preprocess_text = preprocess_result.get('preprocessed_text', '')
            logger.info(f"OHT preprocessing complete. Text length: {len(preprocess_text)}")
        except Exception as e:
            logger.error(f"OHT preprocessing failed: {e}")
    
    elif equipment_type == "CONVEYOR" and CONVEYOR_PREPROCESSOR_AVAILABLE:
        try:
            logger.info("CONVEYOR detected - running preprocessor...")
            preprocess_result = analyze_conveyor(df)
            preprocess_text = preprocess_result.get('preprocessed_text', '')
            logger.info(f"CONVEYOR preprocessing complete. Text length: {len(preprocess_text)}")
        except Exception as e:
            logger.error(f"CONVEYOR preprocessing failed: {e}")
    
    elif equipment_type == "LIFTER" and LIFTER_PREPROCESSOR_AVAILABLE:
        try:
            logger.info("LIFTER detected - running preprocessor...")
            preprocess_result = analyze_lifter(df)
            preprocess_text = preprocess_result.get('preprocessed_text', '')
            logger.info(f"LIFTER preprocessing complete. Text length: {len(preprocess_text)}")
        except Exception as e:
            logger.error(f"LIFTER preprocessing failed: {e}")
    
    # ========================================
    # 프롬프트 생성
    # ========================================
    prompt = create_analysis_prompt(df, analysis, question, preprocess_text)
    
    # 설비별 프롬프트 로드
    equip_prompt_path = os.path.join(EQUIP_PROMPT_DIR, equipment_type)
    if equipment_type != "UNKNOWN" and os.path.exists(equip_prompt_path):
        equip_system, equip_fewshot = get_equipment_prompts(equipment_type)
        system_prompt = equip_system + "\n\n" + equip_fewshot
        logger.info(f"Using equipment-specific prompt for {equipment_type}")
    else:
        system_prompt = get_default_prompt()
        logger.info("Using default prompt")
    
    # LLM 호출
    if LLM_MODE == "api":
        llm_response = call_api_llm(prompt, system_prompt)
    else:
        llm_response = call_local_llm(prompt, system_prompt)
    
    # 결과 구성
    result = {
        "success": True,
        "equipment_type": equipment_type,
        "equipment_details": equip_details,
        "basic_info": {
            "row_count": analysis["row_count"],
            "time_range": analysis["time_range"],
            "message_types": dict(list(analysis["message_types"].items())[:5]),
            "levels": analysis["levels"],
            "machines": analysis["machines"][:5] if analysis["machines"] else [],
            "carriers": analysis["carriers"][:5] if analysis["carriers"] else []
        },
        "analysis": llm_response
    }
    
    # 전처리 결과 상세 정보 추가
    if preprocess_result:
        result["preprocess_details"] = {
            "carrier_id": preprocess_result.get('carrier_id'),
            "total_duration_sec": preprocess_result.get('total_duration_sec', 0),
            "final_status": preprocess_result.get('final_status'),
            "delays": preprocess_result.get('delays', []),
        }
        
        # 설비별 추가 정보
        if equipment_type == "FABJOB":
            result["preprocess_details"]["lot_id"] = preprocess_result.get('lot_id')
            result["preprocess_details"]["hcack_errors"] = len([
                h for h in preprocess_result.get('hcack_events', []) 
                if h.get('hcack') == '2'
            ])
        elif equipment_type == "OHT":
            result["preprocess_details"]["vehicle_id"] = preprocess_result.get('vehicle_id')
            result["preprocess_details"]["source_port"] = preprocess_result.get('source_port')
            result["preprocess_details"]["dest_port"] = preprocess_result.get('dest_port')
        elif equipment_type == "CONVEYOR":
            result["preprocess_details"]["machine_name"] = preprocess_result.get('machine_name')
            result["preprocess_details"]["source_zone"] = preprocess_result.get('source_zone')
            result["preprocess_details"]["dest_zone"] = preprocess_result.get('dest_zone')
        elif equipment_type == "LIFTER":
            result["preprocess_details"]["machine_name"] = preprocess_result.get('machine_name')
            result["preprocess_details"]["source_floor"] = preprocess_result.get('source_floor')
            result["preprocess_details"]["dest_floor"] = preprocess_result.get('dest_floor')
    
    return result