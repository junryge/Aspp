# 10. REST API 명세

본 문서는 `mcslog_web_src` Spring MVC 4.2 기반 웹 애플리케이션이 외부에 노출하는 모든 HTTP 엔드포인트의 상세 명세입니다. 18개의 `@Controller` 클래스(`alarm`, `mat`, `res`, `secs`, `test`, `tot`, `tran` 패키지)와 `servlet-context.xml`의 `<view-controller>` 항목을 분석한 결과입니다.

---

## 개요

### 엔드포인트 수 집계

| 분류 | 엔드포인트 수 |
|------|---------------|
| 전체 `@RequestMapping` 메서드 | **63** |
| `<view-controller>` 매핑 | **2** |
| **총합** | **65** |

### 모듈별 분포

| 모듈 | 컨트롤러 수 | 엔드포인트 수 | 비고 |
|------|------------|---------------|------|
| `alarm` | 1 | 2 | 알람 리포트 조회 |
| `mat` | 1 | 2 | 캐리어 위치 이력 |
| `res` | 6 | 12 | Crane / Machine / Port / Shelf / StorageFull / Vehicle (각 화면+ajax 2개) |
| `secs` | 2 | 9 | SECS / EI 로그, 팝업, 공통 필터(tot/filter/ajax/*) 포함 |
| `test` | 1 | 3 | i18n, monitoring, tmp |
| `tot` | 2 | 25 | Total + TotalNew, 필터 ajax, 대시보드, 팝업 등 |
| `tran` | 5 | 10 | 반송 이력/CMD/JOB/실패, 사유 팝업 |
| `view-controller` | (xml) | 2 | `tot/main`, `tot/index` (정적 매핑) |

### Base URL 규약

| 경로 패턴 | 의미 |
|----------|------|
| `<module>/<page>LogList` | 조회 화면 진입(JSP 뷰) |
| `<module>/ajax/<get*>` | 화면 내 그리드 데이터(JSON) 조회 |
| `<module>/pop/<*Pop>` | 팝업창용 JSP 뷰 |
| `tot/filter/ajax/*` | 다른 모듈에서도 사용하는 공통 필터(area/bay/secs/process 등) 조회 |
| `tot/dashboard/*` | 대시보드 JSP 진입 |
| `tot/{query}` | 동적 path variable 매핑 (`tot/main`으로 라우팅) |

모든 매핑은 컨텍스트 루트(`/mcs_as` 또는 `/`) 직하에 배치되며, 명세 내 URL은 컨텍스트 경로를 생략합니다. `@RequestMapping(value)`에 슬래시 접두어가 있거나 없거나 Spring 내부적으로 동일하게 처리되지만, 원문 표기를 유지했습니다(예: `/mat/ajax/getCarrierLocLogList.do`).

### 응답 형식 분포

| 응답 타입 | 개수 | 설명 |
|----------|------|------|
| JSP 뷰 (`InternalResourceViewResolver`) | 24 | `/WEB-INF/views/<viewName>.jsp` 로 forward |
| `jsonView` (MappingJackson2JsonView 빈으로 추정) | 33 | `ModelAndView`에 담긴 model 키를 JSON 직렬화 |
| `@ResponseBody` (Jackson 직접 직렬화) | 8 | 메서드 반환값을 JSON으로 직접 직렬화 |
| String 반환 (JSP 뷰명) | 3 | `TestController` 의 `i18n`, `monitoring`, `tmp` |
| view-controller (Tiles) | 2 | `main`, `index` (Tiles 정의로 추정) |

---

## 엔드포인트 마스터 목록

| HTTP | URL | 컨트롤러.메서드 | 응답 | 설명 |
|------|-----|----------------|------|------|
| GET, POST | `alarm/alarmReportLogList` | `AlarmReportController.alarmReportLogList` | JSP `alarm/alarmReportLogList` | 알람 리포트 조회 화면 진입 |
| GET, POST | `alarm/ajax/getAlarmReportLogList` | `AlarmReportController.getAlarmReportLogList` | jsonView | 알람 리포트 그리드 데이터(ajax) |
| GET, POST | `mat/carrierLocLogList` | `MaterialController.carrierLocLogList` | JSP `mat/carrierLocLogList` | 캐리어 위치 이력 화면 진입 |
| GET, POST | `/mat/ajax/getCarrierLocLogList.do` | `MaterialController.getList` | jsonView | 캐리어 위치 이력 그리드(ajax) |
| GET, POST | `res/craneLogList` | `ResCraneHistoryController.carrierLocLogList` | JSP `res/craneLogList` | Crane 이력 화면 진입 |
| GET, POST | `res/ajax/getCraneLogList` | `ResCraneHistoryController.getCraneLogList` | jsonView | Crane 이력 그리드(ajax) |
| GET, POST | `res/machineLogList` | `ResMachineHistoryController.carrierLocLogList` | JSP `res/machineLogList` | Machine 이력 화면 진입 |
| GET, POST | `res/ajax/getMachineLogList` | `ResMachineHistoryController.getMachineLogList` | jsonView | Machine 이력 그리드(ajax) |
| GET, POST | `res/portLogList` | `ResPortHistoryController.portLogList` | JSP `res/portLogList` | Port 이력 화면 진입 |
| GET, POST | `res/ajax/getPortLogList` | `ResPortHistoryController.getPortLogList` | jsonView | Port 이력 그리드(ajax) |
| GET, POST | `res/shelfLogList` | `ResShelfHistoryController.portLogList` | JSP `res/shelfLogList` | Shelf 이력 화면 진입 |
| GET, POST | `res/ajax/getShelfLogList` | `ResShelfHistoryController.getShelfLogList` | jsonView | Shelf 이력 그리드(ajax) |
| GET, POST | `res/storageLogList` | `ResStorageFullHistoryController.carrierLocLogList` | JSP `res/storageLogList` | StorageFull 이력 화면 진입 |
| GET, POST | `res/ajax/getStorageLogList` | `ResStorageFullHistoryController.getStorageLogList` | jsonView | StorageFull 이력 그리드(ajax) |
| GET, POST | `res/vehicleLogList` | `ResVehicleHistoryController.portLogList` | JSP `res/vehicleLogList` | Vehicle 이력 화면 진입 |
| GET, POST | `res/ajax/getVehicleLogList` | `ResVehicleHistoryController.getVehicleLogList` | jsonView | Vehicle 이력 그리드(ajax) |
| GET, POST | `ei/eiLogList` | `EiLogController.eiLocLogList` | JSP `ei/eiLogList` | EI 로그 조회 화면 진입 |
| GET, POST | `/ei/ajax/getEiLogList.do` | `EiLogController.getList` | jsonView | EI 로그 그리드(ajax) |
| GET, POST | `tot/filter/ajax/getProcessList` | `EiLogController.getSecsList` | JSON (`@ResponseBody`) | Process 목록(ajax) |
| GET, POST | `ei/pop/textDetailPop` | `EiLogController.filterPop` | JSP `ei/pop/textDetailPop` | EI 텍스트 상세 팝업 |
| GET, POST | `ei/pop/textAreaPop` | `EiLogController.textFilterPop` | JSP `ei/pop/textAreaPop` | EI 텍스트영역 팝업 |
| GET, POST | `tot/filter/ajax/getSelectProcessList` | `EiLogController.getSecsFabList` | jsonView | EI fab별 선택 Process 목록 |
| GET, POST | `ei/ajax/getEiQueryStop` | `EiLogController.getEiQueryStop` | `void` (`@ResponseBody`) | EI 조회 중지 신호 |
| GET, POST | `secs/secsLogList` | `SecsLogController.secsLocLogList` | JSP `secs/secsLogList` | SECS 로그 화면 진입 |
| GET, POST | `/secs/ajax/getsecsLogList.do` | `SecsLogController.getList` | jsonView | SECS 로그 그리드(ajax) |
| GET, POST | `tot/filter/ajax/getSecsList` | `SecsLogController.getSecsList` | JSON (`@ResponseBody`) | SECS 머신 목록(ajax) |
| GET, POST | `tot/filter/ajax/getSecsFabList` | `SecsLogController.getSecsFabList` | jsonView | SECS Fab별 머신 목록(ajax) |
| GET, POST | `ei/ajax/getSecsQueryStop` | `SecsLogController.getSecsQueryStop` | `void` (`@ResponseBody`) | SECS 조회 중지 신호 |
| GET | `/i18n.do` | `TestController.i18n` | JSP `i18n` | i18n 메시지 테스트 |
| GET | `/monitoring.do` | `TestController.monitoring` | JSP `monitoring` | 모니터링 페이지 |
| GET | `/tmp.do` | `TestController.tmp` | JSP `tmp` | tmp 페이지 |
| GET, POST | `tot/totalLogList` | `TotalController.totalLogList` | JSP `tot/totalLogList` | Total 로그 화면 진입 |
| GET, POST | `tot/ajax/getTotalLogList` | `TotalController.getTotalLogList` | jsonView | Total 로그 그리드(ajax) |
| GET, POST | `tot/ajax/getTotalLogListStop` | `TotalController.getTotalLogListStop` | `void` (`@ResponseBody`) | Total 조회 중지 신호 |
| GET, POST | `tot/pop/machineNamePop` | `TotalController.machineNamePop` | JSP `tot/pop/machineNamePop` | Machine 이름 선택 팝업 |
| GET, POST | `tot/ajax/getMachineList` | `TotalController.getMachineList` | jsonView | Machine 이름 목록(ajax) |
| GET, POST | `tot/ajax/getMachineListMachineTypeNotNull` | `TotalController.getMachineListMachineTypeNotNull` | jsonView | MachineType이 NULL이 아닌 Machine 목록 |
| GET, POST | `tot/ajax/getBayFromArea` | `TotalController.getBayFromArea` | jsonView | Area 기준 Bay 목록 |
| GET, POST | `tot/ajax/getAreaFromFab` | `TotalController.getAreaFromFab` | jsonView | Fab 기준 Area 목록 |
| GET, POST | `tot/ajax/getMachineTypeFromFab` | `TotalController.getMachineTypeFromFab` | jsonView | Fab 기준 MachineType 목록 |
| GET, POST | `tot/ajax/getFabFromFabSite` | `TotalController.getFabFromFabSite` | jsonView | FabSite 기준 Fab 목록 |
| GET, POST | `tot/main` | `TotalController.main` | JSP `tot/main` | 메인 페이지(컨트롤러 측) |
| GET | `tot/{query}` | `TotalController.getRequest` | JSP `tot/main` | 동적 path variable 라우팅(`tot/main`로 fallthrough) |
| GET, POST | `tot/filter/ajax/getAreaList` | `TotalController.getAreaList` | JSON (`@ResponseBody`) | Area 이름 목록 |
| GET, POST | `tot/filter/ajax/getBayList` | `TotalController.getBayList` | JSON (`@ResponseBody`) | Bay 이름 목록 |
| GET, POST | `tot/filter/ajax/getMachineNameList` | `TotalController.getMachineNameList` | JSON (`@ResponseBody`) | Machine 이름 목록 |
| GET, POST | `tot/filter/ajax/getCommMsgNameList` | `TotalController.getCommMsgNameList` | JSON (`@ResponseBody`) | CommMsg 이름 목록 |
| GET, POST | `tot/filter/ajax/getMessageNameList` | `TotalController.getMessageNameList` | JSON (`@ResponseBody`) | Message 이름 목록 |
| GET, POST | `tot/filter/ajax/getOperationNameList` | `TotalController.getOperationNameList` | JSON (`@ResponseBody`) | Operation 이름 목록 |
| GET, POST | `tot/pop/filterPop` | `TotalController.filterPop` | JSP `tot/pop/filterPop` | 필터 팝업 |
| GET, POST | `common/pop/settingPop` | `TotalController.settingPop` | JSP `common/pop/settingPop` | 환경 설정 팝업 |
| GET, POST | `tot/dashboard/elapsedAnalysis` | `TotalController.elapsed` | JSP `tot/elapsedAnalysis` | 경과시간 분석 대시보드 |
| GET, POST | `tot/dashboard/compressAnalysis` | `TotalController.elapsed2` | JSP `tot/compressAnalysis` | 압축 분석 대시보드 |
| GET, POST | `tot/dashboard/monitor` | `TotalController.monitor` | JSP `tot/monitor` | 모니터 대시보드 |
| GET, POST | `tot/dashboard/elapsed3` | `TotalController.elapsed3` | JSP `tot/dashboard3` | 대시보드3 |
| GET, POST | `totNew/totalNewLogList` | `TotalNewController.totalNewLogList` | JSP `tot/totalNewLogList` | 신규 로그 조회 화면 진입 |
| GET, POST | `totNew/ajax/totalNewLogList` | `TotalNewController.totalNewLogListAjax` | jsonView | 신규 로그 그리드(ajax) |
| GET, POST | `totNew/pop/machineNamePop` | `TotalNewController.machineNamePop` | JSP `tot/pop/machineNamePop` | 신규 화면용 Machine 팝업 |
| GET, POST | `totNew/ajax/getCarrierElapsed` | `TotalNewController.getCarrierElapsed` | jsonView | Carrier 경과시간 상세 |
| GET, POST | `tran/returnCmdFailLogList` | `TranCmdFailController.returnCmdFailLogList` | JSP `tran/returnCmdFailLogList` | 반송 CMD 실패 화면 진입 |
| GET, POST | `tran/ajax/getReturnCmdFailLogList` | `TranCmdFailController.getReturnCmdFailLogList` | jsonView | 반송 CMD 실패 그리드(ajax) |
| GET, POST | `tran/pop/reasonPop` | `TranCmdFailController.machineNamePop` | JSP `tran/pop/reasonPop` | 반송 사유 팝업 |
| GET, POST | `tran/returnCmdLogList` | `TranCmdHistoryController.returnCmdLogList` | JSP `tran/returnCmdLogList` | 반송 CMD 이력 화면 진입 |
| GET, POST | `tran/ajax/getReturnCmdLogList` | `TranCmdHistoryController.getReturnCmdLogList` | jsonView | 반송 CMD 이력 그리드(ajax) |
| GET, POST | `tran/returnLogList` | `TranController.returnLogList` | JSP `tran/returnLogList` | 반송 이력 화면 진입 |
| GET, POST | `tran/ajax/getReturnLogList` | `TranController.getReturnLogList` | jsonView | 반송 이력 그리드(ajax) |
| GET, POST | `tran/ajax/getTranJobHistoryDetail` | `TranController.getTranJobHistoryDetail` | jsonView | 반송 JOB 이력 상세 |
| GET, POST | `tran/ajax/getReasonList` | `TranController.getReasonList` | jsonView | 반송 사유 목록 |
| GET, POST | `tran/returnJobFailLogList` | `TranJobFailController.tranJobFail` | JSP `tran/returnJobFailLogList` | 반송 JOB 실패 화면 진입 |
| GET, POST | `tran/ajax/getReturnJobFailLogList` | `TranJobFailController.getReturnJobFailLogList` | jsonView | 반송 JOB 실패 그리드(ajax) |
| GET, POST | `tran/returnJobLogList` | `TranJobHistoryController.returnLogList` | JSP `tran/returnJobLogList` | 반송 JOB 이력 화면 진입 |
| GET, POST | `tran/ajax/getReturnJobLogList` | `TranJobHistoryController.getReturnJobLogList` | jsonView | 반송 JOB 이력 그리드(ajax) |
| (view-controller) | `tot/main` | (xml) view-name `main` | Tiles `main` | servlet-context.xml의 정적 매핑 (컨트롤러 매핑과 중복) |
| (view-controller) | `tot/index` | (xml) view-name `index` | Tiles `index` | servlet-context.xml의 정적 매핑 |

> 참고: `tot/main`은 컨트롤러(`TotalController.main`)와 `<view-controller>`에 동시 정의되어 있습니다. Spring은 일반적으로 컨트롤러 매핑을 우선합니다.

---

## 모듈명 (alarm)

### `GET, POST /alarm/alarmReportLogList`

- **컨트롤러**: `AlarmReportController.alarmReportLogList(AlarmReportVo param, HttpServletRequest request)`
- **HTTP**: GET, POST
- **요청 파라미터** (`AlarmReportVo` 필드 자동 바인딩):
  | 이름 | 타입 | 필수 | 기본값 | 설명 |
  |-----|-----|-----|-------|-----|
  | fabSite | String | N | 세션 fabSite | fab site (없으면 세션에서 가져옴, 있으면 세션에 저장) |
  | pageNum | String | N | - | 페이지 번호 |
  | rowNum | String | N | - | 페이지당 행수 |
  | areaName | String | N | - | Area 이름 |
  | bayName | String | N | - | Bay 이름 |
  | machineType | List<String> | N | - | Machine Type 목록 |
  | machineName | List<String> | N | - | Machine Name 목록 |
  | fab | List<String> | N | 기본 fab list | Fab 목록 (서버에서 강제 셋팅) |
  | level | List<String> | N | `[WELL,WARN,ERROR,FATAL]` | 로그 레벨 (서버에서 강제 셋팅) |
  | unit | String | N | - | 유닛 |
  | alarmId | String | N | - | 알람 ID |
  | alarmCode | String | N | - | 알람 코드 |
  | alarmText | String | N | - | 알람 텍스트 |
  | state | String | N | - | 상태 |
  | from | String | N | - | 검색 시작시각 `yyyyMMddHHmmss` |
  | to | String | N | - | 검색 종료시각 `yyyyMMddHHmmss` |
- **응답**: JSP view `alarm/alarmReportLogList`
- **Model 키**: `fabsites`, `fabs`, `levels`, `param`, `params`
- **호출 서비스**: 없음 (단순 화면 진입)
- **설명**: 알람 리포트 조회 화면 초기 진입. fabSite 세션 동기화 및 기본 필터값 셋팅.
- **노트**: 주석된 dead code(bayNameList, machineNameList) 다수. `param.setLevel`이 강제로 4개 레벨로 설정되어 UI에서 전달한 level은 무시됨.

### `GET, POST /alarm/ajax/getAlarmReportLogList`

- **컨트롤러**: `AlarmReportController.getAlarmReportLogList(AlarmReportVo param, HttpServletRequest request)`
- **HTTP**: GET, POST
- **요청 파라미터**: 위 VO 전 필드 + 다음 query string:
  | 이름 | 타입 | 필수 | 기본값 | 설명 |
  |-----|-----|-----|-------|-----|
  | page | String | N | "1" | 현재 페이지 |
  | rows | String | N | "100" | 페이지당 행수 |
  | fab1, fab2, ... | String | N | - | 각 fab 체크박스 값. `fab1=ALL` 인 경우 전체 fab 적용 |
  | level1 ~ levelN | String | N | - | 각 레벨 체크박스 값 |
  | machineTypes | String | N | - | 콤마구분 machineType. `ALL`이면 전체 |
  | areaName | String | N | "ALL" | Area |
  | bayName | String | N | "ALL" | Bay |
  | from | String | N | 현재시각-10분 | 검색 시작 |
  | to | String | N | 현재시각 | 검색 종료 |
- **응답**: `jsonView`
- **Model 키**: `page`, `total`, `records`, `rows`, `fabsites`
- **호출 서비스**: `alarmReportService.getDataList(param)`
- **설명**: 알람 리포트 그리드 데이터를 페이징하여 JSON으로 반환.
- **노트**: jqGrid 스타일 응답 (`page/total/records/rows`). `Paging.nTotalCount` 정적 변수 사용 → 멀티 사용자에서 race condition 가능성.

---

## 모듈명 (mat)

### `GET, POST /mat/carrierLocLogList`

- **컨트롤러**: `MaterialController.carrierLocLogList(MaterialVo param, HttpServletRequest request)`
- **HTTP**: GET, POST
- **요청 파라미터** (`MaterialVo`):
  | 이름 | 타입 | 필수 | 기본값 | 설명 |
  |-----|-----|-----|-------|-----|
  | fabSite | String | N | 세션 | fab site |
  | pageNum, rowNum | String | N | - | 페이징 |
  | areaName, bayName | String | N | - | Area / Bay |
  | machineType, machineName, fab, level | List<String> | N | - | 다중 선택 필터 |
  | carrier | String | N | - | Carrier ID |
  | lotId | String | N | - | Lot ID |
  | commandId | String | N | - | Command ID |
  | unit | String | N | - | 유닛 |
  | from, to | String | N | - | 시간 범위 |
- **응답**: JSP view `mat/carrierLocLogList`
- **Model 키**: `fabsites`, `fabs`, `levels`, `param`, `params`
- **호출 서비스**: 없음
- **설명**: 캐리어 위치 로그 조회 화면 진입.
- **노트**: 알람 화면과 동일한 dead code 패턴.

### `GET, POST /mat/ajax/getCarrierLocLogList.do`

- **컨트롤러**: `MaterialController.getList(MaterialVo param, HttpServletRequest request)`
- **HTTP**: GET, POST
- **요청 파라미터**: VO 전 필드 + `page`, `rows`, `fab1..N`, `level1..N`, `machineTypes`, `areaName`, `bayName`, `from`, `to` (알람과 동일 패턴)
- **응답**: `jsonView`
- **Model 키**: `page`, `total`, `records`, `rows`, `fabsites`
- **호출 서비스**: `materialService.getDataList(param)`
- **설명**: 캐리어 위치 로그 그리드 데이터.
- **노트**: URL에 `.do` 접미사 사용(다른 모듈은 대부분 미사용).

---

## 모듈명 (res)

res 모듈 6개 컨트롤러는 동일한 패턴(화면+ajax)을 가집니다. 각 VO의 고유 필드만 다르고, 페이징·fab·level 처리는 동일합니다.

### `GET, POST /res/craneLogList`

- **컨트롤러**: `ResCraneHistoryController.carrierLocLogList(ResCraneVo param, HttpServletRequest request)`
- **요청 파라미터** (`ResCraneVo` 고유 필드): `craneName`, `state`, `subState`, `processingState`, `transportCommandId` + 공통(fabSite, fab, level, areaName, bayName, machineType, machineName, from, to, pageNum, rowNum)
- **응답**: JSP `res/craneLogList` / **Model 키**: `fabsites`, `fabs`, `levels`, `param`, `params`
- **호출 서비스**: 없음 (`resCraneHistoryServiceImpl` Bean 주입되어 있으나 화면진입 시 미사용)
- **설명**: Crane 이력 조회 화면 초기 진입.
- **노트**: 메서드명이 `carrierLocLogList`이지만 실제는 crane 화면(복붙 흔적). 헤더 주석에 `ResMachineHistoryController` 잘못 표기.

### `GET, POST /res/ajax/getCraneLogList`

- **컨트롤러**: `ResCraneHistoryController.getCraneLogList(ResCraneVo param, HttpServletRequest request)`
- **요청 파라미터**: VO 필드 + `page`, `rows`, `fab1..N`, `level1..N`, `machineTypes`
- **응답**: `jsonView` / **Model 키**: `page`, `total`, `records`, `rows`, `fabsites`
- **호출 서비스**: `resHistoryService.getDataList(param)` (`resCraneHistoryServiceImpl` 빈)
- **설명**: Crane 이력 그리드 데이터.

### `GET, POST /res/machineLogList`

- **컨트롤러**: `ResMachineHistoryController.carrierLocLogList(ResMachineVo param, HttpServletRequest request)`
- **요청 파라미터** (`ResMachineVo` 고유 필드): `state`, `connectionState`, `controlState`, `tscState`, `processingState` + 공통
- **응답**: JSP `res/machineLogList`
- **호출 서비스**: 없음 (진입 단계)
- **설명**: Machine 상태 이력 조회 화면 진입.

### `GET, POST /res/ajax/getMachineLogList`

- **컨트롤러**: `ResMachineHistoryController.getMachineLogList`
- **응답**: `jsonView`
- **호출 서비스**: `resHistoryService.getDataList(param)` (`resMachineHistoryServiceImpl`)
- **설명**: Machine 상태 이력 그리드 데이터.

### `GET, POST /res/portLogList`

- **컨트롤러**: `ResPortHistoryController.portLogList(ResPortVo param, HttpServletRequest request)`
- **요청 파라미터** (`ResPortVo` 고유 필드): `portName`, `state`, `subState`, `processingState`, `banned`, `occupied`, `transportUnitAccessible`, `craneAvailable`, `inOutType`, `manual`, `accessMode`, `idReadState` + 공통
- **응답**: JSP `res/portLogList`
- **설명**: Port 이력 조회 화면 진입.

### `GET, POST /res/ajax/getPortLogList`

- **컨트롤러**: `ResPortHistoryController.getPortLogList`
- **응답**: `jsonView`
- **호출 서비스**: `resHistoryService.getDataList(param)` (`resPortHistoryServiceImpl`)
- **설명**: Port 이력 그리드 데이터.

### `GET, POST /res/shelfLogList`

- **컨트롤러**: `ResShelfHistoryController.portLogList(ResShelfVo param, HttpServletRequest request)`
- **요청 파라미터** (`ResShelfVo` 고유 필드): `shelfName`, `state`, `processingState`, `banned` + 공통
- **응답**: JSP `res/shelfLogList`
- **설명**: Shelf 이력 화면 진입.
- **노트**: 메서드명 `portLogList`이지만 실제는 shelf 화면.

### `GET, POST /res/ajax/getShelfLogList`

- **컨트롤러**: `ResShelfHistoryController.getShelfLogList`
- **응답**: `jsonView`
- **호출 서비스**: `resHistoryService.getDataList(param)` (`resShelfHistoryServiceImpl`)
- **설명**: Shelf 이력 그리드 데이터.

### `GET, POST /res/storageLogList`

- **컨트롤러**: `ResStorageFullHistoryController.carrierLocLogList(ResStorageFullVo param, HttpServletRequest request)`
- **요청 파라미터** (`ResStorageFullVo` 고유 필드): `state`, `fullState`, `processingState` + 공통
- **응답**: JSP `res/storageLogList`
- **설명**: StorageFull 이력 화면 진입.

### `GET, POST /res/ajax/getStorageLogList`

- **컨트롤러**: `ResStorageFullHistoryController.getStorageLogList`
- **응답**: `jsonView`
- **호출 서비스**: `resHistoryService.getDataList(param)` (`resStorageFullHistoryServiceImpl`)
- **설명**: StorageFull 이력 그리드 데이터.

### `GET, POST /res/vehicleLogList`

- **컨트롤러**: `ResVehicleHistoryController.portLogList(ResVehicleVo param, HttpServletRequest request)`
- **요청 파라미터** (`ResVehicleVo` 고유 필드): `vehicleName`, `state`, `subState`, `processingState`, `transportCommandId`, `carrier`, `transportName`, `idReadState` + 공통
- **응답**: JSP `res/vehicleLogList`
- **설명**: Vehicle 이력 화면 진입.

### `GET, POST /res/ajax/getVehicleLogList`

- **컨트롤러**: `ResVehicleHistoryController.getVehicleLogList`
- **응답**: `jsonView`
- **호출 서비스**: `resHistoryService.getDataList(param)` (`resVehicleHistoryServiceImpl`)
- **설명**: Vehicle 이력 그리드 데이터.

---

## 모듈명 (secs)

### `GET, POST /ei/eiLogList`

- **컨트롤러**: `EiLogController.eiLocLogList(EiVo param, HttpServletRequest request)`
- **요청 파라미터** (`EiVo`): `fabSite`, `pageNum`, `rowNum`, `fab`, `level`, `host`, `log`, `process`, `text`, `eiTextConditionCheckBox`, `from`, `to`
- **응답**: JSP `ei/eiLogList`
- **Model 키**: `fabsites`, `fabs`, `levels`, `param`, `params`
- **설명**: EI 로그 조회 화면 진입.

### `GET, POST /ei/ajax/getEiLogList.do`

- **컨트롤러**: `EiLogController.getList(EiVo param, HttpServletRequest request)`
- **요청 파라미터**: VO + `page`, `rows`, `searchDelay`(int, 필수 — `Integer.parseInt` 호출), `eiFab1..N`, `logType1..5`, `host1..3`, `level1..N`
- **응답**: `jsonView` / **Model 키**: `page`, `total`, `records`, `rows`, `fabsites`
- **호출 서비스**: `eiService.getDataList(param)`
- **설명**: EI 로그 그리드 데이터(ajax).
- **노트**: `searchDelay` 파라미터 미전송시 `NumberFormatException` 발생. `Common.searchDelayTime` 전역 static에 저장(멀티 세션 간 간섭).

### `GET, POST /tot/filter/ajax/getProcessList`

- **컨트롤러**: `EiLogController.getSecsList(String fabSite)`
- **요청 파라미터**: `fabSite` (String, query param)
- **응답**: `@ResponseBody List<List>` → JSON 배열
- **호출 서비스**: `eiService.getProcessList(fabSite)`
- **설명**: EI Process 목록(ajax). 반환은 `[[item1, item2, ...]]` 형태.

### `GET, POST /ei/pop/textDetailPop`

- **컨트롤러**: `EiLogController.filterPop(TotalVo param, HttpServletRequest request)`
- **응답**: JSP `ei/pop/textDetailPop`
- **설명**: EI 텍스트 상세 팝업 진입.

### `GET, POST /ei/pop/textAreaPop`

- **컨트롤러**: `EiLogController.textFilterPop(TotalVo param, HttpServletRequest request)`
- **응답**: JSP `ei/pop/textAreaPop`
- **설명**: EI 텍스트영역 팝업 진입.

### `GET, POST /tot/filter/ajax/getSelectProcessList`

- **컨트롤러**: `EiLogController.getSecsFabList(MachineVo param, HttpServletRequest request)`
- **요청 파라미터** (`MachineVo`): `fabSite`, `machineType`, `selectFab`, `selectType`, `areaName`, `bayName`
- **응답**: `jsonView` / **Model 키**: `list`, `fabsites`
- **호출 서비스**: `eiService.getSelectProcessList(param)`
- **설명**: EI 화면 fab/type별 선택된 Process 목록.

### `GET, POST /ei/ajax/getEiQueryStop`

- **컨트롤러**: `EiLogController.getEiQueryStop(HttpServletRequest request)`
- **응답**: `void` + `@ResponseBody` (HTTP 200, body 없음)
- **호출 서비스**: `eiService.getRawLogQueryStop()`
- **설명**: 실행 중인 EI 로그 쿼리를 강제 중지.

### `GET, POST /secs/secsLogList`

- **컨트롤러**: `SecsLogController.secsLocLogList(SecsVo param, HttpServletRequest request)`
- **요청 파라미터** (`SecsVo`): `fabSite`, `pageNum`, `rowNum`, `fab`, `level`, `host`, `carrier`, `vehicle`, `secs`, `carrierLoc`, `commandId`, `transferport`, `sourceport`, `destport`, `text`, `secsTextConditionCheckBox`, `from`, `to`
- **응답**: JSP `secs/secsLogList`
- **Model 키**: `fabsites`, `fabs`, `levels`, `param`, `params`
- **설명**: SECS 로그 조회 화면 진입. levels는 `[TIME, INFO, WARN, RECV, SEND]`로 설정되고 param.level은 `[ALL]` 기본값.

### `GET, POST /secs/ajax/getsecsLogList.do`

- **컨트롤러**: `SecsLogController.getList(SecsVo param, HttpServletRequest request)`
- **요청 파라미터**: VO + `page`, `rows`, `searchDelay`(필수), `secsFab1..N`, `host1..3`, `level1..N`
- **응답**: `jsonView` / **Model 키**: `page`, `total`, `records`, `rows`, `fabsites`
- **호출 서비스**: `secsService.getDataList(param)`
- **설명**: SECS 로그 그리드(ajax).

### `GET, POST /tot/filter/ajax/getSecsList`

- **컨트롤러**: `SecsLogController.getSecsList(String fabSite)`
- **요청 파라미터**: `fabSite` (query)
- **응답**: `@ResponseBody List<List>`
- **호출 서비스**: `secsService.getSecsList(fabSite)`
- **설명**: SECS 머신 목록 ajax.

### `GET, POST /tot/filter/ajax/getSecsFabList`

- **컨트롤러**: `SecsLogController.getSecsFabList(MachineVo param, HttpServletRequest request)`
- **응답**: `jsonView` / **Model 키**: `list`, `fabsites`
- **호출 서비스**: `secsService.getSecsFabList(param)`
- **설명**: SECS Fab별 머신 목록.

### `GET, POST /ei/ajax/getSecsQueryStop`

- **컨트롤러**: `SecsLogController.getSecsQueryStop(HttpServletRequest request)`
- **응답**: `void` + `@ResponseBody`
- **호출 서비스**: `secsService.getRawLogQueryStop()`
- **설명**: SECS 로그 쿼리 강제 중지.
- **노트**: URL이 `/ei/ajax/...`로 시작하지만 실제로는 SECS 처리(URL/모듈 불일치).

---

## 모듈명 (test)

### `GET /i18n.do`

- **컨트롤러**: `TestController.i18n(Locale locale, HttpServletRequest request, Model model)`
- **HTTP**: GET (`method=RequestMethod.GET` 명시)
- **요청 파라미터**: `locale` (Spring 자동), `lang` (LocaleChangeInterceptor가 처리)
- **응답**: String `i18n` (JSP 뷰)
- **Model 키**: `siteCount`, `siteLang`
- **설명**: i18n 메시지/Locale 동작 확인용 테스트 페이지.
- **노트**: 로그 출력 위주의 디버그 페이지.

### `GET /monitoring.do`

- **컨트롤러**: `TestController.monitoring`
- **HTTP**: GET
- **응답**: String `monitoring`
- **설명**: monitoring 페이지(빈 진입 메서드, 단순 뷰 forward).

### `GET /tmp.do`

- **컨트롤러**: `TestController.tmp`
- **HTTP**: GET
- **응답**: String `tmp`
- **설명**: 임시 테스트 페이지.
- **노트**: 본문 전체가 ThreadPool 테스트용 dead code(주석 처리됨).

---

## 모듈명 (tot)

### `GET, POST /tot/totalLogList`

- **컨트롤러**: `TotalController.totalLogList(TotalVo param, HttpServletRequest request)`
- **요청 파라미터** (`TotalVo`): `fabSite`, `pageNum`, `rowNum`, `areaName`, `bayName`, `machineType`, `machineName`, `fab`, `level`, `searchOption`(AND/OR), `process`, `thread`, `gtxnId`, `transactionId`, `messageName`, `comMsgName`, `operationName`, `carrier`, `commandId`, `unit`, `text`, `fulltext`, `key`, `messageName_m`, `comMsgName_m`, `operationName_m`, `from`, `to`
- **응답**: JSP `tot/totalLogList`
- **Model 키**: `fabsites`, `fabs`, `levels`, `param`, `params`
- **설명**: 통합 Total 로그 조회 화면 진입.

### `GET, POST /tot/ajax/getTotalLogList`

- **컨트롤러**: `TotalController.getTotalLogList(TotalVo param, HttpServletRequest request)`
- **요청 파라미터**: VO + `page`, `rows`, `searchDelay`(필수), `fab1..N`, `level1..N`, `machineTypes`
- **응답**: `jsonView` / **Model 키**: `page`, `total`, `records`, `rows`, `fabsites`
- **호출 서비스**: `totService.getDataList(param)`
- **설명**: Total 로그 그리드(ajax).

### `GET, POST /tot/ajax/getTotalLogListStop`

- **컨트롤러**: `TotalController.getTotalLogListStop(HttpServletRequest request)`
- **응답**: `void` + `@ResponseBody`
- **호출 서비스**: `totService.getTotalLogListStop()`
- **설명**: Total 로그 쿼리 중지.

### `GET, POST /tot/pop/machineNamePop`

- **컨트롤러**: `TotalController.machineNamePop(TotalVo param, HttpServletRequest request)`
- **응답**: JSP `tot/pop/machineNamePop` / **Model 키**: `machineTypeInfoList`
- **호출 서비스**: `totService.getMachineTypeFromFab(new MachineVo())`
- **설명**: Machine 선택 팝업. **빈 `MachineVo`** 사용하여 fabSite 미적용 상태로 전체 조회.
- **노트**: fabSite 누락 가능성 — 모든 fab에서 머신 타입 조회됨.

### `GET, POST /tot/ajax/getMachineList`

- **컨트롤러**: `TotalController.getMachineList(MachineVo param, HttpServletRequest request)`
- **응답**: `jsonView` / **Model 키**: `list`, `fabsites`
- **호출 서비스**: `totService.getMachineNameList(param)`
- **설명**: Machine 목록 ajax.

### `GET, POST /tot/ajax/getMachineListMachineTypeNotNull`

- **컨트롤러**: `TotalController.getMachineListMachineTypeNotNull(MachineVo param, HttpServletRequest request)`
- **응답**: `jsonView` / **Model 키**: `list`, `fabsites`
- **호출 서비스**: `totService.getMachineNameListMachineTypeNotNull(param)`
- **설명**: MachineType이 NULL이 아닌 Machine 목록.

### `GET, POST /tot/ajax/getBayFromArea`

- **컨트롤러**: `TotalController.getBayFromArea(MachineVo param, HttpServletRequest request)`
- **응답**: `jsonView` / **Model 키**: `list`, `fabsites`
- **호출 서비스**: `totService.getBayFromAreaList(param)`
- **설명**: 선택된 Area에 대응되는 Bay 목록.

### `GET, POST /tot/ajax/getAreaFromFab`

- **컨트롤러**: `TotalController.getAreaFromFab(MachineVo param, HttpServletRequest request)`
- **응답**: `jsonView` / **Model 키**: `list`, `fabsites`
- **호출 서비스**: `totService.getAreaFromFabList(param)`
- **설명**: 선택된 Fab에 대응되는 Area 목록.

### `GET, POST /tot/ajax/getMachineTypeFromFab`

- **컨트롤러**: `TotalController.getMachineTypeFromFab(MachineVo param, HttpServletRequest request)`
- **응답**: `jsonView` / **Model 키**: `list`, `fabsites`
- **호출 서비스**: `totService.getMachineTypeFromFab(param)`
- **설명**: 선택된 Fab에 대응되는 MachineType 목록.

### `GET, POST /tot/ajax/getFabFromFabSite`

- **컨트롤러**: `TotalController.getFabFromFabSite(FabVo param, HttpServletRequest request)`
- **요청 파라미터** (`FabVo`): `fabSite`, `menu`
- **응답**: `jsonView` / **Model 키**: `list`, `basic_list`
- **호출 서비스**: `Common.getFabList(menu, fabSite)`, `Common.getBasicFabList(menu, fabSite)` (정적 메서드)
- **설명**: FabSite + Menu 기준 fab 전체 목록과 기본 목록.

### `GET, POST /tot/main`

- **컨트롤러**: `TotalController.main(TotalVo param, HttpServletRequest request)`
- **응답**: JSP `tot/main` / **Model 키**: `fabsites`, `param`, `location`
- **설명**: 메인 페이지 진입. 세션 정보 로깅 및 fabSite 처리.
- **노트**: servlet-context.xml의 `<view-controller path="tot/main" view-name="main">`과 URL 중복. 컨트롤러가 우선 적용.

### `GET /tot/{query}`

- **컨트롤러**: `TotalController.getRequest(TotalVo param, @PathVariable String query, HttpServletRequest request)`
- **HTTP**: GET (`method=RequestMethod.GET`)
- **요청 파라미터**:
  | 이름 | 타입 | 필수 | 설명 |
  |-----|-----|-----|-----|
  | query | String (path) | Y | URL 경로변수. 무엇이 들어와도 동일 처리됨 |
- **응답**: JSP `tot/main` (모든 query에 대해 fallthrough)
- **Model 키**: `fabsites`, `param`, `location`
- **설명**: `tot/xxx` 형태의 임의 URL을 메인 페이지로 라우팅하는 catch-all.
- **노트**: 매우 광범위한 패턴 — 다른 `tot/...` 매핑이 우선 매칭되지 않는 경우 모두 이쪽으로. `tot/totalLogList`, `tot/main`, `tot/ajax/...` 등은 더 구체적이므로 우선.

### `GET, POST /tot/filter/ajax/getAreaList`

- **컨트롤러**: `TotalController.getAreaList(String fabSite)`
- **응답**: `@ResponseBody List<List>`
- **호출 서비스**: `totService.getAreaNameList(fabSite)`
- **설명**: Area 이름 목록.

### `GET, POST /tot/filter/ajax/getBayList`

- **컨트롤러**: `TotalController.getBayList(String fabSite)`
- **응답**: `@ResponseBody List<List>`
- **호출 서비스**: `totService.getBayNameList(fabSite)`
- **설명**: Bay 이름 목록.

### `GET, POST /tot/filter/ajax/getMachineNameList`

- **컨트롤러**: `TotalController.getMachineNameList(String fabSite)`
- **응답**: `@ResponseBody List<List>`
- **호출 서비스**: `totService.getMachineNameList(fabSite)`
- **설명**: Machine 이름 목록.

### `GET, POST /tot/filter/ajax/getCommMsgNameList`

- **컨트롤러**: `TotalController.getCommMsgNameList(String fabSite)`
- **응답**: `@ResponseBody List<List>`
- **호출 서비스**: `totService.getCommMsgNameList(fabSite)`
- **설명**: CommMsg 이름 목록.

### `GET, POST /tot/filter/ajax/getMessageNameList`

- **컨트롤러**: `TotalController.getMessageNameList(String fabSite)`
- **응답**: `@ResponseBody List<List>`
- **호출 서비스**: `totService.getMessageNameList(fabSite)`
- **설명**: Message 이름 목록.

### `GET, POST /tot/filter/ajax/getOperationNameList`

- **컨트롤러**: `TotalController.getOperationNameList(String fabSite)`
- **응답**: `@ResponseBody List<List>`
- **호출 서비스**: `totService.getOperationNameList(fabSite)`
- **설명**: Operation 이름 목록.

### `GET, POST /tot/pop/filterPop`

- **컨트롤러**: `TotalController.filterPop(TotalVo param, HttpServletRequest request)`
- **응답**: JSP `tot/pop/filterPop`
- **설명**: 필터 팝업 진입.

### `GET, POST /common/pop/settingPop`

- **컨트롤러**: `TotalController.settingPop(TotalVo param, HttpServletRequest request)`
- **응답**: JSP `common/pop/settingPop`
- **설명**: 환경 설정 팝업.

### `GET, POST /tot/dashboard/elapsedAnalysis`

- **컨트롤러**: `TotalController.elapsed(TotalVo param, HttpServletRequest request)`
- **응답**: JSP `tot/elapsedAnalysis`
- **설명**: 경과시간 분석 대시보드.

### `GET, POST /tot/dashboard/compressAnalysis`

- **컨트롤러**: `TotalController.elapsed2(TotalVo param, HttpServletRequest request)`
- **응답**: JSP `tot/compressAnalysis`
- **설명**: 압축 분석 대시보드.

### `GET, POST /tot/dashboard/monitor`

- **컨트롤러**: `TotalController.monitor(TotalVo param, HttpServletRequest request)`
- **응답**: JSP `tot/monitor`
- **설명**: 모니터 대시보드.

### `GET, POST /tot/dashboard/elapsed3`

- **컨트롤러**: `TotalController.elapsed3(TotalVo param, HttpServletRequest request)`
- **응답**: JSP `tot/dashboard3`
- **설명**: 대시보드3 (뷰 이름은 `tot/dashboard3`).

### `GET, POST /totNew/totalNewLogList`

- **컨트롤러**: `TotalNewController.totalNewLogList(TotalNewVo param, HttpServletRequest request)`
- **요청 파라미터** (`TotalNewVo`): `fabSite`, `pageNum`, `rowNum`, `areaName`, `bayName`, `machineType`, `machineName`, `level`, `searchOption`, `carrier`, `totalElapsedTime`, `elapsedTime`, `command`, `messageName`, `process`, `transactionId`, `commandId`, `unit`, `thread`, `comment`, `from`, `to` + query `machineTypes`
- **응답**: JSP `tot/totalNewLogList`
- **Model 키**: `fabsites`, `list`, `paging`, `param`, `params`
- **호출 서비스**: `totService.getDataList(param)` (`totalNewService` 빈)
- **설명**: 신규 로그 조회 화면 진입. ajax 화면과 달리 진입 시 바로 조회까지 수행.
- **노트**: 페이징 정보 산출 시 `list.get(0).get("count")`를 사용 — 첫 행에 count 칼럼이 포함되어야 함.

### `GET, POST /totNew/ajax/totalNewLogList`

- **컨트롤러**: `TotalNewController.totalNewLogListAjax(TotalNewVo param, HttpServletRequest request)`
- **요청 파라미터**: VO + `page`, `rows`, `machineTypes`
- **응답**: `jsonView` / **Model 키**: `total`, `records`, `paging`, `param`, `params`, `rows`, `fabsites`
- **호출 서비스**: `totService.getDataList(param)`
- **설명**: 신규 로그 그리드 데이터.

### `GET, POST /totNew/pop/machineNamePop`

- **컨트롤러**: `TotalNewController.machineNamePop(TotalVo param, HttpServletRequest request)`
- **요청 파라미터**: `fabSite`
- **응답**: JSP `tot/pop/machineNamePop` / **Model 키**: `list`
- **호출 서비스**: `totService.getSelectList(fabSite)`
- **설명**: 신규 화면용 Machine Name 선택 팝업.

### `GET, POST /totNew/ajax/getCarrierElapsed`

- **컨트롤러**: `TotalNewController.getCarrierElapsed(TotalNewVo param, HttpServletRequest request)`
- **요청 파라미터**: VO + `addQuery` (추가 쿼리문)
- **응답**: `jsonView` / **Model 키**: `list`, `fabsites`
- **호출 서비스**: `totService.getDetailDataList(fabSite, addQuery)`
- **설명**: Carrier 경과시간 상세 — `addQuery`로 추가 WHERE 조건 전달.
- **노트**: **`addQuery` 파라미터를 SQL에 직접 합치는 구조이면 SQL Injection 위험**. `System.out.println` 사용(로그가 stdout으로). 호출 서비스에 fabSite를 인자로 전달.

---

## 모듈명 (tran)

tran 모듈의 ajax 그리드 메서드들은 모두 동일한 파라미터 패턴을 공유합니다: `fab1..N`, `transportMachineTypes`/`fromMachineTypes`/`toMachineTypes`, `transportAreaName/BayName`, `fromAreaName/BayName`, `toAreaName/BayName`, `page`, `rows`, `from`, `to`.

### `GET, POST /tran/returnCmdFailLogList`

- **컨트롤러**: `TranCmdFailController.returnCmdFailLogList(TranCmdFailVo param, HttpServletRequest request)`
- **요청 파라미터** (`TranCmdFailVo`): `fabSite`, `pageNum`, `rowNum`, `fab`, `fromAreaName`, `fromBayName`, `fromUnit`, `toAreaName`, `toBayName`, `toUnit`, `transportAreaName`, `transportBayName`, `transportUnit`, `fromMachineType`, `toMachineType`, `transportMachineType`, `fromMachineName`, `toMachineName`, `transportMachineName`, `from`, `to`, `carrier`, `transportCmdId`, `reason`
- **응답**: JSP `tran/returnCmdFailLogList`
- **Model 키**: `fabsites`, `fabs`, `param`, `params`
- **설명**: 반송 CMD 실패 화면 진입.

### `GET, POST /tran/ajax/getReturnCmdFailLogList`

- **컨트롤러**: `TranCmdFailController.getReturnCmdFailLogList`
- **요청 파라미터**: VO + 공통 ajax 파라미터
- **응답**: `jsonView` / **Model 키**: `page`, `total`, `records`, `rows`, `fabsites`
- **호출 서비스**: `tranService.getDataList(param)` (`tranCmdFailService` 빈)
- **설명**: 반송 CMD 실패 그리드 데이터.

### `GET, POST /tran/pop/reasonPop`

- **컨트롤러**: `TranCmdFailController.machineNamePop(TranCmdFailVo param, HttpServletRequest request)`
- **응답**: JSP `tran/pop/reasonPop`
- **설명**: 반송 실패 사유 선택 팝업.
- **노트**: 메서드명이 `machineNamePop`이나 실제는 reason 팝업(복붙 흔적).

### `GET, POST /tran/returnCmdLogList`

- **컨트롤러**: `TranCmdHistoryController.returnCmdLogList(TranVo param, HttpServletRequest request)`
- **요청 파라미터** (`TranVo`): `fabSite`, 페이징, fab, area/bay/unit (from/to/transport), machineType (from/to/transport), machineName (from/to/transport), `from`, `to`, `carrier`, `lotId`, `transportJobId`, `transportCommandId`, `state`(List)
- **응답**: JSP `tran/returnCmdLogList`
- **설명**: 반송 CMD 이력 화면 진입.

### `GET, POST /tran/ajax/getReturnCmdLogList`

- **컨트롤러**: `TranCmdHistoryController.getReturnCmdLogList`
- **요청 파라미터**: VO + 공통 ajax 파라미터 + `states` (콤마구분), `state1..N` (`ALL` 체크용)
- **응답**: `jsonView` / **Model 키**: `page`, `total`, `records`, `rows`, `fabsites`
- **호출 서비스**: `tranService.getDataList(param)` (`tranCmdHistoryService` 빈)
- **설명**: 반송 CMD 이력 그리드.

### `GET, POST /tran/returnLogList`

- **컨트롤러**: `TranController.returnLogList(TranVo param, HttpServletRequest request)`
- **응답**: JSP `tran/returnLogList`
- **설명**: 반송 이력 화면 진입.

### `GET, POST /tran/ajax/getReturnLogList`

- **컨트롤러**: `TranController.getReturnLogList`
- **요청 파라미터**: VO + 공통 + `state` (단일 String, **`states`가 아님**)
- **응답**: `jsonView` / **Model 키**: `page`, `total`, `records`, `rows`, `fabsites`
- **호출 서비스**: `tranService.getDataList(param)`
- **설명**: 반송 이력 그리드. state는 단일 값을 List로 wrapping.

### `GET, POST /tran/ajax/getTranJobHistoryDetail`

- **컨트롤러**: `TranController.getTranJobHistoryDetail(TranVo param, HttpServletRequest request)`
- **응답**: `jsonView` / **Model 키**: `rows`, `commandListRow`, `historyListRow`
- **호출 서비스**: `tranService.getTranJobHistoryDetail(param)`
- **설명**: 반송 JOB 이력 상세. method 값이 `Common.METHOD_INFO_CREATE_TRANSPORT_COMMAND_HISTORY`이면 commandList로, 그 외는 historyList로 분리.

### `GET, POST /tran/ajax/getReasonList`

- **컨트롤러**: `TranController.getReasonList(FabVo param, HttpServletRequest request)`
- **요청 파라미터** (`FabVo`): `fabSite`, `menu`
- **응답**: `jsonView` / **Model 키**: `list`
- **호출 서비스**: `tranService.getReasonList(fabSite)`
- **설명**: 반송 사유 목록 ajax.

### `GET, POST /tran/returnJobFailLogList`

- **컨트롤러**: `TranJobFailController.tranJobFail(TranJobFailVo param, HttpServletRequest request)`
- **요청 파라미터** (`TranJobFailVo`): TranVo 공통 + `carrier`, `lotId`, `transportJobId`, `reason`(List)
- **응답**: JSP `tran/returnJobFailLogList`
- **설명**: 반송 JOB 실패 화면 진입.

### `GET, POST /tran/ajax/getReturnJobFailLogList`

- **컨트롤러**: `TranJobFailController.getReturnJobFailLogList`
- **응답**: `jsonView` / **Model 키**: `page`, `total`, `records`, `rows`, `fabsites`
- **호출 서비스**: `jobFailService.getDataList(param)` (`jobFailService` 빈)
- **설명**: 반송 JOB 실패 그리드.

### `GET, POST /tran/returnJobLogList`

- **컨트롤러**: `TranJobHistoryController.returnLogList(TranVo param, HttpServletRequest request)`
- **응답**: JSP `tran/returnJobLogList`
- **설명**: 반송 JOB 이력 화면 진입.

### `GET, POST /tran/ajax/getReturnJobLogList`

- **컨트롤러**: `TranJobHistoryController.getReturnJobLogList`
- **응답**: `jsonView` / **Model 키**: `page`, `total`, `records`, `rows`, `fabsites`
- **호출 서비스**: `tranService.getDataList(param)` (`tranJobHistoryService` 빈)
- **설명**: 반송 JOB 이력 그리드.

---

## servlet-context.xml view-controller

```xml
<view-controller path="tot/main"  view-name="main" />
<view-controller path="tot/index" view-name="index" />
```

- **`GET tot/main`** → Tiles 뷰 `main` (Tiles definitions `/WEB-INF/tiles/tiles-layout.xml` 에서 정의)
  - 단, `TotalController.main`이 `tot/main`을 매핑하므로 컨트롤러가 우선됩니다. xml의 매핑은 실제로 도달하지 못할 가능성이 큼.
- **`GET tot/index`** → Tiles 뷰 `index`
  - 컨트롤러 매핑은 없으며 `<view-controller>`만 존재 → Tiles 레이아웃이 렌더링됨.

---

## 공통 파라미터 패턴

### 1) fabSite 처리 패턴 (모든 컨트롤러 메서드 진입부에서 동일)

```java
mav.addObject("fabsites", Common.FabSites);
String sFabSite = param.getFabSite();
if (sFabSite == null || sFabSite.length() == 0) {
    sFabSite = Common.getFabSite(request);   // 세션 → 기본값
    param.setFabSite(sFabSite);
} else {
    sFabSite = Common.setFabSite(request, sFabSite);  // 세션에 저장
}
```

- **`fabSite`**: M14, M16, M16C, C2, C2F 등 사이트 식별자. 세션 키 `FAB_SITE`에 저장.
- 클라이언트는 명시적으로 fabSite를 보내거나(세션 갱신), 생략하면(세션 값 사용) 됨.

### 2) Fab 다중 선택 패턴

| 파라미터 | 형식 | 설명 |
|---------|-----|------|
| `fab1`, `fab2`, ..., `fabN` | String[N] | 각 체크박스의 값을 별도 파라미터로 전송 |
| `fab1=ALL` | String | "전체 선택" 시 → 서버가 자동으로 `Common.getFabList(menu, fabSite)`로 전체 적용 |
| `eiFab1..N` | EI 모듈 전용 | 동일 패턴, prefix만 다름 |
| `secsFab1..N` | SECS 모듈 전용 | 동일 패턴 |

### 3) Level 다중 선택 패턴

`level1`, `level2`, ..., `levelN` 형식. `Common.Levels` 리스트(WELL, WARN, ERROR, FATAL, DEBUG, INFO, FINE 등)에서 선택. `ALL` 처리 로직은 대부분 주석 처리되어 있고 현재는 사용자 선택만 그대로 List에 저장.

### 4) MachineTypes 처리

| 파라미터 | 형식 | 설명 |
|---------|-----|------|
| `machineTypes` | "ALL" 또는 "TYPE1,TYPE2,..." | 콤마로 구분된 단일 string. `ALL`이면 빈 리스트(전체 의미) |
| `transportMachineTypes` / `fromMachineTypes` / `toMachineTypes` | 동일 형식 | tran 모듈 전용 (3-way 분리) |

### 5) 시간 범위

| 파라미터 | 형식 | 기본값 |
|---------|-----|-------|
| `from` | `yyyyMMddHHmmss` | 현재시각 - 10분 |
| `to` | `yyyyMMddHHmmss` | 현재시각 |

### 6) 페이징

| 파라미터 | 기본값 | 설명 |
|---------|-------|-----|
| `page` | "1" | 현재 페이지 (jqGrid 호환) |
| `rows` | "100" | 한 페이지당 row 수 |
| `pageNum` / `rowNum` | VO 필드 (page/rows의 alias) |

### 7) Area / Bay

`areaName`, `bayName` (단일 String). 빈 값이면 서버에서 `Common.sALL`("ALL")로 강제 설정.

tran 모듈은 3-way: `transportAreaName`, `fromAreaName`, `toAreaName` 및 BayName, Unit.

### 8) Source / Destination 식별 (tran)

| 파라미터 prefix | 의미 |
|---------------|-----|
| `from*` | 출발지 |
| `to*` | 목적지 |
| `transport*` | 운반체(차량/크레인) |

### 9) Host / LogType (secs, ei)

| 파라미터 | 형식 | 설명 |
|---------|-----|-----|
| `host1`, `host2`, `host3` | String[] | Primary / Secondary 등 |
| `logType1..5` | String[] | TS, EI, CS, DS (EI 모듈) |

### 10) 검색 지연 (delay)

`searchDelay` (초). 쿼리에 인위적 sleep을 부여하기 위함. `Common.searchDelayTime`이라는 정적 변수에 저장 → **멀티 사용자 간섭 발생 가능**. 또한 EI, SECS, Total 로그의 ajax 메서드는 이 파라미터가 `null`이면 `NumberFormatException`을 발생시킴.

---

## 응답 형식 가이드

### 1) JSP 뷰 (`InternalResourceViewResolver`)

- `servlet-context.xml` 설정:
  ```xml
  <beans:bean class="org.springframework.web.servlet.view.InternalResourceViewResolver">
      <beans:property name="prefix" value="/WEB-INF/views/" />
      <beans:property name="suffix" value=".jsp" />
  </beans:bean>
  ```
- 컨트롤러가 `mav.setViewName("alarm/alarmReportLogList")`로 지정하면 `/WEB-INF/views/alarm/alarmReportLogList.jsp` 가 forward되어 렌더링.
- Model에 추가된 모든 키(`fabsites`, `fabs`, `levels`, `param`, `params`, `list`, `paging`, ...)는 JSP에서 EL(`${param}`, `${list}` 등)로 접근 가능.

### 2) `jsonView` (Jackson 기반 JSON 응답)

- 별도 빈 설정은 코드에서 직접 보이지 않으나, `@EnableWebMvc`/`<annotation-driven/>` 환경에서 `MappingJackson2JsonView`가 `jsonView`라는 이름으로 등록되어 있는 패턴. (`BeanNameViewResolver` 또는 `ContentNegotiatingViewResolver`로 매핑됨)
- 컨트롤러가 `mav.setViewName("jsonView")`를 지정하면 `ModelAndView`에 담긴 키-값 쌍이 단일 JSON 객체로 직렬화됨:
  ```json
  {
    "page": "1",
    "total": 250,
    "records": 100,
    "rows": [ ... ],
    "fabsites": [ ... ]
  }
  ```
- 응답 헤더: `Content-Type: application/json`.

### 3) `@ResponseBody`

- 메서드에 `@ResponseBody` 또는 클래스에 `@RestController`(본 프로젝트는 미사용)를 지정한 경우.
- 메서드 반환값(예: `List<List>`)이 Jackson을 통해 직접 JSON으로 직렬화됨. ModelAndView를 거치지 않음.
- 본 프로젝트는 두 가지 사용 패턴:
  - **데이터 반환**: `List<List>` 반환 → `[[...]]` 형식의 JSON 배열.
  - **void 반환**: 단순 명령 전달(쿼리 중지 등). HTTP 200 + 빈 body.

### 4) Tiles (`tilesViewResolver`)

- `servlet-context.xml`:
  ```xml
  <beans:bean id="tilesViewResolver" ... order="1">
      <beans:property name="viewClass" value="...TilesView" />
  </beans:bean>
  ```
- Tiles 리졸버의 `order=1` 이므로 InternalResourceViewResolver보다 우선 평가됨. 뷰 이름이 `/WEB-INF/tiles/tiles-layout.xml`의 `<definition name="...">`과 일치하면 Tiles로, 일치하지 않으면 다음 리졸버(`InternalResourceViewResolver`)로 fallback.
- `<view-controller>`의 `view-name="main"`, `view-name="index"`는 Tiles definition으로 추정.

### 5) String 반환 (Test 컨트롤러)

- `return "i18n";` → 뷰 리졸버 체인에 의해 Tiles definition `i18n` (있으면) 또는 `/WEB-INF/views/i18n.jsp`로 해석.

---

## 부록: 알려진 이슈 / 노트 요약

1. **dead code / 복붙 흔적**: 거의 모든 컨트롤러 상단에 `bayNameList`, `machineNameList` 주석 코드와 사용되지 않는 `totService` 의존성이 남아있음.
2. **잘못된 메서드명/주석**: `ResShelfHistoryController.portLogList`, `ResStorageFullHistoryController.carrierLocLogList`, `TranCmdFailController.machineNamePop` 등 메서드명이 동작과 불일치.
3. **URL/모듈 불일치**: `SecsLogController.getSecsQueryStop`은 URL이 `/ei/ajax/getSecsQueryStop` (ei prefix이지만 secs 서비스 호출).
4. **`searchDelay` 필수 + NPE 위험**: 누락 시 `Integer.parseInt(null)` → `NumberFormatException`.
5. **전역 static 변수 공유**: `Common.searchDelayTime`, `Paging.nTotalCount` 등은 정적 변수라 멀티 세션 간섭 가능.
6. **SQL Injection 가능성**: `TotalNewController.getCarrierElapsed`의 `addQuery` 파라미터가 그대로 service에 전달되어 SQL에 합쳐지면 위험.
7. **`tot/main`이 컨트롤러+view-controller 양쪽에 정의**됨. 컨트롤러가 우선.
8. **`/tot/{query}` catch-all 매핑**: 다른 매핑이 없으면 모두 `tot/main`으로 fallthrough.
9. **page == 1, rows == 100 디폴트는 두 곳에서 분기**됨 (request param vs VO 필드). 일관성 없음.
10. **응답 키 `fabsites`**가 ajax 응답에도 무조건 포함되어 페이로드가 커짐.
