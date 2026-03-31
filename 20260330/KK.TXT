public class OhtMsgWorkerRunnable implements Runnable {
    private static final Logger logger = LoggerFactory.getLogger(OhtMsgWorkerRunnable.class);
    private static final int MSG_ID_IDX = 0;
    private final String message;
    private final long receivedMilli;
    private final long messageSequence;
    private final String fabId;
    private final String facId;
    private final String mcpName;

    public OhtMsgWorkerRunnable(
            String fabId,
            String message,
            String mcpName,
            long receivedMilli,
            long messageSequence
    ) {
        FabProperties fabProperties = DataService.getInstance().getFabPropertiesMap().get(fabId);

        this.fabId              = fabId;
        this.mcpName            = mcpName;
        this.facId              = fabProperties.getFacId();
        this.message            = message;
        this.receivedMilli      = receivedMilli;
        this.messageSequence    = messageSequence;
    }

    @Override
    public void run() {
        String[] tokens = StringUtils.splitPreserveAllTokens(message, ',');

        if (tokens.length >= 2) {
            String messageId = tokens[MSG_ID_IDX];

            if (messageId.equals(MSG_ID.VHL_STATE_REPORT)) {
                this._processOhtReport(tokens);
            }
        }
    }

    @Override
    public String toString() {
        return String.format("%s %s %s %s", getClass(), fabId, mcpName, message);
    }

    private void _processOhtReport(String[] tokens) {
        Vhl vehicle;
        ReentrantLock vhlLock;
        String keyPrefix = this.fabId + ":" + DataSet.VHL_PREFIX + ":" + this.mcpName;
        ConcurrentMap<String, Vhl> vehicleMap = DataService.getDataSet().getVhlMap();

        if (vehicleMap == null || vehicleMap.isEmpty()) {
            logger.error("... mapping data of vehicle is empty or null");

            return;
        }

        if (tokens.length <= VHL_STATE_REPORT.VHL_ID_IDX) {
            logger.error("... invalid token context, it's not matched index [token length: {}] [message: {}]", tokens.length, message);

            return;
        }

        String vhlName      = tokens[VHL_STATE_REPORT.VHL_ID_IDX];
        String vehicleKey   = StringUtils.isNotEmpty(vhlName) ? keyPrefix + ":" + vhlName : "";
        vehicle             = vehicleMap.get(vehicleKey);

        if (vehicle == null) {
            logger.error("... vehicle data is null [key: {}]", vehicleKey);
        } else {
            synchronized (vehicle) {
                try {
                    long lastMessageSequence = vehicle.getLastMessageSequenceNo().longValue();

                    if (lastMessageSequence > this.messageSequence) {
                        // sequence 역전 현상 발생시 과정 생략
                        logger.error("... this sequence of message precedes the last sequence of it [current: {} | last: {}] [fab: {} | mcp: {} | message: {}]", this.messageSequence, lastMessageSequence, this.fabId, this.mcpName, this.message);

                        return;
                    } else {
                        vehicle.getLastMessageSequenceNo().set(messageSequence);
                    }
                } catch (Exception e) {
                    logger.error("", e);
                }
            }

            vhlLock = vehicle.getLock();

            if (VHL_STATE.getValue(tokens[VHL_STATE_REPORT.STATE_IDX]) == VHL_STATE.REMOVING) {
                // `REMOVING` 으로 받을 땐 해당 Vhl 상태 초기화
                this._updateRemovingVehicle(tokens, vehicle);
            } else {
                this._updateVehicle(tokens, vehicle, System.currentTimeMillis());
            }

            if (vhlLock != null && vhlLock.isHeldByCurrentThread()) {
                vhlLock.unlock();
            }
        }
    }

    /*
    vehicle 초기화
     */
    private void _updateRemovingVehicle(String[] tokens, Vhl vehicle) {
        vehicle.copyCurrentVhlUdpStateToLast();

        vehicle.setRailNodeId("");
        vehicle.setUdpCarrierId("");
        vehicle.setDestStationId("");
        vehicle.setDestPortId("");
        vehicle.setDistance(0);
        vehicle.setFull(false);
        vehicle.setOnline(false);
        vehicle.setNextRailNodeId("");
        vehicle.setRunCycle(RUN_CYCLE.NONE);
        vehicle.setVhlCycle(VHL_CYCLE.NONE);
        vehicle.setState(VHL_STATE.REMOVING);
        vehicle.setReceivedTime(receivedMilli);
        vehicle.setEmStatus(Util.binaryStringToByte(tokens[VHL_STATE_REPORT.EM_STATUS_IDX]));
        vehicle.setGroupId(tokens[VHL_STATE_REPORT.GROUP_ID_IDX]);
        vehicle.setSourcePortId("");
        vehicle.setDestPortId("");
        vehicle.setPriority(0);
        vehicle.setDetailState(VHL_DET_STATE.NONE);
        vehicle.setRunDistance(0);
        vehicle.setRailEdgeId("");
        vehicle.setCommandId("");

        if (StringUtils.isNotEmpty(vehicle.getLastUdpState().railEdgeId)) {
            RailEdge lastRailEdge = DataService.getDataSet().getRailEdgeMap().get(vehicle.getLastUdpState().railEdgeId);
            lastRailEdge.removeVhlId(vehicle.getId());
            lastRailEdge.addHistory();
        }
    }

    /**
     * vehicle 정보의 업데이트 & udp 메세지 를 통해 railEdge 정보 획득
     *
     * @param tokens  Data obtained by udp message
     * @param vehicle vehicle to reflect message information
     */
    private void _updateVehicle(String[] tokens, Vhl vehicle, long systemsDateTime) {
        RailEdge railEdge;
        VHL_DET_STATE detailStatus = VHL_DET_STATE.getValue(Util.getTokenSafely(tokens, VHL_STATE_REPORT.DET_STATUS_IDX, ""));
        int address = Util.getIntOrZero(tokens[VHL_STATE_REPORT.ADDRESS_IDX]);
        String railNodeId = address != 0
                ? DataSet.address2RailNodeId(this.fabId, this.mcpName, address)
                : "";
        int nextAddress = Util.getIntOrZero(tokens[VHL_STATE_REPORT.NEXT_ADDRESS_IDX]);
        String nextRailNodeId = nextAddress != 0
                ? DataSet.address2RailNodeId(this.fabId, this.mcpName, nextAddress)
                : "";
        String destinationPortId = Util.getTokenSafely(tokens, VHL_STATE_REPORT.DEST_PORT_IDX, "");
        String railEdgeId = DataSet.address2RailEdgeId(
                this.fabId,
                this.mcpName,
                railNodeId,
                nextRailNodeId
        );
//        String carrierId = StringUtils.isNotEmpty(tokens[VHL_STATE_REPORT.CARRIER_ID_IDX])
//                ? DataSet.CARRIER_PREFIX + ":" + tokens[VHL_STATE_REPORT.CARRIER_ID_IDX]
//                : "";
//        String destinationStationId = StringUtils.isNotEmpty(tokens[VHL_STATE_REPORT.DESTINATION_IDX])
//                ? this.fabId + ":" + DataSet.STATION_PREFIX + ":" + mcpName + ":" + String.format("%05d", Util.getIntOrZero(tokens[VHL_STATE_REPORT.DESTINATION_IDX]))
//                : "";
        double distance = Util.getDoubleOrZero(StringUtils.isNotEmpty(tokens[VHL_STATE_REPORT.DISTANCE_IDX])
                ? tokens[VHL_STATE_REPORT.DISTANCE_IDX]
                : ""
        ) * 100L;
        String errorCode = Util.getTokenSafely(tokens, VHL_STATE_REPORT.ERROR_CODE_IDX, "");

//        boolean isFull = 0 < Util.getIntOrZero(tokens[VHL_STATE_REPORT.FULL_IDX]);
//        boolean isOnline = "1".equals(tokens[VHL_STATE_REPORT.ONLINE_IDX]);
        RUN_CYCLE runCycle = RUN_CYCLE.getValue(tokens[VHL_STATE_REPORT.RUN_CYCLE_IDX]);
        VHL_CYCLE vhlCycle = VHL_CYCLE.getValue(tokens[VHL_STATE_REPORT.VHL_CYCLE_IDX]);
        VHL_STATE vhlState = VHL_STATE.getValue(tokens[VHL_STATE_REPORT.STATE_IDX]);
//        String sourcePortId = StringUtils.isNotEmpty(tokens[VHL_STATE_REPORT.SOURCE_PORT_IDX])
//                ? DataService.getDataSet().getCarrierContainableByCarrierLoc(tokens[VHL_STATE_REPORT.SOURCE_PORT_IDX], fabId).getId()
//                : "";
//        String destinationPortId = StringUtils.isNotEmpty(tokens[VHL_STATE_REPORT.DEST_PORT_IDX])
//                ? DataService.getDataSet().getCarrierContainableByCarrierLoc(tokens[VHL_STATE_REPORT.DEST_PORT_IDX], fabId).getId()
//                : "";
//        byte emStatus = Util.binaryStringToByte(tokens[VHL_STATE_REPORT.EM_STATUS_IDX]);
//        String groupId = tokens[VHL_STATE_REPORT.GROUP_ID_IDX];
//        int priority = StringUtils.isNotEmpty(tokens[VHL_STATE_REPORT.PRIORITY_IDX])
//                ? Util.getIntOrZero(tokens[VHL_STATE_REPORT.PRIORITY_IDX])
//                : -1;
        long runDistance = Long.parseLong(tokens[VHL_STATE_REPORT.RUN_DISTANCE_IDX]);

        // setter
        vehicle.copyCurrentVhlUdpStateToLast();
        vehicle.setCurrentAddress(address);
        vehicle.setNextAddress(nextAddress);
        vehicle.setRailNodeId(railNodeId);
//        vehicle.setUdpCarrierId(carrierId);
//        vehicle.setDestStationId(destinationStationId);
        vehicle.setDistance(distance);
        vehicle.setErrorCode(errorCode);
//        vehicle.setFull(isFull);
//        vehicle.setOnline(isOnline);
        vehicle.setNextRailNodeId(nextRailNodeId);
        vehicle.setRunCycle(runCycle);
        vehicle.setVhlCycle(vhlCycle);
        vehicle.setState(vhlState);
        vehicle.setReceivedTime(receivedMilli);
//        vehicle.setEmStatus(emStatus);
//        vehicle.setGroupId(groupId);
//        vehicle.setSourcePortId(sourcePortId);
        vehicle.setDestPortId(destinationPortId);
//        vehicle.setPriority(priority);
        vehicle.setDetailState(detailStatus);
        vehicle.setRunDistance(runDistance);
        vehicle.setRailEdgeId(railEdgeId);
        //~setter

        ConcurrentMap<String, AbstractEdge> edgeMap = DataService.getDataSet().getEdgeMap();

        if (edgeMap.get(railEdgeId) instanceof RailEdge) {
            railEdge = (RailEdge) edgeMap.get(railEdgeId);

            this._buildRailVelocity(vehicle, railEdge);
        } else {
            logger.error("... `railEdgeId` selected is not supported [rail edge id: {}]", railEdgeId);

            return;
        }

        int hidId = railEdge.getHIDId();
        String machineId = vehicle.getName();
        String requiredKey = this.fabId + ":" + this.mcpName;
        String hidOffKey = requiredKey + ":" + String.format("%03d", hidId);
        String machineKey = requiredKey + ":" + machineId;
        List<Map<String, String>> messageDataList = new ArrayList<>();
        FunctionItem functionItem = Env.getSwitchMap().get(requiredKey);

        // HID IN/OUT 엣지 집계
        if (functionItem.getUseFunction(FunctionType.HID_INOUT)) {
            this._processHidInout(hidId, vehicle, functionItem);
        }
        //~HID IN/OUT 엣지 집계
        
        // HID 구간 별 VHL 수 계산
        if (functionItem.getUseFunction(FunctionType.VHL_CNT)) {
            this._calculatedVhlCnt(
                    hidId,
                    requiredKey,
                    vehicle,
                    functionItem
            );
        }
        //~HID 구간 별 VHL 수 계산
        
        // Stage Command Monitoring
        if (functionItem.getUseFunction(FunctionType.MAP_FILE_REFRESH)) {
            this._processStageCommandMonitoring(
                    detailStatus,
                    machineKey,
                    machineId,
                    destinationPortId,
                    systemsDateTime
            );
        }
        //~Stage Command Monitoring

        // HIDOFF
        if (functionItem.getUseFunction(FunctionType.HID_OFF)) {
            messageDataList.add(
                    this._processHidOff(
                            hidId,
                            hidOffKey,
                            errorCode,
                            address,
                            nextAddress,
                            systemsDateTime
                    )
            );
        }
        //~HIDOFF

        // VHLOFF
        if (functionItem.getUseFunction(FunctionType.VHL_OFF)) {
            messageDataList.addAll(
                    this._processVhlOff(
                            machineKey,
                            vehicle,
                            errorCode,
                            address,
                            nextAddress,
                            railEdge,
                            systemsDateTime
                    )
            );
        }
        //~VHLOFF

        // HIDOFF & VHLOFF Tibrv 송신
        if (!messageDataList.isEmpty()) {
            for (String tibrvKey : DataService.getInstance().getTibrvSenderLikeMap(fabId + ":send:").keySet()) {
                // 위의 과정을 통해 구성한 Map 데이터로 tib/rv 메세지를 만든 후 송신
                for (Map<String, String> messageData : messageDataList) {
                    if (messageData != null && !messageData.isEmpty()) {
                        String type = messageData.get(LayoutUtil.LAYOUT_MEMBER.DEVICE_TYPE);

                        if (type == null) continue;

                        DataService.getInstance().addTibrvMessageQueue(tibrvKey, type, messageData);
                    }
                }
            }
        }

    }

    /*
     Stage Command Monitoring
     `작업 상태 상세` 값이 103 인 경우 적재 <-> 그외 치유 혹은 생략
     alarm 발생 조건에 대한 논의가 필요
    */
    private void _processStageCommandMonitoring(
            VHL_DET_STATE detailState,
            String key,
            String machineId,
            String portId,
            long systemsDateTime
    ) {
        ConcurrentMap<String, StageCommandRecordItem> recordMap = DataService.getDataSet().getStageCommandMap();
        String deviceId = "";

        if (recordMap.containsKey(key)) {
            StageCommandRecordItem recordItem = recordMap.get(key);

            if (detailState.equals(VHL_DET_STATE.STAGE_MOVING)) {
                recordItem.setState(OHT_TIB_STATE.ABNORMAL);
                recordItem.setEventDateTime(systemsDateTime);
                recordItem.setDestinationPortId(portId);
            } else {
                recordItem.setState(OHT_TIB_STATE.NORMAL);  // 조건에서 벗어난 것을 표시하기 위함
            }
        } else {
            if (detailState.equals(VHL_DET_STATE.STAGE_MOVING)) {
                StageCommandRecordItem recordItem = new StageCommandRecordItem(
                        key,
                        fabId,
                        mcpName,
                        facId,
                        deviceId,
                        machineId,
                        portId,
                        systemsDateTime
                );

                recordMap.put(key, recordItem);
            }
        }
    }

    // ========================================================================================
    // [수정] _calculatedVhlCnt() — 기존 코드 유지 + 엣지 집계 추가
    // ========================================================================================

    /**
     * HID 구간별 VHL 재적수
     * @param currentHidId 현재 vehicle 이 위치한 railEdge 의 hid 값
     * @param key DataSet 에서 특정 데이터를 호출하기 위한 key 값
     * @param vehicle vehicle 객체
     */
    private void _calculatedVhlCnt(int currentHidId, String key, Vhl vehicle, FunctionItem functionItem) {
        long timer = System.currentTimeMillis();
        int previousHidId = vehicle.getHidId();

        if (previousHidId != currentHidId) {
            if (currentHidId > 0) {
                String v = String.format("%03d", currentHidId);
                DataService.getDataSet().increaseHidVehicleCnt(key + ":" + v);
            }

            if (previousHidId > 0) {
                String v = String.format("%03d", previousHidId);
                DataService.getDataSet().decreaseHidVehicleCnt(key + ":" + v);
            }

            vehicle.setHidId(currentHidId);
        }

        long checkingTime = System.currentTimeMillis() - timer;

        if (checkingTime >= 60000) {
            logger.info("... `number of vehicles per hid section` process took more than 1 minute to complete [elapsed time: {}min]", checkingTime / 60000);
        }
    }
    //~HID 구간별 VHL 재적수

    // ========================================================================================
    // HID IN/OUT 엣지 집계 (HID_INOUT 스위치 전용)
    // ========================================================================================

    /**
     * HID IN/OUT 엣지 전환 카운트 집계, 1분 배치 플러시, 하루 1회 마스터 테이블 업데이트
     */
    private void _processHidInout(int currentHidId, Vhl vehicle, FunctionItem functionItem) {
        int previousHidId = vehicle.getHidId();
        
        // 엣지 전환 카운트 집계 → 테이블 3
        if (previousHidId != currentHidId) {
            String vhlIdFull = vehicle.getId();
            String vhlName = vhlIdFull.substring(vhlIdFull.lastIndexOf(':') + 1);
            String eqpIdFull = vehicle.getEqpId();
            String eqpName = eqpIdFull.substring(eqpIdFull.lastIndexOf(':') + 1);
            String edgeKey = String.format("%03d:%03d:%s:%s:%s:%s:%s",
                    previousHidId, currentHidId, this.fabId, this.mcpName,
                    vehicle.getFabId(), vhlName, eqpName);
            
            int transCnt = DataService.getDataSet().getEdgeInOutCountMap()
            		.merge(edgeKey, 1, Integer::sum);
            
            // add tib                        
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:00");
            SimpleDateFormat dateOnlyFormat = new SimpleDateFormat("yyyy-MM-dd");
            Date now = new Date();
            String eventDt = dateFormat.format(now);
            String eventDate = dateOnlyFormat.format(now);
            String type = MSG_TYP.OHT.toString() + ".HID.INOUT";
            Map<String, Object> dataMap = new HashMap<>();
            
            dataMap.put("TYPE", type);
            dataMap.put("FAB_ID", this.fabId);
            dataMap.put("EVENT_DT", eventDt);
            dataMap.put("EVENT_DATE", eventDate);  
            dataMap.put("FROM_HIDID", previousHidId);  
            dataMap.put("TO_HIDID", currentHidId);  
            dataMap.put("VHL_ID", vhlName);  
            dataMap.put("EQP_ID", eqpName);
            dataMap.put("TRANS_CNT", transCnt);
            dataMap.put("MCP_NM", this.mcpName);
            dataMap.put("ENV", Env.getEnv());

            // VHL_COUNT_LIMIT, VHL_PRECAUTION → RawHid (layout.xml VEHICLE_MAX, VEHICLE_PRECAUTION)
            int vhlCountLimit = 0;
            int vhlPrecaution = 0;
            McpProperties mcpProperties = DataService.getInstance().getFabPropertiesMap().get(this.fabId).getMcpPropertiesMap().get(this.mcpName);
            if (mcpProperties != null && mcpProperties.getMcp75Config() != null) {
                for (RawHid rawHid : mcpProperties.getMcp75Config().getRawHidMap().values()) {
                    if (rawHid.getId() == currentHidId) {
                        vhlCountLimit = rawHid.getVhlMax();
                        vhlPrecaution = rawHid.getVhlPreCaution();
                        break;
                    }
                }
            }
            dataMap.put("VHL_COUNT_LIMIT", vhlCountLimit);
            dataMap.put("VHL_PRECAUTION", vhlPrecaution);

            for (String tibrvKey : DataService.getInstance().getTibrvSenderLikeMap(fabId + ":send:amos").keySet()) {
    			DataService.getInstance().addTibrvMessageQueue(
    					tibrvKey,
    					type,
    					dataMap
    			);
    		}
        }
    }

    /**
     * HID OFF
     * @param hidId 현재 hid 값
     * @param hidOffKey DataSet 에서 특정 데이터를 호출하기 위한 key 값
     * @param errorCode 오류 코드
     * @param currentAddress 현재 주소
     * @param nextAddress 다음 주소
     * @return normal/abnormal 인 경우, tib/rv message 를 반환
     */
    private Map<String, String> _processHidOff(
            int hidId,
            String hidOffKey,
            String errorCode,
            int currentAddress,
            int nextAddress,
            long systemsDateTime
    ) {
        long timer = System.currentTimeMillis();

        if (hidId < 0) return new HashMap<>();

        Map<String, String> dataMap = new HashMap<>();
        ConcurrentMap<String, HidOffRecordItem> hidOffRecordMap = DataService.getDataSet().getHidOffRecordMap(); // HID OFF 에 대한 기록 호출, 참조
        HidOffRecordItem recordItem;
        ConcurrentMap<String, List<String>> errorCodeList = DataService.getInstance().getOhtAlarmCodeListMap();

        if (errorCodeList != null && errorCodeList.containsKey(FunctionType.HID_OFF.getKey())) {
            if (errorCodeList.get(FunctionType.HID_OFF.getKey()).contains(errorCode)) {
                logger.info("[HID OFF] The fault has occurred [fab: {} | mcp: {} | error code: {} | hid: {}]", this.fabId, this.mcpName, errorCode, hidId);

                Set<String> addressSet = this._getAddressSet(hidOffKey);
                ConcurrentMap<String, List<String>> hid2PortMap = DataService.getDataSet().getHid2PortMap();
                List<String> portList = hid2PortMap.getOrDefault(hidOffKey, Collections.emptyList());
                String deviceId = String.valueOf(hidId);
                String alarmCode = String.format("HID%03d", hidId);

                recordItem = new HidOffRecordItem(
                        hidOffKey,
                        this.fabId,
                        this.facId,
                        this.mcpName,
                        deviceId,
                        hidId,
                        currentAddress,
                        nextAddress,
                        addressSet,
                        portList,
                        OHT_TIB_STATE.ABNORMAL,
                        errorCode,
                        alarmCode,
                        systemsDateTime
                );

                if (!hidOffRecordMap.containsKey(hidOffKey)) {
                    hidOffRecordMap.put(hidOffKey, recordItem);    // 해당 hid 구간의 hidOff 현상이 회복된 경우를 대응
                    logger.info("[HID OFF] This fault was saved as a record on the server in case it was resolved [fab: {} | mcp: {} | hid: {}]", this.fabId, this.mcpName, hidId);
                } else {
                    logger.info("[HID OFF] This fault has occurred, but it has not been saved to the server [fab: {} | mcp: {} | hid: {}] - faults in the same hid area that occurred previously are not resolved !", this.fabId, this.mcpName, hidId);
                }
            } else if (hidOffRecordMap.containsKey(hidOffKey)) {
                // hidOff 현상이 해소된 경우 (ABNORMAL -> NORMAL)
                recordItem = hidOffRecordMap.get(hidOffKey);

                recordItem.setState(OHT_TIB_STATE.NORMAL);
            } else {
                return new HashMap<>();
            }

            dataMap = LayoutUtil.buildLayoutMessageDataMap(recordItem);

            if (_insertHidOffLogpresso(recordItem, systemsDateTime) && recordItem.getState().equals(OHT_TIB_STATE.NORMAL)) {
                hidOffRecordMap.remove(hidOffKey);

                logger.info("[HID OFF] The fault is recovered [fab: {} | mcp: {} | hid: {}]", this.fabId, this.mcpName, hidId);
            }
        } else {
            logger.warn("[HID OFF] Not exist error code for hid off [fab: {} | mcp: {} | hid: {}] !", this.fabId, this.mcpName, hidId);
        }

        long checkingTime = System.currentTimeMillis() - timer;

        // 소요된 시간이 1분 이상인 경우 로그 표시
        if (checkingTime >= 60000) {
            logger.info("... `HID OFF` process took more than 1 minute to complete [elapsed time: {}min]", checkingTime / 60000);
        }

        return dataMap;
    }

    private Set<String> _getAddressSet(String hidOffKey) {
        ConcurrentMap<String, RailEdge> railEdgeMap = DataService.getDataSet().getRailEdgeMap();
        ConcurrentMap<String, List<String>> railEdge4HidMap = DataService.getDataSet().getRailEdge4HidMap();
        List<String> addressList = new ArrayList<>();

        try {
            for (String railEdgeId : railEdge4HidMap.get(hidOffKey)) {
                if (railEdgeMap.containsKey(railEdgeId)) {
                    RailEdge railEdge = railEdgeMap.get(railEdgeId);

                    addressList.add(railEdge.getAddress());
                }
            }
        } catch (Exception e) {
            logger.error("", e);
        }

        return Set.copyOf(addressList);
    }

    private boolean _insertHidOffLogpresso(HidOffRecordItem recordItem, long currentMilli) {
        Tuple tuple = new Tuple();
        String state = recordItem.getState();

        tuple.put("FAB_ID", recordItem.getFabId());
        tuple.put("MCP_NM", recordItem.getMcpName());
        tuple.put("ALARM_CD", recordItem.getErrorCode());
        tuple.put("EVENT_DT", recordItem.getEventDateTimeString());
        tuple.put("HID_ID", recordItem.getHidId());
        tuple.put("ADDR_LST", recordItem.getHidAreaAddressString());
        tuple.put("PORT_LST", recordItem.getAffectedPortString());
        tuple.put("ENV", Env.getEnv());
        tuple.put("STATE", state);

        if (state.equals(OHT_TIB_STATE.NORMAL)) {
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
            tuple.put("RECOVERY_DT", dateFormat.format(new Date(currentMilli)));
        }

        return LogpressoAPI.setInsertTuples("ATLAS_OHT_HID_OFF", List.of(tuple), 20);
    }
    //~HID OFF

    /**
     * VHL OFF
     * @param vehicle vehicle object <--- it had to set property by udp message
     * @param errorCode error code (ef. OhtVhlOffAlarmCodeList.txt)
     * @param currentAddress current address (=`from` rail node)
     * @param nextAddress next address (=`to` rail node)
     * @param railEdge railEdge object <--- it had made by udp message
     * @return give back the tib/rv message, when message state is normal/abnormal
     */
    private List<Map<String, String>> _processVhlOff(
            String vhlOffKey,
            Vhl vehicle,
            String errorCode,
            int currentAddress,
            int nextAddress,
            RailEdge railEdge,
            long systemsDateTime
    ) {
        long timer = System.currentTimeMillis();
        List<VhlOffRecordItem> bufferList = new ArrayList<>();
        List<Map<String, String>> dataList = new ArrayList<>();

        ConcurrentMap<String, VhlOffRecordItem> vhlOffRecordMap = DataService.getDataSet().getVhlOffRecordMap();
        ConcurrentMap<String, VhlOffRecordItem> vhlOffMonitoringMap = DataService.getDataSet().getVhlOffMonitoringMap();
        ConcurrentMap<String, List<String>> errorCodeList = DataService.getInstance().getOhtAlarmCodeListMap();
        String machineId = vehicle.getName();   // "V00001"
//        VHL_STATE vehicleState = vehicle.getState();
        String deviceId = machineId + ":" + currentAddress + ":" + errorCode;

        if (errorCodeList != null && errorCodeList.containsKey(FunctionType.VHL_OFF.getKey())) {
            VhlOffRecordItem temp = this._buildVhlOff(
                    vhlOffKey,
                    deviceId,
                    machineId,
                    currentAddress,
                    nextAddress,
                    errorCode,
                    railEdge,
                    errorCodeList,
                    systemsDateTime
            );

            if (vhlOffRecordMap.containsKey(vhlOffKey)) {
                /*
                    1. 새로운 device 명칭인 경우
                    2. 장애가 해소된 상태
                 */
                VhlOffRecordItem previousRecordItem = vhlOffRecordMap.get(vhlOffKey);

                if (previousRecordItem != null) {
                    if (temp == null) {
                        // 2. 장애가 해소된 상태
                        previousRecordItem.setState(OHT_TIB_STATE.NORMAL);
                        previousRecordItem.setRecoveryDateTime(systemsDateTime);

                        bufferList.add(previousRecordItem);

                        vhlOffMonitoringMap.put(vhlOffKey, previousRecordItem); //@

                        vhlOffRecordMap.remove(vhlOffKey);  //@
                    } else {
                        final String previousDeviceName = previousRecordItem.getDeviceId();

                        if (!temp.getDeviceId().equals(previousDeviceName)) {
                            // 1. 새로운 device 명칭인 경우
                            // 1-1. 최근에 기록된 정보를 처리
                            previousRecordItem.setState(OHT_TIB_STATE.NORMAL);
                            previousRecordItem.setRecoveryDateTime(systemsDateTime);

                            bufferList.add(previousRecordItem);
                            //~1-1
                            bufferList.add(temp);

                            vhlOffMonitoringMap.put(vhlOffKey, temp);

                            vhlOffRecordMap.put(vhlOffKey, temp);
                        }
                    }
                }
            } else if (temp != null) {
                /*
                    1. 새로운 장애 발생
                 */
                bufferList.add(temp);

                vhlOffMonitoringMap.put(vhlOffKey, temp);

                vhlOffRecordMap.put(vhlOffKey, temp);
            }

            for (VhlOffRecordItem bufferItem : bufferList) {
                dataList.add(LayoutUtil.buildLayoutMessageDataMap(bufferItem));
            }
        } else {
            logger.warn("[VHL OFF] Not exist error code for vhl off [fab: {} | mcp: {} | machine: {}] !", this.fabId, this.mcpName, machineId);
        }

        long checkingTime = System.currentTimeMillis() - timer;

        if (checkingTime >= 60000) {
            logger.info("... `VHL OFF` process took more than 1 minute to complete [elapsed time: {}min]", checkingTime / 60000);
        }

        return dataList;
    }

    /**
     *
     * @param vhlOffKey {fabId}:{mcpName}:{machineId}
     * @param deviceId {machineId}:{address}:{errorCode}
     * @param machineId ex) V00001
     * @param currentAddress ex) 1001
     * @param nextAddress ex) 1002
     * @param errorCode ex) 0000
     * @param railEdge current rail located
     * @param alarmCodeMap list of error code with `vhl off`
     * @param systemsDateTime occupation time
     * @return if return to null, it was normal state
     */
    private VhlOffRecordItem _buildVhlOff(
            String vhlOffKey,
            String deviceId,
            String machineId,
            int currentAddress,
            int nextAddress,
            String errorCode,
            RailEdge railEdge,
            ConcurrentMap<String, List<String>> alarmCodeMap,
            long systemsDateTime
    ) {
        if (alarmCodeMap.get(FunctionType.VHL_OFF.getKey()).contains(errorCode)) {
            logger.info("*[VHL OFF] defect(VHL OFF) has occurred [fab: {} | mcp: {} | error code: {} | machine: {}]", this.fabId, this.mcpName, errorCode, machineId);

            Navigator navigator     = new Navigator(railEdge);
            Set<String> addressSet  = navigator.getAffectedRailSet();
            List<String> portList   = navigator.getAffectedPortSortedList();

            return new VhlOffRecordItem(
                    vhlOffKey,
                    deviceId,
                    this.fabId,
                    this.facId,
                    this.mcpName,
                    machineId,
                    currentAddress,
                    nextAddress,
                    portList,
                    addressSet,
                    OHT_TIB_STATE.ABNORMAL,
                    errorCode,
                    systemsDateTime
            );
        }

        return null;
    }
    //~VHL OFF

    /**
     * 속도 값(velocity) 계산 전, 현재와 이전 데이터를 산출
     * @param vehicle updated vehicle data (reflected udp message)
     * @param railEdge current railEdge data (reflected udp message)
     */
    private void _buildRailVelocity(Vhl vehicle, RailEdge railEdge){
        RailEdge lastRailEdge = null;
        String vehicleId = vehicle.getId();
        String railEdgeId = railEdge.getId();
        String lastRailEdgeId = vehicle.getLastUdpState().railEdgeId;

        if (StringUtils.isNotEmpty(lastRailEdgeId)) {
            lastRailEdge = DataService.getDataSet().getRailEdgeMap().get(lastRailEdgeId);
            String fromNodeId = railEdge.getFromNodeId();
            String toNodeId = railEdge.getToNodeId();
            String lastFromNodeId = lastRailEdge.getFromNodeId();
            String lastToNodeId = lastRailEdge.getToNodeId();

            if (lastFromNodeId.equals(fromNodeId) && !lastToNodeId.equals(toNodeId)) {
                lastRailEdge.removeVhlId(vehicleId);

                vehicle.getLastUdpState().railEdgeId = railEdgeId;
                lastRailEdge = DataService.getDataSet().getRailEdgeMap().get(railEdgeId);

                if (!lastRailEdgeId.equals(vehicle.getRailEdgeId())) {
                    lastRailEdge.removeVhlId(vehicleId);
                }
            }
        }

        this._setRailEdgeVelocity(vehicle, railEdge, lastRailEdge);
    }

    private void _setRailEdgeVelocity(
            Vhl vehicle,
            RailEdge railEdge,
            RailEdge lastRailEdge
    ) {
        String vehicleId = vehicle.getId();

        if (
                lastRailEdge != null
                        && lastRailEdge.getToNodeId().equals(railEdge.getFromNodeId())
        ) {
            if (this._checkVehicleMovement(vehicle)) {
                double ran_distance = lastRailEdge.getLength() - vehicle.getLastUdpState().distance + vehicle.getDistance();
                long elapsed = vehicle.getReceivedTime() - vehicle.getLastUdpState().receivedTime;
                double speed = ran_distance / (double)elapsed * 60.0;

                lastRailEdge.addVelocity(speed);
            }

            lastRailEdge.addHistory();
            lastRailEdge.getVhlIdMap().remove(vehicleId);

            railEdge.addVhlId(vehicleId);
        } else if (lastRailEdge != null && !railEdge.getId().equals(lastRailEdge.getId())) {
            lastRailEdge.addHistory();
            lastRailEdge.getVhlIdMap().remove(vehicleId);

            railEdge.addVhlId(vehicleId);

            if (this._checkVehicleMovement(vehicle)) {
                ConcurrentLinkedQueue<RailEdge> predictedEdges;
                RailNode sourceNode = (RailNode) DataService.getDataSet().getNodeMap().get(lastRailEdge.getFromNodeId());
                RailNode destinationNode = (RailNode) DataService.getDataSet().getNodeMap().get(railEdge.getFromNodeId());
                predictedEdges = new DijkstraVhlRouteFind(vehicle, sourceNode, destinationNode).getRailEdgeList();

                if(!predictedEdges.isEmpty()) {
                    double currentSumSpeed = this._getCurrentSumSpeed(vehicle, predictedEdges);

                    for (RailEdge predictedEdge : predictedEdges) {
                        if(!lastRailEdge.getId().equals(predictedEdge.getId())) {
                            predictedEdge.addHistory();
                        }

                        predictedEdge.addVelocity(currentSumSpeed);
                    }
                }
            } else {
                ConcurrentLinkedQueue<RailEdge> predictedEdges;
                RailNode source = (RailNode)DataService.getDataSet().getNodeMap().get(lastRailEdge.getFromNodeId());
                RailNode dest = (RailNode)DataService.getDataSet().getNodeMap().get(railEdge.getFromNodeId());
                predictedEdges = new DijkstraVhlRouteFind(vehicle, source, dest).getRailEdgeList();

                if (!predictedEdges.isEmpty()) {
                    for(RailEdge pre : predictedEdges) {
                        pre.addHistory();
                    }
                }
            }

        } else {
            if(this._checkVehicleMovement(vehicle)) {
                //속도 계산시 이동 하지 않은 vehicle 에 대해 추후 이동시 실제 속도를 반영 하기 위함.
                vehicle.setReceivedTime(vehicle.getLastUdpState().receivedTime);
                vehicle.setDistance(vehicle.getLastUdpState().distance);
            }

            railEdge.addVhlId(vehicleId);
        }
    }

    private double _getCurrentSumSpeed(Vhl vehicle, ConcurrentLinkedQueue<RailEdge> predictedEdges) {
        double distanceSum = 0;
        long lastReceivedMilli = vehicle.getLastUdpState().receivedTime;
        long totalElapsedMilli = vehicle.getReceivedTime() - lastReceivedMilli;

        for(RailEdge pre : predictedEdges) {
            distanceSum += pre.getLength();
        }

        distanceSum -= vehicle.getLastUdpState().distance;
        distanceSum += vehicle.getDistance();

        return distanceSum / (double)totalElapsedMilli * 60.0;
    }

    private boolean _checkVehicleMovement(Vhl vehicle) {
        return (vehicle.getReceivedTime() - vehicle.getLastUdpState().receivedTime) < (60 * 1000)
                && (
                VHL_STATE.RUN == vehicle.getState()
                        || VHL_STATE.OBS_BZ_STOP == vehicle.getState()
                        || VHL_STATE.JAM == vehicle.getState()
                        || VHL_STATE.E84_TIMEOUT == vehicle.getState()
        ) && vehicle.getRunCycle() == vehicle.getLastUdpState().runCycle
                && vehicle.getVhlCycle() == vehicle.getLastUdpState().vhlCycle
                && (
                RUN_CYCLE.ACQUIRE == vehicle.getRunCycle()
                        || RUN_CYCLE.DEPOSIT == vehicle.getRunCycle()
        ) && (
                VHL_CYCLE.ACQUIRE_MOVING == vehicle.getVhlCycle()
                        || VHL_CYCLE.DEPOSIT_MOVING == vehicle.getVhlCycle()
        );
    }

    private static class MSG_ID {
        public static final String MCP_ONLINE_REPORT = "1";
        public static final String VHL_STATE_REPORT = "2";
        public static final String STATION_STATE_REPORT = "3";
        public static final String MACHINE_STATE_REPORT = "4";
        public static final String MCP7_RESTART_REPORT = "5";
        public static final String POWER_STATE_REPORT = "13";
        public static final String POWER_STATE_HISTORY_REPORT = "14";
        public static final String VHL_ROUTE_REPORT = "15";
        public static final String STATE_REQUEST = "51";
    }

    public static class VHL_STATE_REPORT {
        public static final int TXT_ID_IDX = 0;             // 텍스트 id
        public static final int MCP_NM_IDX = 1;             // mcp 명칭
        public static final int VHL_ID_IDX = 2;             // vehicle 명
        public static final int STATE_IDX = 3;              // 상태
        public static final int FULL_IDX = 4;               // 재하 정보
        public static final int ERROR_CODE_IDX = 5;         // error code
        public static final int ONLINE_IDX = 6;             // 통신 상태
        public static final int ADDRESS_IDX = 7;            // 현재 번지
        public static final int DISTANCE_IDX = 8;           // 현재 번지로부터의 거리
        public static final int NEXT_ADDRESS_IDX = 9;       // 다음 번지
        public static final int RUN_CYCLE_IDX = 10;         // 실행 cycle
        public static final int VHL_CYCLE_IDX = 11;         // vehicle 실행 cycle 진척
        public static final int CARRIER_ID_IDX = 12;        // carrier id
        public static final int DESTINATION_IDX = 13;       // destination
        public static final int EM_STATUS_IDX = 14;         // e/m 상태
        public static final int GROUP_ID_IDX = 15;          // group id
        public static final int SOURCE_PORT_IDX = 16;       // 반송원 port
        public static final int DEST_PORT_IDX = 17;         // 반송처 port
        public static final int PRIORITY_IDX = 18;          // 반송 우선도
        public static final int DET_STATUS_IDX = 19;        // 작업 상태 상세
        public static final int RUN_DISTANCE_IDX = 20;      // 대차 주행거리
        public static final int CMD_ID_IDX = 21;            // command id
        public static final int BAY_NM_IDX = 22;            // bay 명칭
    }

    // 상태값
    public static class OHT_TIB_STATE {
        public static final String NORMAL = "NORMAL";
        public static final String ABNORMAL = "ABNORMAL";

        public static List<String> getStates() {
            List<String> states = new ArrayList<>();

            states.add(NORMAL);
            states.add(ABNORMAL);

            return states;
        }
    }
}
