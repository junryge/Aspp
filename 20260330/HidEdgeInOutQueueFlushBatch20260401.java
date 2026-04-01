public class HidEdgeInOutQueueFlushBatch implements Job {

	private static final Logger logger = LoggerFactory.getLogger(HidEdgeInOutQueueFlushBatch.class);
	
	@Override
	public void execute(JobExecutionContext context) throws JobExecutionException {
		if (DataService.getInstance() == null || !DataService.getInstance().getInitialized()) {
			return;
		}
		
		logger.info("HID Edge flush start");
		
		var copyMap = new HashMap<String, Integer>();
		
		DataService.getDataSet().getEdgeInOutCountMap().forEach((k, v) -> {
			copyMap.put(new String(k), v.intValue());
		});
		
		DataService.getDataSet().setEdgeInOutCountMap(new ConcurrentHashMap<>());
		
        logger.info("HID Edge flush copied: {}", copyMap.size());

        // 현재 시간 (1분 단위로 정렬)
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:00");
        SimpleDateFormat dateOnlyFormat = new SimpleDateFormat("yyyy-MM-dd");
        Date now = new Date();
        String eventDt = dateFormat.format(now);
        String eventDate = dateOnlyFormat.format(now);

        var fabIdTuples = new HashMap<String, List<Tuple>>();

        for (Map.Entry<String, Integer> entry : copyMap.entrySet()) {
            String[] parts = entry.getKey().split(":");
            var fromHidId = Integer.parseInt(parts[0]);
            var toHidId = Integer.parseInt(parts[1]);
            var fabId = parts[2];
            String mcpName = parts[3];
            String vhlFabId = parts[4];
            String vhlId = parts[5];
            String eqpId = parts[6];
            int transCnt = entry.getValue();

            Tuple tuple = new Tuple();
            tuple.put("EVENT_DATE", eventDate);
            tuple.put("EVENT_DT", eventDt);
            tuple.put("FROM_HIDID", fromHidId);
            tuple.put("TO_HIDID", toHidId);
            tuple.put("TRANS_CNT", transCnt);
            tuple.put("FAB_ID", vhlFabId);
            tuple.put("VHL_ID", vhlId);
            tuple.put("EQP_ID", eqpId);
            tuple.put("MCP_NM", mcpName);
            tuple.put("ENV", Env.getEnv());

            // FREE_FLOW_SPEED → 현재 HID 구간 RailEdge velocity 평균 (실시간 1분 구간 평균속도)
            double sumVelocity = 0.0;
            int velCount = 0;
            for (AbstractEdge ae : DataService.getDataSet().getEdgeMap().values()) {
                if (ae instanceof RailEdge) {
                    RailEdge re = (RailEdge) ae;
                    if (re.getHIDId() == toHidId && re.getVelocity() > 0) {
                        sumVelocity += re.getVelocity();
                        velCount++;
                    }
                }
            }
            double freeFlowSpeed = velCount > 0 ? sumVelocity / velCount : 0.0;
            tuple.put("FREE_FLOW_SPEED", freeFlowSpeed);

            // HID_VALUE → 현재 HID 대기 차량 수 (DataSet.hidVehicleCountMap)
            String hidKey = fabId + ":" + mcpName + ":" + String.format("%03d", toHidId);
            int hidValue = DataService.getDataSet().getHidVehicleCountMap().getOrDefault(hidKey, 0);
            tuple.put("HID_VALUE", hidValue);

            if (fabIdTuples.get(fabId) == null) {
            	fabIdTuples.put(fabId, new ArrayList<Tuple>());
            }
            
            fabIdTuples.get(fabId).add(tuple);
        }

        for (var entry : fabIdTuples.entrySet()) {
        	var fabId = entry.getKey();
        	var tuples = entry.getValue();
        	
            if (Strings.isBlank(fabId)) {
            	return;
            }
            
            // 테이블명: {FAB}_ATLAS_HID_INOUT (예: M14A_ATLAS_HID_INOUT)
            String tableName = fabId + "_ATLAS_HID_INOUT";

            boolean success = LogpressoAPI.setInsertTuples(tableName, tuples, 100);

            if (success) {
                logger.info("HID Edge flush: {} - {} records", tableName, tuples.size());
            }
        }
	}
	
}
