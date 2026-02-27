/**
 * develop history
 * ...
 * 2025-11-27 변수에 대한 스위치 추가(absoluteVelocity, maxVelocity, passCnt, vhlCnt)
 * 2026-02-27 HID_INOUT 스위치 추가
 */
public class FunctionItem {

    private final Logger logger = LoggerFactory.getLogger(getClass());

    private final String fabId;
    private final String mcpName;

    private Boolean useHidOff = null;
    private Boolean useHidInout = null;
    private Boolean useVhlOff = null;
    private Boolean useRailCut = null;
    private Boolean useMapFileRefresh = null;
    private Boolean useVhlCnt = null;
    private Boolean useVhlCnt10 = null;
    private Boolean useVhlCnt30 = null;
    private Boolean useVhlCnt60 = null;
    private Boolean useStageCommandMonitoring = null;
    private Boolean useUdpMessageMonitoring = null;
    private Boolean useRailVibration = null;
    // TrafficBatch.class
    private Boolean useRailTraffic = null;
    private Boolean useRailTrafficSub = null;
    private Boolean useRailTrafficAbsoluteVelocity = null;
    private Boolean useRailTrafficMaxVelocity = null;
    private Boolean useRailTrafficPassCnt = null;
    private Boolean useRailTrafficVhlCnt = null;

    public FunctionItem(String fabId, String mcpName) {
        this.fabId      = fabId;
        this.mcpName    = mcpName;
    }

    public FunctionItem(String fabId, String mcpName, Map<FunctionType, Boolean> functionTypeMapper) {
        this.fabId      = fabId;
        this.mcpName    = mcpName;

        if (functionTypeMapper == null) {
            logger.error("... mapping data is null value, so composing `{}` construction exit [functionTypeMapper=null]", this.getClass().getSimpleName());
        } else {
            for (Map.Entry<FunctionType, Boolean> entry : functionTypeMapper.entrySet()) {
                setUseFunction(entry.getKey(), entry.getValue());
            }
        }
    }

    public String getFabId() {
        return fabId;
    }

    public String getMcpName() {
        return mcpName;
    }

    public boolean isUseHidOff() {
        return useHidOff != null && useHidOff;
    }

    public boolean isUseHidInout() {
        return useHidInout != null && useHidInout;
    }

    public boolean isUseVhlOff() {
        return useVhlOff != null && useVhlOff;
    }

    public boolean isUseRailCut() {
        return useRailCut != null && useRailCut;
    }

    public boolean isUseMapFileRefresh() {
        return useMapFileRefresh != null && useMapFileRefresh;
    }

    public boolean isUseVhlCnt() {
        return useVhlCnt != null && useVhlCnt;
    }

    public boolean isUseStageCommandMonitoring() {
        return useStageCommandMonitoring != null && useStageCommandMonitoring;
    }

    public boolean isUseUdpMessageMonitoring() {
        return useUdpMessageMonitoring != null && useUdpMessageMonitoring;
    }

    public boolean isUseRailVibration() {
        return useRailVibration != null && useRailVibration;
    }

    public boolean isUseRailTraffic() {
        return useRailTraffic != null && useRailTraffic;
    }

    public boolean isUseRailTrafficSub() {
        return useRailTrafficSub != null && useRailTrafficSub;
    }

    public boolean isUseVhlCnt10() {
        return useVhlCnt10 != null && useVhlCnt10;
    }

    public boolean isUseVhlCnt30() {
        return useVhlCnt30 != null && useVhlCnt30;
    }

    public boolean isUseVhlCnt60() {
        return useVhlCnt60 != null && useVhlCnt60;
    }

    public boolean isUseRailTrafficAbsoluteVelocity() {
        return useRailTrafficAbsoluteVelocity != null && useRailTrafficAbsoluteVelocity;
    }

    public boolean isUseRailTrafficMaxVelocity() {
        return useRailTrafficMaxVelocity != null && useRailTrafficMaxVelocity;
    }

    public boolean isUseRailTrafficPassCnt() {
        return useRailTrafficPassCnt != null && useRailTrafficPassCnt;
    }

    public boolean isUseRailTrafficVhlCnt() {
        return useRailTrafficVhlCnt != null && useRailTrafficVhlCnt;
    }

    public void setUseFunction(FunctionType key, Boolean isAvailable) {
        switch (key) {
            case HID_OFF: {
                this.useHidOff = isAvailable;
            } break;
            case HID_INOUT: {
                this.useHidInout = isAvailable;
            } break;
            case VHL_OFF: {
                this.useVhlOff = isAvailable;
            } break;
            case RAIL_CUT: {
                this.useRailCut = isAvailable;
            } break;
            case MAP_FILE_REFRESH: {
                this.useMapFileRefresh = isAvailable;
            } break;
            case VHL_CNT: {
                this.useVhlCnt = isAvailable;
            } break;
            case VHL_CNT_10: {
                this.useVhlCnt10 = isAvailable;
            } break;
            case VHL_CNT_30: {
                this.useVhlCnt30 = isAvailable;
            } break;
            case VHL_CNT_60: {
                this.useVhlCnt60 = isAvailable;
            } break;
            case RAIL_VIBRATION: {
                this.useRailVibration = isAvailable;
            } break;
            case UDP_MESSAGE_MONITORING: {
                this.useUdpMessageMonitoring = isAvailable;
            } break;
            case STAGE_COMMAND_MONITORING: {
                this.useStageCommandMonitoring = isAvailable;
            } break;
            case RAIL_TRAFFIC: {
                this.useRailTraffic = isAvailable;
            } break;
            case RAIL_TRAFFIC_SUB: {
                this.useRailTrafficSub = isAvailable;
            } break;
            case RAIL_TRAFFIC_ABSOLUTE_VELOCITY: {
                this.useRailTrafficAbsoluteVelocity = isAvailable;
            } break;
            case RAIL_TRAFFIC_MAX_VELOCITY: {
                this.useRailTrafficMaxVelocity = isAvailable;
            } break;
            case RAIL_TRAFFIC_PASS_COUNT: {
                this.useRailTrafficPassCnt = isAvailable;
            }
            case RAIL_TRAFFIC_VEHICLE_COUNT: {
                this.useRailTrafficVhlCnt = isAvailable;
            } break;
            default: {
                this._notRegistered(key);
            } break;
        }
    }

    public boolean getUseFunction(FunctionType key) {
        return getUseFunction(key, false);
    }

    public Boolean getUseFunction(FunctionType key, boolean nullable) {
        switch (key) {
            case HID_OFF: {
                if (nullable) {
                    return useHidOff;
                } else {
                    return isUseHidOff();
                }
            }
            case HID_INOUT: {
                if (nullable) {
                    return useHidInout;
                } else {
                    return isUseHidInout();
                }
            }
            case VHL_OFF: {
                if (nullable) {
                    return useVhlOff;
                } else {
                    return isUseVhlOff();
                }
            }
            case RAIL_CUT: {
                if (nullable) {
                    return useRailCut;
                } else {
                    return isUseRailCut();
                }
            }
            case MAP_FILE_REFRESH: {
                if (nullable) {
                    return useMapFileRefresh;
                } else {
                    return isUseMapFileRefresh();
                }
            }
            case VHL_CNT: {
                if (nullable) {
                    return useVhlCnt;
                } else {
                    return isUseVhlCnt();
                }
            }
            case VHL_CNT_10: {
                if (nullable) {
                    return useVhlCnt10;
                } else {
                    return isUseVhlCnt10();
                }
            }
            case VHL_CNT_30: {
                if (nullable) {
                    return useVhlCnt30;
                } else {
                    return isUseVhlCnt30();
                }
            }
            case VHL_CNT_60: {
                if (nullable) {
                    return useVhlCnt60;
                } else {
                    return isUseVhlCnt60();
                }
            }
            case RAIL_VIBRATION: {
                if (nullable) {
                    return useRailVibration;
                } else {
                    return isUseRailVibration();
                }
            }
            case UDP_MESSAGE_MONITORING: {
                if (nullable) {
                    return useUdpMessageMonitoring;
                } else {
                    return isUseUdpMessageMonitoring();
                }
            }
            case STAGE_COMMAND_MONITORING: {
                if (nullable) {
                    return useStageCommandMonitoring;
                } else {
                    return isUseStageCommandMonitoring();
                }
            }
            case RAIL_TRAFFIC: {
                if (nullable) {
                    return useRailTraffic;
                } else {
                    return isUseRailTraffic();
                }
            }
            case RAIL_TRAFFIC_SUB: {
                if (nullable) {
                    return useRailTrafficSub;
                } else {
                    return isUseRailTrafficSub();
                }
            }
            case RAIL_TRAFFIC_ABSOLUTE_VELOCITY: {
                if (nullable) {
                    return useRailTrafficAbsoluteVelocity;
                } else {
                    return isUseRailTrafficAbsoluteVelocity();
                }
            }
            case RAIL_TRAFFIC_MAX_VELOCITY: {
                if (nullable) {
                    return useRailTrafficMaxVelocity;
                } else {
                    return isUseRailTrafficMaxVelocity();
                }
            }
            case RAIL_TRAFFIC_PASS_COUNT: {
                if (nullable) {
                    return useRailTrafficPassCnt;
                } else {
                    return isUseRailTrafficPassCnt();
                }
            }
            case RAIL_TRAFFIC_VEHICLE_COUNT: {
                if (nullable) {
                    return useRailTrafficVhlCnt;
                } else {
                    return isUseRailTrafficVhlCnt();
                }
            }
            default: {
                this._notRegistered(key);
                return false;
            }
        }
    }

    private void _notRegistered(FunctionType functionType) {
        logger.error("... entered function type is not registered [key: {}] [fab: {} | mcp: {}]", functionType.getKey(), fabId, mcpName);
    }

    public enum FunctionType {
        HID_OFF("HID_OFF"),
        HID_INOUT("HID_INOUT"),
        VHL_OFF("VHL_OFF"),
        RAIL_CUT("RAIL_CUT"),
        MAP_FILE_REFRESH("MAP_FILE_REFRESH"),
        VHL_CNT("VHL_CNT"),
        VHL_CNT_10("VHL_CNT_10"),
        VHL_CNT_30("VHL_CNT_30"),
        VHL_CNT_60("VHL_CNT_60"),
        STAGE_COMMAND_MONITORING("STAGE_COMMAND_MONITORING"),
        UDP_MESSAGE_MONITORING("UDP_MESSAGE_MONITORING"),
//        ITSM_SCHEDULE_MONITORING("ITSM_SCHEDULE_MONITORING"),
        RAIL_VIBRATION("RAIL_VIBRATION"),
        RAIL_TRAFFIC("RAIL_TRAFFIC"),
        RAIL_TRAFFIC_SUB("RAIL_TRAFFIC_SUB"),
        RAIL_TRAFFIC_ABSOLUTE_VELOCITY("RAIL_TRAFFIC_ABSOLUTE_VELOCITY"),
        RAIL_TRAFFIC_MAX_VELOCITY("RAIL_TRAFFIC_MAX_VELOCITY"),
        RAIL_TRAFFIC_PASS_COUNT("RAIL_TRAFFIC_PASS_COUNT"),
        RAIL_TRAFFIC_VEHICLE_COUNT("RAIL_TRAFFIC_VEHICLE_COUNT");

        private final String key;

        FunctionType(String key) {
            this.key = key;
        }

        public String getKey() {
            return key;
        }
    }
}
