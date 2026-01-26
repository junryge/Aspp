public class BranchJoinEdge extends AbstractEdge {
    private ConcurrentLinkedQueue<String> railEdgeIds 	= new ConcurrentLinkedQueue<String>();
    private long cost 									= -1;
    private boolean isAvailable 						= false;
    private double velocity 							= -1;
    private double maxVelocity 							= -1;
	private int vhlCount 								= -1;
	
    public boolean changed(BranchJoinEdge oe) {
		if(Util.isContentsEqualsCollection(this.railEdgeIds, oe.railEdgeIds) == false) {
			return true;
		}
		
		return super.changed(oe);
	}
    
    public String getMcpName() {
    	return this.getFromNode().getMcpName(DataService.getDataSet());
    }
    
	public BranchJoinEdge(
							String fabId, 
							String id, 
							String fromNodeId, 
							String toNodeId, 
							EDGE_TYPE type, 
							double length, 
							ConcurrentLinkedQueue<String> railEdgeIds,
							boolean isUpdate
	) {
		super(fabId, id, fromNodeId, toNodeId, type, length, isUpdate);
		
		this.railEdgeIds.addAll(railEdgeIds);
		
	    this.batchFlush = true;	    
	}
	
	public long internalGetCost() {
		return this.cost;
	}
	
	@Override
	public long getCost(String carrierId) {
		long cost = 0;
		
		for (String reId : railEdgeIds) {
			RailEdge re = DataService.getDataSet().getRailEdgeMap().get(reId);
			
			cost 		+= re.getCost(carrierId);
		}
		
		this.cost = cost;
		
		return cost;
	}

	@Override
	public long getFutureCost(String carrierId, long after) {
		return 0;
	}

	@Override
	public int getFutureTransCount(String carrierId, long after) {
		return 0;
	}

	public boolean internalisAvailable() {
		return this.isAvailable;
	}
	
	@Override
	public boolean isAvailable() {
		for(String edgeId : railEdgeIds) {
			AbstractEdge ae = DataService.getDataSet().getEdgeMap().get(edgeId);
			if(ae.isAvailable() == false) {
				this.isAvailable = false;
				return false;
			}
		}
		this.isAvailable = true;
		return true;
	}
	
	@Override
	public boolean isAvailable(PROCESS_TYPE carrierType) {
		for(String edgeId : railEdgeIds) {
			AbstractEdge ae = DataService.getDataSet().getEdgeMap().get(edgeId);
			if(ae.isAvailable(carrierType) == false)
				return false;
		}
		return true;
	}
	
	public double internalGetVelocity() {
		return this.velocity;
	}
	
	public double getVelocity() {
		this.velocity = (double)this.getLength() / (double)this.getCost("") * 60.0;
		return this.velocity; 
	}
	
	public ConcurrentLinkedQueue<String> getRailEdgeIds() {
		return railEdgeIds;
	}

	public void setRailEdgeIds(ConcurrentLinkedQueue<String> railEdgeIds) {
		this.railEdgeIds = railEdgeIds;
	}
	
	public double internalGetMaxVelocity() {
		return this.maxVelocity;
	}
	
	public double getMaxVelocity() {
		if(maxVelocity < 0) {
			double mv = Double.MIN_VALUE;
			for(String railEdgeId : railEdgeIds) {
				RailEdge re = DataService.getDataSet().getRailEdgeMap().get(railEdgeId);
				if(re != null && re.getMaxVelocity() > mv) {
					mv = re.getMaxVelocity();
				}
			}
			maxVelocity = mv;
		}
		return maxVelocity;
	}
	
	public int internalGetVhlCount() {
		return this.vhlCount;
	}
	
	public int getVhlCount() {
		int cnt = 0;
		for(String reId : railEdgeIds) {
			RailEdge re = DataService.getDataSet().getRailEdgeMap().get(reId);
			cnt += re.getVhlIdMap().size();
		}
		this.vhlCount = cnt;
		return cnt;
	}
	
	public float getDensity() {
		float vhlLengthSum = 0;
		float vhlLength = 0;
		
		if(fabId.startsWith("M14")) {
			vhlLength = 784 + 300;
		}else if(fabId.startsWith("M16"))
			vhlLength = 943 + 300;
		else
			vhlLength = 784 + 300;
		int vhlCntSum = 0;
		
		for(String edgeId : railEdgeIds) {
			vhlCntSum += DataService.getDataSet().getRailEdgeMap().get(edgeId).getVhlIdMap().size();			
		}
		vhlLengthSum = vhlCntSum * vhlLength;
		
		float railLength = (float)getLength() - ((float)getLength() % vhlLength);
		if(railLength <= vhlLength) {
			railLength = vhlLength;
		}

		float returnValue = vhlLengthSum / ((float)railLength) * 100f;
		
		return returnValue>=100f?100f:returnValue;
	}
}