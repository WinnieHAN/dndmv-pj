package depparsing.constraints;

import gnu.trove.TIntArrayList;
import util.Alphabet;
import depparsing.data.DepCorpus;
import depparsing.data.DepInstance;

/**
 * An object for mapping between constraint types expressed as Parent-type, child-type and
 * edge IDs used by L1LMax.  Essentially a dictionary of constraints. 
 * @author kuzman
 *
 */
public class ConstraintEnumerator{
	private final PCType childType, parentType;
	public final boolean useRoot;
	public final boolean useDirection;
	private final DepCorpus c;
	private final Alphabet<String> types2indices;
	TIntArrayList[] parentIdsPerChild;
	TIntArrayList edge2childType;
	TIntArrayList edge2parentType;
	
	public ConstraintEnumerator(DepCorpus c, PCType childType, PCType parentType, boolean useRoot, boolean useDirection){
		this.c = c;
		this.childType = childType;
		this.parentType = parentType;
		this.useRoot = useRoot;
		this.useDirection = useDirection;
		this.types2indices = new Alphabet<String>();
		parentIdsPerChild = new TIntArrayList[numIdsChild()];
		edge2childType = new TIntArrayList();
		edge2parentType = new TIntArrayList();
		for (int child = 0; child < parentIdsPerChild.length; child++) {
			parentIdsPerChild[child] = new TIntArrayList();
		}
	}
	
	public int root2cid(DepInstance di, int rootIndex){
		if(!useRoot) return -1; 
		int childId = index2id(di, rootIndex, getChildType());
		String childName = getChildType().id2string(c, childId);
		int res = types2indices.lookupObject("root="+childName);
		if (edge2childType.size() <= res){
			assert edge2childType.size() == res && edge2parentType.size() == res;
			edge2childType.add(-2);
			edge2parentType.add(-2);
			edge2childType.set(res, childId);
			edge2parentType.set(res, -1);
		}
		return res;
	}
			
	public int edge2cid(DepInstance di, int child, int parent){
		String dir = "";
		if (useDirection) dir = child>parent? "right":"left";
		int childId = index2id(di,child,getChildType());
		int parentId = index2id(di,parent,getParentType());
		String childName = getChildType().id2string(c, childId);
		String parentName = getParentType().id2string(c, parentId);
		int res=types2indices.lookupObject("edge="+childName+","+parentName+":"+dir);
		if (!parentIdsPerChild[childId].contains(res)) parentIdsPerChild[childId].add(res);
		if (edge2childType.size() <= res){
			assert edge2childType.size() == res && edge2parentType.size() == res;
			edge2childType.add(-2);
			edge2parentType.add(-2);
			edge2childType.set(res, childId);
			edge2parentType.set(res, parentId);
		}
		return res;
	}
	
	/**
	 * Only gets the id of the edge if it's been observed before; otherwise it returns -1. 
	 * @param child
	 * @param parent
	 * @param dir should be "right" or "left"
	 * @return
	 */
	public int getEdgeId(String child, String parent, String dir){
		if (!useDirection) dir = "";
		String edgeName = "edge="+child+","+parent+":"+dir;
		if (!types2indices.feat2index.contains(edgeName)) return -1;
		return types2indices.lookupObject(edgeName);
	}
	
	/** 
	 * @return a list of indices corresponding to different edge types for each child type.
	 * used by the stats class {@code L1LMaxStats}. 
	 */
	public TIntArrayList[] cidAsMatrix(){
		return parentIdsPerChild;
	}

	public String constraint2string(int c){
		return types2indices.lookupIndex(c);
	}
	
	private int numIds(PCType t){
		switch (t) {
		case WORD: return c.getNrWordTypes();
		case TAG: return c.getNrTags();
		}
		throw new RuntimeException("unknwon tag type");
	}
	
	int numIdsChild(){
		return useDirection? numIds(getChildType()) : numIds(getChildType())*2;
	}
	
	int numIdsParent(){
		return useRoot? numIds(getParentType()) : numIds(getParentType())+1;
	}

	private int index2id(DepInstance di, int ind, PCType t){
		switch (t) {
		case WORD: return  di.words[ind];
		case TAG: return di.postags[ind];
		}
		throw new RuntimeException("unknwon tag type");
	}
	
	public PCType getChildType() {
		return childType;
	}

	public PCType getParentType() {
		return parentType;
	}
	
	public int getChildType(int edgeType){
		return edge2childType.get(edgeType);
	}
	
	public int getParentType(int edgeType){
		return edge2parentType.get(edgeType);
	}
}