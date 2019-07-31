package depparsing.constraints;

import gnu.trove.TIntArrayList;
import gnu.trove.TIntObjectHashMap;
import gnu.trove.TIntObjectIterator;

import java.io.IOException;
import java.util.ArrayList;

import data.WordInstance;
import depparsing.constraints.PCType;
import depparsing.data.DepCorpus;
import depparsing.data.DepInstance;
import depparsing.model.DepModel;

/**
 * 
 * @author kuzman
 * 
 * This version of {@link L1LMax} constraint groups the parents by their type. 
 * So, for example, the probability of "the second Determiner in the sentence" 
 * is dominated by different Nouns would be combined into a single mass.  
 * 
 */
public class GroupedL1LMax extends L1LMax {
	
	public GroupedL1LMax(DepCorpus corpus, DepModel model, ArrayList<WordInstance> toProject, PCType cType, PCType pType, 
			boolean useRoot, boolean useDirection, double constraintStrength, int minOccurrencesForProjection, String fileOfAllowedTypes) throws IOException{
		super(corpus, model, toProject, cType, pType, useRoot, useDirection, constraintStrength, minOccurrencesForProjection, fileOfAllowedTypes);
	}
	
	/**
	 * count the number of indices necessary for each child-parent type in the ragged array.  For Fernando 
	 * style constraints, this will be the number of 
	 * @param toProject
	 * @return
	 */
	public ArrayList<Integer> countIndicesForChildParentType(ArrayList<WordInstance> toProject){
		ArrayList<Integer> indicesforcp = new ArrayList<Integer>();

		// compute how many of each childType-parentType pair there are. 
		for (int s = 0; s < toProject.size(); s++) {
			DepInstance di = (DepInstance) toProject.get(s);
			for (int childIndex = 0; childIndex < di.numWords; childIndex++) {
				// reserve a space for each child being the root
				int roottype = cstraints.root2cid(di, childIndex);
				if (roottype >= 0){ 
					while (roottype >=indicesforcp.size()) indicesforcp.add(0);
					indicesforcp.set(roottype, 1+indicesforcp.get(roottype));
				}
				// group parents by their type; parsByEdgeType contains a map from 
				// a parent type (for the current parent) to a list of indices in the 
				// sentence that correspond to parents of that type. 
				TIntObjectHashMap<TIntArrayList> parsByEdgeType = new TIntObjectHashMap<TIntArrayList>();
				for (int parentIndex = 0; parentIndex < di.numWords; parentIndex++) {
					int edgetype = cstraints.edge2cid(di,childIndex, parentIndex);
					if (!parsByEdgeType.contains(edgetype)) parsByEdgeType.put(edgetype, new TIntArrayList());
					// we never actually get the edge types; 
					parsByEdgeType.get(edgetype).add(parentIndex);
				}
				for (TIntObjectIterator<TIntArrayList> itr = parsByEdgeType.iterator(); itr.hasNext();) {
					itr.advance();
					int edgetype = itr.key();
					while (edgetype >=indicesforcp.size()) indicesforcp.add(0);
					indicesforcp.set(edgetype, 1+indicesforcp.get(edgetype));
				}
			}
		}
		return indicesforcp;
	}

	public void makeEdge2SentenceChildParent(ArrayList<WordInstance> toProject, ArrayList<Integer> indicesforcp){
		// fill in the matrices
		for (int s = 0; s < toProject.size(); s++) {
			DepInstance di = (DepInstance) toProject.get(s);
			for (int childIndex = 0; childIndex < di.numWords; childIndex++) {
				// make an SentenceChildParent object for each child being the root
				int roottype = cstraints.root2cid(di, childIndex);
				if (roottype >= 0) {
					int index = indicesforcp.get(roottype);
					edge2scp[roottype][index] = new SentenceChildParent(s,childIndex,new int[] {-1});
					indicesforcp.set(roottype, 1+index);
				}
				// group parents by their type; parsByEdgeType contains a map from 
				// a parent type (for the current parent) to a list of indices in the 
				// sentence that correspond to parents of that type. 
				TIntObjectHashMap<TIntArrayList> parsByEdgeType = new TIntObjectHashMap<TIntArrayList>();
				for (int parentIndex = 0; parentIndex < di.numWords; parentIndex++) {
					int edgetype = cstraints.edge2cid(di,childIndex, parentIndex);
					if (!parsByEdgeType.contains(edgetype)) parsByEdgeType.put(edgetype, new TIntArrayList());
					parsByEdgeType.get(edgetype).add(parentIndex);
				}
				for (TIntObjectIterator<TIntArrayList> itr = parsByEdgeType.iterator(); itr.hasNext();) {
					itr.advance();
					int edgetype = itr.key();
					TIntArrayList parents = itr.value();
					int index = indicesforcp.get(edgetype);
					edge2scp[edgetype][index] = new SentenceChildParent(s,childIndex,parents.toNativeArray());
					indicesforcp.set(edgetype, 1+index);
				}
			}
		}
	}
	
}
