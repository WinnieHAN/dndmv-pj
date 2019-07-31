package depparsing.constraints;

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
 * This version of {@link L1LMax} constraint does not group the parents by their type. 
 * So, for example, the probability of "the second Determiner in the sentence" 
 * is dominated by different Nouns would be separate masses: "the second Determiner in the 
 * sentence is dominated by the first Noun",  "the second Determiner in the 
 * sentence is dominated by the second Noun" and so on. 
 * 
 */
public class UngroupedL1LMax extends L1LMax{

	public UngroupedL1LMax(DepCorpus corpus, DepModel model,
			ArrayList<WordInstance> toProject, PCType cType, PCType pType,
			boolean useRoot, boolean useDirection, double constraintStrength,
			int minOccurrencesForProjection, String fileOfAllowedTypes)
			throws IOException {
		super(corpus, model, toProject, cType, pType, useRoot, useDirection,
				constraintStrength, minOccurrencesForProjection, fileOfAllowedTypes);
	}
	
	@Override
	public ArrayList<Integer> countIndicesForChildParentType(
			ArrayList<WordInstance> toProject) {
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
				// don't group parents by their type; so every potential parent has 
				// its own space. 
				for (int parentIndex = 0; parentIndex < di.numWords; parentIndex++) {
					int edgetype = cstraints.edge2cid(di,childIndex, parentIndex);
					while (edgetype >=indicesforcp.size()) indicesforcp.add(0);
					indicesforcp.set(edgetype, 1+indicesforcp.get(edgetype));
				}
			}
		}
		return indicesforcp;
	}

	@Override
	public void makeEdge2SentenceChildParent(ArrayList<WordInstance> toProject,
			ArrayList<Integer> indicesforcp) {
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
				// don't group parents by their type; so every potential parent has 
				// its own space. 
				for (int parentIndex = 0; parentIndex < di.numWords; parentIndex++) {
					int edgetype = cstraints.edge2cid(di,childIndex, parentIndex);
					int index = indicesforcp.get(edgetype);
					edge2scp[edgetype][index] = new SentenceChildParent(s,childIndex,new int[] {parentIndex});
					indicesforcp.set(edgetype, 1+index);
				}
			}
		}

	}

}
