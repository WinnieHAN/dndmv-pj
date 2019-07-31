package depparsing.model;

import static depparsing.globals.Constants.RIGHT;

import data.WordInstance;
import depparsing.data.DepInstance;

public class RightBranchingInit {

	private RightBranchingInit() {
	}

	/**
	 * Initializes parameters with right branching.
	 */
	public static void initRightBranching(DepProbMatrix model,double backoff) {
		model.fill(0);		
		for(WordInstance depInst : model.corpus.trainInstances.instanceList)
			initRightBranching((DepInstance)depInst, model);
		
		// Convert to log and normalize		
		model.convertRealToLog();
		model.backoff(-1e1);
		model.logNormalize();
		
	}
	
	/**
	 * Adds counts for dependencies observed in the given sentence to the model.
	 * We assume all dependencies are right so the head is the first word in the sentence.
	 */
	private static void initRightBranching(DepInstance depInst, DepProbMatrix model) {
		int[] posTags = depInst.postags;
		int numWords = posTags.length;
		int[][] histogram = new int[numWords][2];

		// Cycle through all words in the sentence, building up a histogram
		for(int i = 1; i < numWords; i++) {
			model.child[posTags[i]][posTags[i-1]][RIGHT][0]++;	
		}
		model.root[0]++;

		MaxLikelihoodEstimator.processHistogram(model, histogram, depInst);
	}
}
