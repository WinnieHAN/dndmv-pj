package depparsing.model;

import static depparsing.globals.Constants.LEFT;

import java.io.IOException;

import data.WordInstance;
import depparsing.data.DepInstance;

public class LeftBranchingInit {

	private LeftBranchingInit() {}

	/**
	 * Initializes parameters with left branching.
	 */
	public static void initLeftBranching(DepProbMatrix model,double backoff) throws IOException {
		model.fill(0);		
		for(WordInstance wInst : model.corpus.trainInstances.instanceList)
			initLeftBranching((DepInstance) wInst, model);
		
		// Convert to log and normalize		
		model.convertRealToLog();
		model.backoff(-1e1);
		model.logNormalize();
	}
	
	/**
	 * Adds counts for dependencies observed in the given sentence to the model.
	 * We assume all dependencies are left so the head is the last word in the sentence.
	 */
	private static void initLeftBranching(DepInstance depInst, DepProbMatrix model) throws IOException {
		int[] posTags = depInst.postags;
		int numWords = posTags.length;
		int[][] histogram = new int[numWords][2];

		// Cycle through all words in the sentence, building up a histogram
		for(int i = 1; i < numWords; i++) {
			model.child[posTags[i-1]][posTags[i]][LEFT][0]++;	
		}
		model.root[posTags[numWords-1]]++;

		MaxLikelihoodEstimator.processHistogram(model, histogram, depInst);
	}
}
