package depparsing.model;

import static depparsing.globals.Constants.*;

import depparsing.data.DepInstance;
import data.InstanceList;
import data.WordInstance;

import java.util.Arrays;

public class MaxLikelihoodEstimator {
	
	private MaxLikelihoodEstimator() {}

	/**
	 * Computes max likelihood counts for the given corpus,
	 * additionally incorporating the given backoffs.
	 */
	public static void computeMLModel(DepProbMatrix model, double backoff, double childBackoff) {
		computeMLModel(model, backoff, childBackoff, model.corpus.trainInstances);
	}
	
	public static void computeMLCounts(DepProbMatrix counts, InstanceList dlist) {
		counts.fill(0);
		for(WordInstance depInst : dlist.instanceList)
			countDependencies((DepInstance)depInst, counts);	
		counts.convertRealToLog();
	}
	
	public static void computeMLModel(DepProbMatrix model, double backoff, double childBackoff, InstanceList dlist){
		computeMLCounts(model, dlist);
		if(backoff > 0)
			model.backoff(Math.log(backoff));
		if(childBackoff > 0)
			model.addChildBackoff(childBackoff);
		model.logNormalize();
	}
	
	/**
	 * Adds counts for dependencies observed in the given sentence to the model.
	 */
	private static void countDependencies(DepInstance depInst, DepProbMatrix model) {
		int[] posTags = depInst.postags;
		int numWords = posTags.length;
		int[][] histogram = new int[numWords][2];

		// Cycle *left-to-right* through all words in the sentence
		int[] headValences = new int[numWords];
		for(int i = 0; i < numWords; i++) {
			int head = depInst.parents[i] - 1;

			// Update lambda_root
			if(head == -1)
				model.root[posTags[i]]++;
			else {
				if(i < head) { // i is to the left of head
					// Fill in left child counts and update histogram for use in setting decision counts
					model.child[posTags[i]][posTags[head]][LEFT][headValences[head]]++;
					if(headValences[head] < model.nontermMap.childValency - 1) headValences[head]++;
					histogram[head][LEFT]++;
				}
			}
		}
		
		// Cycle *right-to-left* through all words in the sentence
		Arrays.fill(headValences, 0);
		for(int i = numWords - 1; i >= 0; i--) {
			int head = depInst.parents[i] - 1;
			if(i > head && head != -1) {
				// Fill in right child counts and update histogram for use in setting decision counts
				model.child[posTags[i]][posTags[head]][RIGHT][headValences[head]]++;
				if(headValences[head] < model.nontermMap.childValency - 1) headValences[head]++;
				histogram[head][RIGHT]++;
			}
		}

		processHistogram(model, histogram, depInst);
	}
	
	/**
	 * Process the histogram to update the stop/continue arrays.
	 */
	public static void processHistogram(DepProbMatrix model, int[][] histogram, DepInstance depInst) {
		for(int i = 0; i < depInst.numWords; i++) {
			int headPosNum = depInst.postags[i];
			for(int dir = 0; dir < 2; dir++) {
				int current = histogram[i][dir];

				int maxValence = Math.min(current, model.nontermMap.decisionValency - 1);
				model.decision[headPosNum][dir][maxValence][END]++;
				if(current > 0) {
					for(int prevCont = 0; prevCont < maxValence; prevCont++)
						model.decision[headPosNum][dir][prevCont][CONT]++;
					model.decision[headPosNum][dir][maxValence][CONT] += (current - maxValence);
				}
			}
		}
	}
} 
