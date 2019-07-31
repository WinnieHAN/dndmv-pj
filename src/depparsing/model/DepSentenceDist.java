package depparsing.model;

import static depparsing.globals.Constants.*;
import depparsing.data.DepInstance;
import model.AbstractSentenceDist;
import depparsing.model.DepProbMatrix;
import depparsing.model.NonterminalMap;
import util.ArrayMath;
import util.LogSummer;
import depparsing.model.Insider;
import depparsing.model.Outsider;

public class DepSentenceDist extends AbstractSentenceDist {
	public final DepInstance depInst;
	
	// Sentence likelihood
	public double insideRoot;
	
	public double inside[][][][];
	public double outside[][][][];

	// One probability per word
	public double root[];
	
	// 1st index = child, 2nd index = parent, 3rd index = valency
	public double child[][][];
	
	// 1st index = parent, 2nd index = direction, 3rd index = valency,
	// 4th index = choice (0 for stop, 1 for continue)
	public double decision[][][][];

	private boolean posteriorsComputed;
	private double childPosteriors[][][];
	private double decisionPosteriors[][][][];
	
	public double insideRoot1;
	public double inside1[][][][];
	public double outside1[][][][];
	public double root1[];
	public double child1[][][];
	public double decision1[][][][];
	private double childPosteriors1[][][];
	private double decisionPosteriors1[][][][];
	
	public final NonterminalMap nontermMap;
	
	public DepSentenceDist(DepInstance depInst, NonterminalMap ntMap) {
		this.depInst = depInst;
		nontermMap = ntMap;
		posteriorsComputed = false;
	}
	
	/**
	 * Copies model parameters to local caches so that we don't have to
	 * constantly look up mappings between the indices of words 
	 * in the sentence and their corresponding POS numbers
	 * when computing the posteriors and running inside outside.
	 */
	public void cacheModel(DepProbMatrix model) {
		// FIXME -- is this necessary
		initSentenceDist();
		int[] i2tag = depInst.postags;
		for (int c = 0; c < i2tag.length; c++) {
			int ctag = i2tag[c];
			root[c] = model.root[ctag];
			for (int p = 0; p < i2tag.length; p++) {
				if(c == p) continue;
				int ptag = i2tag[p];
				int dir = (c < p ? LEFT : RIGHT);
				for (int v = 0; v < model.nontermMap.childValency; v++) {
					child[c][p][v] = model.child[ctag][ptag][dir][v];
				}
			}
			for (int dir = 0; dir < 2; dir++) {
				for (int v = 0; v < model.nontermMap.decisionValency; v++) {
					for (int choice = 0; choice < 2; choice++) {
						decision[c][dir][v][choice] = model.decision[ctag][dir][v][choice];
					}
				}
				
			}
		}
	}
	
	public void cacheModelAndComputeIO(DepProbMatrix model) {
		cacheModel(model);
		computeIO();
	}
	public void exponentiateAndCacheModel(DepProbMatrix model, double sigma) {
		initSentenceDist();
		int[] i2tag = depInst.postags;
		for (int c = 0; c < i2tag.length; c++) {
			int ctag = i2tag[c];
			root[c] = model.root[ctag] * sigma;
			for (int p = 0; p < i2tag.length; p++) {
				if(c == p) continue;
				int ptag = i2tag[p];
				int dir = (c < p ? LEFT : RIGHT);
				for (int v = 0; v < model.nontermMap.childValency; v++) {
					child[c][p][v] = model.child[ctag][ptag][dir][v] * sigma;;
				}
			}
			for (int dir = 0; dir < 2; dir++) {
				for (int v = 0; v < model.nontermMap.decisionValency; v++) {
					for (int choice = 0; choice < 2; choice++) {
						decision[c][dir][v][choice] = model.decision[ctag][dir][v][choice] * sigma;;
					}
				}
				
			}
		}
	}
	public void computeIO(){
		posteriorsComputed = false;
		
		insideRoot = Double.NEGATIVE_INFINITY;
		ArrayMath.set(inside, Double.NEGATIVE_INFINITY);
		ArrayMath.set(outside, Double.NEGATIVE_INFINITY);
		Insider.computeCounts(this);
		Outsider.computeCounts(this);
		
		InOutUtils.checkIOAgreement(1e-10, this);
	}
	
	public double getRootPosterior(int wordIndex) {
		assert(!Double.isInfinite(insideRoot));
		return ArrayMath.safeAdd(new double[]{root[wordIndex],
				inside[0][depInst.numWords - 1][nontermMap.getNontermIndex(RIGHT, CHOICE)][wordIndex],
				-insideRoot});
	}
	//////
	public double getRoot1Posterior(int wordIndex) {
		assert(!Double.isInfinite(insideRoot1));
		return ArrayMath.safeAdd(new double[]{root1[wordIndex],
				inside1[0][depInst.numWords - 1][nontermMap.getNontermIndex(RIGHT, CHOICE)][wordIndex],
				-insideRoot1});
	}

	public void cachePosteriors() {
		computeUnormalizedChildPosteriors();
		computeUnormalizedDecisionPosteriors();

		// Normalize
		assert(!Double.isInfinite(insideRoot));
		assert(!Double.isNaN(insideRoot));
		for(int i = 0; i < depInst.numWords; i++) {
			for(int j = 0; j < depInst.numWords; j++)
				for(int v = 0; v < nontermMap.childValency; v++)
					childPosteriors[i][j][v] -= insideRoot;
		
			for(int dir = 0; dir < 2; dir++)
				for(int v = 0; v < nontermMap.decisionValency; v++)
					for(int choice = 0; choice < 2; choice++)
						decisionPosteriors[i][dir][v][choice] -= insideRoot;
		}
		
		posteriorsComputed = true;
	}
	
	private void computeUnormalizedChildPosteriors() {
		ArrayMath.set(childPosteriors, Double.NEGATIVE_INFINITY);
		ArrayMath.set(decisionPosteriors, Double.NEGATIVE_INFINITY);
		// Compute childPosteriors and decisionPosteriors tables
		for(int begin = 0; begin < depInst.numWords; begin++) {
			for(int end = begin + 1; end < depInst.numWords; end++) {
				for(int split = begin; split < end; split++) {
					for(int l = begin; l <= split; l++) {
						for(int r = split + 1; r <= end; r++) {
							int vIncr = 1;
							for(int v = 0; v < nontermMap.maxValency; v++) {
								int cv = Math.min(v, nontermMap.childValency - 1);
								if(v == nontermMap.maxValency - 1) vIncr = 0;
								
								// Left child
								childPosteriorsHelper(l, r, nontermMap.getNontermIndex(LEFT, CHILD) + v,
										nontermMap.getNontermIndex(RIGHT, CHOICE),
										nontermMap.getNontermIndex(LEFT, CHOICE) + v + vIncr,
										begin, end, split, cv);

								// Right child
								childPosteriorsHelper(r, l, nontermMap.getNontermIndex(RIGHT, CHILD) + v,
										nontermMap.getNontermIndex(RIGHT, CHOICE) + v + vIncr,
										nontermMap.getNontermIndex(RIGHT, CHOICE),
										begin, end, split, cv);
							}
						}
					}
				}
			}
		}
	}
	
	private void childPosteriorsHelper(int c, int p, int A, int B, int C,
			int begin, int end, int split, int cv) {
		int l, r;
		if(c < p) {
			l = c; r = p;
		} else {
			l = p; r = c;
		}
		
		LogSummer.sum(childPosteriors[c][p], cv,
				ArrayMath.safeAdd(new double[]{child[c][p][cv], inside[begin][split][B][l],
				inside[split + 1][end][C][r], outside[begin][end][A][p]}));
	}
	
	private void computeUnormalizedDecisionPosteriors() {
		// stop(x,left,j): Lj[x] -> x
		for(int p = 0; p < depInst.numWords; p++)
			for(int v = 0; v < nontermMap.maxValency; v++) {
				int dv = Math.min(v, nontermMap.decisionValency - 1);
				decisionPosteriorsHelper(p, p, p, dv, LEFT, END,
						nontermMap.getNontermIndex(LEFT, CHOICE) + v, -1);
			}
		
		for(int begin = 0; begin < depInst.numWords; begin++)
			for(int end = begin; end < depInst.numWords; end++)
				for(int p = begin; p <= end; p++) {

					for(int v = 0; v < nontermMap.maxValency; v++) {
						int dv = Math.min(v, nontermMap.decisionValency - 1);
						
						// stop(x,right,j): Rj[x] -> L0[x]
						decisionPosteriorsHelper(p, begin, end, dv, RIGHT, END,
								nontermMap.getNontermIndex(RIGHT, CHOICE) + v,
								nontermMap.getNontermIndex(LEFT, CHOICE));
					}
					
					for(int v = 0; v < nontermMap.maxValency; v++) {
						int dv = Math.min(v, nontermMap.decisionValency - 1);
						
						// continue(x,right,j): Rj[x] -> Rcj[x]
						decisionPosteriorsHelper(p, begin, end, dv, RIGHT, CONT,
								nontermMap.getNontermIndex(RIGHT, CHOICE) + v,
								nontermMap.getNontermIndex(RIGHT, CHILD) + v);

						// continue(x,left,j): Lj[x] -> Lcj[x]
						decisionPosteriorsHelper(p, begin, end, dv, LEFT, CONT,
								nontermMap.getNontermIndex(LEFT, CHOICE) + v,
								nontermMap.getNontermIndex(LEFT, CHILD) + v);
					}
				}
	}
	
	private void decisionPosteriorsHelper(int p, int begin, int end, int v, int dir, int choice, int A, int B) {
		// If we're at a terminal rule, inside prob is "certainty" (1 real prob -> 0 log prob)
		LogSummer.sum(decisionPosteriors[p][dir][v], choice,
				ArrayMath.safeAdd(new double[]{(B == -1? 0 : inside[begin][end][B][p]),
						outside[begin][end][A][p],
						decision[p][dir][v][choice]}));
	}

	
	public double getChildPosterior(int wantedChildIndex, int wantedParentIndex, int wantedValency) {
		if(!posteriorsComputed) {
			cachePosteriors();
		}
		return childPosteriors[wantedChildIndex][wantedParentIndex][wantedValency];
	}
	public double getChild1Posterior(int wantedChildIndex, int wantedParentIndex, int wantedValency) {
//		if(!posteriorsComputed) {
//			cachePosteriors();
//		}
		return childPosteriors1[wantedChildIndex][wantedParentIndex][wantedValency];
	}
	public double getDecisionPosterior(int wantedParentIndex, int wantedDirection, int wantedValency, int wantedChoice) {
		if(!posteriorsComputed) {
			cachePosteriors();
		}
		return decisionPosteriors[wantedParentIndex][wantedDirection][wantedValency][wantedChoice];
	}
	public double getDecision1Posterior(int wantedParentIndex, int wantedDirection, int wantedValency, int wantedChoice) {
//		if(!posteriorsComputed) {
//			cachePosteriors();
//		}
		return decisionPosteriors1[wantedParentIndex][wantedDirection][wantedValency][wantedChoice];
	}
	@Override
	public void clearCaches() {
		// We can't get rid of the root cache because it's necessary for the root posterior
		// this.root = null;
//		this.child = null;
//		this.decision = null;
	}

	@Override
	public void clearPosteriors() {
		/////////////////////////////////////2.18
		int numWords = depInst.getNrWords();
		insideRoot1 = this.insideRoot;
		
		inside1 = new double[numWords][numWords][nontermMap.numNontermTypes][numWords];
		outside1 = new double[numWords][numWords][nontermMap.numNontermTypes][numWords];
		inside1 = this.inside.clone();
		outside1 = this.outside.clone();
		
		root1 = new double[numWords];
		child1 = new double[numWords][numWords][nontermMap.childValency];
		decision1 = new double[numWords][2][nontermMap.decisionValency][2];
		root1 = this.root.clone();
		child1 = this.child.clone();
		decision1 = this.decision.clone();
		
		childPosteriors1 = new double[numWords][numWords][nontermMap.childValency];
		decisionPosteriors1 = new double[numWords][2][nontermMap.decisionValency][2];
		childPosteriors1 = this.childPosteriors.clone();
		decisionPosteriors1 = this.decisionPosteriors.clone();
		
		posteriorsComputed = false;
		this.childPosteriors = null;
		this.decisionPosteriors = null;
		this.inside = null;
		this.outside = null;
		this.root = null;
		this.child = null;
		this.decision = null;
	}

	@Override
	public double getLogLikelihood() {
		if(!posteriorsComputed) cachePosteriors();
		return insideRoot;
	}

	@Override
	public void initSentenceDist() {
		int numWords = depInst.getNrWords();
		
		// Create caches for model probabilities
		// (note root/child/decision indices mean different things here than in the model)
		root = new double[numWords];
		child = new double[numWords][numWords][nontermMap.childValency];
		decision = new double[numWords][2][nontermMap.decisionValency][2];
		
		inside = new double[numWords][numWords][nontermMap.numNontermTypes][numWords];
		outside = new double[numWords][numWords][nontermMap.numNontermTypes][numWords];
		
		posteriorsComputed = false;
		childPosteriors = new double[numWords][numWords][nontermMap.childValency];
		decisionPosteriors = new double[numWords][2][nontermMap.decisionValency][2];
	}
	
	
	public void cacheModelFromOutside(double[] root, double[][][] child, double[][][][] decision){
		this.root = root;
		this.child = child;
		this.decision = decision;
	}
}
