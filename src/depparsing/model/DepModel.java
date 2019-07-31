package depparsing.model;

import static depparsing.globals.Constants.LEFT;
import static depparsing.globals.Constants.RIGHT;
import model.AbstractCountTable;
import model.AbstractModel;
import model.AbstractSentenceDist;
import util.DigammaFunction;
import util.LogSummer;
import data.InstanceList;
import depparsing.data.DepCorpus;
import depparsing.data.DepInstance;
import depparsing.util.Lambda;

public class DepModel extends AbstractModel {

	public enum UpdateType {TABLE_UP, VB};
	private final UpdateType updateType;
	
	public final DepCorpus corpus;
	public final DepProbMatrix params;
	public final double childBackoff;
	public final int decisionValency;
	public final int childValency;
	private final double prior;
	
	public DepModel(DepProbMatrix params, DepCorpus corpus, int decisionV, int childV, double childBackoff, UpdateType ut) {
		this.params = params;
		this.corpus = corpus;
		this.decisionValency = decisionV;
		this.childValency = childV;
		this.childBackoff = childBackoff;
		this.updateType = ut;
		this.prior = 0;
	}
	
	public DepModel(DepProbMatrix params, DepCorpus corpus, int decisionV, int childV, double childBackoff, UpdateType ut, double prior) {
		this.params = params;
		this.corpus = corpus;
		this.decisionValency = decisionV;
		this.childValency = childV;
		this.childBackoff = childBackoff;
		this.updateType = ut;
//		if(ut != UpdateType.VB) throw new RuntimeException("Constructor with a prior is only applicable for varaitional Bayes");
		this.prior = prior;
	}
	
	@Override
	public void addToCounts(AbstractSentenceDist asd, AbstractCountTable acounts) {
		DepSentenceDist sd = (DepSentenceDist)asd;
		DepProbMatrix counts = (DepProbMatrix)acounts;
		int numWords = sd.depInst.getNrWords();
		
		// Root probabilities
		for(int parent = 0; parent < numWords; parent++) {
			int parentPos = sd.depInst.postags[parent];
			LogSummer.sum(counts.root, parentPos, sd.getRootPosterior(parent));
		}
		
		// Child probabilities
		for(int parent = 0; parent < numWords; parent++) {
			int parentPos = sd.depInst.postags[parent];

			for(int child = 0; child < numWords; child++) {
				int dir = LEFT;
				if(child == parent) continue;
				if(child > parent) dir = RIGHT;
				int childPos = sd.depInst.postags[child];

				for(int v = 0; v < sd.nontermMap.childValency; v++) {
					double posterior = sd.getChildPosterior(child, parent, v);
					LogSummer.sum(counts.child[childPos][parentPos][dir], v, posterior);
				}
			}
		}

		// Stop and continue probabilities	
		for(int parent = 0; parent < numWords; parent++) {
			int parentPos = sd.depInst.postags[parent];
			for(int choice = 0; choice < 2; choice++) {
				for(int dir = 0; dir < 2; dir++) {
					for(int v = 0; v < sd.nontermMap.decisionValency; v++) {
						LogSummer.sum(counts.decision[parentPos][dir][v], choice,
								sd.getDecisionPosterior(parent, dir, v, choice));
					}
				}
			}
		}
	}

	@Override
	public void computePosteriors(AbstractSentenceDist adist) {
		DepSentenceDist dist = (DepSentenceDist) adist;
		dist.cacheModelAndComputeIO(params);
//		((DepSentenceDist)dist).cachePosteriors();
	}

	@Override
//	public AbstractCountTable getCountTable() {
//		return new DepProbMatrix(corpus, decisionValency, childValency);
//	}
	public AbstractCountTable getCountTable() {
		return this.params;
	}

	@Override
	public AbstractSentenceDist[] getSentenceDists() {
		return getSentenceDists(corpus.trainInstances);
	}
	
	@Override
	public AbstractSentenceDist[] getSentenceDists(InstanceList list) {
		DepSentenceDist[] sentences = new DepSentenceDist[list.instanceList.size()];
		for (int i = 0; i < sentences.length; i++) {
			sentences[i] = new DepSentenceDist((DepInstance)list.instanceList.get(i), params.nontermMap);
		}
		return sentences;
	}

	@Override
	public void updateParameters(AbstractCountTable counts) {
		params.fill(counts);
		
		if(childBackoff > 0)
			params.addChildBackoff(childBackoff);
		
		if(updateType == UpdateType.TABLE_UP) {
			params.logNormalize();
		} else if(updateType == UpdateType.VB) {
			params.backoff(Math.log(prior));
			
			double[][][] childTotals = new double[params.numTags][2][params.nontermMap.childValency];
			double[][][] decisionTotals = new double[params.numTags][2][params.nontermMap.decisionValency];
			double rootTotal = params.getNormalizers(childTotals, decisionTotals);
			rootTotal = params.checkForNegInfTotals(rootTotal, childTotals, decisionTotals);
					
			// Subtract normalizers
			params.applyToNormGroups(new Lambda.Two<Double, Double, Double>() {
				public Double call(Double a, Double b) {
					assert(!Double.isInfinite(b) || !Double.isNaN(b)) : "total = " + b;
					double newCounts = Math.log(DigammaFunction.expDigamma(Math.exp(a)));
					double newSum = Math.log(DigammaFunction.expDigamma(Math.exp(b)));
					assert(!Double.isNaN(newCounts)) : "New counts are NAN : " + a + newCounts;
					assert(!Double.isNaN(newSum)) : "New Sum are NAN : " + b + newSum;
					if(Double.isInfinite(newSum)) {
						return Double.NEGATIVE_INFINITY;
					} else {
						return  newCounts - newSum;
					}
				}
			}, rootTotal, childTotals, decisionTotals);
			
			// Note that params are not normalized now and we should not normalize
		}
	}
	
	
	/**
	 * Exponentiate all the probabilities to the power of <code>exponent</code>.
	 * Used in softmax-EM.
	 * 
	 * @param exponent
	 */
	public void exponentiateParameters(double exponent) { // TODO add this
															// method to
															// AbstractModel
		params.apply(new Lambda.Two<Double, Double, Double[]>() {
			public Double call(Double p1, Double[] p2) {
				return p1 * p2[0]; // p1 is log probability
			}
		}, new Double[] { exponent });
	}	
}
