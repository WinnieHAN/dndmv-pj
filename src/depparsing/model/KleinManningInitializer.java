package depparsing.model;

import java.io.IOException;
import java.util.Arrays;

import data.WordInstance;
import depparsing.util.Lambda;
import depparsing.data.DepInstance;
import static depparsing.globals.Constants.*;

public class KleinManningInitializer {

	private KleinManningInitializer() {}
	
	/**
	 * Noah Smith's version of K&M initialization.
	 */
	public static void initNoah(DepProbMatrix model) throws IOException {
		model.fill(0);
		DepProbMatrix norm = new DepProbMatrix(model.corpus, model.nontermMap.decisionValency, model.nontermMap.childValency);
		if(model.nontermMap.decisionValency < 2) throw new RuntimeException("K&M initialization is only for models of decision valence 2 or greater");
		
		for(WordInstance wInst : model.corpus.trainInstances.instanceList) {
			DepInstance depInst = (DepInstance) wInst;
			int numWords = depInst.postags.length;
			double change[][] = new double[2][numWords];
			
			// Update root counts
			for(int i = 0; i < numWords; i++) {
				model.root[depInst.postags[i]] += 1.0 / numWords;
			}
			
			// Update child counts
			for(int chi = 0; chi < numWords; chi++) {
				int childNum = depInst.postags[chi];
				double sum = 0;
				
				// Normalization so all child counts in whole sentence sum to n - 1
				for(int par = 0; par < numWords; par++) {
					if(par != chi)
						sum += 1.0 / Math.abs(par - chi);
				}
				double scale = ((numWords - 1) / (double) numWords) * (1 / (double) sum); 
			
				for(int par = 0; par < numWords; par++) {
					if(par == chi) continue;
					double update = scale * (1 / (double) Math.abs(chi - par));
					int dir = (par > chi ? LEFT : RIGHT);
					change[dir][par] += update;
					
					model.child[childNum][depInst.postags[par]][dir][0] += update;
				}
			}
			
			// Copy 0-valence parameters to all higher valences
			for(int c = 0; c < model.numTags; c++)
				for(int p = 0; p < model.numTags; p++)
					for(int dir = 0; dir < 2; dir++)
						Arrays.fill(model.child[c][p][dir], model.child[c][p][dir][0]);
						
			// Update stop/continue counts
			stopContinueNoah(change[LEFT], LEFT, norm, depInst.postags, model);
			stopContinueNoah(change[RIGHT], RIGHT, norm, depInst.postags, model);
		}

		// Smoothing: add backoff to all counts
		model.addConstant(Math.pow(10, -1));
		
		// A crazy heuristic for setting a first child probability?
		double Es[] = firstChildNoah(norm, model);
		double prFirstKid = 0.9 * Es[0]; //+ 0.1 * Es[1];
		
		// Finalize the normalization counts --
		// should decrease probability of stopping with no kids and continuing with kids if both are large
		norm.applyToDecisions(new Lambda.Two<Double, Double, Double>() {
			public Double call(Double a, Double b) {
				return a*b;
			}
		}, prFirstKid); // multiply all norm decision probs by prFirstKid
		model.applyToDecisions(new Lambda.Two<Double, Double, Double>() {
            public Double call(Double a, Double b) {
                return a + b;
            }
		}, norm.decision);
		
		// Just set all other decision probabilities equal to those for valence 1
		for(int posNum = 0; posNum < model.numTags; posNum++)
			for(int dir = 0; dir < 2; dir++)
				for(int valence = 2; valence < model.nontermMap.decisionValency; valence++) {
					model.decision[posNum][dir][valence][CONT] = model.decision[posNum][dir][1][CONT];
					model.decision[posNum][dir][valence][END] = model.decision[posNum][dir][1][END];
				}
		
		// Convert to log probabilities and normalize
		model.convertRealToLog();
		model.logNormalize();
	}
	
	/**
	 * Helper method for computing decision probabilities in K&M initialization.
	 */
	private static void stopContinueNoah(double change[], int dir, DepProbMatrix norm,
			int posNums[], DepProbMatrix model) {
		for(int i = 0; i < change.length; i++) {
			int posNum = posNums[i];
			
			if(change[i] > 0) {
				// model.decision[posNum][dir][NO_KIDS][CONT] += 0.0;
				norm.decision[posNum][dir][0][CONT] += 1.0;
				model.decision[posNum][dir][1][CONT] += change[i];
				norm.decision[posNum][dir][1][CONT] += -1.0;

				model.decision[posNum][dir][0][END] += 1.0;
				norm.decision[posNum][dir][0][END] += -1.0;
				// model.decision[posNum][dir][KIDS][END] += 0.0;
				norm.decision[posNum][dir][1][END] += 1.0;
			} else {
				model.decision[posNum][dir][0][END] += 1.0;
			}
		}
	}
	
	/**
	 * Helper method for computing the "no kids" case for K&M initialization.
	 */
	private static double[] firstChildNoah(DepProbMatrix norm, DepProbMatrix model) {
		double[] Es = new double[]{1, 0};

		for(int kids = 0; kids < model.nontermMap.decisionValency; kids++)
			for(int choice = 0; choice < 2; choice++)
				for(int i = 0; i < model.numTags; i++) {
					for(int dir = 0; dir < 2; dir++) {
						double num = model.decision[i][dir][kids][choice];
						if(num > 0){ // will be > 0 for continue with kids and end without kids
							double denom = norm.decision[i][dir][kids][choice];
							double ratio = -1 * num/denom;
							if(denom < 0 && Es[0] > ratio)
								Es[0] = ratio;  // big only if you have high prob of stopping without kids AND high prob of continuing with kids
							//if(denom > 0 && Es[1] < ratio)
								//Es[1] = ratio;  // this code would never execute, so don't know why it was in Noah's script
						}
					}
				}

		return Es;
	}
}
