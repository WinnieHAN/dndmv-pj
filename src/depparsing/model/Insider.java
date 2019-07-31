package depparsing.model;

import static depparsing.globals.Constants.*;
import util.ArrayMath;
import util.LogSummer;

/* inside[wordIndexA][wordIndexB][nonterminal-type][head index in sentence]
 * = total probability of generating words A to B from specified nonterminal
 */
public class Insider {
	
	// Order is important here, since we want to work from the bottom of the tree up;
	// the left-branching rules are below the right-branching rules in the tree,
	// so we always want to fill left in first
	public static void computeCounts(DepSentenceDist sd) {
		int numWords = sd.depInst.numWords;

		// Base case: Spans covering a single word
		// (plus accounting for any unaries directly above words)
		for(int i = 0; i < numWords; i++) {
			for(int v = sd.nontermMap.maxValency - 1; v >= 0; v--) {
				int dv = Math.min(v, sd.nontermMap.decisionValency - 1);
				
				// stop(x,left,j): Lj[x] -> x
				if(v <= i)
					LogSummer.sum(sd.inside[i][i][sd.nontermMap.getNontermIndex(LEFT, CHOICE) + v], i,
							sd.decision[i][LEFT][dv][END]);
			}
			
			for(int v = sd.nontermMap.maxValency - 1; v >= 0; v--) {
				int dv = Math.min(v, sd.nontermMap.decisionValency - 1);
				
				// stop(x,right,j): Rj[x] -> L0[x]
				if(v <= numWords - 1 - i)
					LogSummer.sum(sd.inside[i][i][sd.nontermMap.getNontermIndex(RIGHT, CHOICE) + v], i,
							ArrayMath.safeAdd(new double[]{sd.decision[i][RIGHT][dv][END],
									sd.inside[i][i][sd.nontermMap.getNontermIndex(LEFT, CHOICE)][i]}));
			}
		}

		for(int span = 1; span < numWords; span++) {
			for(int begin = 0; begin < numWords - span; begin++) {
				int end = begin + span;
				for(int split = begin; split < end; split++) {

					// Take care of binary rules						
					for(int l = begin; l <= split; l++) {
						for(int r = split + 1; r <= end; r++) {
							int vIncr = 0;
							for(int v = sd.nontermMap.maxValency - 1; v >= 0; v--) {
								if(v <= sd.nontermMap.maxValency - 2) vIncr = 1;
								int cv = Math.min(v, sd.nontermMap.childValency - 1);
								
								// Left child
								if(v <= l) { // Must have at least #(valence - 1) other items further to the left than this child
									double additionalProb = ArrayMath.safeAdd(new double[]{sd.child[l][r][cv],
											sd.inside[begin][split][sd.nontermMap.getNontermIndex(RIGHT, CHOICE)][l],
											sd.inside[split + 1][end][sd.nontermMap.getNontermIndex(LEFT, CHOICE) + v + vIncr][r]});
									LogSummer.sum(sd.inside[begin][end][sd.nontermMap.getNontermIndex(LEFT, CHILD) + v], r, additionalProb);
								}

								// Right child
								if(v <= numWords - 1 - r) {  // Must have at least #(valence - 1) other items further to the right than this child
									double additionalProb = ArrayMath.safeAdd(new double[]{sd.child[r][l][cv],
											sd.inside[begin][split][sd.nontermMap.getNontermIndex(RIGHT, CHOICE) + v + vIncr][l],
											sd.inside[split + 1][end][sd.nontermMap.getNontermIndex(RIGHT, CHOICE)][r]});
									LogSummer.sum(sd.inside[begin][end][sd.nontermMap.getNontermIndex(RIGHT, CHILD) + v], l, additionalProb);
								}
							}
						}
					}
				}

				// Take care of unary rules
				for(int head = begin; head <= end; head++) {
					for(int v = sd.nontermMap.maxValency - 1; v >= 0; v--) {
						int dv = Math.min(v, sd.nontermMap.decisionValency - 1);
						
						// continue(x,left,j): Lj[x] -> Lcj[x]
						if(head > v)
							LogSummer.sum(sd.inside[begin][end][sd.nontermMap.getNontermIndex(LEFT, CHOICE) + v], head, 
									ArrayMath.safeAdd(new double[]{
											sd.decision[head][LEFT][dv][CONT],
											sd.inside[begin][end][sd.nontermMap.getNontermIndex(LEFT, CHILD) + v][head]}));
					}
					
					for(int v = sd.nontermMap.maxValency - 1; v >= 0; v--) {
						int dv = Math.min(v, sd.nontermMap.decisionValency - 1);
						
						// stop(x,right,j): Rj[x] -> L0[x]
						if(head <= numWords - v - 1)
							LogSummer.sum(sd.inside[begin][end][sd.nontermMap.getNontermIndex(RIGHT, CHOICE) + v], head, 
									ArrayMath.safeAdd(new double[]{
											sd.decision[head][RIGHT][dv][END],
											sd.inside[begin][end][sd.nontermMap.getNontermIndex(LEFT, CHOICE)][head]}));
					}
					
					for(int v = sd.nontermMap.maxValency - 1; v >= 0; v--) {
						int dv = Math.min(v, sd.nontermMap.decisionValency - 1);
						
						// continue(x,right,j): Rj[x] -> Rcj[x]
						if(head < numWords - v - 1)
							LogSummer.sum(sd.inside[begin][end][sd.nontermMap.getNontermIndex(RIGHT, CHOICE) + v], head, 
									ArrayMath.safeAdd(new double[]{
											sd.decision[head][RIGHT][dv][CONT],
											sd.inside[begin][end][sd.nontermMap.getNontermIndex(RIGHT, CHILD) + v][head]}));
					}
				}
			}
		}
		
		// root(x): S -> R0[x]
		for(int head = 0; head < numWords; head++) {
			sd.insideRoot = LogSummer.sum(sd.insideRoot,
					ArrayMath.safeAdd(new double[]{sd.inside[0][numWords - 1][sd.nontermMap.getNontermIndex(RIGHT, CHOICE)][head], sd.root[head]}));
		}
	}
	
}