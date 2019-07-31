package depparsing.model;

import util.ArrayMath;
import util.LogSummer;
import static depparsing.globals.Constants.*;

/* outside[wordIndexA][wordIndexB][nonterminal-type][head index in sentence]
 * = total probability of beginning with the start symbol S and
 *   generating the specified nonterminal and all the words outside of its A to B scope
 */
public class Outsider {
	
	// Order is important here, since we want to work from the top of the tree down;
	// the right-branching rules are above the left-branching rules in the tree,
	// so we always want to fill right in first
	public static void computeCounts(DepSentenceDist sd) {
		int numWords = sd.depInst.numWords;
		
		// Start symbol S gets outside prob 1 (0 in log prob)
		// so it's not necessary to store this
		
		// Take care of unary directly below root
		for(int i = 0; i < numWords; i++) {
			// outside[0][numWords - 1][S][posNum] + p.root[posNum] = p.root[posNum]
			sd.outside[0][numWords - 1][sd.nontermMap.getNontermIndex(RIGHT, CHOICE)][i] = sd.root[i];

			// stop(x,right,0): R0[x] -> L0[x]
			sd.outside[0][numWords - 1][sd.nontermMap.getNontermIndex(LEFT, CHOICE)][i] =
				ArrayMath.safeAdd(new double[]{sd.decision[i][RIGHT][0][END],
											  sd.outside[0][numWords - 1][sd.nontermMap.getNontermIndex(RIGHT, CHOICE)][i]});
			// continue(x,left,0): L0[x] -> Lc0[x]
			if(i > 0) // If not leftmost word, can have left children
				sd.outside[0][numWords - 1][sd.nontermMap.getNontermIndex(LEFT, CHILD)][i] = 
					ArrayMath.safeAdd(new double[]{sd.decision[i][LEFT][0][CONT],
							sd.outside[0][numWords - 1][sd.nontermMap.getNontermIndex(LEFT, CHOICE)][i]});

			// continue(x,right,0): R0[x] -> Rc0[x]
			if(i < numWords - 1) // If not rightmost word, can have right children
				sd.outside[0][numWords - 1][sd.nontermMap.getNontermIndex(RIGHT, CHILD)][i] =
					ArrayMath.safeAdd(new double[]{sd.decision[i][RIGHT][0][CONT],
							sd.outside[0][numWords - 1][sd.nontermMap.getNontermIndex(RIGHT, CHOICE)][i]});
		}
		
		for(int span = numWords - 1; span > 0; span--) {
			for(int begin = 0; begin < numWords - span; begin++) {
				int end = begin + span;

				// Take care of unary rules
				if(span < numWords - 1) {
					for(int head = begin; head <= end; head++) {
						for(int v = 0; v < sd.nontermMap.maxValency; v++) {
							int dv = Math.min(v, sd.nontermMap.decisionValency - 1);
							
							// stop(x,right,j): Rj[x] -> L0[x]
							if(head <= numWords - v - 1)
								LogSummer.sum(sd.outside[begin][end][sd.nontermMap.getNontermIndex(LEFT, CHOICE)], head,
										ArrayMath.safeAdd(new double[]{sd.decision[head][RIGHT][dv][END],
												sd.outside[begin][end][sd.nontermMap.getNontermIndex(RIGHT, CHOICE) + v][head]}));
						}
						
						for(int v = 0; v < sd.nontermMap.maxValency; v++) {
							int dv = Math.min(v, sd.nontermMap.decisionValency - 1);
							
							// continue(x,right,j): Rj[x] -> Rcj[x]
							if(head < numWords - v - 1)
								LogSummer.sum(sd.outside[begin][end][sd.nontermMap.getNontermIndex(RIGHT, CHILD) + v], head,
										ArrayMath.safeAdd(new double[]{sd.decision[head][RIGHT][dv][CONT],
												sd.outside[begin][end][sd.nontermMap.getNontermIndex(RIGHT, CHOICE) + v][head]}));

							// continue(x,left,j): Lj[x] -> Lcj[x]
							if(head > v)
								LogSummer.sum(sd.outside[begin][end][sd.nontermMap.getNontermIndex(LEFT, CHILD) + v], head,
										ArrayMath.safeAdd(new double[]{sd.decision[head][LEFT][dv][CONT],
												sd.outside[begin][end][sd.nontermMap.getNontermIndex(LEFT, CHOICE) + v][head]}));
						}
					}
				}

				// Take care of binary rules
				int A, B, C, par, chi;
				for(int split = begin; split < end; split++) {
					for(int l = begin; l <= split; l++) {
						for(int r = split + 1; r <= end; r++) {
							int vIncr = 1;
							for(int v = 0; v < sd.nontermMap.maxValency; v++) {
								if(v == sd.nontermMap.maxValency - 1) vIncr = 0;
								int cv = Math.min(v, sd.nontermMap.childValency - 1);
								
								// Right child: Rcj[x] -> R(j+valenceIncr)[x] Rc0[x']
								if(v <= numWords - 1 - r) {
									A = sd.nontermMap.getNontermIndex(RIGHT, CHILD) + v;
									B = sd.nontermMap.getNontermIndex(RIGHT, CHOICE) + v + vIncr;
									C = sd.nontermMap.getNontermIndex(RIGHT, CHOICE);
									par = l;
									chi = r;
									LogSummer.sum(sd.outside[begin][split][B], l,
											ArrayMath.safeAdd(new double[]{sd.outside[begin][end][A][par],
													sd.child[chi][par][cv],
													sd.inside[split + 1][end][C][r]}));
									LogSummer.sum(sd.outside[split + 1][end][C], r,
											ArrayMath.safeAdd(new double[]{sd.outside[begin][end][A][par] +
													sd.child[chi][par][cv] +
													sd.inside[begin][split][B][l]}));
								}

								// Left child: Lcj[x] -> Rc0[x'] L(j+valenceIncr)[x]
								if(v <= l) {
									A = sd.nontermMap.getNontermIndex(LEFT, CHILD) + v;
									B = sd.nontermMap.getNontermIndex(RIGHT, CHOICE);
									C = sd.nontermMap.getNontermIndex(LEFT, CHOICE) + v + vIncr;
									par = r;
									chi = l;
									LogSummer.sum(sd.outside[begin][split][B], l,
											ArrayMath.safeAdd(new double[]{sd.outside[begin][end][A][par],
													sd.child[chi][par][cv],
													sd.inside[split + 1][end][C][r]}));
									LogSummer.sum(sd.outside[split + 1][end][C], r,
											ArrayMath.safeAdd(new double[]{sd.outside[begin][end][A][par] +
													sd.child[chi][par][cv] +
													sd.inside[begin][split][B][l]}));
								}
							}
						}
					}
				}
			}
		}

		// Apply zero-span rules
		for(int i = 0; i < numWords && numWords > 1; i++) {
			for(int v = 0; v < sd.nontermMap.maxValency; v++) {
				int dv = Math.min(v, sd.nontermMap.decisionValency - 1);
		
				// stop(x,right,j): Rj[x] -> L0[x]
				if(i <= numWords - v - 1)
					LogSummer.sum(sd.outside[i][i][sd.nontermMap.getNontermIndex(LEFT, CHOICE)], i,
							ArrayMath.safeAdd(new double[]{sd.decision[i][RIGHT][dv][END],
									sd.outside[i][i][sd.nontermMap.getNontermIndex(RIGHT, CHOICE) + v][i]}));
			}
		}
	}
}