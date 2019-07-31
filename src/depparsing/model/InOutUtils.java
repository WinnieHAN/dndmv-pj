package depparsing.model;

import static depparsing.globals.Constants.*;
import java.util.Arrays;

import depparsing.model.NonterminalMap;
import util.LogSummer;

public class InOutUtils {
	
	public static void printInsideOutside(DepSentenceDist sd) {
		System.out.println("Inside:");
		printIOHelper(sd.depInst.numWords, sd.inside, sd.nontermMap);
		System.out.println("Outside:");
		printIOHelper(sd.depInst.numWords, sd.outside, sd.nontermMap);
	}

	private static void printIOHelper(int numWords, double[][][][] matrix, NonterminalMap nontermMap) {
		int numNonterms = matrix[0][0].length;
		for(int i = 0; i < numWords; i++) {
			for(int j = i; j < numWords; j++) {
				System.out.println("(word " + i + ", word " + j + ")");
				for(int k = 0; k < numNonterms; k++) {
					System.out.println("nonterm = " + nontermMap.nontermIndex2String(k));
					for(int l = 0; l < numWords; l++) {
						System.out.print(matrix[i][j][k][l] + " ");
					}
					System.out.println();
				}
			}
		}
	}
	
	/**
	 * Inside and outside arrays should be identical for certain entries and sums of entries.
	 * This method asserts that all of these identities hold.
	 */
	public static double checkIOAgreement(double threshold, DepSentenceDist sd) {
		double avgDiff = 0;
		
		// Check that the outside array has internal consistency
		
		// Compute all the outside predictions of sentence likelihood - sum over terminal rules
		double outsideFull[] = new double[sd.depInst.numWords];
		Arrays.fill(outsideFull, Double.NEGATIVE_INFINITY);
		for(int i = 0; i < sd.depInst.numWords; i++)
			for(int v = 0; v < sd.nontermMap.maxValency; v++) {
				if(i >= v) { // If left children at this valence are possible
					int dv = Math.min(v, sd.nontermMap.decisionValency - 1);
					LogSummer.sum(outsideFull, i,
							sd.outside[i][i][sd.nontermMap.getNontermIndex(LEFT, CHOICE) + v][i] +
							sd.decision[i][LEFT][dv][END]);
				}
			}
		
		// Average the outside predictions
		double avgOutsideFull = 0;
		for(double prob : outsideFull)
			avgOutsideFull += prob;
		avgOutsideFull /= sd.depInst.numWords;
		
		// Check that no outside prediction deviates too much from the average
		for(double prob : outsideFull) {
			double diff = Math.abs(avgOutsideFull - prob);
			avgDiff += diff;
			assert(diff < threshold) :
				"Outside sentence likelihoods are not consistent; found difference of " + diff + ".";
		}
		
		// Check that the inside and outside predictions agree
		double diff = Math.abs(sd.insideRoot - avgOutsideFull);
		avgDiff += diff;
		assert(diff < threshold) :
			"Inside/outside arrays disagree on sentence prob by " + diff + ".";
		
		return avgDiff;
	}
}
