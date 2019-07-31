package yong.depmodel;

import java.util.ArrayList;
import java.util.List;

import depparsing.data.DepCorpus;
import depparsing.model.NonterminalMap;
import depparsing.util.Lambda;



public class test {
	public final double root[];

	// Child probabilities,
	// indexed by child POS tag, parent POS tag, direction, and child existence
	public final double child[][][][];

	// Stop/continue probabilities,
	// indexed by POS tag, direction, child existence, and the decision (stop/continue)
	public final double decision[][][][];
	public final int numTags = 50;
	public final int childValency = 2;
	public final int decisionValency = 2;
	
	private static List<Integer> chdAndDesicionANNList;
	
	public test(/*DepCorpus corpus, */int decisionValency, int childValency) {
		//this.corpus = corpus;
		//this.numTags = corpus.getNrTags();
		root = new double[numTags];
		child = new double[numTags][numTags][2][childValency];
		decision = new double[numTags][2][decisionValency][2];
		//nontermMap = new NonterminalMap(decisionValency, childValency);
		double ini = 0.5;
		for(int i = 0; i < root.length; i++){
			root[i] = ini;
		}
		for(int i = 0; i < numTags; i++){
			for(int j = 0; j < numTags; j++){
				for(int m = 0; m < 2; m++){
					for(int n = 0; n < childValency; n++){
						child[i][j][m][n] = ini;
					}
				}
			}
		}
		for(int i = 0; i < numTags; i++){
			for(int j = 0; j < 2; j++){
				for(int m = 0; m < decisionValency; m++){
					for(int n = 0; n < 2; n++){
						decision[i][j][m][n] = ini;
					}
				}
			}
		}
		double sigma = 0.1;//viterbi
		exponentiateParameters(1 / (1 - sigma));
		System.out.println("test");
	}
	/**
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		System.out.println("testbegin!");
	    //test test1 = new test(2,2);
		chdAndDesicionANNList = new ArrayList<>();
		for (int i = 0 ; i < 10 ; i ++){
			chdAndDesicionANNList.add(Integer.valueOf(i));
		}
		System.out.println(chdAndDesicionANNList.toString());
		chdAndDesicionANNList.remove(0);
		System.out.println(chdAndDesicionANNList.toString());
		System.out.println(chdAndDesicionANNList.get(0));
		chdAndDesicionANNList.add(20);
		System.out.println(chdAndDesicionANNList.toString());
	}

	/**
	 * 
	 * @param exponent
	 */
	public  void exponentiateParameters(double exponent) { // TODO add this
		// method to
		// AbstractModel
		apply(new Lambda.Two<Double, Double, Double[]>() {
		public Double call(Double p1, Double[] p2) {
		return p1 * p2[0]; // p1 is log probability
		}
		}, new Double[] { exponent });
	}	
	/**
	 * 
	 * @param function
	 * @param args
	 */
	public  void apply(Lambda.Two<Double, Double, Double[]> function, Double[] args) {
		for(int i = 0; i < numTags; i++) {
			root[i] = function.call(root[i], args);
			
			for(int dir = 0; dir < 2; dir++) {
				for(int j = 0; j < numTags; j++)
					for(int v = 0; v < childValency; v++)
						child[i][j][dir][v] = function.call(child[i][j][dir][v], args);
				
				for(int v = 0; v < decisionValency; v++)
					for(int choice = 0; choice < 2; choice++)
						decision[i][dir][v][choice] = function.call(decision[i][dir][v][choice], args);
			}
		}
	}
}
