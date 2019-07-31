package nn;

import java.util.ArrayList;
import java.util.HashMap;
import org.ejml.simple.SimpleMatrix;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import org.ejml.alg.dense.mult.MatrixDimensionException;
import org.ejml.simple.SimpleMatrix;
import org.kohsuke.args4j.CmdLineException;

import depparsing.util.GrammarMath;
import depparsing.util.Pair;
import dic.Word;

public class Net2SoftTwoLayersNoBias {
	public SimpleMatrix weight1Para;
	public SimpleMatrix weight2Para;
	private Node parNode;
	
	private SimpleMatrix inputVec;
	private SimpleMatrix hidVec1;
	private SimpleMatrix hidVec2;
	//private SimpleMatrix valencyVec; not used
	
	private Node childNode;
	
	
	private int vlcNum;
	public final int dim;
	private int valencyDim;
	private int tagDim;
	private int dirDim;
	
	
	public final double outputProb;
	public ArrayList<Word> negSet;
	public HashMap<String, SimpleMatrix> gradOfWordsInNegSet;//child name and tag
	public HashMap<String, SimpleMatrix> gradOfTagsInNegSet;
	private final int dicSize;
	
	public double logSum;
	
	public ArrayList<Double> scoresInDic;
	private ArrayList<SimpleMatrix> gradOfFunctionOfInput;
	private ArrayList<SimpleMatrix> gradOfFunctionOfW1;
	private ArrayList<SimpleMatrix> gradOfFunctionOfW2;
	
	// no root net.
//	public Net2SoftTwoLayers(SimpleMatrix rootVec, Node childNode, 
//			SimpleMatrix initWeight, int negSize){
//		this.childNode = childNode;
//		this.tagDim = this.childNode.word.tagVec.numRows();
//		this.dim = childNode.dimension;
//		this.weightPara = initWeight;
//		this.inputVec = new SimpleMatrix(dim, 1);
//		for(int i = 0; i < dim ; i++){
//			this.inputVec.set(i, 0, rootVec.get(i, 0));
//		}
//		hidVec = this.weightPara.mult(this.inputVec);
//		// add nonlinear mapping
//		for(int i = 0; i < hidVec.numRows(); i++){
//			if(hidVec.get(i, 0) <= 0)
//				hidVec.set(i, 0, 0);
//		}
//		
//		this.dicSize = negSize;
//		SimpleMatrix vec = this.combineVector(childNode.word.wordVec, childNode.word.tagVec);
//		this.outputProb = hidVec.transpose().mult(vec).trace();
//	}
	
	public Net2SoftTwoLayersNoBias(Node parNode, Node childNode, SimpleMatrix initWeight1, SimpleMatrix initWeight2, SimpleMatrix valencyv, SimpleMatrix dirVec,
			int valanceNum, int valanceInx, int dicSize) throws Exception{
		//this.valencyVec = valencyv;
		this.parNode = parNode;
		this.childNode = childNode;
		
		this.valencyDim = valencyv.numRows();
		this.dim = parNode.dimension;

		this.dirDim = dirVec.numRows();
		
		this.tagDim = this.childNode.word.tagVec.numRows();

		this.vlcNum = valanceNum;
		this.weight1Para = initWeight1;
		this.weight2Para = initWeight2;
		this.inputVec = new SimpleMatrix(dim + tagDim + valencyDim + dirDim, 1);
		for(int i = 0; i < dim ; i++){
			this.inputVec.set(i, 0, this.parNode.word.wordVec.get(i, 0));
		}
		
		for(int i = dim; i < dim + tagDim; i++){
			this.inputVec.set(i, 0, this.parNode.word.tagVec.get(i - dim, 0));
		}
		
		for(int i = dim + tagDim; i < dim + tagDim + valencyDim; i++){
			this.inputVec.set(i, 0, valencyv.get(i - dim - tagDim, 0));
		}
		
		for(int i = dim + tagDim + valencyDim; i < dim + tagDim + valencyDim + dirDim; i++){
			this.inputVec.set(i, 0, dirVec.get(i - dim - tagDim - valencyDim, 0));
			
		}
		
		hidVec1 = this.weight1Para.mult(this.inputVec);
		
		//add nonlinear mapping
		for(int i = 0; i < hidVec1.numRows(); i++){
			if(hidVec1.get(i, 0) < 0)
				hidVec1.set(i, 0, 0);                                 //hidVec is pow(3)?
		}
		
		hidVec2 = this.weight2Para.mult(this.hidVec1);
		
		for(int i = 0; i < hidVec2.numRows(); i++){
			if(hidVec2.get(i, 0) < 0)
				hidVec2.set(i, 0, 0);                                 //hidVec is pow(3)?
		}
		//this.negSize = negSize;		
		this.dicSize = dicSize;
		SimpleMatrix chdWordAndTagVec = this.combineVector(this.childNode.word.wordVec, this.childNode.word.tagVec);
		this.outputProb = hidVec2.transpose().mult(chdWordAndTagVec).trace();
		

		
	}
	
	public void genNegSet(HashMap<String, Integer> dic, HashMap<String, String> word2tag,
			HashMap<String, SimpleMatrix> word2vec, HashMap<String, SimpleMatrix> tag2vec){
		
		//this.dicSize  = dic.size();
		negSet = new ArrayList<>();
		
		for(Map.Entry<String, Integer> wname : dic.entrySet()){
			String name = wname.getKey();
			if(word2tag.get(name) != null){
				Word w = new Word(name, word2tag.get(name), word2vec.get(name), tag2vec.get(word2tag.get(name)));
				negSet.add(w);
			}
		}
		scoresInDic = new ArrayList<>();
		logSum = GrammarMath.LOGZERO;
		for(int i = 0; i < this.negSet.size(); i++){
			SimpleMatrix chi = this.combineVector(this.negSet.get(i).wordVec, this.negSet.get(i).tagVec);
			double s = this.hidVec2.transpose().mult(chi).trace();
			scoresInDic.add(s);
//			if(s > 1)
//				System.out.print(s + " ");
			
			logSum = GrammarMath.log_sum_exp(s, logSum);
		}
		//System.out.print("\n");
	}
	
	public double forward(Word w){
		double p = 0.0;
		SimpleMatrix vec = null;
		vec = this.combineVector(w.wordVec, w.tagVec);
		if(w.equals( this.childNode.word )){//softmax + log here!	
			p = (1 - Math.exp(this.softmaxFunction(this.hidVec2.transpose().mult(vec).trace())));//(1.0 /this.dicSize) delected?
		}
		else{
			p = - Math.exp(this.softmaxFunction(this.hidVec2.transpose().mult(vec).trace()));
		}
		return p;
	}
	/**
	 * 
	 * @throws Exception
	 */
	public void backwardOfChd() throws Exception{
		if(this.negSet == null){
			System.out.println("Negtive set should be generated before running backward!");
			System.exit(0);
		}
		this.gradOfWordsInNegSet = new HashMap<>();
		this.gradOfTagsInNegSet = new HashMap<>();
		
		
		for(int i = 0; i < this.negSet.size(); i++){
			
			SimpleMatrix grad = this.gradientOfchd(this.negSet.get(i));
			if(Double.isNaN(grad.get(1,0))){
				System.out.println("NAN in backwardOfChd");
				throw new Exception();	
			}
			this.gradOfWordsInNegSet.put(this.negSet.get(i).getWordName(), grad.extractMatrix(0, this.dim, 0, 1));
			if(!this.gradOfTagsInNegSet.containsKey(this.negSet.get(i).tagName))
				this.gradOfTagsInNegSet.put(this.negSet.get(i).tagName, grad.extractMatrix(this.dim, this.dim + this.tagDim, 0, 1));
			else
				this.gradOfTagsInNegSet.put(this.negSet.get(i).tagName, 
						this.gradOfTagsInNegSet.get(this.negSet.get(i).tagName).plus(grad.extractMatrix(this.dim, this.dim + this.tagDim, 0, 1)));
		}
		
	}
	
	public SimpleMatrix gradientOfchd(Word w){
		//SimpleMatrix g = this.weightPara.mult(this.inputVec).scale( forward(w));
		// add nonlinear mapping
//		SimpleMatrix tmp = this.weightPara.mult(this.inputVec);
//		for(int i = 0; i < tmp.numRows(); i++){
//			if(tmp.get(i, 0) < 0)
//				tmp.set(i, 0, Math.abs(tmp.get(i, 0)));
//		}
		//SimpleMatrix g = tmp.scale(forward(w));
		SimpleMatrix g = this.hidVec2.scale(forward(w));
		
		return g;
	}

	public SimpleMatrix gradientOfinput(HashMap<String, Integer> dic){//dic is not used?
		
		SimpleMatrix g = null;
		//int index = dic.get(this.childNode.word.wordName);// childnode here?
		int index = 0;
		for(int i = 0; i < this.dicSize; i++){
			if(this.childNode.word.wordName.equals(this.negSet.get(i).wordName)){
				index = i;
				break;
			}
		}
		for(int i = 0; i < this.dicSize; i++){
			if(i == 0){
				double sc = Math.exp(this.scoresInDic.get(i));
//				if(sc > 100)
	//				System.out.println(sc);
				g = (this.gradOfFunctionOfInput.get(index).minus(this.gradOfFunctionOfInput.get(i))).scale(sc); 
			}
			else{
				double sc = Math.exp(this.scoresInDic.get(i));
		//		if(sc > 100){
			//		System.out.println(sc);
				//	System.exit(0);
				//}
				SimpleMatrix tmp = (this.gradOfFunctionOfInput.get(index).minus(this.gradOfFunctionOfInput.get(i))).scale(sc); 
				g = g.plus(tmp);
			}
		}
		
		g = g.scale(1.0 / Math.exp(logSum));
		return g;
	}
	
	public SimpleMatrix gradientOfW1(Word w, HashMap<String, Integer> dic) throws Exception{//dic is not used
		SimpleMatrix g = null;
		//int index = dic.get(w.wordName);
		int index = 0;
		for(int i = 0; i < this.dicSize; i++){
			if(w.wordName.equals(this.negSet.get(i).wordName)){
				index = i;
				break;
			}
		}
		for(int i = 0; i < this.dicSize; i++){
			if(i == 0){
				double sc = Math.exp(this.scoresInDic.get(i));
				g = (this.gradOfFunctionOfW1.get(index).minus(this.gradOfFunctionOfW1.get(i))).scale(sc); 
			}
			else{
				double sc = Math.exp(this.scoresInDic.get(i));
				SimpleMatrix tmp = (this.gradOfFunctionOfW1.get(index).minus(this.gradOfFunctionOfW1.get(i))).scale(sc); 
				g = g.plus(tmp);
			}
		}
		
		g = g.scale(1.0 / Math.exp(logSum));                          
		return g;
	}
	
	public SimpleMatrix gradientOfW2(Word w, HashMap<String, Integer> dic) throws Exception{//dic is not used
		SimpleMatrix g = null;
		//int index = dic.get(w.wordName);
		int index = 0;
		for(int i = 0; i < this.dicSize; i++){
			if(w.wordName.equals(this.negSet.get(i).wordName)){
				index = i;
				break;
			}
		}
		for(int i = 0; i < this.dicSize; i++){
			if(i == 0){
				double sc = Math.exp(this.scoresInDic.get(i));
				g = (this.gradOfFunctionOfW2.get(index).minus(this.gradOfFunctionOfW2.get(i))).scale(sc); 
			}
			else{
				double sc = Math.exp(this.scoresInDic.get(i));
				SimpleMatrix tmp = (this.gradOfFunctionOfW2.get(index).minus(this.gradOfFunctionOfW2.get(i))).scale(sc); 
				g = g.plus(tmp);
			}
		}
		
		g = g.scale(1.0 / Math.exp(logSum));                          
		return g;
	}
	
    /**
     * 
     * @return
     * @throws Exception
     */
	public SimpleMatrix backwardOfW1(HashMap<String, Integer> dic) throws Exception{
		return gradientOfW1(this.childNode.word, dic);

	}

	public SimpleMatrix backwardOfW2(HashMap<String, Integer> dic) throws Exception{
		return gradientOfW2(this.childNode.word, dic);

	}
	/**
	 * 
	 * @param val
	 * @return
	 */
	public double sigmoidFunction(double val){
		return 1.0 / (1 + Math.exp(-1 * val));
	}
	
	public double softmaxFunction(double val){
		if(this.negSet == null){
			System.out.println("Negtive set should be generated before running backward!");
			System.exit(0);
		}

		return val - logSum;
	}
	public SimpleMatrix combineVector(SimpleMatrix vec1, SimpleMatrix vec2){
		int row1 = vec1.numRows();
		int row2 = vec2.numRows();
		
		SimpleMatrix vec = new SimpleMatrix(vec1.numRows() + vec2.numRows(), 1);
		
		for(int i = 0; i < row1; i++){
			vec.set(i, 0, vec1.get(i, 0));
		}
		
		for(int i = 0; i < row2; i++){
			vec.set(i + row1, 0, vec2.get(i, 0)); 
		}
		
		
		return vec;
	}
	
	public void calGrads(){
		gradOfFunctionOfInput = new ArrayList<>();
		gradOfFunctionOfW1 = new ArrayList<>();
		gradOfFunctionOfW2 = new ArrayList<>();
		
		
		for(int i = 0; i < this.dicSize; i++){
			
			SimpleMatrix vec = this.combineVector(negSet.get(i).wordVec, negSet.get(i).tagVec);
		    assert(vec.numRows() == this.hidVec2.numRows()): "Dim wrong!";
			SimpleMatrix gw2 = new SimpleMatrix(vec);
			for(int ind = 0; ind < vec.numRows(); ind++){
				if(this.hidVec2.get(ind, 0) <= 0){
					gw2.set(ind, 0, 0);
				}
			}
			this.gradOfFunctionOfW2.add(gw2.mult(this.hidVec1.transpose()));
			
			SimpleMatrix p1 = new SimpleMatrix(vec);
			for(int ind = 0; ind < vec.numRows(); ind++){
				if(this.hidVec2.get(ind, 0) <= 0){
					p1.set(ind, 0, 0);
				}
			}
			SimpleMatrix p2 = new SimpleMatrix(this.weight2Para.transpose().mult(p1));
			assert(p2.numRows() == this.hidVec1.numRows()): "Dim wrong!";
			for(int ind = 0; ind < p2.numRows(); ind++){
				if(this.hidVec1.get(ind, 0) <= 0){
					p2.set(ind, 0, 0);
				}
			}
			this.gradOfFunctionOfInput.add(this.weight1Para.transpose().mult(p2));
			
			this.gradOfFunctionOfW1.add(p2.mult(this.inputVec.transpose()));
		}
		
		
	}
	

	public double gradChildCheck(double eps, int row, int col){
		
		SimpleMatrix chdWordAndTagVec = this.combineVector(this.childNode.word.wordVec, this.childNode.word.tagVec);	
		SimpleMatrix newVec = new SimpleMatrix(chdWordAndTagVec.numRows(), 1);
		newVec.set(row, col, eps);
		SimpleMatrix nextChildVec = chdWordAndTagVec.plus(newVec);
		SimpleMatrix lastChildVec = chdWordAndTagVec.minus(newVec);
		

		

		double nextLogSum = GrammarMath.LOGZERO;
		double lastLogSum = GrammarMath.LOGZERO;
		
		int index = 0;
		for(int i = 0; i < this.negSet.size(); i++){
			if(this.negSet.get(i).wordName.equals(this.childNode.word.wordName))
				index = i;
		}
		
		for(int i = 0; i < this.negSet.size(); i++){
			if(i == index){
				double nexts = this.hidVec2.transpose().mult(nextChildVec).trace();
				double lasts = this.hidVec2.transpose().mult(lastChildVec).trace();
				nextLogSum = GrammarMath.log_sum_exp(nexts, nextLogSum);
				lastLogSum = GrammarMath.log_sum_exp(lasts, lastLogSum);
			}else{
				SimpleMatrix chi = this.combineVector(this.negSet.get(i).wordVec, this.negSet.get(i).tagVec);
				double nexts = this.hidVec2.transpose().mult(chi).trace();
				double lasts = this.hidVec2.transpose().mult(chi).trace();
				nextLogSum = GrammarMath.log_sum_exp(nexts, nextLogSum);
				lastLogSum = GrammarMath.log_sum_exp(lasts, lastLogSum);
				
			}
		}
		
		//this.negSize = negSize;
		double nextOutputScore = this.hidVec2.transpose().mult(nextChildVec).trace();
		double lastOutputScore = this.hidVec2.transpose().mult(lastChildVec).trace();
		
		double nextLogOutputProb = nextOutputScore - nextLogSum;
		double lastLogOutputProb = lastOutputScore - lastLogSum;
		
		return (nextLogOutputProb - lastLogOutputProb) / (2 * eps);
		
	}
	
	public double gradInputCheck(double eps, int row, int col){
		SimpleMatrix newVec = new SimpleMatrix(
				this.inputVec.numRows(), 1);
		newVec.set(row, col, eps);
		SimpleMatrix nextInputVec = this.inputVec.plus(newVec);
		SimpleMatrix lastInputVec = this.inputVec.minus(newVec);
		
		SimpleMatrix nextHidVec1 = this.weight1Para.mult(nextInputVec);
		SimpleMatrix lastHidVec1 = this.weight1Para.mult(lastInputVec);
		
		//add nonlinear mapping
		for(int i = 0; i < nextHidVec1.numRows(); i++){
			if(nextHidVec1.get(i, 0) < 0)
				nextHidVec1.set(i, 0, 0);
			if(lastHidVec1.get(i, 0) < 0)
				lastHidVec1.set(i, 0, 0);
		}
		SimpleMatrix nextHidVec2 = this.weight2Para.mult(nextHidVec1);
		SimpleMatrix lastHidVec2 = this.weight2Para.mult(lastHidVec1);
		
		//add nonlinear mapping
		for(int i = 0; i < nextHidVec2.numRows(); i++){
			if(nextHidVec2.get(i, 0) < 0)
				nextHidVec2.set(i, 0, 0);
			if(lastHidVec2.get(i, 0) < 0)
				lastHidVec2.set(i, 0, 0);
		}		

		double nextLogSum = GrammarMath.LOGZERO;
		double lastLogSum = GrammarMath.LOGZERO;
		for(int i = 0; i < this.negSet.size(); i++){
			SimpleMatrix chi = this.combineVector(this.negSet.get(i).wordVec, this.negSet.get(i).tagVec);
			double nexts = nextHidVec2.transpose().mult(chi).trace();
			double lasts = lastHidVec2.transpose().mult(chi).trace();
			nextLogSum = GrammarMath.log_sum_exp(nexts, nextLogSum);
			lastLogSum = GrammarMath.log_sum_exp(lasts, lastLogSum);
		}
		
		//this.negSize = negSize;
		SimpleMatrix chdWordAndTagVec = this.combineVector(this.childNode.word.wordVec, this.childNode.word.tagVec);
		double nextOutputScore = nextHidVec2.transpose().mult(chdWordAndTagVec).trace();
		double lastOutputScore = lastHidVec2.transpose().mult(chdWordAndTagVec).trace();
		
		double nextLogOutputProb = nextOutputScore - nextLogSum;
		double lastLogOutputProb = lastOutputScore - lastLogSum;
		
		return (nextLogOutputProb - lastLogOutputProb) / (2 * eps);
		
	}
	
	
	public double gradW1Check(double eps, int row, int col){
		SimpleMatrix newWeight = new SimpleMatrix(this.weight1Para.numRows(), this.weight1Para.numCols());
		newWeight.set(row, col, eps);
		SimpleMatrix nextWeight = this.weight1Para.plus(newWeight);
		SimpleMatrix lastWeight = this.weight1Para.minus(newWeight);
		
		SimpleMatrix nextHidVec1 = nextWeight.mult(this.inputVec);
		SimpleMatrix lastHidVec1 = lastWeight.mult(this.inputVec);
		
		//add nonlinear mapping
		for(int i = 0; i < nextHidVec1.numRows(); i++){
			if(nextHidVec1.get(i, 0) < 0)
				nextHidVec1.set(i, 0, 0);
			if(lastHidVec1.get(i, 0) < 0)
				lastHidVec1.set(i, 0, 0);
		}
		SimpleMatrix nextHidVec2 = this.weight2Para.mult(nextHidVec1);
		SimpleMatrix lastHidVec2 = this.weight2Para.mult(lastHidVec1);
		
		//add nonlinear mapping
		for(int i = 0; i < nextHidVec2.numRows(); i++){
			if(nextHidVec2.get(i, 0) < 0)
				nextHidVec2.set(i, 0, 0);
			if(lastHidVec2.get(i, 0) < 0)
				lastHidVec2.set(i, 0, 0);
		}
		

		double nextLogSum = GrammarMath.LOGZERO;
		double lastLogSum = GrammarMath.LOGZERO;
		for(int i = 0; i < this.negSet.size(); i++){
			SimpleMatrix chi = this.combineVector(this.negSet.get(i).wordVec, this.negSet.get(i).tagVec);
			double nexts = nextHidVec2.transpose().mult(chi).trace();
			double lasts = lastHidVec2.transpose().mult(chi).trace();
			nextLogSum = GrammarMath.log_sum_exp(nexts, nextLogSum);
			lastLogSum = GrammarMath.log_sum_exp(lasts, lastLogSum);
		}
		
		//this.negSize = negSize;
		SimpleMatrix chdWordAndTagVec = this.combineVector(this.childNode.word.wordVec, this.childNode.word.tagVec);
		double nextOutputScore = nextHidVec2.transpose().mult(chdWordAndTagVec).trace();
		double lastOutputScore = lastHidVec2.transpose().mult(chdWordAndTagVec).trace();
		
		double nextLogOutputProb = nextOutputScore - nextLogSum;
		double lastLogOutputProb = lastOutputScore - lastLogSum;
		
		return (nextLogOutputProb - lastLogOutputProb) / (2 * eps);
		
	}
	
	public double gradW2Check(double eps, int row, int col){
		SimpleMatrix newWeight = new SimpleMatrix(this.weight2Para.numRows(), this.weight2Para.numCols());
		newWeight.set(row, col, eps);
		SimpleMatrix nextWeight = this.weight2Para.plus(newWeight);
		SimpleMatrix lastWeight = this.weight2Para.minus(newWeight);
		
		SimpleMatrix nextHidVec1 = this.weight1Para.mult(this.inputVec);
		SimpleMatrix lastHidVec1 = this.weight1Para.mult(this.inputVec);
		
		//add nonlinear mapping
		for(int i = 0; i < nextHidVec1.numRows(); i++){
			if(nextHidVec1.get(i, 0) < 0)
				nextHidVec1.set(i, 0, 0);
			if(lastHidVec1.get(i, 0) < 0)
				lastHidVec1.set(i, 0, 0);
		}
		SimpleMatrix nextHidVec2 = nextWeight.mult(nextHidVec1);
		SimpleMatrix lastHidVec2 = lastWeight.mult(lastHidVec1);
		
		//add nonlinear mapping
		for(int i = 0; i < nextHidVec2.numRows(); i++){
			if(nextHidVec2.get(i, 0) < 0)
				nextHidVec2.set(i, 0, 0);
			if(lastHidVec2.get(i, 0) < 0)
				lastHidVec2.set(i, 0, 0);
		}
		

		double nextLogSum = GrammarMath.LOGZERO;
		double lastLogSum = GrammarMath.LOGZERO;
		for(int i = 0; i < this.negSet.size(); i++){
			SimpleMatrix chi = this.combineVector(this.negSet.get(i).wordVec, this.negSet.get(i).tagVec);
			double nexts = nextHidVec2.transpose().mult(chi).trace();
			double lasts = lastHidVec2.transpose().mult(chi).trace();
			nextLogSum = GrammarMath.log_sum_exp(nexts, nextLogSum);
			lastLogSum = GrammarMath.log_sum_exp(lasts, lastLogSum);
		}
		
		//this.negSize = negSize;
		SimpleMatrix chdWordAndTagVec = this.combineVector(this.childNode.word.wordVec, this.childNode.word.tagVec);
		double nextOutputScore = nextHidVec2.transpose().mult(chdWordAndTagVec).trace();
		double lastOutputScore = lastHidVec2.transpose().mult(chdWordAndTagVec).trace();
		
		double nextLogOutputProb = nextOutputScore - nextLogSum;
		double lastLogOutputProb = lastOutputScore - lastLogSum;
		
		return (nextLogOutputProb - lastLogOutputProb) / (2 * eps);
		
	}
	
}


