package nn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;

import org.ejml.alg.dense.mult.MatrixDimensionException;
import org.ejml.simple.SimpleMatrix;
import org.kohsuke.args4j.CmdLineException;

import depparsing.util.Pair;
import dic.Word;

public class Net2 {
	public SimpleMatrix weightPara;
	private Node parNode;
	
	private SimpleMatrix inputVec;
	private SimpleMatrix hidVec;
	private SimpleMatrix valencyVec;
	
	private Node childNode;
	
	
	private int vlcNum;
	public final int dim;
	private int valencyDim;
	private int tagDim;
	private int dirDim;
	
	
	public final double outputProb;
	private ArrayList<Word> negSet;
	public HashMap<String, SimpleMatrix> gradOfWordsInNegSet;
	public HashMap<String, SimpleMatrix> gradOfTagsInNegSet;
	private final int negSize;

	/**
	 * this class is for root -> child
	 * @param rootVec
	 * @param childNode
	 * @param initWeight
	 * @param negSize
	 */
	
	public Net2(SimpleMatrix rootVec, Node childNode, 
			SimpleMatrix initWeight, int negSize){
		this.childNode = childNode;
		this.tagDim = this.childNode.word.tagVec.numRows();
		this.dim = childNode.dimension;
		this.weightPara = initWeight;
		this.inputVec = new SimpleMatrix(dim, 1);
		for(int i = 0; i < dim ; i++){
			this.inputVec.set(i, 0, rootVec.get(i, 0));
		}
		hidVec = this.weightPara.mult(this.inputVec);
		this.negSize = negSize;
		SimpleMatrix vec = this.combineVector(childNode.word.wordVec, childNode.word.tagVec);
		this.outputProb = hidVec.transpose().mult(vec).trace();
	}
	
	/**
	 *  constructor of the class Net2
	 * @param parNode Parent Node
	 * @param childNode Child Node
	 * @param initWeight parameters of the network
	 * @param valanceNum 
	 * @param valanceInx
	 * @param negSize
	 * @throws Exception 
	 */
	public Net2(Node parNode, Node childNode, SimpleMatrix initWeight, SimpleMatrix valencyv, SimpleMatrix dirVec,
			int valanceNum, int valanceInx, int negSize) throws Exception{
		this.valencyVec = valencyv;
		this.parNode = parNode;
		this.childNode = childNode;
		
		this.valencyDim = valencyv.numRows();
		this.dim = parNode.dimension;

		this.dirDim = dirVec.numRows();
		
		this.tagDim = this.childNode.word.tagVec.numRows();

		this.vlcNum = valanceNum;
		this.weightPara = initWeight;
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
		
		if(valanceInx > valanceNum)
			throw new Exception("valance index bigger than valance number!");

		
		hidVec = this.weightPara.mult(this.inputVec);
		this.negSize = negSize;		
		SimpleMatrix chdWordAndTagVec = this.combineVector(this.childNode.word.wordVec, this.childNode.word.tagVec);
		this.outputProb = hidVec.transpose().mult(chdWordAndTagVec).trace();
	}
	
	
	/**
	 * generate negtive set from the negtive set dictionary
	 * @param size
	 * @param dic
	 */
	public void genNegSet(ArrayList<String> negDic, HashMap<String, String> word2tag,
			HashMap<String, SimpleMatrix> word2vec, HashMap<String, SimpleMatrix> tag2vec){
		double scope = 1.0 / negDic.size();
		
		int tmpSize = negSize;
		negSet = new ArrayList<>();
		Word chd = new Word(this.childNode.word);
		negSet.add(chd);
		Random rand =new Random(1);//
		for(int idx = 0; idx < tmpSize; idx++){
		//    Random generator = new Random(1);
		//    double g = generator.nextDouble();
			
			int boxInx = (int) ( rand.nextDouble() / scope );
			String name = negDic.get(boxInx);
			Word w = null;
			
			if(word2vec.get(name) == null){
				System.out.println(name);
			}
			
		/*	if( !tag2vec.containsKey(name) ){
				throw new NullPointerException("tag name:" + name);
			}*/
			
			w = new Word(name, word2tag.get(name), word2vec.get(name), tag2vec.get(word2tag.get(name)));
			
			if(negSet.size() == negSize + 1){
				System.out.println("negSet size equals negSize!!!");
				return;
			}
			
			if(negSet.contains(w)){
				tmpSize++;
		//		System.out.println(tmpSize);
				
			}
			else
				negSet.add(w);	
			
		} 
	}
	


	public double outputScore(){
		double score = 0.0;
		score += this.forward(this.childNode.word);
			
		double forwardScore = 0.0;
		for(Word negword : this.negSet){
			forwardScore  = this.forward(negword);
			if(forwardScore < 1e-20)
				forwardScore = 1e-20;
			
			score += Math.log(forwardScore);
		}
		
		score = -1 * score;
		return score;
	}
	
	public double forward(Word w){
		double p = 0.0;
		SimpleMatrix vec = null;
		vec = this.combineVector(w.wordVec, w.tagVec);
		if(w.equals( this.childNode.word )){	
			p = 1 - this.sigmoidFunction(this.hidVec.transpose().mult(vec).trace());
		}
		else{
			p = - this.sigmoidFunction(this.hidVec.transpose().mult(vec).trace());
		}
		return p;
	}
	/**
	 * 
	 * @param w
	 * @return
	 */
	public double getFunctionScore(){
		if(this.negSet == null){
			System.out.println("Negtive set should be generated before running backward!");
			System.exit(0);
		}
		double score = getonewordFunctionScore(this.negSet.get(0));
		
		for(int i = 1; i < this.negSet.size(); i++){
			score += getonewordFunctionScore(this.negSet.get(i));
		}
		return score;
	}
	
	public double getonewordFunctionScore(Word w){
		double lss = 0.0;
		SimpleMatrix vec = null;
		vec = this.combineVector(w.wordVec, w.tagVec);
		if(w.equals( this.childNode.word )){	
			double a = this.hidVec.transpose().mult(vec).trace();
			double b = 1 - this.sigmoidFunction(a);
			lss = Math.log(b);
			if(lss < -1e10){
				lss = -1e10;
			}
			//lss = Math.log(1 - this.sigmoidFunction(this.hidVec.transpose().mult(vec).trace()));
		}
		else{
			double a = this.hidVec.transpose().mult(vec).trace();
			double b = this.sigmoidFunction(a);
			lss = Math.log(b);
			if(lss < -1e10){
				lss = -1e10;
			}
			//lss = Math.log(this.sigmoidFunction(this.hidVec.transpose().mult(vec).trace()));
		}
		return lss;
	}
	/**
	 *  backward of gradient W
	 * @param rate
	 * @return
	 * @throws Exception 
	 */
	public SimpleMatrix backwardOfW() throws Exception{
		if(this.negSet == null){
			System.out.println("Negtive set should be generated before running backward!");
			System.exit(0);
		}
		
		SimpleMatrix grad = this.gradientOfW(this.negSet.get(0));
		
		for(int i = 1; i < this.negSet.size(); i++){
			if(Double.isNaN( grad.get(0, 0) )){
				System.out.println("NAN in backwardOfW!");
				throw new Exception();
			}
			grad = grad.plus(this.gradientOfW(this.negSet.get(i)));
		}

		
		return grad;
	}
	
	/**
	 *  backward of Input vector
	 * @param rate
	 * @return
	 * @throws Exception 
	 */
	public SimpleMatrix backwardOfInput() throws Exception{
		if(this.negSet == null){
			System.out.println("Negtive set should be generated before running backward!");
			System.exit(0);
		}
		
		SimpleMatrix grad = this.gradientOfinput(this.negSet.get(0));
		
		for(int i = 1; i < this.negSet.size(); i++){
			if(Double.isNaN( grad.get(1,0) )){
				System.out.println("NAN in backwardOfInput!");
				throw new Exception();
			}
			grad = grad.plus(this.gradientOfinput(this.negSet.get(i)));
		}
		
		return grad;
	}
	
	
	/**
	 *  backward of child vector, which is in the negtive set.
	 * @param rate
	 * @return
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
			this.gradOfTagsInNegSet.put(this.negSet.get(i).tagName, grad.extractMatrix(this.dim, this.dim + this.tagDim, 0, 1));
		}
		
	}
	
	public double sigmoidFunction(double val){
		return 1.0 / (1 + Math.exp(-1 * val));
	}
	
	/**
	 *  This is gradient of parameter W
	 * @param w
	 * @return
	 * @throws Exception 
	 */
	public SimpleMatrix gradientOfW(Word w) throws Exception{
		SimpleMatrix gv = this.combineVector(w.wordVec, w.tagVec);

		SimpleMatrix g = gv.mult(inputVec.transpose()).scale( forward(w) );
		if(Double.isNaN(g.get(1, 0))){
			System.out.println("NAN! need debug!\t" + w.getWordName() + "\ttag name:\t" + w.tagName);
			throw new Exception();
		}
		return g;
	}
	
	/**
	 *  This is gradient of input
	 * @param w
	 * @return
	 */
	public SimpleMatrix gradientOfinput(Word w){
		
		SimpleMatrix gv = new SimpleMatrix(this.dim + this.tagDim, 1);
		for(int i = 0; i < this.dim; i++){
			gv.set(i, 0, w.wordVec.get(i, 0));
		}
		
		for(int i = 0; i < this.tagDim; i++){
			gv.set(i + this.dim, 0, w.tagVec.get(i, 0));
		}
		
		SimpleMatrix g = this.weightPara.transpose().mult(gv).scale( forward(w));
		
		return g;
	}
	
	/**
	 *  This is gradient of child vector
	 * @param w
	 * @return
	 */
	public SimpleMatrix gradientOfchd(Word w){
		SimpleMatrix g = this.weightPara.mult(this.inputVec).scale( forward(w));
		return g;
	}
	
	/**
	 *  Combine two vectors into one vector. 
	 *  Just like the operation (vec = [vec1 ; vec2])
	 * @param vec1
	 * @param vec2
	 * @return
	 */
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
}
