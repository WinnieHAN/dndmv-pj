package nn;
import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;
import org.kohsuke.args4j.CmdLineException;

import depparsing.util.Pair;

/**
 * This class is a neural network for predicting the output between STOP or CONTINUE.
 * Input: a parent, left or right, valancy number.  
 * @author Jy
 *
 */
public class Net {
	public final Node parNode;
	public SimpleMatrix weightPara;
	//public SimpleMatrix hid2outputPara;

	private SimpleMatrix diffPara;
	private SimpleMatrix input;
	private SimpleMatrix hidVec;
	public final double output;
	public final double target; // Defining CONTINUE is 1, STOP is 0.
	
	private int dim;
	private int tagDim;
	private int valencyDim;
	private int dirDim;
	
	public final int valanceNum;
	public final int valanceInx;
	public final boolean isLeft;
	
	private SimpleMatrix continueVec;
	private SimpleMatrix stopVec;
	public Net(Node input, SimpleMatrix weight, SimpleMatrix valencyv, 
			SimpleMatrix dirVec, int vlcNum, int vlcInx, boolean isLeft/*, double t*/,
			SimpleMatrix stopVec, SimpleMatrix continueVec, boolean isStop){
		this.parNode = input;
		
		this.tagDim = input.word.tagVec.numRows();
		this.dim = input.dimension;
		this.valencyDim = valencyv.numRows();
		this.dirDim = dirVec.numRows();
		
		this.valanceInx = vlcInx;
		this.valanceNum = vlcNum;
		this.input = new SimpleMatrix(dim + tagDim + valencyDim + dirDim, 1);
		//weightPara = new SimpleMatrix(dim + valanceNum + 2, 1);
		
		this.weightPara = weight;
	//	this.hid2outputPara = isStop == true ? stopVec : continueVec;
		
		this.continueVec = continueVec;
		this.stopVec = stopVec;
		
		//this.diffPara = isStop == true ? (stopVec.minus(continueVec)) : (continueVec.minus(stopVec));
		this.diffPara = continueVec.minus(stopVec);
		this.target = isStop == true ? 0 : 1;//gold target
		this.isLeft = isLeft;
	//	this.target = t;
		//this.target = isStop == true ? 0 : 1;
		
		// OK, signing the input vector is not funny.
		for(int i = 0; i < dim; i++){
			this.input.set(i, 0, parNode.word.wordVec.get(i, 0));// args: row, col, value
		}
		for(int i = dim; i < dim + tagDim; i++){
			this.input.set(i, 0, parNode.word.tagVec.get(i - dim, 0));
		}
		for(int i = dim + tagDim; i < dim + tagDim + valencyDim; i++){
			this.input.set(i, 0, valencyv.get(i - dim - tagDim, 0));
		}
		for(int i = dim + tagDim + valencyDim; i < dim + tagDim + valencyDim + dirDim; i++){
			this.input.set(i, 0, dirVec.get(i - dim - tagDim - valencyDim, 0));
		}
		
		assert(!Double.isInfinite(valencyv.get(0, 0))): "valencyv is Infinite!";
		assert(!Double.isInfinite(dirVec.get(0, 0))): "dirVec is Infinite!" + dirVec.toString();

		//hidden Vector
		this.hidVec = weightPara.mult(this.input); //this.input.transpose().mult(); //vector multiply
		for(int i = 0; i < hidVec.numRows(); i++){
			if(hidVec.get(i, 0) < 0)
				hidVec.set(i, 0, 0);                                 
		}
		assert(!Double.isNaN(this.hidVec.get(0, 0))): 
			"hid vec is NAN!" + this.input.toString();
		//output value
		//double currScore = this.hidVec.transpose().mult(hid2outputPara).trace(); //this is a value, not a vector
		//double currOtherScore = this.hidVec.transpose().mult(isStop == false ? stopVec : continueVec).trace();
		double currScore = this.hidVec.transpose().mult(stopVec).trace(); //this is a value, not a vector
		double currOtherScore = this.hidVec.transpose().mult(continueVec).trace();
		assert(!Double.isInfinite(currOtherScore)) : "currOtherScore is infinity!!!" + currOtherScore;
		assert(!Double.isInfinite(currScore)) : "currScore is infinity!!!" + currScore;
		double diff = currOtherScore - currScore;
		
	//	if(diff > 100)
	//		this.output = 1e-20;
	//	else
			this.output = 1.0 / (1 + Math.exp(- diff));
		assert(!Double.isNaN(diff)):
			"NAN number:\t" + Double.toString(diff) + "\tcurrOtherScore :\t" + currOtherScore;
		assert(!Double.isInfinite(this.output)) : "output is infinity!!!" + this.output + "\n\n + diff:" + diff;
	}
	
	/**
	 * return the log value of forward process
	 * @return
	*/
	public double forward(){
		/*double result = output;
		if(result < 1e-20)
			result = 1e-20;

		assert(!Double.isNaN(result)):
			"NAN number:\t" + Double.toString(result) + "\toriginal number:\t" + output;
		return -1 * Math.log(result);*/
		return this.output;
	} 
	
	public SimpleMatrix gradientOfhidPara(){//hid2out v1
		double error = this.target - this.output; 
		
		return this.hidVec.scale(error);
	}
	
	public SimpleMatrix gradientOfWeight(){
		
		double error = this.target - this.output; 
		for(int i = 0; i < this.diffPara.numRows(); i++){
			if(this.hidVec.get(i, 0) <= 0){
				this.diffPara.set(i, 0, 0);
			}
		}
		SimpleMatrix g =  this.diffPara.mult(this.input.transpose()).scale(error);
		return g;
	}
	
	
	public SimpleMatrix gradientOfinput(){
		double error = this.target - this.output;

		for(int i = 0; i < this.diffPara.numRows(); i++){
			if(this.hidVec.get(i, 0) <= 0){
				this.diffPara.set(i, 0, 0);
			}
		}
		SimpleMatrix g = this.weightPara.transpose().mult(this.diffPara).scale(error);
		
		

		assert(!Double.isInfinite(g.get(0, 0))) : "g is infinity" + g.toString() + "\n\n" + this.diffPara.toString() + "\n\n" + this.output;

		
		return g;
	}
	
	/**
	 *  Util function: gradient of sigmoid function
	 * @param val
	 * @return
	 */
	public double gradientOfSigmoid(double val){
		return this.sigmoidFunction(val) * (1 - this.sigmoidFunction(val));
	}
	
	/**
	 *  Util function: sigmoid function
	 * @param val
	 * @return
	 */
	public double sigmoidFunction(double val){
		return 1.0 / (1 + Math.exp(-1 * val));
	}

	public double getFunctionScore() {
		// TODO Auto-generated method stub
		double score = 0;
		if(this.target > 0.5){
			score =  Math.log(this.output);
			if(score < -1e10){
				score = -1e10;
			}
		}else{
			score = Math.log(1 - this.output);
			if(score < -1e10){
				score = -1e10;
			}
		}
		return score;
	}

	public double gradCheckW(double eps, int row, int col) {
		// TODO Auto-generated method stub
		SimpleMatrix newWeight = new SimpleMatrix(this.weightPara.numRows(), this.weightPara.numCols());
		newWeight.set(row, col, eps);
		SimpleMatrix nextWeight = this.weightPara.plus(newWeight);
		SimpleMatrix lastWeight = this.weightPara.minus(newWeight);
		
		SimpleMatrix nextHidVec = nextWeight.mult(this.input);
		SimpleMatrix lastHidVec = lastWeight.mult(this.input);
		for(int i = 0; i < nextHidVec.numRows(); i++){
			if(nextHidVec.get(i, 0) < 0)
				nextHidVec.set(i, 0, 0);
		}
		for(int i = 0; i < lastHidVec.numRows(); i++){
			if(lastHidVec.get(i, 0) < 0)
				lastHidVec.set(i, 0, 0);
		}
		
		double nextDiffScore;
		double lastDiffScore;
		if(this.target == 1){
			nextDiffScore = nextHidVec.transpose().mult(this.diffPara).trace();
		}else{
			nextDiffScore = -nextHidVec.transpose().mult(this.diffPara).trace();
		}
		if(this.target == 1){
			lastDiffScore = lastHidVec.transpose().mult(this.diffPara).trace();
		}else{
			lastDiffScore = -lastHidVec.transpose().mult(this.diffPara).trace();
		}
		
		double nextLogSigma, lastLogSigma;
		nextLogSigma = Math.log(sigmoidFunction(nextDiffScore ));
		lastLogSigma = Math.log(sigmoidFunction(lastDiffScore ));
		
		return (nextLogSigma - lastLogSigma) / (2 * eps);
	}

	public double gradCheckCont(double eps, int row, int col) {//check ContinueVector
		// TODO Auto-generated method stub
		SimpleMatrix newContinueVec = new SimpleMatrix(this.continueVec.numRows(), this.continueVec.numCols());
		newContinueVec.set(row, col, eps);
		SimpleMatrix nextContinueVec = this.continueVec.plus(newContinueVec);
		SimpleMatrix lastContinueVec = this.continueVec.minus(newContinueVec);
		
		double nextDiffScore;
		double lastDiffScore;
		if(this.target == 1){
			nextDiffScore = this.hidVec.transpose().mult(nextContinueVec.minus(this.stopVec)).trace();
		}else{
			nextDiffScore = -this.hidVec.transpose().mult(nextContinueVec.minus(this.stopVec)).trace();
		}
		if(this.target == 1){
			lastDiffScore = this.hidVec.transpose().mult(lastContinueVec.minus(this.stopVec)).trace();
		}else{
			lastDiffScore = -this.hidVec.transpose().mult(lastContinueVec.minus(this.stopVec)).trace();
		}

		double nextLogSigma, lastLogSigma;
		nextLogSigma = Math.log(sigmoidFunction(nextDiffScore ));
		lastLogSigma = Math.log(sigmoidFunction(lastDiffScore ));
		
		return (nextLogSigma - lastLogSigma) / (2 * eps);
	}

	public double gradCheckInput(double eps, int row, int col) {
		// TODO Auto-generated method stub
		SimpleMatrix newInput = new SimpleMatrix(this.input.numRows(), this.input.numCols());
		newInput.set(row, col, eps);
		SimpleMatrix nextInput = this.input.plus(newInput);
		SimpleMatrix lastInput = this.input.minus(newInput);
		
		SimpleMatrix nextHidVec = this.weightPara.mult(nextInput);
		SimpleMatrix lastHidVec = this.weightPara.mult(lastInput);
		for(int i = 0; i < nextHidVec.numRows(); i++){
			if(nextHidVec.get(i, 0) < 0)
				nextHidVec.set(i, 0, 0);
		}
		for(int i = 0; i < lastHidVec.numRows(); i++){
			if(lastHidVec.get(i, 0) < 0)
				lastHidVec.set(i, 0, 0);
		}
		
		double nextDiffScore;
		double lastDiffScore;
		if(this.target == 1){
			nextDiffScore = nextHidVec.transpose().mult(this.diffPara).trace();
		}else{
			nextDiffScore = -nextHidVec.transpose().mult(this.diffPara).trace();
		}
		if(this.target == 1){
			lastDiffScore = lastHidVec.transpose().mult(this.diffPara).trace();
		}else{
			lastDiffScore = -lastHidVec.transpose().mult(this.diffPara).trace();
		}

		double nextLogSigma, lastLogSigma;
		nextLogSigma = Math.log(sigmoidFunction(nextDiffScore ));
		lastLogSigma = Math.log(sigmoidFunction(lastDiffScore ));
		
		return (nextLogSigma - lastLogSigma) / (2 * eps);
	}


}
