package depparsing.constraints;

import java.util.Arrays;
import java.util.Random;

import model.AbstractSentenceDist;
import optimization.gradientBasedMethods.ProjectedObjective;
import util.ArrayMath;
import util.LogSummer;
import depparsing.constraints.L1LMax.SentenceChildParent;
import depparsing.model.DepSentenceDist;

public class L1LMaxObjective extends ProjectedObjective {

	private final L1LMax parent;
	private Cache cache;
	static final double epsilon = 0.0001;

	class Cache{
		public double[] expectations;
		AbstractSentenceDist[] posteriors;
		public double logLikelihood;
		boolean stale;
		
		Cache(AbstractSentenceDist[] posteriors){
			stale = true;
			this.posteriors = posteriors;
			refreshExpectationsAndLikelihood();
		}
		
		private void refreshExpectationsAndLikelihood() {
			if (expectations==null) expectations = new double[getNumParameters()];
			logLikelihood = 0;
			int index=0;
			for (int constr = 0; constr < parent.edge2scp.length; constr++){
				for (int i = 0; i < parent.edge2scp[constr].length; i++) {
					expectations[index] = 0;
					SentenceChildParent scp = parent.edge2scp[constr][i];
					assert parent.param2scp[index]==scp;
					// now we need to sum all the parents for each 
					// sentence-childid corresponding to constraint feature i
					for(int pid:scp.parents){
						if (pid < 0){
							expectations[index] += Math.exp(((DepSentenceDist)posteriors[scp.s]).getRootPosterior(scp.c));
						} else {
							double p = Double.NEGATIVE_INFINITY;
							DepSentenceDist sd = (DepSentenceDist) posteriors[scp.s];
							for(int v = 0; v < sd.nontermMap.childValency; v++)
								p = LogSummer.sum(p, sd.getChildPosterior(scp.c, pid, v));
							expectations[index] += Math.exp(p);
						}
					}
					index ++;
				}
			}
			assert (index == parameters.length);
			for (int sent = 0; sent < posteriors.length; sent++) {
				logLikelihood += ((DepSentenceDist)posteriors[sent]).insideRoot;
			}
		}

		/** runs inside-outside on the entire corpus, and updates expectations and likelihood */
		void update() {
			// initialize the parameters to the original children. 
			for (int sent = 0; sent < posteriors.length; sent++) {
				DepSentenceDist sd = ((DepSentenceDist)posteriors[sent]);
				double[][][] child = sd.child;
				int numWords = sd.depInst.numWords;
				for (int c = 0; c < numWords; c++) {
					sd.root[c] = parent.originalRoots[sent][c];
					for (int p = 0; p < numWords; p++) {
						System.arraycopy(parent.originalChildren[sent][c][p], 0, child[c][p], 0, child[c][p].length);
					}
				}
				
			}
			int index = 0;
			// update them with the lambda values
			for (int constr = 0; constr < parent.edge2scp.length; constr++){
				for (int i = 0; i < parent.edge2scp[constr].length; i++) {
					SentenceChildParent scp = parent.edge2scp[constr][i];
					assert parent.param2scp[index] == scp;
					DepSentenceDist sd = (DepSentenceDist) posteriors[scp.s];
					for(int pid:scp.parents)
					if (pid < 0){
						sd.root[scp.c] -= parameters[index];
					} else {
						for(int v = 0; v < sd.nontermMap.childValency; v++)
							sd.child[scp.c][pid][v] -= parameters[index];
					}
					index++;
				}
			}
			if (index != parameters.length) throw new AssertionError("Looks like we didn't use all the parameters!");

			for (int sent = 0; sent < posteriors.length; sent++) {
				((DepSentenceDist)posteriors[sent]).computeIO();
			}

			refreshExpectationsAndLikelihood();
			gradient = cache.expectations.clone();
			projectGradient(gradient);
			// FIXME: do we need -ve gradient or gradient?
			for (int i = 0; i < gradient.length; i++) {
				gradient[i] = -gradient[i];
			}
			stale = false;
		}
		
		void setStale(){ stale = true;}
	}
	
	public L1LMaxObjective(double[] lambda, L1LMax parent, AbstractSentenceDist[] posteriors) {
		this.parameters = lambda;
		this.parent = parent;
		cache = new Cache(posteriors);
		cache.update();
	}
	
	@Override
	public double[] getGradient() {
		if (cache.stale) {
			cache.update();
		}
		return this.gradient;
	}

	@Override
	public double getValue() {
		if (cache.stale) cache.update();
		return cache.logLikelihood;
	}

	@Override
	public void setParameter(int index, double value) {
		parameters[index]=value;
		cache.setStale();
		doLambdaProjection(parameters);
		cache.update();
	}

	@Override
	public double[] projectPoint(double[] point) {
		// FIXME: do we really need to copy the point, or can we destroy it?
		double[] res = point.clone();
		doLambdaProjection(res);
		return res;
	};
	
	boolean doTestGradient = false;
	@Override
	public void setParameters(double[] params) {
		if(parameters == null){
			parameters = new double[params.length];
		}
		updateCalls++;
		if(doTestGradient){
			doTestGradient = false;
			testGradient();
			doTestGradient = true;
		}
		cache.setStale();
		System.arraycopy(params, 0, parameters, 0, parameters.length);
		doLambdaProjection(parameters);
	}

	@Override
	public  void setInitialParameters(double[] params){
		parameters = params;
		doLambdaProjection(parameters);
	}

	double testGradientStep = epsilon;
	int numdimstocheck = 10;
	//@Override
	/**
	 * uses sampling to see whether the gradient is the same as taking a small step. 
	 * FIXME: not yet implemented. 
	 */
	public void testGradient() {
		// choose some dimensions
		double[] grad = getGradient().clone();
		projectGradient(grad);
		double origObjective = getValue();
		double[] origLambda = parameters.clone();
		double[] newLambda = parameters.clone();
		Random r = new Random(0);
//		int[] dims = new int[numdimstocheck];
		int[] indices = new int[numdimstocheck];
		double[] algebraicGradient = new double[numdimstocheck];
		double[] numericalGradientP = new double[numdimstocheck];
		double[] numericalGradientN = new double[numdimstocheck];
 		for (int i = 0; i < numdimstocheck; i++) {
//			dims[i] = r.nextInt(grad.length);
			indices[i] = r.nextInt(grad.length);
			algebraicGradient[i] = grad[indices[i]];
			double myStepSize = this.testGradientStep*grad.length/((double)parent.edge2scp.length);
			{
				newLambda[indices[i]] += myStepSize;
				setParameters(newLambda);
				double currObjective = getValue();
				numericalGradientP[i] = (currObjective - origObjective)/myStepSize;
				newLambda[indices[i]] = origLambda[indices[i]];
			} {
				newLambda[indices[i]] -= myStepSize;
				setParameters(newLambda);
				double currObjective = getValue();
				numericalGradientN[i] = (origObjective - currObjective)/myStepSize;
				newLambda[indices[i]] = origLambda[indices[i]];
			}
			
		}
 		System.out.println(String.format("2norm^2= %.2f",ArrayMath.twoNormSquared(parameters))+
 				String.format("   value=%.2f",origObjective));
 		double [] numerical = new double[numdimstocheck];
 		for (int i = 0; i < numerical.length; i++) {
			double min = Math.min(numericalGradientN[i], numericalGradientP[i]);
			double max = Math.max(numericalGradientN[i], numericalGradientP[i]);
			if (min <= algebraicGradient[i] && algebraicGradient[i] <= max) numerical[i] = algebraicGradient[i];
			if (min > algebraicGradient[i]) numerical[i] = min;
			if (max < algebraicGradient[i]) numerical[i] = max;
			// if alg > 0, then we will add. In this case we want numP > 0
			// if alg < 0, tehn we will subtract, and we want numM < 0
			if (algebraicGradient[i] > 0 && numericalGradientP[i] <= 0){
				System.out.println(" *** ASSERT FAILED ***"+
						String.format(" - %%11s  nP=%.4f  nN=%.4f  a=%.4f  l%.4f (%.4f)    p%.4f (%.4f)", 
						//parent.cstraints.constraint2string(dims[i]), // FIXME
						numericalGradientP[i], numericalGradientN[i], 
						algebraicGradient[i],
						origLambda[indices[i]],
						ArrayMath.sum(origLambda), // FIXME .value[dims[i]]
						cache.expectations[indices[i]],
						ArrayMath.max(cache.expectations) // FIXME .value[dims[i]]
						));
			}
		}
 		double cosine = ArrayMath.cosine(numerical, algebraicGradient);
 		System.out.println("Test gradient (num ,algebraic) cos = "+cosine+" (angle "+Math.acos(cosine)+")");
 		if(cosine < 0.9999){
 	 		double cosineP = ArrayMath.cosine(numericalGradientP, algebraicGradient);
 	 		System.out.println("Test gradient (num+,algebraic) cos = "+cosineP+" (angle "+Math.acos(cosineP)+")");
 	 		double cosineN = ArrayMath.cosine(numericalGradientN, algebraicGradient);
 	 		System.out.println("Test gradient (num-,algebraic) cos = "+cosineN+" (angle "+Math.acos(cosineN)+")");
 			System.out.println("     type      num+  num-  algebraic lambda (sum) posteriors (max)");
 			for (int i = 0; i < numericalGradientP.length; i++) {
				System.out.println(String.format(" - %%11s  %.4f  %.4f  %.4f  l%.4f (%.4f)    p%.4f (%.4f)", 
						//parent.cstraints.constraint2string(dims[i]),
						numericalGradientP[i], numericalGradientN[i], 
						algebraicGradient[i],
						origLambda[indices[i]],
						ArrayMath.sum(origLambda), //FIXME .value[dims[i]]
						cache.expectations[indices[i]], 
						ArrayMath.max(cache.expectations) // FIXME .value[dims[i]]
						));
			}
 		}
 		setParameters(origLambda);
 		getValue();
	}

	
	/**
	 * This projects the gradient by doing:
	 * gradient <= (project(parameters + epsilon * gradient) - parameters)/epsilon
	 * which is identical to:
	 *  - take a small step in the gradien direction
	 *  - do the projection 
	 *  - see in what direction you actually took a small step. 
	 */
	private void projectGradient(double[] gradient) {
		ArrayMath.timesEquals(gradient, epsilon);
		ArrayMath.plusEquals(gradient, parameters, 1);
		doLambdaProjection(gradient);
		ArrayMath.plusEquals(gradient, parameters, -1);
		ArrayMath.timesEquals(gradient, 1/epsilon);
	}

	private void doLambdaProjection(double[] lambda) {
		int index = 0;
		for (int edgeType = 0; edgeType < parent.edge2scp.length; edgeType++) {
			assert parent.param2scp[index] == parent.edge2scp[edgeType][0];
			double[] ds = new double[parent.edge2scp[edgeType].length];
			assert parent.param2scp[index+ds.length-1] == parent.edge2scp[edgeType][ds.length-1];
			System.arraycopy(lambda, index, ds, 0, ds.length);
			assert(ArrayMath.sumPart(lambda, index, index+ds.length) == ArrayMath.sum(ds));
			double theta = doSimplexProjection(ds, parent.getConstraintStrength(edgeType));
			for (int i = 0; i < ds.length; i++) {
				double v = lambda[index+i];
				lambda[index+i] = Math.max(0, v-theta);
			}
			// project a second time, in case numerical imprecision made the projection inexact.
			// Note: without this numerical gradient didn't quite equal computed gradient; not sure why though. 
			System.arraycopy(lambda, index, ds, 0, ds.length);
			theta = doSimplexProjection(ds, parent.getConstraintStrength(edgeType));
			for (int i = 0; i < ds.length; i++) {
				double v = lambda[index+i];
				lambda[index+i] = Math.max(0, v-theta);
			}
			// test that simplex projection is not broken... 
			if (theta > 0 && !almost(ArrayMath.sumPart(lambda, index, index+ds.length),parent.getConstraintStrength(edgeType), 1e-2 ) ){
				System.out.println("  ***** simplex projection failure! theta="+theta);
				System.out.println(" length = "+ds.length+" sum="+ArrayMath.sumPart(lambda, index, index+ds.length)+" cstrength"+parent.getConstraintStrength(edgeType));
				// FIXME: temp just for testing..
				throw new RuntimeException();
			}
			index += ds.length;
		}
		if (index != lambda.length) throw new AssertionError("We're not using all the lambda values in projection!");
	}

	
	/**
	 * Computes the theta that we need to subtract from all the entries of 
	 * v in order to project it onto the scaled simplex. 
	 * WARNING : this destructively modifies ds! 
	 * @param ds
	 * @return
	 */
	protected double doSimplexProjection(double[] ds, double scale) {
		for (int i = 0; i < ds.length; i++) ds[i] = ds[i]>0? ds[i]:0;
		double sum = ArrayMath.sum(ds);
		if (sum <= scale && !almost(sum,scale)) return 0;
		sortDescending(ds);
//		double dsum = MathUtil.sum(ds);
//		if (Math.abs(dsum-sum)>0) System.out.println("sum = "+sum+" sorted = "+MathUtil.sum(ds)+"diff = "+(dsum-sum));
		sum=0;
//		System.out.println("mu = "+Arrays.toString(ds));
		double oldSum = Double.NaN;
		for (int i =  0; i <ds.length; i++) {
//			System.out.println("mu["+i+"] = "+ds[i]);
			sum+= ds[i];
			// number used so far is i+1.
			double theta = (sum-scale)/(i+1);
//			System.out.println("theta["+i+"] = "+theta);
			if (ds[i]<theta) 
				return (oldSum-scale)/(i);
			oldSum = sum;
		}
		// they're all too big!
		return (sum - scale)/ds.length;
	}

	private void sortDescending(double[] ds){
		for (int i = 0; i < ds.length; i++) ds[i] = -ds[i];
		Arrays.sort(ds);
		for (int i = 0; i < ds.length; i++) ds[i] = -ds[i];
	}
	
	protected boolean almost(double a, double b, double prec){
		return Math.abs(a-b)/Math.abs(a+b) <= prec || (almostZero(a) && almostZero(b));
	}

	protected boolean almost(double a, double b){
		return Math.abs(a-b)/Math.abs(a+b) <= 1e-10 || (almostZero(a) && almostZero(b));
	}

	boolean almostZero(double a) {
		return Math.abs(a) <= 1e-30;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(getClass().getCanonicalName()).append(" with ");
		sb.append(parameters.length).append(" parameters and ");
		sb.append(parent.edge2scp.length).append(" constraints");
		return sb.toString();
	}

	
}
