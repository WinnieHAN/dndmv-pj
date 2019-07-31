package depparsing.constraints;

import java.util.ArrayList;


public  class ProjectionStats {

	public boolean directionNotAscentDirection;
	/** No step was found */
	public boolean noZoomStepFound;
	public boolean noBracketingStepFound;
	public boolean noConvergence;
	public boolean objectiveBecomeNotANumber;
	/** Maximum evaluations for picking a step when a step was actually found*/
	public int maxStepEval=-1;
	
	public ArrayList<Integer> _numberOfSteps;
	
	/** Number of steps to converge */
	public int numberOfProjections;
//	/** Posteriors after projections
//	 * usefull to look at when no projection was possible 
//	 */
//	public Trellis[] posteriors;
	
	/** Value of the objective after the projection */
	public double finalObjective;
	/** Value of the objective before the projection */
	public double originalObjective;
	
	/** Likelihood before projection */
	public double originalLikelihood;
	
	/** Likelihood after projection */
	public double likelihood;
	
	/** The sentence already satisfied all the constrains */
	public boolean noConstrains;
	
	/** Number of forward backward*/
	public int fbcall;
	
	/** time spend in projectin*/
	long start = -1;
	long totalTime = 0;
	
	
	
//	public SentenceConstrainedProjectionStats(){
//		
//	}
	
	public ProjectionStats() {
		noZoomStepFound = false;
		_numberOfSteps = new ArrayList<Integer>();
		numberOfProjections=0;
	}
	
	public String prettyString(){
		StringBuffer res = new StringBuffer();
		res.append(" ProjectionStats:  time " + totalTime);
		res.append( " proj:"+numberOfProjections + " obj " + finalObjective + " fbCalls = "+fbcall);
		res.append("\nsteps: ");
		for(int nStep : _numberOfSteps){
			res.append(nStep + " ");
		}
		if(noZoomStepFound){
			res.append( " no step");
		}
		return res.toString();
	}
	
	public void startTime() {
		start = System.currentTimeMillis();
	}
	public void stopTime() {
		totalTime += System.currentTimeMillis() - start;
	}
	
}
