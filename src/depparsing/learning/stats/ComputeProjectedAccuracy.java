package depparsing.learning.stats;

import learning.EM;
import learning.stats.TrainStats;
import constraints.CorpusConstraints;
import depparsing.model.DepModel;
import depparsing.model.DepSentenceDist;

public class ComputeProjectedAccuracy extends TrainStats<DepModel,DepSentenceDist> {

	@Override
	public String getPrefix() {
		return "ProjAcc";
	}
	
	@Override
	 public void eStepAfterConstraints(DepModel model,
			 EM em,
			 CorpusConstraints constraints,
			 DepSentenceDist[] sentenceDists){
		int[][] parses = new int[sentenceDists.length][];
		//hanwj10.15
		//double accuracies[] = depparsing.decoding.CKYParser.computeAccuracy(sentenceDists, parses, model.corpus);
		//System.out.println("Iter:" + em.getCurrentIterationNumber()+" Accuracy project for train "  +  ": directed " + accuracies[0] + ", undirected " + accuracies[1]);
        
    }
}
