package depparsing.learning.stats;

import java.util.ArrayList;

import learning.EM;
import learning.stats.CompositeTrainStats;
import learning.stats.TrainStats;
import data.WordInstance;
import depparsing.data.DepInstance;
import depparsing.model.DepModel;
import depparsing.model.DepProbMatrix;
import depparsing.model.DepSentenceDist;


/**
 * Computes the likelihood of the model for a development corpus at each iteration.
 * The likelihood is computed before the E-Step is computed. 
 * @author javg
 *
 */
public class DevLikelihoodStats extends TrainStats<DepModel,DepSentenceDist> {

	private double likelihood;
	private double prevLikelihood;
	
	public String getPrefix(){
        return "DevLogL: ";
	}
	
	@Override
	public String printEndMStep(DepModel model,EM em) {
		prevLikelihood = likelihood;
		likelihood = 0;
		ArrayList<WordInstance> devSentences = model.corpus.devInstances.instanceList;
		CompositeTrainStats<DepModel,DepSentenceDist> stats = new CompositeTrainStats<DepModel,DepSentenceDist>();
		stats.addStats(new L1LMaxStats("0"));
		stats.eStepStart(model, em);
		
		// Save model, add backoff and normalize
		DepProbMatrix smooth = new DepProbMatrix(model.corpus, model.decisionValency, model.childValency);
		smooth.copyFrom(model.params);
		smooth.backoff(-1e3);
		smooth.logNormalize();
		
		for(int i = 0; i < devSentences.size(); i++){		
			DepSentenceDist sd = new DepSentenceDist((DepInstance)devSentences.get(i), smooth.nontermMap);
			sd.cacheModelAndComputeIO(smooth);
			stats.eStepSentenceEnd(model,em,sd);
			likelihood += sd.insideRoot;
		}
		
		String results = "devCorpus -log(likelihood) = " + util.Printing.prettyPrint(-likelihood, "0.000000E00", 9) +
				", difference = " + util.Printing.prettyPrint(likelihood - prevLikelihood, "0.000000E00", 9);		
		results+=stats.printEndEStep(model,em).replace("\n", "\n"+getPrefix());
		return results;
	}
}
