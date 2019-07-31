package mycode;

import learning.EM;
import learning.stats.TrainStats;
import model.AbstractModel;
import model.AbstractSentenceDist;

public class LikelihoodStatsWithTermination<K extends AbstractModel, J extends AbstractSentenceDist>
		extends TrainStats<K, J> {

	protected double likelihood;
	protected double prevLikelihood = Double.NEGATIVE_INFINITY;

	public static boolean terminate = false;

	public String getPrefix() {
		return "LogLES::";
	}

	@Override
	public void eStepSentenceEnd(K model, EM em, J sd) {
		likelihood += sd.getLogLikelihood();
	}

	@Override
	public String printEndEStep(K model, EM em) {
		double change = likelihood - prevLikelihood;
		if (change / -prevLikelihood < 1e-5) {
			// System.out.println("Terminated by train set likelihood: \t-log(L)="
			// + util.Printing.prettyPrint(-likelihood, "0.000000E00", 9));
			// System.exit(0);
			terminate = true;
		}

		StringBuffer s = new StringBuffer();
		s.append("Iter " + em.getCurrentIterationNumber() + "\t-log(L)="
				+ util.Printing.prettyPrint(-likelihood, "0.000000E00", 9));
		s.append(" diff=" + util.Printing.prettyPrint(change, "0.000000E00", 9));
		// Perform clean up
		prevLikelihood = likelihood;
		likelihood = 0;
		return s.toString();
	}
}
