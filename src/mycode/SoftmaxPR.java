package mycode;

import learning.CorpusPR;
import learning.stats.TrainStats;
import model.AbstractCountTable;
import model.AbstractModel;
import model.AbstractSentenceDist;
import constraints.CorpusConstraints;
import depparsing.model.DepModel;

public class SoftmaxPR extends CorpusPR {

	protected double sigma = 0;

	public SoftmaxPR(AbstractModel model, CorpusConstraints cstraints,
			double sigma) {
		super(model, cstraints);
		this.sigma = sigma;
	}

	@Override
	public void corpusEStep(AbstractCountTable counts,
			AbstractSentenceDist[] sentenceDists, TrainStats stats) {
		((DepModel) model).exponentiateParameters(1 / (1 - sigma));
		super.corpusEStep(counts, sentenceDists, stats);
	}
}
