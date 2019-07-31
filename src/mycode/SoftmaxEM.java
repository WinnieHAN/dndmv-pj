package mycode;

import java.io.IOException;
import java.io.UnsupportedEncodingException;

import learning.EM;
import learning.stats.TrainStats;
import model.AbstractCountTable;
import model.AbstractModel;
import model.AbstractSentenceDist;
import depparsing.model.DepModel;

public class SoftmaxEM extends EM {

	protected double likelihood = 0;
	protected double prevLikelihood = Double.NEGATIVE_INFINITY;

	protected double sigma = 0;

	public SoftmaxEM(AbstractModel model, double sigma) {
		super(model);
		this.sigma = sigma;
	}

	@Override
	public void em(int numIters, TrainStats stats)
			throws UnsupportedEncodingException, IOException {
		stats.emStart(this.model, this);

		AbstractSentenceDist[] sentenceDists = model.getSentenceDists();

		AbstractCountTable counts = model.getCountTable();

		for (iter = 0; iter < numIters; iter++) {
			System.out.println("\nEM Iteration " + (iter + 1));
			System.out.flush();
			stats.emIterStart(this.model, this);

			((DepModel) model).exponentiateParameters(1 / (1 - sigma));
			// TODO add this method to AbstractModel

			corpusEStep(counts, sentenceDists, stats);

			double change = likelihood - prevLikelihood;
			System.out.print("Iter " + getCurrentIterationNumber()
					+ "\t-log(L)="
					+ util.Printing.prettyPrint(-likelihood, "0.000000E00", 9));
			System.out.println(" diff="
					+ util.Printing.prettyPrint(change, "0.000000E00", 9));
			prevLikelihood = likelihood;
			likelihood = 0;
			if (change / -prevLikelihood < 1e-5) {
				break;
			}

			// M step
			stats.mStepStart(this.model, this);
			mStep(counts);
			stats.mStepEnd(this.model, this);
			System.out.print(stats.printEndMStep(this.model, this));

			stats.emIterEnd(this.model, this);
			System.out.print(stats.printEndEMIter(this.model, this));
		}
		stats.emEnd(this.model, this);
		System.out.print(stats.printEndEM(this.model, this));
		System.out.println();
	}

	@Override
	public void sentenceEStep(AbstractSentenceDist sd,
			AbstractCountTable counts, TrainStats stats) {
		sd.initSentenceDist();
		stats.eStepSentenceStart(model,this,sd);
		model.computePosteriors(sd);
		sd.clearCaches();
		model.addToCounts(sd,counts);	
		stats.eStepSentenceEnd(model,this,sd);
		likelihood += sd.getLogLikelihood();
		sd.clearPosteriors();
		stats.printEndSentenceEStep(model,this);
	}
}
