package mycode;

import static depparsing.globals.Constants.CONT;
import static depparsing.globals.Constants.END;
import static depparsing.globals.Constants.LEFT;
import static depparsing.globals.Constants.RIGHT;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;

import learning.EM;
import learning.stats.TrainStats;
import model.AbstractCountTable;
import model.AbstractModel;
import model.AbstractSentenceDist;
import depparsing.decoding.CKYParser;
import depparsing.model.DepModel;
import depparsing.model.DepProbMatrix;
import depparsing.model.DepSentenceDist;

/**
 * Softmax-EM with the sigma value changing with the iterations
 * 
 */
public class AnnealingSoftmaxEM extends EM {

	protected double likelihood = 0;
	protected double prevLikelihood = Double.NEGATIVE_INFINITY;

	protected double sigma_0 = 0;
	protected double sigma_k = 0;
	protected double sigma_e = 0;

	public AnnealingSoftmaxEM(AbstractModel model, double sigma_0,
			double sigma_k, double sigma_e) {
		super(model);
		this.sigma_0 = sigma_0;
		this.sigma_k = sigma_k;
		this.sigma_e = sigma_e;
	}

	@Override
	public void em(int numIters, TrainStats stats)
			throws UnsupportedEncodingException, IOException {
		stats.emStart(this.model, this);

		AbstractSentenceDist[] sentenceDists = model.getSentenceDists();

		DepProbMatrix counts = (DepProbMatrix) model.getCountTable();

		for (iter = 0; iter < numIters; iter++) {
			System.out.println("\nEM Iteration " + (iter + 1));
			System.out.flush();
			stats.emIterStart(this.model, this);

			double sigma = sigma_0 + iter * sigma_k;
			sigma = Math.max(sigma_e, sigma);
			System.out.println("sigma=" + sigma);
			if (sigma > 0.95) // 0.95 instead of 1 to avoid overflow
				corpusEStepV(counts, sentenceDists, stats);
			else {
				((DepModel) model).exponentiateParameters(1 / (1 - sigma));
				corpusEStep(counts, sentenceDists, stats);
				likelihood *= 1 - sigma;
			}

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

	public void sentenceEStep(AbstractSentenceDist sd,
			AbstractCountTable counts, TrainStats stats) {
		// sd.initSentenceDist();
		stats.eStepSentenceStart(model, this, sd);
		model.computePosteriors(sd);
		sd.clearCaches();
		model.addToCounts(sd, counts);
		stats.eStepSentenceEnd(model, this, sd);
		likelihood += sd.getLogLikelihood();
		sd.clearPosteriors();
		stats.printEndSentenceEStep(model, this);
	}

	public void corpusEStepV(DepProbMatrix counts,
			AbstractSentenceDist[] sentenceDists, TrainStats stats) {
		stats.eStepStart(this.model, this);

		// Clear model for accumulating E step counts
		System.out
				.println("Using senteces on training:" + sentenceDists.length);
		counts.fill(0);
		for (AbstractSentenceDist sd : sentenceDists) {
			sentenceEStepV((DepSentenceDist) sd, counts, stats);
		}
		counts.convertRealToLog();

		stats.eStepEnd(this.model, this);
		System.out.print(stats.printEndEStep(this.model, this));
	}

	public void sentenceEStepV(DepSentenceDist dsd, DepProbMatrix counts,
			TrainStats stats) {
		stats.eStepSentenceStart(model, this, dsd);

		dsd.cacheModel(((DepModel) model).params);
		int[] parse = new int[dsd.depInst.postags.length];
		//hanwj10.15
		//likelihood += CKYParser.parseSentence(dsd, parse);
		countDependencies(parse, dsd.depInst.postags, counts);
		dsd.clearPosteriors();

		stats.eStepSentenceEnd(model, this, dsd);
		stats.printEndSentenceEStep(model, this);
	}

	/**
	 * adapted from MaxLikelihoodEstimator.countDependencies
	 * 
	 * @param postags
	 */
	protected void countDependencies(int[] parse, int[] postags,
			DepProbMatrix model) {
		int[] posTags = postags;
		int numWords = posTags.length;
		int[][] histogram = new int[numWords][2];

		// Cycle *left-to-right* through all words in the sentence
		int[] headValences = new int[numWords];
		for (int i = 0; i < numWords; i++) {
			int head = parse[i] - 1;

			// Update lambda_root
			if (head == -1)
				model.root[posTags[i]]++;
			else {
				if (i < head) { // i is to the left of head
					// Fill in left child counts and update histogram for use in
					// setting decision counts
					model.child[posTags[i]][posTags[head]][LEFT][headValences[head]]++;
					if (headValences[head] < model.nontermMap.childValency - 1)
						headValences[head]++;
					histogram[head][LEFT]++;
				}
			}
		}

		// Cycle *right-to-left* through all words in the sentence
		Arrays.fill(headValences, 0);
		for (int i = numWords - 1; i >= 0; i--) {
			int head = parse[i] - 1;
			if (i > head && head != -1) {
				// Fill in right child counts and update histogram for use in
				// setting decision counts
				model.child[posTags[i]][posTags[head]][RIGHT][headValences[head]]++;
				if (headValences[head] < model.nontermMap.childValency - 1)
					headValences[head]++;
				histogram[head][RIGHT]++;
			}
		}

		processHistogram(model, histogram, postags);
	}

	/**
	 * adapted from MaxLikelihoodEstimator.processHistogram
	 */
	public static void processHistogram(DepProbMatrix model, int[][] histogram,
			int[] postags) {
		for (int i = 0; i < postags.length; i++) {
			int headPosNum = postags[i];
			for (int dir = 0; dir < 2; dir++) {
				int current = histogram[i][dir];

				int maxValence = Math.min(current,
						model.nontermMap.decisionValency - 1);
				model.decision[headPosNum][dir][maxValence][END]++;
				if (current > 0) {
					for (int prevCont = 0; prevCont < maxValence; prevCont++)
						model.decision[headPosNum][dir][prevCont][CONT]++;
					model.decision[headPosNum][dir][maxValence][CONT] += (current - maxValence);
				}
			}
		}
	}
}
