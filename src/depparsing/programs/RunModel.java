package depparsing.programs;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Random;

import learning.CorpusPR;
import learning.EM;
import learning.stats.CompositeTrainStats;
import learning.stats.LikelihoodStats;
import learning.stats.TrainStats;
import mycode.AnnealingDynamicSoftmaxEM;
import mycode.AnnealingSoftmaxEM;
import mycode.DynamicSoftmaxEM;
import mycode.SoftmaxEM;
import mycode.SoftmaxPR;
import mycode.ViterbiEM;

import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import constraints.CorpusConstraints;
import data.InstanceList;
import data.WordInstance;
import depparsing.constraints.GroupedL1LMax;
import depparsing.constraints.L1LMax;
import depparsing.constraints.PCType;
import depparsing.constraints.UngroupedL1LMax;
import depparsing.data.DepCorpus;
import depparsing.data.DepInstance;
import depparsing.decoding.CKYParser;
import depparsing.io.CONLLWriter;
import depparsing.io.DepProbReader;
import depparsing.io.DepProbWriter;
import depparsing.model.DepModel;
import depparsing.model.DepModel.UpdateType;
import depparsing.model.DepProbMatrix;
import depparsing.model.DepSentenceDist;
import depparsing.model.KleinManningInitializer;
import depparsing.model.LeftBranchingInit;
import depparsing.model.MaxLikelihoodEstimator;
import depparsing.model.RightBranchingInit;

public final class RunModel {

	// Train, dev, test files and output files
	@Option(name = "-corpus-params", usage = "<filename> = Corpus parameters file")
	private String corpusParams;
	// Min and max sentence sizes to include for training
	@Option(name = "-min-sentence-size", usage = "Minimum sentence size to use for training")
	private int minSentenceSize = 0;
	@Option(name = "-max-sentence-size", usage = "Maximum sentence size to use for training")
	private int maxSentenceSize = Integer.MAX_VALUE;
	@Option(name = "-max-number-sentences", usage = "Maximum number of sentences to load")
	private int maxNumberOfSentences = Integer.MAX_VALUE;

	// Model initialization
	private enum ModelInit {
		K_AND_M, ZERO, RANDOM, SUPERVISED, SAVED, RANDOM_POOL, RANDOM_SUPERVISED, LEFT_BRANCHING, RIGHT_BRANCHING
	};

	@Option(name = "-model-init", usage = "K_AND_M = Klein and Manning (default), ZERO = Uniform, RANDOM = Random, SUPERVISED = Max likelihood, or SAVED = Load saved model"
			+ ", RANDOM_POOL = Randomly initalized a pool of models of size andom_pool_size runs each method for random-pool_burn_in and return the model "
			+ "with highest likelihood on the development corpus"
			+ "LEFT_BRANCHING = Initalizes with always left edges tree"
			+ "RIGHT_BRANCHING = Initalizes with always right edges tree"
			+ "RANDOM_SUPERVISED = randomly choose n sentences and initialize supervised (n is set with the option)")
	private ModelInit initType = ModelInit.K_AND_M;
	@Option(name = "-dvalency", usage = "decision parameters' valency: 1 = no distinctions, 2 = kids vs. no kids, 3 = 2 or more kids vs 1 kid vs no kids, 4 = ... (default = 2)")
	private int dvalency = 2;
	@Option(name = "-cvalency", usage = "child parameters' valency: 1 = no distinctions, 2 = kids vs. no kids, 3 = 2 or more kids vs 1 kid vs no kids, 4 = ... (default = 1)")
	private int cvalency = 1;
	@Option(name = "-child-backoff", usage = "strength of backoff for the child parameters; must be in [0,1] range")
	private double childBackoff = 0;
	@Option(name = "-seed", usage = "Seed for random number generator for random start (default = 1)")
	private Long seed; // Note: Default value is set in main method instead of
						// here, to allow for more error checking
	@Option(name = "-model-to-load", usage = "<filename> = Model to load")
	private String modelFile;
	@Option(name = "-backoff", usage = "Smoothing to add to each model parameter before testing (default = 4.5 x 10^-5)")
	private double backoff = Math.exp(-1e1);
	@Option(name = "-random-pool-size", usage = "The size of the random pool of models. Default value is 10")
	private int random_pool_size = 50;
	@Option(name = "-random-pool-burn-in", usage = "Number of iterations to run each model in the pool. Default is 20")
	private int random_pool_burn_in = 10;
	@Option(name = "-supervised-sample-size", usage = "Number of supervised sentences for random_supervised initialization")
	private int supervised_sample_size = 10;
	@Option(name = "-supervised-sample-weight", usage = "only usable when we do RANDOM_SUPERVISED initialization. 0 means normal EM/PR, 1 means fully supervised. default=0")
	private double supervised_sample_weight = 0;
	private InstanceList supervisedSample;

	// Basic EM options
	@Option(name = "-num-em-iters", usage = "Number of EM iterations (default = 0)")
	private int numEMIters = 0;
	@Option(name = "-stats-file", usage = "Training statistics file")
	private String statsFile = "";
	@Option(name = "-trainingType", usage = "Training Type: 0 EM Training; 1 PR with L1LMax; 2 Variational (Some options need extra arguments); "
			+ "3 Softmax-EM; 4 Viterbi EM; 5 Dynamic-Softmax-EM; 6 Annealing-Softmax-EM; "
			+ "7 Annealing-Softmax-VB; 8 Annealing-Dynamic-Softmax-EM; "
			+ "9 Softmax-PR with L1LMax")
	private int trainingType = 0;

	// PR options
	// Note: Some default values are set in main method instead of here, to
	// allow for more error checking
	@Option(name = "-constraint-strength", usage = "Sparsity constraint strength (default = 10)")
	private Double cstrength;
	@Option(name = "-num-warmup-iters", usage = "Number of EM warm-up iterations before starting projection (default = 0, total iters = num-em-iters + num-warmup-iters)")
	private Integer warmupIters;
	@Option(name = "-use-child-words", usage = "Use child words for computing L1LMax (default = just use tags not words; note: option does not take any parameters)")
	private boolean useChildWords = false;
	@Option(name = "-use-parent-words", usage = "Use parent words for computing L1LMax (default = just use tags not words; note: option does not take any parameters)")
	private boolean useParentWords = false;
	@Option(name = "-constrain-direction", usage = "Take into account edge direction for L1LMax (default = don't consider direction; note: option does not take any parameters)")
	private boolean constrainDir = false;
	@Option(name = "-constrain-root", usage = "Constrain root posteriors (default = don't constrain them; note: option does not take any parameters)")
	private boolean constrainRoot = false;

	@Option(name = "-output-prefix", usage = "save decoded predictions in <output-prefix><test-file-name> (default=predictions.).")
	private String outputPrefix; // = "predictions.";

	// Model output; note that models are saved before backoff is applied
	@Option(name = "-save-model", usage = "File to which to save the learned model")
	private String savefile;

	@Option(name = "-prior", usage = "Parameter prior to use with Dirichlet prior")
	private double prior = 1;

	@Option(name = "-project-at-test", usage = "Perform projection before running CKY (default = false)")
	private boolean projectAtTest = false;

	@Option(name = "-project-iters-at-pool", usage = "how many max projection iterations to run for the random pool")
	private Integer projectItersAtPool;

	@Option(name = "-scale-cstrength", usage = "scale the constraint strength based on the number of contexts of parent and child (tag-tag constraints only)")
	private boolean scaleCstrength = false;

	@Option(name = "-min-occurrences-for-projection", usage = "do not project edges that occur less than this many times (default = 0)")
	private int minOccurrencesForProjection = 0;

	@Option(name = "-use-fernando-constraints", usage = "Use Fernando-style constraints: sum edges of identical child-ID and parent-TYPE.")
	private boolean useFernandoConstraints = false;

	@Option(name = "-do-not-project-file", usage = "Do not project edges of the types specified in this file")
	private String doNotProjectFile = "/dev/null";

	@Option(name = "-sigma", usage = "Parameter to use with softmax-EM")
	private double sigma = 0;

	@Option(name = "-sigma_k", usage = "Parameter to use with dynamic/annealing softmax-EM")
	private double sigma_k = 0;

	@Option(name = "-sigma_e", usage = "Parameter to use with annealing softmax-EM")
	private double sigma_e = 0;

	@Option(name = "-tmpfile", usage = "a tmp field for Pascal competition")
	private String tmpfile; //XXX

	/**
	 * print out the list of options so that we can reconstruct how the script
	 * was run.
	 */
	private void printOptions(PrintStream out) {
		out.println("-corpus-params " + corpusParams);
		out.println("-min-sentence-size " + minSentenceSize);
		out.println("-max-sentence-size " + maxSentenceSize);
		out.println("-max-number-of-sentences " + maxNumberOfSentences);
		out.println("-model-init " + initType);
		out.println("-dvalency " + dvalency);
		out.println("-cvalency " + cvalency);
		out.println("-child-backoff " + childBackoff);
		out.println("-seed " + seed);
		out.println("-supervised-sample-size " + supervised_sample_size);
		out.println("-supervised-sample-weight " + supervised_sample_weight);
		out.println("-model-to-load " + modelFile);
		out.println("-backoff " + backoff);
		out.println("-num-em-iters " + numEMIters);
		out.println("-stats-file " + statsFile);
		out.println("-trainingType " + trainingType);
		out.println("-constraint-strength " + cstrength);
		out.println("-num-warmup-iters " + warmupIters);
		out.println("-use-child-words " + useChildWords);
		out.println("-use-parent-words " + useParentWords);
		out.println("-constrain-direction " + constrainDir);
		out.println("-constrain-root " + constrainRoot);
		out.println("-prior " + prior);
		out.println("-output-prefix " + outputPrefix);
		out.println("-save-model " + savefile);
		out.println("-random-pool-size " + random_pool_size);
		out.println("-random-pool-burn-in " + random_pool_burn_in);
		out.println("-project-at-test " + projectAtTest);
		out.println("-project-iters-at-pool " + projectItersAtPool);
		out.println("-scale-cstrength " + scaleCstrength);
		out.println("-min-occurrences-for-projection "
				+ minOccurrencesForProjection);
		out.println("-use-fernando-constraints " + useFernandoConstraints);
		out.println("-do-not-project-file " + doNotProjectFile);
		out.println("-sigma " + sigma);
		out.println("-sigma_k " + sigma_k);
		out.println("-sigma_e " + sigma_e);
	}

	// For additional unexpected arguments, so can print them in error message
	@Argument
	private final ArrayList<String> arguments = new ArrayList<String>();

	public static void main(String[] args) throws IOException,
			ClassNotFoundException, InstantiationException,
			IllegalAccessException, InvocationTargetException {
		new RunModel().parseCommandLineArguments(args);
	}

	public void parseCommandLineArguments(String[] args)
			throws ClassNotFoundException, InvocationTargetException,
			IllegalAccessException, InstantiationException, IOException {

		CmdLineParser parser = new CmdLineParser(this);
		try {
			parser.setUsageWidth(120);

			// Read options
			parser.parseArgument(args);

			// If any additional arguments were given, throw exception
			if (!arguments.isEmpty())
				throw new CmdLineException(
						"Unrecognized command line arguments: "
								+ arguments.toString());

			printOptions(System.out);

			// Read corpus params, the only option that's always required
			DepCorpus corpus;
			if (corpusParams == null)
				throw new CmdLineException(
						"Must always specify a corpus params file");
			else
				corpus = new DepCorpus(corpusParams, minSentenceSize,
						maxSentenceSize, maxNumberOfSentences);

			// Tnitialize model and train
			// Note that even if loading a saved model, the corpus from the
			// current corpus-params file will be used
			if (dvalency < 1 || cvalency < 1)
				throw new CmdLineException("All valencies must be >= 1");
			if (childBackoff < 0 || childBackoff > 1)
				throw new CmdLineException(
						"Child backoff strength must be in the [0,1] range");
			DepProbMatrix params = new DepProbMatrix(corpus, dvalency, cvalency);
			initializeModel(params);
			TrainStats<DepModel, DepSentenceDist> stats = CompositeTrainStats
					.buildTrainStats(statsFile);
			DepModel model = null;
			model = trainModel(params, numEMIters, stats);
			if (numEMIters == 0 && warmupIters != null)
				throw new CmdLineException(
						"Cannot have warmup iterations without other iterations");

			// Save model to file
			if (savefile != null) {
				System.out.println("Saving model to file " + savefile + "\n");
				DepProbWriter.writeToFile(savefile, model.params);
			}

			// Add backoff to all model params before running testing
			if (backoff > 0) {
				System.out.println("Adding backoff " + backoff
						+ " to all model parameters\n");
				model.params.backoff(Math.log(backoff));
			}

			// Test model (will test for all test sets available in corpus)
			//testModel(model.params);              //hanwj10.15
		} catch (CmdLineException e) {
			// Print exception's message
			System.err.println(e.getMessage() + "\n");

			// Print usage message
			System.err
					.println("Full set of available options:\njava programs.Train [options...]");
			parser.printUsage(System.err);
		}
	}

	private void initializeModel(DepProbMatrix model) throws CmdLineException,
			IOException, ClassNotFoundException, InvocationTargetException,
			IllegalAccessException, InstantiationException {
		if (!(initType == ModelInit.RANDOM || initType == ModelInit.RANDOM_POOL
				|| initType == ModelInit.RANDOM_SUPERVISED || seed == null))
			throw new CmdLineException(
					"-seed option should only be specified when using random initialization");
		if (!(initType == ModelInit.SAVED || modelFile == null))
			throw new CmdLineException(
					"-model-to-load option should only be specified when using saved model initialization");
		if (initType != ModelInit.RANDOM_SUPERVISED
				&& supervised_sample_weight != 0)
			throw new CmdLineException(
					"-supervised-sample-weight can only be used when using RANDOM_SUPERVISED initialization");
		if (supervised_sample_weight >= 1 || supervised_sample_weight < 0)
			throw new CmdLineException(
					"-supervised-sample-weight must be >= 0 and <1, so "
							+ supervised_sample_weight + " is invalid.");

		switch (initType) {
		case K_AND_M:
			System.out.println("Klein and Manning model initialization");
			KleinManningInitializer.initNoah(model);
			break;
		case ZERO:
			System.out
					.println("Uniform (all parameters set to log prob 0) initialization");
			model.fill(0);
			break;
		case RANDOM:
			seed = (seed == null ? 1 : seed);
			System.out.println("Random initialization with seed " + seed);
			model.setRandom(new Random(seed));
			break;
		case SUPERVISED:
			System.out.println("Supervised (max likelihood) initialization");
			MaxLikelihoodEstimator.computeMLModel(model, 0, childBackoff);
			break;
		case LEFT_BRANCHING:
			System.out.println("Left branching tree initialization");
			LeftBranchingInit.initLeftBranching(model, backoff);
			break;
		case RIGHT_BRANCHING:
			System.out.println("Right branching tree initialization");
			RightBranchingInit.initRightBranching(model, backoff);
			break;
		case RANDOM_SUPERVISED:
			seed = (seed == null ? 1 : seed);
			System.out
					.println("initialization from random supervised sample of size "
							+ supervised_sample_size + " with seed " + seed);
			// note sampleSupervisedSet also adds the sample to the list of
			// testing sets.
			supervisedSample = sampleSupervisedSet(model.corpus);
			MaxLikelihoodEstimator.computeMLModel(model, backoff, childBackoff,
					supervisedSample);
			break;
		case SAVED:
			System.out.println("Initialization from saved model\n");
			model.copyFrom(DepProbReader.readFromFile(modelFile));
			break;
		case RANDOM_POOL:
			System.out.println("Initialization from random pool. Pool size "
					+ random_pool_size + " burn in iters "
					+ random_pool_burn_in);
			seed = (seed == null ? 1 : seed);
			initFromRandomPool(model, random_pool_size, random_pool_burn_in,
					seed);
			break;
		default:
			assert (false) : "Why didn't args4j catch this model-init options error?";
			break;
		}
		model.logNormalize();
		System.out.println();
	}

	public DepModel trainModel(DepProbMatrix params, int numIters,
			TrainStats<DepModel, DepSentenceDist> stats)
			throws IllegalArgumentException, IOException,
			ClassNotFoundException, InstantiationException,
			IllegalAccessException, InvocationTargetException, CmdLineException {
		DepModel model = null;
		if (trainingType == 1) {
			model = new DepModel(params, params.corpus, dvalency, cvalency,
					childBackoff, UpdateType.TABLE_UP);
			if (numIters > 0) {
				runPostRegEM(model, numIters, stats, stats);
			}
		} else if (trainingType == 0) {
			// Check that no command line options for PR were passed for a run
			// of standard EM
			model = new DepModel(params, params.corpus, dvalency, cvalency,
					childBackoff, UpdateType.TABLE_UP);
			if (warmupIters == null && !useChildWords && !useParentWords
					&& !constrainRoot && !constrainDir && cstrength == null) {
				if (numIters > 0) {
					runStandardEM(model, numIters, stats);
				}
			} else
				throw new CmdLineException(
						"Cannot use options constraint-strength, num-warmup-iters, use-child-words, "
								+ "use-parent-words, constrain-direction, or constrain-root when running standard EM");
		} else if (trainingType == 2) {
			if (prior < 0)
				throw new CmdLineException(
						"Cannot use variational EM without a prior or with a negative prior");
			model = new DepModel(params, params.corpus, dvalency, cvalency,
					childBackoff, UpdateType.VB, prior);
			if (numIters > 0) {
				runStandardEM(model, numIters, stats);
			}
		} else if (trainingType == 3) {
			if (prior < 1)
				throw new CmdLineException("Cannot use EM with a <1 prior");
			model = new DepModel(params, params.corpus, dvalency, cvalency,
					childBackoff, UpdateType.TABLE_UP, prior);
			if (numIters > 0) {
				runSoftmaxEM(model, numIters, stats);
			}
		} else if (trainingType == 4) {
			if (prior < 1)
				throw new CmdLineException("Cannot use EM with a <1 prior");
			model = new DepModel(params, params.corpus, dvalency, cvalency,
					childBackoff, UpdateType.TABLE_UP, prior);
			if (numIters > 0) {
				runViterbiEM(model, numIters, stats);
			}
		} else if (trainingType == 5) {
			if (prior < 1)
				throw new CmdLineException("Cannot use EM with a <1 prior");
			model = new DepModel(params, params.corpus, dvalency, cvalency,
					childBackoff, UpdateType.TABLE_UP, prior);
			if (numIters > 0) {
				runDynamicSoftmaxEM(model, numIters, stats);
			}
		} else if (trainingType == 6) {
			if (prior < 1)
				throw new CmdLineException("Cannot use EM with a <1 prior");
			model = new DepModel(params, params.corpus, dvalency, cvalency,
					childBackoff, UpdateType.TABLE_UP, prior);
			if (numIters > 0) {
				runAnnealingSoftmaxEM(model, numIters, stats);
			}
		} else if (trainingType == 7) {
			if (prior < 0)
				throw new CmdLineException(
						"Cannot use variational EM without a prior or with a negative prior");
			model = new DepModel(params, params.corpus, dvalency, cvalency,
					childBackoff, UpdateType.VB, prior);
			if (numIters > 0) {
				runAnnealingSoftmaxEM(model, numIters, stats);
			}
		} else if (trainingType == 8) {
			if (prior < 1)
				throw new CmdLineException("Cannot use EM with a <1 prior");
			model = new DepModel(params, params.corpus, dvalency, cvalency,
					childBackoff, UpdateType.TABLE_UP, prior);
			if (numIters > 0) {
				runAnnealingDynamicSoftmaxEM(model, numIters, stats);
			}
		} else if (trainingType == 9) {
			model = new DepModel(params, params.corpus, dvalency, cvalency,
					childBackoff, UpdateType.TABLE_UP);
			if (numIters > 0) {
				runSoftmaxPostRegEM(model, numIters, stats, stats);
			}
		} else {
			throw new CmdLineException("Not a valid training type");
		}

		return model;
	}

	/**
	 * choose maximally long sentences from the training data
	 * 
	 * @return
	 */
	private InstanceList sampleSupervisedSet(DepCorpus corpus) {
		int maxlength = 0;
		ArrayList<Integer> indices = new ArrayList<Integer>();
		for (int i = 0; i < corpus.trainInstances.instanceList.size(); i++) {
			DepInstance di = (DepInstance) corpus.trainInstances.instanceList
					.get(i);
			if (di.numWords > maxlength) {
				maxlength = di.numWords;
				indices = new ArrayList<Integer>();
			}
			if (di.numWords == maxlength) {
				indices.add(i);
			}
		}
		if (indices.size() < supervised_sample_size)
			throw new RuntimeException("not enough sentences of length "
					+ maxlength + ". want " + supervised_sample_size + " have "
					+ indices.size());
		Random r = new Random(seed);
		// shuffle the indices
		for (int i = 0; i < indices.size(); i++) {
			int valati = indices.get(i);
			int other = r.nextInt(indices.size() - i) + i;
			indices.set(i, indices.get(other));
			indices.set(other, valati);
		}
		InstanceList res = new InstanceList("subsample-of-"
				+ corpus.trainInstances.name);
		for (int i = 0; i < supervised_sample_size; i++) {
			res.add(corpus.trainInstances.instanceList.get(indices.get(i)));
		}
		corpus.testInstances.add(res);
		return res;
	}

	/**
	 * the variable below is a real hack to make it possible to set the
	 * projection to only do e.g. 10 steps while we are evaluating the pool
	 * instead of doing 100.
	 */
	private boolean tmpDoingRandomPoolInit = false;

	public void initFromRandomPool(DepProbMatrix model, int randomPoolSize,
			int randomPoolBurnIn, long seed) throws ClassNotFoundException,
			InvocationTargetException, IllegalAccessException,
			InstantiationException, IOException, CmdLineException {
		DepCorpus corpus = model.corpus;
		DepProbMatrix[] pool = new DepProbMatrix[randomPoolSize];
		Random r = new Random(seed);
		CompositeTrainStats<DepModel, DepSentenceDist> stats = new CompositeTrainStats<DepModel, DepSentenceDist>();
		stats.addStats(new LikelihoodStats<DepModel, DepSentenceDist>());
		long[] myseeds = new long[randomPoolSize];
		for (int i = 0; i < randomPoolSize; i++) {
			myseeds[i] = r.nextLong();
			pool[i] = new DepProbMatrix(corpus, dvalency, cvalency);
			pool[i].setRandom(new Random(myseeds[i]));
			System.out.println("Training pool model " + i);
			tmpDoingRandomPoolInit = true;
			trainModel(pool[i], randomPoolBurnIn, stats);
			tmpDoingRandomPoolInit = false;
		}

		// Calculate likelihood for each devSentences
		ArrayList<WordInstance> devSentences = corpus.devInstances.instanceList;
		double minNegLogLikelihood = Double.POSITIVE_INFINITY;
		int modelNumber = -1;
		for (int j = 0; j < randomPoolSize; j++) {
			DepProbMatrix m = pool[j];
			double negLogLikelihood = 0;
			for (int i = 0; i < devSentences.size(); i++) {
				DepSentenceDist sd = new DepSentenceDist(
						(DepInstance) devSentences.get(i), model.nontermMap);
				sd.cacheModelAndComputeIO(m);
				negLogLikelihood += sd.insideRoot;
			}
			System.out.println("Model " + j + " dev likelihood "
					+ negLogLikelihood);
			// Check if its max Model
			if (negLogLikelihood < minNegLogLikelihood) {
				minNegLogLikelihood = negLogLikelihood;
				modelNumber = j;
			}
		}
		model.copyFrom(pool[modelNumber]);
		// Leaves with the max model assign to model
		System.out.println("Picked model " + modelNumber + " with seed "
				+ myseeds[modelNumber] + " devLikelihood "
				+ minNegLogLikelihood);
	}

	private void runPostRegEM(DepModel model, int numIters,
			TrainStats<DepModel, DepSentenceDist> stats,
			TrainStats<DepModel, DepSentenceDist> warmupStats)
			throws ClassNotFoundException, InvocationTargetException,
			IllegalAccessException, InstantiationException, IOException {
		System.out.println("EM with posterior regularization");
		// Get posterior regularized EM parameters
		PCType childType = (useChildWords ? PCType.WORD : PCType.TAG);
		System.out.println("Using child " + (useChildWords ? "words" : "tags")
				+ " in L1LMax constraints");
		PCType parentType = (useParentWords ? PCType.WORD : PCType.TAG);
		System.out.println("Using parent "
				+ (useParentWords ? "words" : "tags")
				+ " in L1LMax constraints");
		System.out.println("Constraining root posteriors? " + constrainRoot);
		System.out.println("Differentiating L1LMax based on edge direction? "
				+ constrainDir);
		if (cstrength == null)
			cstrength = 10.0;
		System.out.println("Constraint strength: " + cstrength);
		System.out.println();
		CorpusConstraints constraints = null;
		if (useFernandoConstraints)
			constraints = new GroupedL1LMax(model.corpus, model,
					model.corpus.trainInstances.instanceList, childType,
					parentType, constrainRoot, constrainDir, cstrength,
					minOccurrencesForProjection, doNotProjectFile);
		else
			constraints = new UngroupedL1LMax(model.corpus, model,
					model.corpus.trainInstances.instanceList, childType,
					parentType, constrainRoot, constrainDir, cstrength,
					minOccurrencesForProjection, doNotProjectFile);
		if (tmpDoingRandomPoolInit) {
			if (projectItersAtPool != null) {
				if (useFernandoConstraints) {
					((L1LMax) constraints)
							.setMaxProjectionSteps(projectItersAtPool);
				} else {
					((UngroupedL1LMax) constraints)
							.setMaxProjectionSteps(projectItersAtPool);
				}

			} else {
				if (useFernandoConstraints) {
					((L1LMax) constraints).setMaxProjectionSteps(0);
				} else {
					((UngroupedL1LMax) constraints).setMaxProjectionSteps(0);
				}

			}
		}

		CorpusPR EMer;
		if (supervised_sample_weight > 0) {
			if (supervisedSample == null)
				throw new RuntimeException(
						"null supervised sample, but non-zero weight!");
			if (supervisedSample.instanceList.size() == 0)
				throw new RuntimeException(
						"empyt supervised sample, but non-zero weight!");
			DepProbMatrix initialCounts = new DepProbMatrix(model.corpus,
					model.params.nontermMap.decisionValency,
					model.params.nontermMap.childValency);
			MaxLikelihoodEstimator.computeMLCounts(initialCounts,
					supervisedSample);
			double numUnsup = model.corpus.trainInstances.instanceList.size();
			double weight = (numUnsup * supervised_sample_weight);
			weight /= supervisedSample.instanceList.size()
					* (1 - supervised_sample_weight);
			initialCounts.scaleBy(weight);
			model.updateParameters(initialCounts);
		}
		EMer = new CorpusPR(model, constraints);

		// Warmup with standard EM
		if (warmupIters != null && warmupIters > 0) {
			System.out.println("Beginning " + warmupIters
					+ " warmup EM iterations");
			runStandardEM(model, warmupIters, warmupStats);
			System.out.println();
		}

		// Then run PR EM
		System.out.println("Beginning PR EM iterations");
		EMer.em(numIters, stats);
	}

	private void runSoftmaxPostRegEM(DepModel model, int numIters,
			TrainStats<DepModel, DepSentenceDist> stats,
			TrainStats<DepModel, DepSentenceDist> warmupStats)
			throws ClassNotFoundException, InvocationTargetException,
			IllegalAccessException, InstantiationException, IOException {
		System.out.println("EM with posterior regularization");
		// Get posterior regularized EM parameters
		PCType childType = (useChildWords ? PCType.WORD : PCType.TAG);
		System.out.println("Using child " + (useChildWords ? "words" : "tags")
				+ " in L1LMax constraints");
		PCType parentType = (useParentWords ? PCType.WORD : PCType.TAG);
		System.out.println("Using parent "
				+ (useParentWords ? "words" : "tags")
				+ " in L1LMax constraints");
		System.out.println("Constraining root posteriors? " + constrainRoot);
		System.out.println("Differentiating L1LMax based on edge direction? "
				+ constrainDir);
		if (cstrength == null)
			cstrength = 10.0;
		System.out.println("Constraint strength: " + cstrength);
		System.out.println();
		CorpusConstraints constraints = null;
		if (useFernandoConstraints)
			constraints = new GroupedL1LMax(model.corpus, model,
					model.corpus.trainInstances.instanceList, childType,
					parentType, constrainRoot, constrainDir, cstrength,
					minOccurrencesForProjection, doNotProjectFile);
		else
			constraints = new UngroupedL1LMax(model.corpus, model,
					model.corpus.trainInstances.instanceList, childType,
					parentType, constrainRoot, constrainDir, cstrength,
					minOccurrencesForProjection, doNotProjectFile);
		if (tmpDoingRandomPoolInit) {
			if (projectItersAtPool != null) {
				if (useFernandoConstraints) {
					((L1LMax) constraints)
							.setMaxProjectionSteps(projectItersAtPool);
				} else {
					((UngroupedL1LMax) constraints)
							.setMaxProjectionSteps(projectItersAtPool);
				}

			} else {
				if (useFernandoConstraints) {
					((L1LMax) constraints).setMaxProjectionSteps(0);
				} else {
					((UngroupedL1LMax) constraints).setMaxProjectionSteps(0);
				}

			}
		}

		SoftmaxPR EMer;
		if (supervised_sample_weight > 0) {
			if (supervisedSample == null)
				throw new RuntimeException(
						"null supervised sample, but non-zero weight!");
			if (supervisedSample.instanceList.size() == 0)
				throw new RuntimeException(
						"empyt supervised sample, but non-zero weight!");
			DepProbMatrix initialCounts = new DepProbMatrix(model.corpus,
					model.params.nontermMap.decisionValency,
					model.params.nontermMap.childValency);
			MaxLikelihoodEstimator.computeMLCounts(initialCounts,
					supervisedSample);
			double numUnsup = model.corpus.trainInstances.instanceList.size();
			double weight = (numUnsup * supervised_sample_weight);
			weight /= supervisedSample.instanceList.size()
					* (1 - supervised_sample_weight);
			initialCounts.scaleBy(weight);
			model.updateParameters(initialCounts);
		}
		EMer = new SoftmaxPR(model, constraints, sigma);

		// Warmup with standard EM
		if (warmupIters != null && warmupIters > 0) {
			System.out.println("Beginning " + warmupIters
					+ " warmup EM iterations");
			runStandardEM(model, warmupIters, warmupStats);
			System.out.println();
		}

		// Then run PR EM
		System.out.println("Beginning PR EM iterations");
		EMer.em(numIters, stats);
	}

	private void runStandardEM(DepModel model, int numIters,
			TrainStats<DepModel, DepSentenceDist> stats)
			throws ClassNotFoundException, InvocationTargetException,
			IllegalAccessException, InstantiationException, IOException {
		EM EMer;
		if (supervised_sample_weight > 0) {
			if (supervisedSample == null)
				throw new RuntimeException(
						"null supervised sample, but non-zero weight!");
			if (supervisedSample.instanceList.size() == 0)
				throw new RuntimeException(
						"empyt supervised sample, but non-zero weight!");
			DepProbMatrix initialCounts = new DepProbMatrix(
					model.params.corpus,
					model.params.nontermMap.decisionValency,
					model.params.nontermMap.childValency);
			MaxLikelihoodEstimator.computeMLCounts(initialCounts,
					supervisedSample);
			double numUnsup = model.params.corpus.trainInstances.instanceList
					.size();
			double weight = (numUnsup * supervised_sample_weight);
			weight /= supervisedSample.instanceList.size()
					* (1 - supervised_sample_weight);
			initialCounts.scaleBy(weight);
			model.updateParameters(initialCounts);
		}
		EMer = new EM(model);
		EMer.em(numIters, stats);
	}

	private void runSoftmaxEM(DepModel model, int numIters,
			TrainStats<DepModel, DepSentenceDist> stats)
			throws ClassNotFoundException, InvocationTargetException,
			IllegalAccessException, InstantiationException, IOException {
		SoftmaxEM EMer;
		if (supervised_sample_weight > 0) {
			if (supervisedSample == null)
				throw new RuntimeException(
						"null supervised sample, but non-zero weight!");
			if (supervisedSample.instanceList.size() == 0)
				throw new RuntimeException(
						"empyt supervised sample, but non-zero weight!");
			DepProbMatrix initialCounts = new DepProbMatrix(
					model.params.corpus,
					model.params.nontermMap.decisionValency,
					model.params.nontermMap.childValency);
			MaxLikelihoodEstimator.computeMLCounts(initialCounts,
					supervisedSample);
			double numUnsup = model.params.corpus.trainInstances.instanceList
					.size();
			double weight = (numUnsup * supervised_sample_weight);
			weight /= supervisedSample.instanceList.size()
					* (1 - supervised_sample_weight);
			initialCounts.scaleBy(weight);
			model.updateParameters(initialCounts);
		}
		EMer = new SoftmaxEM(model, sigma);
		EMer.em(numIters, stats);
	}

	private void runViterbiEM(DepModel model, int numIters,
			TrainStats<DepModel, DepSentenceDist> stats)
			throws ClassNotFoundException, InvocationTargetException,
			IllegalAccessException, InstantiationException, IOException {
		ViterbiEM EMer;
		if (supervised_sample_weight > 0) {
			if (supervisedSample == null)
				throw new RuntimeException(
						"null supervised sample, but non-zero weight!");
			if (supervisedSample.instanceList.size() == 0)
				throw new RuntimeException(
						"empyt supervised sample, but non-zero weight!");
			DepProbMatrix initialCounts = new DepProbMatrix(
					model.params.corpus,
					model.params.nontermMap.decisionValency,
					model.params.nontermMap.childValency);
			MaxLikelihoodEstimator.computeMLCounts(initialCounts,
					supervisedSample);
			double numUnsup = model.params.corpus.trainInstances.instanceList
					.size();
			double weight = (numUnsup * supervised_sample_weight);
			weight /= supervisedSample.instanceList.size()
					* (1 - supervised_sample_weight);
			initialCounts.scaleBy(weight);
			model.updateParameters(initialCounts);
		}
		EMer = new ViterbiEM(model);
		EMer.em(numIters, stats);
	}

	private void runDynamicSoftmaxEM(DepModel model, int numIters,
			TrainStats<DepModel, DepSentenceDist> stats)
			throws ClassNotFoundException, InvocationTargetException,
			IllegalAccessException, InstantiationException, IOException {
		DynamicSoftmaxEM EMer;
		if (supervised_sample_weight > 0) {
			if (supervisedSample == null)
				throw new RuntimeException(
						"null supervised sample, but non-zero weight!");
			if (supervisedSample.instanceList.size() == 0)
				throw new RuntimeException(
						"empyt supervised sample, but non-zero weight!");
			DepProbMatrix initialCounts = new DepProbMatrix(
					model.params.corpus,
					model.params.nontermMap.decisionValency,
					model.params.nontermMap.childValency);
			MaxLikelihoodEstimator.computeMLCounts(initialCounts,
					supervisedSample);
			double numUnsup = model.params.corpus.trainInstances.instanceList
					.size();
			double weight = (numUnsup * supervised_sample_weight);
			weight /= supervisedSample.instanceList.size()
					* (1 - supervised_sample_weight);
			initialCounts.scaleBy(weight);
			model.updateParameters(initialCounts);
		}
		EMer = new DynamicSoftmaxEM(model, sigma, sigma_k);
		EMer.em(numIters, stats);
	}

	private void runAnnealingSoftmaxEM(DepModel model, int numIters,
			TrainStats<DepModel, DepSentenceDist> stats)
			throws ClassNotFoundException, InvocationTargetException,
			IllegalAccessException, InstantiationException, IOException {
		AnnealingSoftmaxEM EMer;
		if (supervised_sample_weight > 0) {
			if (supervisedSample == null)
				throw new RuntimeException(
						"null supervised sample, but non-zero weight!");
			if (supervisedSample.instanceList.size() == 0)
				throw new RuntimeException(
						"empyt supervised sample, but non-zero weight!");
			DepProbMatrix initialCounts = new DepProbMatrix(
					model.params.corpus,
					model.params.nontermMap.decisionValency,
					model.params.nontermMap.childValency);
			MaxLikelihoodEstimator.computeMLCounts(initialCounts,
					supervisedSample);
			double numUnsup = model.params.corpus.trainInstances.instanceList
					.size();
			double weight = (numUnsup * supervised_sample_weight);
			weight /= supervisedSample.instanceList.size()
					* (1 - supervised_sample_weight);
			initialCounts.scaleBy(weight);
			model.updateParameters(initialCounts);
		}
		EMer = new AnnealingSoftmaxEM(model, sigma, sigma_k, sigma_e);
		EMer.em(numIters, stats);
	}

	private void runAnnealingDynamicSoftmaxEM(DepModel model, int numIters,
			TrainStats<DepModel, DepSentenceDist> stats)
			throws ClassNotFoundException, InvocationTargetException,
			IllegalAccessException, InstantiationException, IOException {
		AnnealingDynamicSoftmaxEM EMer;
		if (supervised_sample_weight > 0) {
			if (supervisedSample == null)
				throw new RuntimeException(
						"null supervised sample, but non-zero weight!");
			if (supervisedSample.instanceList.size() == 0)
				throw new RuntimeException(
						"empyt supervised sample, but non-zero weight!");
			DepProbMatrix initialCounts = new DepProbMatrix(
					model.params.corpus,
					model.params.nontermMap.decisionValency,
					model.params.nontermMap.childValency);
			MaxLikelihoodEstimator.computeMLCounts(initialCounts,
					supervisedSample);
			double numUnsup = model.params.corpus.trainInstances.instanceList
					.size();
			double weight = (numUnsup * supervised_sample_weight);
			weight /= supervisedSample.instanceList.size()
					* (1 - supervised_sample_weight);
			initialCounts.scaleBy(weight);
			model.updateParameters(initialCounts);
		}
		EMer = new AnnealingDynamicSoftmaxEM(model, sigma, sigma_k);
		EMer.em(numIters, stats);
	}
//hanwj10.15
//	private double[] testModel(DepProbMatrix model) throws IOException {
//		//XXX for (InstanceList il : model.corpus.testInstances) {
//		InstanceList il = model.corpus.testInstances.get(0);
//			int[][] parses = new int[il.instanceList.size()][];
//			int maxSentLength = 0;
//			for (WordInstance di : il.instanceList)
//				if (di.getNrWords() > maxSentLength)
//					maxSentLength = di.getNrWords();
//			System.out.println("max sentence length = " + maxSentLength);
//
//			double[] accuracies;
//			long starttime = System.currentTimeMillis();
//			accuracies = CKYParser.computeAccuracy(model, il.instanceList,
//					parses);
//			System.out
//					.println("Accuracy no-project for testfile "
//							+ il.name
//							+ ": direct accuracy & undirect accuracy for all, <=10, <=20:\t"
//							+ accuracies[0] + "\t" + accuracies[1] + "\t");
////							+ accuracies[2] + "\t" + accuracies[3] + "\t"
////							+ accuracies[4] + "\t" + accuracies[5]);
//			long endtime = System.currentTimeMillis();
//			System.out.println("Decoding took "
//					+ util.Printing.formatTime(endtime - starttime));
//			if( outputPrefix != null ) {
//				System.out.println("Saving predictions in " + outputPrefix
//						+ il.name);
//				CONLLWriter.printConll(parses, outputPrefix + il.name, il,
//						model.corpus);
//			}
//
//			if (projectAtTest && maxSentLength <= 10) {
//				if (true)
//					throw new RuntimeException("TODO: Fix constraints");
//				// int[][] parsesP = new int[il.instanceList.size()][];
//				// starttime = System.currentTimeMillis();
//				// DepSentenceDist[] sentenceDists = new
//				// DepSentenceDist[il.instanceList.size()];
//				// for(int i = 0; i < il.instanceList.size(); i++){
//				// sentenceDists[i] = new
//				// DepSentenceDist((DepInstance)il.instanceList.get(i),
//				// model.nontermMap);
//				// sentenceDists[i].cacheModelAndComputeIO(model);
//				// }
//				//
//				// // actually do the projection...
//				// PCType childType = (useChildWords? PCType.WORD : PCType.TAG);
//				// System.out.println("Using child " + (useChildWords? "words" :
//				// "tags") + " in L1LMax constraints");
//				// PCType parentType = (useParentWords? PCType.WORD :
//				// PCType.TAG);
//				// System.out.println("Using parent " + (useParentWords? "words"
//				// : "tags") + " in L1LMax constraints");
//				// CorpusConstraints constraints = new L1Lmax(model.corpus,
//				// il.instanceList, childType, parentType, constrainRoot,
//				// constrainDir, projectAtTestStrength, scaleCstrength,
//				// minOccurrencesForProjection,doNotProjectFile);
//				// constraints.project(sentenceDists);
//				//
//				// // do the parsing
//				// accuracies = CKYParser.computeAccuracy(sentenceDists,
//				// parsesP, model.corpus);
//				// System.out.println("Accuracy project for testfile " + il.name
//				// + ": directed " + accuracies[0] + ", undirected " +
//				// accuracies[1]);
//				// endtime = System.currentTimeMillis();
//				// System.out.println("Decoding took "+util.Printing.formatTime(endtime
//				// - starttime));
//				// System.out.println("Saving predictions in "+outputPrefix+"proj."+il.name);
//				// CONLLWriter.printConll(parsesP, outputPrefix+"proj."+il.name,
//				// il, model.corpus);
//			}
//		//}
//			
//			if (tmpfile != null) {
//				FileWriter fw = new FileWriter(tmpfile);
//				fw.write(accuracies[0] + "\t" + accuracies[1] + "\t"
//						+ accuracies[2] + "\t" + accuracies[3] + "\t"
//						+ accuracies[4] + "\t" + accuracies[5]);
//				fw.close();
//			}
//			
//			return accuracies;
//	}

}
