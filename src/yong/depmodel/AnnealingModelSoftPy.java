package yong.depmodel;
import static depparsing.globals.Constants.CONT;
import static depparsing.globals.Constants.END;
import static depparsing.globals.Constants.LEFT;
import static depparsing.globals.Constants.RIGHT;

import java.io.BufferedWriter;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.FileReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;

import java.util.regex.Pattern;
import java.util.Scanner;

import nn.Net;
import nn.Net2;
import nn.Net2Soft;
import nn.Net2SoftTwoLayers;
import nn.Node;
import util.Array;
import util.ArrayMath;

import org.ejml.simple.SimpleMatrix;

import data.WordInstance;
import depparsing.util.GrammarMath;
import depparsing.util.Pair;
import depparsing.data.DepCorpus;
import depparsing.data.DepInstance;
import depparsing.decoding.CKYParser;
import depparsing.model.DepModel;
import depparsing.model.DepProbMatrix;
import depparsing.model.DepSentenceDist;
import depparsing.model.KleinManningInitializer;
import depparsing.model.NonterminalMap;
import dic.Corpus;
import dic.Sentence;
import dic.Word;
import learning.EM;
import learning.stats.CompositeTrainStats;
import learning.stats.TrainStats;
import model.AbstractCountTable;
import model.AbstractModel;
import model.AbstractSentenceDist;

public class AnnealingModelSoftPy extends EM {//
	
	//public HashMap<String, Integer> tagIdx;
	public final int decisionValency;
	public final int childValency;
	public final int wordDim;
	public final int tagDim;
	public final int valencyDim;
	public final int dirDim;
	public final int decisionDim;
	
	public SimpleMatrix weight;
	public SimpleMatrix weight2;
	public SimpleMatrix rootWeight;
	public SimpleMatrix decisionWeight;
	
	public SimpleMatrix baisVec;
	
	public ArrayList<SimpleMatrix> valencyVecs;
	
	public SimpleMatrix leftDirVec;
	public SimpleMatrix rightDirVec;
	public SimpleMatrix stopVec;
	public SimpleMatrix continueVec;
	
	// Initial parameters
	public SimpleMatrix weightInit;
	public SimpleMatrix weight2Init;
	public SimpleMatrix rootWeightInit;
	public SimpleMatrix decisionWeightInit;
	
	public ArrayList<SimpleMatrix> valencyVecsInit;
	
	public SimpleMatrix leftDirVecInit;
	public SimpleMatrix rightDirVecInit;
	public SimpleMatrix stopVecInit;
	public SimpleMatrix continueVecInit;
	
	private double ChildBackoff = 0.33;
	
	private List<List<Integer>> chdANN;
	private List<List<Integer>> rootANN;
	private List<List<Integer>> decisionANN;
///////////////////////////////////////
	private List<List<Double>> chdANNSoft;
	private List<List<Double>> rootANNSoft;
	private List<List<Double>> decisionANNSoft;
////////////////For NN Mstep///////////////////////
	private List<List<Integer>> chdAndDesicionANN;
	private List<List<List<Integer>>> chdAndDesicionANNList;
///////////////////////////////////////
	private Integer[][][][] chdANNCountTable;
	private Integer[] rootANNCountTable;
	private Integer[][][][] decisionANNCountTable;
////////////////For Count NN Mstep////////////////////////
	private List<Integer[][][][]> chdANNCountTableList;
	private List<Integer[]> rootANNCountTableList;
	private List<Integer[][][][]> decisionANNCountTableList;
	
	private int onlineBatch;
	private int onlineNum;
	private int onlineWinSize;
	private double onlineEta;
	private int onlineIdx;
	private int[][] onlineStsSplit;
	private int accIdx;
	
	private double validPerc;
	
	public double sigma;//sigma>0 to protect with overfit
	private double sigma_l1;
	//public final int negSetSize;
	private int dicSize;
	//private double[][][][] childCount;
	//private double[] rootCount;
	//private double[][][][] decisionCount;
	private Corpus c;
	//private double learnRate = 0.001;
	
	private HashMap<int[], Pair<int[], Integer>> allTrees;
	private HashMap<int[], int[]> allValidTrees;
//	private HashMap<int[], int[]> allTreesForCount;
    private ArrayList<Pair<int[], int[]>> allTreesForCount;
	
	private int iteration;
	/////////////////////////////
	double[][][][] childPy;
	double[] rootPy;
	double[][][][] decisionPy;

	private double[][][][] childPyStcsSpec_train = null;
	private double[][][][][] decisionPyStcsSpec_train = null;
	private double[][][][] childPyStcsSpec_test = null;
	private double[][][][][] decisionPyStcsSpec_test = null;
	private double[][][][] childPyStcsSpec_val = null;
	private double[][][][][] decisionPyStcsSpec_val = null;
	private double[][][][] childPyStcsSpec_test_all = null;
	private double[][][][][] decisionPyStcsSpec_test_all = null;
	/////////////////////////////
	double[][][][] childCountForComp;
	double[] rootCountForComp;
	double[][][][] decisionCountForComp;
	///////////////////For DMV Mstep////////////////
	private List<double[][][][]> childCountForCompList;
	private List<double[]> rootCountForCompList;
	private List<double[][][][]> decisionCountForCompList;
	///////////////////////////////////
	
	double[][][][] childCountForComp1;
	double[] rootCountForComp1;
	double[][][][] decisionCountForComp1;
	//************DEBUGE!!
	int DEBUGE1 = 1;  // child
	int DEBUGE2 = 0;  // root
	int DEBUGE3 = 0;  // decision
	//////////////////////////////
	double wRate;
	double dirRate;
	double wordRate = 0.0001;
	double valencyRate = 0.0001;
	//////////////////////////////
	double wInitRate;
	double dirInitRate;
	double wordInitRate;
	double valencyInitRate;
	//////////////////////////
	protected double sigma_0 = 0;
	protected double sigma_k = 0;
	protected double sigma_e = 0;
	/////////////////////////////
	AbstractSentenceDist[] sentenceDists;
	DepProbMatrix counts;
	TrainStats<DepModel, DepSentenceDist> stats;		
	boolean lastIsViterbi = false;
	//////////////////////////////////////////
	private HashMap<Pair<String, String>,Pair<Integer, Integer>> universalRule;
	private double universalValue;
	private double regulationValue;
	///////////////////////////////////////////
	public final double prior;
	public boolean isEarlyStopping;
	public boolean isShuffle = false;  //

	int pascalIdx = 0;

	boolean ischeck = false;
	
	public AnnealingModelSoftPy(int decisionValency, int childValency, int negSetSize, 
			SimpleMatrix W, SimpleMatrix W2, SimpleMatrix rw, SimpleMatrix dw, SimpleMatrix stop, SimpleMatrix cont,
			int dim, int tagDim, int valVecDim, int dirVecDim, Corpus c, double sigma, double sigma_l1, 
			AbstractModel model, double sigma_0, double sigma_k, double sigma_e, double backoff, double prior, int pascalIdx) throws ClassNotFoundException, IllegalArgumentException, InstantiationException, IllegalAccessException, InvocationTargetException, IOException{//sigma_0=1, double sigma_k=-0.1, double sigma_e=0
		super(model);
		this.childValency = childValency;
		this.decisionValency = decisionValency;
		//this.negSetSize = negSetSize;
		
		this.wordDim = dim;
		this.tagDim = tagDim;
		this.valencyDim = valVecDim;
		this.dirDim = dirVecDim;
		this.decisionDim = dirVecDim; 
		this.sigma = sigma;
		this.sigma_l1 = sigma_l1;
		this.ChildBackoff = backoff;
		//weight = new SimpleMatrix(this.wordDim, this.wordDim + this.valencySize);
		this.weight = W;
		this.weight2 = W2;
		this.rootWeight = rw;
		this.decisionWeight = dw;
		this.stopVec = stop;// initialize outside
		this.continueVec = cont;// initialize outside
		
		allTrees = new HashMap<>();
//		allTreesForCount = new HashMap<>();
		allTreesForCount = new ArrayList<>();
		allValidTrees = new HashMap<>();
		this.dicSize = c.dic.size();
		//childCount = new double[dicSize][dicSize][2][this.valencySize];// no use
		//rootCount = new double[dicSize];
		//decisionCount = new double[dicSize][2][decisionValency][2];
		this.c = c;
		
		this.valencyVecs = new ArrayList<>();;// initialize inside

		//int[] vecs = new int[]{-1, -2, -3};
		for(int i = 0; i < this.childValency; i++){
			SimpleMatrix vvec = getUnusualVec(this.valencyDim);			
		//	vvec.set(0, 0, vecs[i]);
			this.valencyVecs.add(vvec);
		}
		

		
		leftDirVec = this.getUnusualVec(dirVecDim);//// initialize inside    *2!
		//leftDirVec.set(0, 0, -0.5);
		
		rightDirVec = this.getUnusualVec(dirVecDim);
		//rightDirVec.set(0, 0, 0.5);
		this.rePara2Init();
		iteration = 0;
		this.sigma_0 = sigma_0;
		this.sigma_k = sigma_k;
		this.sigma_e = sigma_e;
		//for annealing python
		this.sentenceDists = new AbstractSentenceDist[this.c.stcsWithLengthLessThanTen.size()];//allTrees.size()
		this.counts = new DepProbMatrix(dicSize,  decisionValency, childValency);//
		this.stats = CompositeTrainStats.buildTrainStats("");//statsFile		
		boolean lastIsViterbi = true;
		
		this.chdAndDesicionANNList = new ArrayList<>();
		this.childCountForCompList = new ArrayList<>();
		this.decisionCountForCompList = new ArrayList<>();
		this.rootCountForCompList = new ArrayList<>();
		
		this.chdANNCountTableList = new ArrayList<>();
		this.decisionANNCountTableList = new ArrayList<>();
		this.rootANNCountTableList = new ArrayList<>();
		//this.tagIdx = new HashMap<>();
		universalRule = new HashMap<>();
		universalValue = 0;
		regulationValue = 0;
		
		this.prior = prior;
		this.isEarlyStopping = false;
		//this.universalRule.
		universalRuleAdd();

		this.pascalIdx = pascalIdx;
	}
	private void universalRuleAdd() {
		List<Pair<String, String>> univsRule= new ArrayList<>();//[new Pair<"n", "m">];
		String[] verbs = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"};
		String[] nouns = {"NN", "NNS", "NNP", "NNPS"};
		String[] adj = {"JJ", "JJS", "JJR"};
		String[] pron = {"PRP", "PRP$", "WP"};//wp
		//String[] det = {"DT"};//"PDT" "WDT"//no
		String[] adv = {"RB", "RBR", "RBS"};//WRB except the clausal negation not and reduced forms of it, 
		String[] num = {"CD"};
		String[] adp = {"IN"};//in,"RP", "TO" when not a complementizer or subordinating conjunction.  IN is	Preposition or subordinating conjunction 
		//String[] conj = {"CC"};//"IN"

		String[] unitag = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
				"NN", "NNS", "NNP", "NNPS",
				"JJ", "JJS", "JJR",
				"PRP", "PRP$", "WP",
				"RB", "RBR", "RBS",
				"CD",
				"IN"};
		List<List<Integer>> unitagList = new ArrayList<>(unitag.length);
		for (int i = 0; i < unitag.length; i++){
			unitagList.add(i, new ArrayList<>());
		}
		for(Entry<Pair<String, String>, Integer> entry : this.c.dic.entrySet()){
			int loc = tagInSeq(entry.getKey().b, unitag);
			if(loc > -0.5){
//				if(unitagList.get(loc) == null){
//					
//				}
				//each array is a list of dic word according to a tag
				unitagList.get(loc).add(entry.getValue());
				//unitagList.set(index, element)
			}
		}
		
		for(int p = 0; p < verbs.length; p ++){
			for(int c = 0; c < verbs.length; c++){
				univsRule.add(new Pair<>(verbs[p], verbs[c]));
			}
		}

		for(int p = 0; p < verbs.length; p ++){
			for(int c = 0; c < nouns.length; c++){
				univsRule.add(new Pair<>(verbs[p], nouns[c]));
			}
		}

		for(int p = 0; p < verbs.length; p ++){
			for(int c = 0; c < pron.length; c++){
				univsRule.add(new Pair<>(verbs[p], pron[c]));
			}
		}
		for(int p = 0; p < verbs.length; p ++){
			for(int c = 0; c < adv.length; c++){
				univsRule.add(new Pair<>(verbs[p], adv[c]));
			}
		}


		for(int p = 0; p < nouns.length; p ++){
			for(int c = 0; c < nouns.length; c++){
				univsRule.add(new Pair<>(nouns[p], nouns[c]));
			}
		}
		for(int p = 0; p < nouns.length; p ++){
			for(int c = 0; c < adj.length; c++){
				univsRule.add(new Pair<>(nouns[p], adj[c]));
			}
		}
//		for(int p = 0; p < nouns.length; p ++){
//			for(int c = 0; c < det.length; c++){
//				univsRule.add(new Pair<>(nouns[p], det[c]));
//			}
//		}
		for(int p = 0; p < nouns.length; p ++){
			for(int c = 0; c < num.length; c++){
				univsRule.add(new Pair<>(nouns[p], num[c]));
			}
		}
//		for(int p = 0; p < nouns.length; p ++){
//			for(int c = 0; c < conj.length; c++){
//				univsRule.add(new Pair<>(nouns[p], conj[c]));
//			}
//		}
		for(int p = 0; p < adj.length; p ++){
			for(int c = 0; c < adv.length; c++){
				univsRule.add(new Pair<>(adj[p], adv[c]));
			}
		}
		for(int p = 0; p < adp.length; p ++){
			for(int c = 0; c < nouns.length; c++){
				univsRule.add(new Pair<>(adp[p], nouns[c]));
			}
		}
		for (int i = 0; i < univsRule.size(); i ++){
			Pair<String,String> rl = univsRule.get(i);
			int pidx = tagInSeq(rl.a, unitag);
			int cidx = tagInSeq(rl.b,unitag);
			if(pidx > -0.5 & cidx > -0.5){
				for (int j = 0; j < unitagList.get(pidx).size(); j++){
					for(int k = 0; k < unitagList.get(cidx).size(); k++){
						Pair<Integer, Integer> value = new Pair<>(unitagList.get(pidx).get(j),unitagList.get(cidx).get(k)); 
						Pair<String, String> dicrl = new Pair<>(rl.a+String.valueOf(j), rl.b+String.valueOf(k));// TAG + dic idx
						this.universalRule.put(dicrl, value);
						//System.out.println(dicrl.a + "\t" + dicrl.b + "\t" + value.a.toString() + "\t" + value.b.toString());
					}
				}
			}else
				System.out.println("Pair<String,String> rl error!");
		}
		// Root -> Verbs
		for(int i = 0; i < verbs.length; i ++){
			int root = -1;
			int cidx = tagInSeq(verbs[i],unitag);
			
			for(int k = 0; k < unitagList.get(cidx).size(); k++){
				Pair<String, String> dicrl = new Pair<>("ROOT", verbs[i]+String.valueOf(k));
				Pair<Integer, Integer> value = new Pair<>(root,unitagList.get(cidx).get(k)); 
				this.universalRule.put(dicrl, value);
				//System.out.println(dicrl.a + "\t" + dicrl.b + "\t" + value.a.toString() + "\t" + value.b.toString());
			}
		}

		//this.universalRule.
	}
	private int tagInSeq(String b, String[] unitag) {
		int loc = -1;
		for (int i = 0; i < unitag.length; i ++){
			if(b.equals(unitag[i])){
				loc = i;
			}
		}
		return loc;
	}
	public int getMaxX(){
		return this.dicSize * this.childValency * 2;
	}
	
	public void printTags(){
		for(int i = 0; i < this.c.dic.size(); i++){
			System.out.println(i + ":\t" + c.idx2dic.get(i));
		}
	}
	

	public void setChdAndDesPy(ArrayList<ArrayList<Double>> chdPy, ArrayList<ArrayList<Double>> desPy){

		this.decisionPy = new double[this.dicSize][2][this.decisionValency][2];

		this.childPy = new double[this.dicSize][this.dicSize][2][this.childValency];
		
		
		for(int p = 0; p < this.dicSize; p++)
			for(int dirChd = 0; dirChd < 2; dirChd++)
				for(int vc = 0; vc < this.childValency; vc++){
							int num = p * 2 * this.childValency + dirChd * this.childValency + vc;
							for(int chd = 0; chd < this.dicSize; chd++){
								this.childPy[chd][p][dirChd][vc] = Math.log(chdPy.get(num).get(chd)); 
							}
							
						}
		for(int p = 0; p < this.dicSize; p++)
			for(int dirChd = 0; dirChd < 2; dirChd++)
				for(int vc = 0; vc < this.decisionValency; vc++){
							int num = p * 2 * this.decisionValency + dirChd * this.decisionValency + vc;

							
							for(int cc = 0;cc < 2; cc++){
								this.decisionPy[p][dirChd][vc][cc] = Math.log(desPy.get(num).get(cc));
							}
							
						}
		
		System.out.println("transform from python: done!");
	}


	
	public void rateSetting(double wrate, double dirrate, double wordrate, double valencyrate){
		this.wRate = wrate;
		this.dirRate = dirrate;
		this.wordRate = wordrate;
		this.valencyRate = valencyrate;
		/////////////////////////////////////////////////
		wInitRate = wRate;
		dirInitRate = dirRate;
		wordInitRate = wordRate;
		valencyInitRate = valencyRate;
		//////////////////////////////////////////////
	}
	
	
	public void reSetRates(){
		wRate = wInitRate;
		dirRate = dirInitRate;
		wordRate = wordInitRate;
		valencyRate = valencyInitRate;
	}

	public void AnnealingEM(boolean normalizedGredient, int rateChoose, int EMiter, boolean isAlternate, int initType, int pre_acc_idx) throws Exception{
		int iter = 0;                                                                  //iter = 0; unsupervised iter
		double llh = Double.MAX_VALUE, pre_llh = Double.MIN_VALUE;
		HashMap<int[], int[]> allGoldTrees = new HashMap<>();
		for(Sentence s : c.stcsWithLengthLessThanTen){
			DepInstance depins = s.tran2DepIns();
			int[] gold = new int[s.goldTree.size()];
			for(int i = 0; i < gold.length; i++){
				gold[i] = s.goldTree.get(i);
			}
			allGoldTrees.put(depins.postags,  gold);
		}

		//this.kmInit();
	    //this.goldInit();
		///this.randomInit();
		//this.normPara();
		switch(initType){
			case 1:
				this.kmInit(this.pascalIdx);
				System.out.println("init type KM");
				break;
			case 2:
//				IF want annealing_init, just delete comments of the following two lines. You can also make some changes in Estep(if(iteration == 5)...), and DEBUGE1,3 = 0.!!
				this.afterAnnealingInit();
				this.normPara(Math.exp(-1e1));
				System.out.println("init type GOOD");
				break;
			case 3:
				this.goldInit();
				System.out.println("init type GOLD");
				break;
			case 4:
				this.randomInit();
				this.normPara(0);//For tuning.//for random it sometime is 1(such as in reproof)
				System.out.println("init type RANDOM");
				break;
			case 5:
				this.uniformInit();
				this.normPara(0);//For tuning.
				System.out.println("init type UNIFORM");
				break;
                        case 6:
				this.ruleProbInit(pre_acc_idx);  
				System.out.println("init type ruleProb");
				break;
			default:
				System.out.println("init type error!!! NOT FING!");
			
		}

		//AbstractSentenceDist[] sentenceDists = model.getSentenceDists();//Annealing
		AbstractSentenceDist[] sentenceDists = new AbstractSentenceDist[this.c.stcsWithLengthLessThanTen.size()];//allTrees.size()
		//DepProbMatrix  counts = (DepProbMatrix) model.getCountTable();
		DepProbMatrix counts = new DepProbMatrix(dicSize, this.decisionValency, this.childValency);//
		TrainStats<DepModel, DepSentenceDist> stats = CompositeTrainStats.buildTrainStats("");//statsFile
		
		boolean lastIsViterbi = true;
		
		double rate = 1;

	}

	public void ruleProbInit(int initAccIdx) throws Exception{
		this.childCountForComp = new double[dicSize][dicSize][2][this.childValency];
		this.rootCountForComp = new double[dicSize];
		this.decisionCountForComp = new double[dicSize][2][decisionValency][2];
		
		String tempfilechd = System.getProperty("user.dir") + "/temp/" + "ChdCountForComp" + String.valueOf(initAccIdx) +  ".txt";
		String tempfiledec = System.getProperty("user.dir") + "/temp/" + "DecCountForComp" + String.valueOf(initAccIdx) +  ".txt";
		String tempfileroot = System.getProperty("user.dir") + "/temp/" + "RootCountForComp" + String.valueOf(initAccIdx) +  ".txt";
		
		BufferedReader outputBwchd = new BufferedReader(new FileReader(tempfilechd));
		String line = outputBwchd.readLine();
		String[] toks = line.split("\t");
		int index = 0;
		for(int c = 0; c < dicSize; c++){
		  for(int p = 0; p < dicSize; p++){
			for(int d = 0; d < 2; d++){
				for(int v = 0; v < this.childValency; v++){
					    this.childCountForComp[c][p][d][v] = Double.valueOf(toks[index]);
					    index++;
				    }
			    }			
		    }
	    }
		outputBwchd.close();
		
		BufferedReader outputBwdec = new BufferedReader(new FileReader(tempfiledec));
		String line1 = outputBwdec.readLine();
		String[] toks1 = line1.split("\t");
		index = 0;
		for(int p = 0; p < dicSize; p++){
			for(int d = 0; d < 2; d++){
				for(int v = 0; v < decisionValency; v++){
					for(int s = 0; s < 2; s++){
						this.decisionCountForComp[p][d][v][s] = Double.valueOf(toks1[index]);
						index++;
					    }
				    }
				
			 }
	    }
		outputBwdec.close();
		
		BufferedReader outputBwroot = new BufferedReader(new FileReader(tempfileroot));
		String line2 = outputBwroot.readLine();
		String[] toks2 = line2.split("\t");
		index = 0;
		for(int r = 0; r < dicSize;r++){
			this.rootCountForComp[r] = Double.valueOf(toks2[index]);
			index++;
		}
		outputBwroot.close();

	}	

	public void ViterbiEStep() throws Exception{
		this.iteration++;
		System.out.println("=======================Now iteration:\t" + this.iteration);
		this.ViterbiEStep(false);
	}

	public void ViterbiEStep(Boolean is_pytorch, String train_chd, String train_dec, String val_chd, String val_dec, String test_chd, String test_dec, String test_all_chd, String test_all_dec) throws Exception{
		this.iteration++;
		System.out.println("=======================Now iteration:\t" + this.iteration);
		this.ViterbiEStep(train_chd,  train_dec,  val_chd,  val_dec,  test_chd,  test_dec, test_all_chd, test_all_dec);
	}
	public void universalization(double universalValue){

		this.universalValue = universalValue;//?
		for(int c = 0; c < dicSize; c++){
			for(int p = 0; p < dicSize; p++){
				for(int d = 0; d < 2; d++){
					for(int v = 0; v < this.childValency; v++){
						Pair<Integer, Integer> rule = new Pair<Integer, Integer>(Integer.valueOf(p) ,Integer.valueOf(c));
						if(this.universalRule.containsValue(rule)){
							childCountForComp[c][p][d][v] = Math.log((Math.exp(childCountForComp[c][p][d][v]) + universalValue));
						}
						
					}
				}			
			}
		}
		
		double[][][] childCountForCompDen = new double[dicSize][2][this.childValency];// no use

		
		
		for(int c = 0; c < dicSize; c++){
			for(int p = 0; p < dicSize; p++){
				for(int d = 0; d < 2; d++){
					for(int v = 0; v < this.childValency; v++){

						childCountForCompDen[p][d][v] += Math.exp(childCountForComp[c][p][d][v]);//

					}
				}			
			}
		}

		
		// Normalization!!!!
		for(int c = 0; c < dicSize; c++){
			for(int p = 0; p < dicSize; p++){
				for(int d = 0; d < 2; d++){
					for(int v = 0; v < this.childValency; v++){
							childCountForComp[c][p][d][v] = Math.log(Math.exp(childCountForComp[c][p][d][v]) / (childCountForCompDen[p][d][v]));
					}
				}			
			}
		}
		

	}

	//public byte[][] ChdAndDecSoftMStep(){}
    public List<List<Double>> ChdSoftMStep() throws Exception{
		chdANNSoft = new ArrayList<>();
		rootANNSoft = new ArrayList<>();
		decisionANNSoft = new ArrayList<>();

		for(AbstractSentenceDist oneStce1 : sentenceDists){
			//DepSentenceDist dist = (DepSentenceDist) adist;
			//AbstractSentenceDist
			DepSentenceDist oneStce = (DepSentenceDist)oneStce1;
			//int[] sentence = oneStce.depInst.postags;//word??
			//int[] parser   = oneStce.getValue();
			NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
			HashMap<Integer, Double> rootMap = new HashMap<>();
			HashMap<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Double> child = new HashMap<>();
			HashMap<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Double> decision = new HashMap<>();
			

			countDependenciesPairSoft(oneStce, rootMap, child, decision, nonMap);
			double ruleThread = 1e-40;//ruleThread = 1e-40
			for(Entry<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Double> element : child.entrySet()){
			  //Map.Entry<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Integer>
//				Integer dd = 1;
//				Double d = Double.valueOf(dd);
				if(element.getValue() > ruleThread){
					Double[] arrIns = new Double[]{Double.valueOf(element.getKey().a.a), Double.valueOf(element.getKey().b.a),Double.valueOf(element.getKey().b.b), Double.valueOf(element.getKey().a.b), Double.valueOf(element.getValue())};
					List<Double> ins =  (List<Double>) Arrays.asList(arrIns);
					chdANNSoft.add(ins);
				}

			}
			
			for(Entry<Integer, Double> element : rootMap.entrySet()){

				Double[] arrIns = new Double[]{Double.valueOf(element.getKey())};
				List<Double> ins = (List<Double>) Arrays.asList(arrIns);
				rootANNSoft.add(ins);
				
			}
			
			for(Entry<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Double> element : decision.entrySet()){
                if(element.getValue() > ruleThread){
    				Double[] arrIns = new Double[]{Double.valueOf(element.getKey().a.a), Double.valueOf(element.getKey().a.b),
    						Double.valueOf(element.getKey().b.a), Double.valueOf(element.getKey().b.b), Double.valueOf(element.getValue())};
    				List<Double> ins = (List<Double>) Arrays.asList(arrIns);
    				decisionANNSoft.add(ins);
                }

			}
		}
		return chdANNSoft;
	}

	public double getPerc(){
		return this.validPerc;
	}
	

	/**
	 * did not use this.allValidTrees, cause we actuallty did not use it in neural network.
	 * @param filechd
	 * @param filedec
	 * @throws IOException 
	 */
	public void MStepTxt(String filechd, String filedec) throws IOException{
		FileWriter wf_chd = new FileWriter(filechd);
		FileWriter wf_dec = new FileWriter(filedec);

		for(Map.Entry<int[], Pair<int[], Integer>> entry : this.allTrees.entrySet()){
			childCountForComp1 = new double[dicSize][dicSize][2][this.childValency];// no use
			rootCountForComp1 = new double[dicSize];
			decisionCountForComp1 = new double[dicSize][2][decisionValency][2];

			int[] w = entry.getKey();
			int[] p = entry.getValue().a;
			int sts_idx = entry.getValue().b;
			
			NonterminalMap nonMapPri = new NonterminalMap(this.decisionValency, this.childValency);//  c p d v
			this.countDependencies(p, w, childCountForComp1, rootCountForComp1, decisionCountForComp1, nonMapPri);
			// maybe waste a lot of time
			for(int i = 0; i < dicSize; i++){
				for(int j = 0; j < dicSize; j++){
					for(int k = 0; k < 2; k ++){
						for(int l = 0; l < this.childValency; l ++){
							for(int eva = 0; eva < (int)childCountForComp1[i][j][k][l]; eva ++){  // id 0 p c l r v
							    wf_chd.write(String.valueOf(sts_idx) + "\t" +String.valueOf("0") + "\t"+ String.valueOf(j) +
										"\t" + String.valueOf(i) + "\t" + String.valueOf(1-k) + "\t"+ String.valueOf(k) + "\t" + String.valueOf(l));
							    wf_chd.write("\n");
							}


						}
					}
				}
			}
			for(int i = 0; i < dicSize; i++){
				for(int j = 0; j < 2; j++){   //direction
					for(int k = 0; k < decisionValency; k ++){ // p d v stop
						for(int l = 0; l < 2; l ++){  // stop and conti     //     i j  k  l
							for(int eva = 0; eva < (int)decisionCountForComp1[i][j][k][l]; eva ++) {  // id 1 p stop l r v

								wf_dec.write(String.valueOf(sts_idx) + "\t" + String.valueOf("1") + "\t" + String.valueOf(i) +
										"\t" + String.valueOf(l) + "\t" + String.valueOf(1-j) + "\t" + String.valueOf(j) + "\t"+ String.valueOf(k));
								wf_dec.write("\n");
							}

						}
					}
				}
			}


		}

		wf_chd.close();
		wf_dec.close();
		//NeuDMVaddPrior();//adding prior information need other extra steps. Should we add prior in total countTable?
//		chdRule2file(childCountForComp1 ,filechd);
//		decRule2file(decisionCountForComp1,filedec);
		
		this.allTreesForCount.clear();
		this.allTrees.clear();
		this.allValidTrees.clear();
	}

	
	public Integer[][][][] getValidChdCountTable(){
		return this.chdANNCountTable;
	}
	
	public Integer[][][][] getValidDecisionCountTable(){
		return this.decisionANNCountTable;
	}
	
	public List<Integer[][][][]> getDecisionCountTable(){
		return this.decisionANNCountTableList;
	}
	
	public Integer[] getRootCountTable(){
		return this.rootANNCountTable;
	}
	
	public List<List<Integer>> getRoot(){
		return this.rootANN;
	}
	public List<List<Integer>> getDecision(){
		return this.decisionANN;
	}
	public List<List<Double>> getSoftRoot(){
		return this.rootANNSoft;
	}
	public List<List<Double>> getSoftDecision(){
		return this.decisionANNSoft;
	}


	private void updateModel(double[][][][] child, double[] root, double[][][][] decision) {
		DepProbMatrix para = new DepProbMatrix(this.dicSize, this.decisionValency, this.childValency);
		ArrayMath.setEqual(para.child, child);
		ArrayMath.setEqual(para.root, root);
		ArrayMath.setEqual(para.decision, decision);
//		para.addChildBackoffAnneal(backoff, this.c.tag2idx.size());
//		para.logNormalize();
//		para.child = child.clone();
//		para.root = root.clone();
//		para.decision = decision.clone();
		// TODO Auto-generated method stub
		model.updateParameters(para);
//		((DepModel)model).params.fill(para);
	}


	public void setChildPyStcsSpec(String f_chd, int train_test_val_testall){ // here 10 is a parameter, if test longer sts, should be change
		if(train_test_val_testall==0)
		    this.childPyStcsSpec_train = new double[this.c.stcsWithLengthLessThanTen.size()][10][10][this.childValency];  // stc_idx c p v prob
		else if(train_test_val_testall==1)
		    this.childPyStcsSpec_test = new double[this.c.testStcsLessThanTen.size()][10][10][this.childValency];  // stc_idx c p v prob
		else if(train_test_val_testall==2)
			this.childPyStcsSpec_val = new double[this.c.validStcsLessThanTen.size()][10][10][this.childValency];  // stc_idx c p v prob
		else if(train_test_val_testall==3)
			this.childPyStcsSpec_test_all = new double[this.c.testStcs.size()][60][60][this.childValency];  // stc_idx c p v prob

		try (BufferedReader br = new BufferedReader(new FileReader(f_chd))) {
			String line;
			int sentID = 0, head = 0, chd = 0, dir = 0, val = 0;
			double prob = 0.0;
			while ((line = br.readLine()) != null) {
//				System.out.println(line);
				String[] tokens = line.split("\t");
				sentID = Integer.parseInt(tokens[0]);// idx c_i p_i v prob
				chd = Integer.parseInt(tokens[1]);
				head = Integer.parseInt(tokens[2]);
				val = Integer.parseInt(tokens[3]);
				prob = Double.parseDouble(tokens[4]);
				if(train_test_val_testall==0)
				    this.childPyStcsSpec_train[sentID][chd][head][val] = Math.log(prob);
				if(train_test_val_testall==1)
					this.childPyStcsSpec_test[sentID][chd][head][val] = Math.log(prob);
				if(train_test_val_testall==2)
					this.childPyStcsSpec_val[sentID][chd][head][val] = Math.log(prob);
				if(train_test_val_testall==3)
					this.childPyStcsSpec_test_all[sentID][chd][head][val] = Math.log(prob);
			}
		} catch(IOException io) { }
	}

	public void setDecisionPyStcsSpec(String f_dec, int train_test_val_testall){
		if (train_test_val_testall==0)
		    this.decisionPyStcsSpec_train = new double[this.c.stcsWithLengthLessThanTen.size()][10][2][this.decisionValency][2];  // sts_idx, p d v s prob
		if (train_test_val_testall==1)
		    this.decisionPyStcsSpec_test = new double[this.c.testStcsLessThanTen.size()][10][2][this.decisionValency][2];  // sts_idx, p d v s prob
		if (train_test_val_testall==2)
		    this.decisionPyStcsSpec_val = new double[this.c.validStcsLessThanTen.size()][10][2][this.decisionValency][2];  // sts_idx, p d v s prob
		if (train_test_val_testall==3)
			this.decisionPyStcsSpec_test_all = new double[this.c.testStcs.size()][60][2][this.decisionValency][2];  // sts_idx, p d v s prob

		try (BufferedReader br = new BufferedReader(new FileReader(f_dec))) {
			String line;
			int sentID = 0, head = 0, chd = 0, dir = 0, val = 0;
			double prob = 0.0;
			while ((line = br.readLine()) != null) {
//				System.out.println(line);
				String[] tokens = line.split("\t");
				sentID = Integer.parseInt(tokens[0]); // idx p_i d v s prob
				head = Integer.parseInt(tokens[1]);
				dir = Integer.parseInt(tokens[2]);
				val = Integer.parseInt(tokens[3]);
				chd = Integer.parseInt(tokens[4]);
				prob = Double.parseDouble(tokens[5]);
                if (train_test_val_testall==0)
				    this.decisionPyStcsSpec_train[sentID][head][dir][val][chd] = Math.log(prob);
				if (train_test_val_testall==1)
					this.decisionPyStcsSpec_test[sentID][head][dir][val][chd] = Math.log(prob);
				if (train_test_val_testall==2)
					this.decisionPyStcsSpec_val[sentID][head][dir][val][chd] = Math.log(prob);
				if (train_test_val_testall==3)
					this.decisionPyStcsSpec_test_all[sentID][head][dir][val][chd] = Math.log(prob);
			}
		} catch(IOException io) { }
	}

	public double[][][][] setchCount(String f_chd){

		double[][][][] chdc = new double[this.dicSize][this.dicSize][2][2];  // c p d v
		setChildPyStcsSpec(f_chd, 0);
//		this.childPyStcsSpec_train[sentID][chd][head][val] = Math.log(prob);
//		int dir = (c < p ? LEFT : RIGHT);
		for(int i=0; i<c.stcsWithLengthLessThanTen.size(); i++){
			int len = c.testStcsLessThanTen.get(i).wordIdxInDic.size();
			ArrayList<Integer> words = c.testStcsLessThanTen.get(i).wordIdxInDic;
			for(int c = 0; c < len; c++){
				for (int p = 0; p < len; p++) {
					if(c!=p) {
						for (int v = 0; v < 2; v++) {
							int d = 0;
							if (c < p)
								d = 0;
							else
								d = 1;
							if (chdc[words.get(c)][words.get(p)][d][v] == 0.0)
								chdc[words.get(c)][words.get(p)][d][v] = this.childPyStcsSpec_train[i][c][p][v];
							else {
								if (Math.abs(chdc[words.get(c)][words.get(p)][d][v] - this.childPyStcsSpec_train[i][c][p][v]) > 1e-4)
									System.out.println("chd error!! " + String.valueOf(chdc[words.get(c)][words.get(p)][d][v]) + "  " + String.valueOf(this.childPyStcsSpec_train[i][c][p][v]));
							}
						}
					}

				}
			}
		}



		//		try (BufferedReader br = new BufferedReader(new FileReader(f_dec))) {
//			String line;
//			int c1=0, h=0, d=0, v=0;
//			double prob = 0.0;
//			while ((line = br.readLine()) != null) {
//				String[] tokens = line.split("\t");
//				c1 = Integer.parseInt(tokens[0]);
//				h = Integer.parseInt(tokens[1]);
//				d = Integer.parseInt(tokens[2]);
//				v = Integer.parseInt(tokens[3]);
//				prob = Double.parseDouble(tokens[4]);
//
//				chdc[c1][h][d][v] = Math.log(prob);
//			}
//		} catch(IOException io) { }
		return chdc;
	}
	public double[][][][] setdecCount(String f_dec){
		double[][][][] decc = new double[this.dicSize][2][2][2];  // p d v s
		setDecisionPyStcsSpec(f_dec, 0);
		for(int i=0; i<c.stcsWithLengthLessThanTen.size(); i++){
			int len = c.testStcsLessThanTen.get(i).wordIdxInDic.size();
			ArrayList<Integer> words = c.testStcsLessThanTen.get(i).wordIdxInDic;
			for(int p = 0; p < len; p++){
				for(int d = 0; d < 2; d++) {
					for (int v = 0; v < 2; v++) {
						for (int s = 0; s < 2; s++) {
							if (decc[words.get(p)][d][v][s] == 0.0)
								decc[words.get(p)][d][v][s] = this.decisionPyStcsSpec_train[i][p][d][v][s];  // p d v s
							else {
								if (Math.abs(decc[words.get(p)][d][v][s] - this.decisionPyStcsSpec_train[i][p][d][v][s]) > 1e-4)
									System.out.println("dec error!!  "+String.valueOf(decc[words.get(p)][d][v][s]) + "  "+ String.valueOf(this.decisionPyStcsSpec_train[i][p][d][v][s]));
							}
						}
					}
				}
			}
		}


//		try (BufferedReader br = new BufferedReader(new FileReader(f_dec))) {
//			String line;
//			int p=0, d=0, v=0, s=0;
//			double prob = 0.0;
//			while ((line = br.readLine()) != null) {
//				String[] tokens = line.split("\t");
//				p = Integer.parseInt(tokens[0]);
//				d = Integer.parseInt(tokens[1]);
//				v = Integer.parseInt(tokens[2]);
//				s = Integer.parseInt(tokens[3]);
//				prob = Double.parseDouble(tokens[4]);
//
//				decc[p][d][v][s] = Math.log(prob);
//
//			}
//		} catch(IOException io) { }
		return decc;
	}

	public void ViterbiEStep(String train_chd, String train_dec, String val_chd, String val_dec, String test_chd, String test_dec, String test_all_chd, String test_all_dec) throws Exception{
		double llh = 0.0;

		int count = 0;
		int times = 0;

		long tic, toc;
		tic = System.currentTimeMillis();
		int acc = 0, wordNum = 0;//accurate edge & word number

		double[][][] sentenceChild;
		double[] sentenceRoot;
		double[][][][] sentenceDecision;

		if(iteration == 1)
			System.out.println("DMV");
		else
			System.out.println("ANN");
//		System.out.println("Test Scale!");

		double[][][][] chCount = null, decCount = null;
		double[] rootCount = null;
		if(DEBUGE1 == 0 ||  iteration == 1)
			chCount = this.childCountForComp;
		else{
			if(ischeck) {
				chCount = setchCount(train_chd);

			}else{
				chCount = this.childCountForComp;
				setChildPyStcsSpec(train_chd, 0);
				setChildPyStcsSpec(test_chd, 1);
				setChildPyStcsSpec(val_chd, 2);
				setChildPyStcsSpec(test_all_chd, 3);
			}
		}
		if(DEBUGE2 == 0 ||  iteration == 1)
			rootCount = this.rootCountForComp;
		else{
			rootCount = this.rootPy;
		}
		if(DEBUGE3 == 0 ||  iteration == 1)
			decCount = this.decisionCountForComp;
		else{
			if(ischeck) {
				decCount = setdecCount(train_dec);
			}else{
				decCount = this.decisionCountForComp;
				setDecisionPyStcsSpec(train_dec, 0);
				setDecisionPyStcsSpec(test_dec, 1);
				setDecisionPyStcsSpec(val_dec, 2);
				setDecisionPyStcsSpec(test_all_dec, 3);
			}

		}
		if(this.iteration > 1){
			for(int i = this.onlineStsSplit[this.onlineIdx][0]; i < this.onlineStsSplit[this.onlineIdx][1]; i++){// !!!
				// Using the CYK parsing function in the library,
				// We need to prepare the parameters first!
				Sentence s = c.stcsWithLengthLessThanTen.get(i);

				NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
				DepInstance depins = s.tran2DepIns();
				DepSentenceDist sd = new DepSentenceDist(depins, nonMap);

				int len = s.getLength();
				sentenceChild = new double[len][len][this.childValency]; // c p v
				sentenceRoot = new double[len];
				sentenceDecision = new double[len][2][this.decisionValency][2]; // p d v s

//				double[][][] sentenceChild1 = new double[len][len][this.childValency]; // c p v

//				double[][][][] sentenceDecision1 = new double[len][2][this.decisionValency][2]; // p d v s

				this.cacheModel(s.wordIdxInDic, chCount, rootCount, decCount,
						sentenceChild, sentenceRoot, sentenceDecision);  //only using root
				if(ischeck){
					for (int si = 0; si < len; si++) {
						if (this.DEBUGE1 == 1) {
							for (int sj = 0; sj < len; sj++) {
								if(si!=sj) {
									for (int sk = 0; sk < this.childValency; sk++) {
										if (sentenceChild[si][sj][sk] != this.childPyStcsSpec_train[i][si][sj][sk])
											System.out.println("chd error!!  "+ String.valueOf(sentenceChild[si][sj][sk]) +"  "+String.valueOf( this.childPyStcsSpec_train[i][si][sj][sk]));
									}
								}
							}
						}
						if (this.DEBUGE3 == 1) {
							for (int sl = 0; sl < 2; sl++) {
								for (int sm = 0; sm < this.decisionValency; sm++) {
									for (int sn = 0; sn < 2; sn++) {
										if(sentenceDecision[si][sl][sm][sn] != this.decisionPyStcsSpec_train[i][si][sl][sm][sn])
											System.out.println("dec error!!  "+String.valueOf(sentenceDecision[si][sl][sm][sn])+"  " +String.valueOf( this.decisionPyStcsSpec_train[i][si][sl][sm][sn]));
									}
								}
							}
						}
					}

				}
				else {
					for (int si = 0; si < len; si++) {
						if (this.DEBUGE1 == 1) {
							for (int sj = 0; sj < len; sj++) {
								for (int sk = 0; sk < this.childValency; sk++) {
									sentenceChild[si][sj][sk] = this.childPyStcsSpec_train[i][si][sj][sk];
								}
							}
						}
						if (this.DEBUGE3 == 1) {
							for (int sl = 0; sl < 2; sl++) {
								for (int sm = 0; sm < this.decisionValency; sm++) {
									for (int sn = 0; sn < 2; sn++) {
										sentenceDecision[si][sl][sm][sn] = this.decisionPyStcsSpec_train[i][si][sl][sm][sn];
									}
								}
							}
						}
					}
				}
				sd.cacheModelFromOutside(sentenceRoot, sentenceChild, sentenceDecision);
				int[] parser = new int[s.getLength()];
				double score = CKYParser.parseSentence(sd, parser, null, null, universalRule, regulationValue);//10.7
				Pair<int[], Integer> temp = new Pair<>(parser, i);
				this.allTrees.put(depins.postags, temp);

				llh += score;
				acc += s.accWords(parser);
				wordNum += s.getLength();

				count ++;
				if((count - times * 1000) > 0){
					toc = System.currentTimeMillis();
					//System.out.println("Iteration 1000 times! with total:" + count + "time consuming:" + (toc - tic));
					tic = toc;
					times++;

				}
			}
		}else{
			for(int i=0; i<c.stcsWithLengthLessThanTen.size(); i++){
				Sentence s = c.stcsWithLengthLessThanTen.get(i);
				NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
				DepInstance depins = s.tran2DepIns();
				DepSentenceDist sd = new DepSentenceDist(depins, nonMap);

				int len = s.getLength();
				sentenceChild = new double[len][len][this.childValency];
				sentenceRoot = new double[len];
				sentenceDecision = new double[len][2][this.decisionValency][2];

				this.cacheModel(s.wordIdxInDic, chCount, rootCount, decCount,
						sentenceChild, sentenceRoot, sentenceDecision);


				sd.cacheModelFromOutside(sentenceRoot, sentenceChild, sentenceDecision);
				int[] parser = new int[s.getLength()];
				double score = CKYParser.parseSentence(sd, parser, null, null, universalRule, regulationValue);//10.7
				Pair<int[], Integer> temp= new Pair<>(parser, i);
				allTrees.put(depins.postags, temp);

				llh += score;
				acc += s.accWords(parser);
				wordNum += s.getLength();

				count ++;
				if((count - times * 1000) > 0){
					toc = System.currentTimeMillis();
					//System.out.println("Iteration 1000 times! with total:" + count + "time consuming:" + (toc - tic));
					tic = toc;
					times++;

				}
			}
			this.onlineIdx--;
		}

//		for(Sentence s : c.stcsWithLengthLessThanTen){ //
		for(int i = 0; i < c.stcsWithLengthLessThanTen.size(); i++){ // allTreesForCount is used for one kind of online em: for mstep of ROOT
			Sentence s = c.stcsWithLengthLessThanTen.get(i);
			NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
			DepInstance depins = s.tran2DepIns();
			DepSentenceDist sd = new DepSentenceDist(depins, nonMap);

			int len = s.getLength();
			sentenceChild = new double[len][len][this.childValency];
			sentenceRoot = new double[len];
			sentenceDecision = new double[len][2][this.decisionValency][2];

			this.cacheModel(s.wordIdxInDic, chCount, rootCount, decCount,
					sentenceChild, sentenceRoot, sentenceDecision);
			if(ischeck){

			}
			else {
				if(this.iteration>1) {
					for (int si = 0; si < len; si++) {
						if (this.DEBUGE1 == 1) {
							for (int sj = 0; sj < len; sj++) {
								for (int sk = 0; sk < this.childValency; sk++) {
									sentenceChild[si][sj][sk] = this.childPyStcsSpec_train[i][si][sj][sk];
								}
							}
						}
						if (this.DEBUGE3 == 1) {
							for (int sl = 0; sl < 2; sl++) {
								for (int sm = 0; sm < this.decisionValency; sm++) {
									for (int sn = 0; sn < 2; sn++) {
										sentenceDecision[si][sl][sm][sn] = this.decisionPyStcsSpec_train[i][si][sl][sm][sn];
									}
								}
							}
						}
					}
				}
			}

			sd.cacheModelFromOutside(sentenceRoot, sentenceChild, sentenceDecision);
			int[] parser = new int[s.getLength()];
			double score = CKYParser.parseSentence(sd, parser, null, null, universalRule, regulationValue);//10.7
//			allTreesForCount.put(depins.postags, parser);
			allTreesForCount.add(new Pair(depins.postags, parser));

			llh += score;
			acc += s.accWords(parser);
			wordNum += s.getLength();

			count ++;
			if((count - times * 1000) > 0){
				toc = System.currentTimeMillis();
				//System.out.println("Iteration 1000 times! with total:" + count + "time consuming:" + (toc - tic));
				tic = toc;
				times++;

			}

		}
		if(this.iteration>1) {
			saveSentenceAndParsing(this.accIdx, this.allTreesForCount, c.stcsWithLengthLessThanTen);
		}


		if(this.onlineIdx < this.onlineNum - 1)
			this.onlineIdx ++;
		else{
			this.onlineIdx = 0;
			if(isShuffle)
				trainStsShuffle();
		}


		if(0 == 0) {//test10 //this.onlineIdx == 0
			acc = 0;
			wordNum = 0;
			llh = 0.0;
			count = 0;
			times = 0;

//			for (Sentence s : c.testStcsLessThanTen) {
			for (int i=0; i < c.testStcsLessThanTen.size(); i++) {
				Sentence s = c.testStcsLessThanTen.get(i);
				// Using the CYK parsing function in the library,
				// We need to prepare the parameters first!

				NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
				DepInstance depins = s.tran2DepIns();
				DepSentenceDist sd = new DepSentenceDist(depins, nonMap);
				//	sd.cacheModelFromOutside(root, child, decision);
				int len = s.getLength();
				sentenceChild = new double[len][len][this.childValency];
				sentenceRoot = new double[len];
				sentenceDecision = new double[len][2][this.decisionValency][2];

				this.cacheModel(s.wordIdxInDic, chCount, rootCount, decCount,
						sentenceChild, sentenceRoot, sentenceDecision);

				///
				if(ischeck){}
				else {
					if (this.iteration > 1) {
						for (int si = 0; si < len; si++) {
							if (this.DEBUGE1 == 1) {
								for (int sj = 0; sj < len; sj++) {
									for (int sk = 0; sk < this.childValency; sk++) {
										sentenceChild[si][sj][sk] = this.childPyStcsSpec_test[i][si][sj][sk];
									}
								}
							}
							if (this.DEBUGE3 == 1) {
								for (int sl = 0; sl < 2; sl++) {
									for (int sm = 0; sm < this.decisionValency; sm++) {
										for (int sn = 0; sn < 2; sn++) {
											sentenceDecision[si][sl][sm][sn] = this.decisionPyStcsSpec_test[i][si][sl][sm][sn];
										}
									}
								}
							}
						}
					}
				}
				////
				sd.cacheModelFromOutside(sentenceRoot, sentenceChild, sentenceDecision);
				int[] parser = new int[s.getLength()];
				double score = CKYParser.parseSentence(sd, parser, null, null, universalRule, regulationValue);
				//this.allTrees.put(depins.postags, parser);
				llh += score;
				acc += s.accWords(parser);
				wordNum += s.getLength();

				count++;
				if ((count - times * 1000) > 0) {
					toc = System.currentTimeMillis();
					//System.out.println("Iteration 1000 times! with total:" + count + "time consuming:" + (toc - tic));
					tic = toc;
					times++;

				}
			}

			System.out.println("Debuge! Testing accuracy performance(10):\t" + (double) acc / wordNum);
			acc2File("acc" + String.valueOf(accIdx) + ".txt", String.valueOf((double) acc / wordNum));
			System.out.println("Testing data llh(10):\t" + llh);
		}
		if(0 == 0) {
			acc = 0;
			wordNum = 0;
			llh = 0.0;
			count = 0;
			times = 0;
//			for(Sentence s : c.testStcs){
			for (int i = 0; i < c.testStcs.size(); i++) {
				// Using the CYK parsing function in the library,
				// We need to prepare the parameters first!
				Sentence s = c.testStcs.get(i);


				NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
				DepInstance depins = s.tran2DepIns();
				DepSentenceDist sd = new DepSentenceDist(depins, nonMap);
				//	sd.cacheModelFromOutside(root, child, decision);
				int len = s.getLength();
				sentenceChild = new double[len][len][this.childValency];
				sentenceRoot = new double[len];
				sentenceDecision = new double[len][2][this.decisionValency][2];


				this.cacheModel(s.wordIdxInDic, chCount, rootCount, decCount,
						sentenceChild, sentenceRoot, sentenceDecision);

				///
				if(ischeck){}
				else {
					if (this.iteration > 1) {
						for (int si = 0; si < len; si++) {
							if (this.DEBUGE1 == 1) {
								for (int sj = 0; sj < len; sj++) {
									for (int sk = 0; sk < this.childValency; sk++) {
										sentenceChild[si][sj][sk] = this.childPyStcsSpec_test_all[i][si][sj][sk];
									}
								}
							}
							if (this.DEBUGE3 == 1) {
								for (int sl = 0; sl < 2; sl++) {
									for (int sm = 0; sm < this.decisionValency; sm++) {
										for (int sn = 0; sn < 2; sn++) {
											sentenceDecision[si][sl][sm][sn] = this.decisionPyStcsSpec_test_all[i][si][sl][sm][sn];
										}
									}
								}
							}
						}
					}
				}
				////


				sd.cacheModelFromOutside(sentenceRoot, sentenceChild, sentenceDecision);
				int[] parser = new int[s.getLength()];
				double score = CKYParser.parseSentence(sd, parser, null, null, universalRule, regulationValue);
				//allTrees.put(depins.postags, parser);

				llh += score;
				acc += s.accWords(parser);
				wordNum += s.getLength();

				count++;
				if ((count - times * 1000) > 0) {
					toc = System.currentTimeMillis();
					//System.out.println("Iteration 1000 times! with total:" + count + "time consuming:" + (toc - tic));
					tic = toc;
					times++;

				}
			}
			System.out.println("Debuge! Testing accuracy performance(all):\t" + (double) acc / wordNum);
			acc2File("accall" + String.valueOf(accIdx) + ".txt", String.valueOf((double) acc / wordNum));
			System.out.println("Testing data llh(all):\t" + llh);
		}
		if(0==0) {
			acc = 0;
			wordNum = 0;
			llh = 0.0;
			count = 0;
			times = 0;

//			for (Sentence s : c.validStcsLessThanTen) {
			for(int i =0; i < c.validStcsLessThanTen.size(); i++) {
				Sentence s = c.validStcsLessThanTen.get(i);
				// Using the CYK parsing function in the library,
				// We need to prepare the parameters first!

				NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
				DepInstance depins = s.tran2DepIns();
				DepSentenceDist sd = new DepSentenceDist(depins, nonMap);
				//	sd.cacheModelFromOutside(root, child, decision);
				int len = s.getLength();
				sentenceChild = new double[len][len][this.childValency];
				sentenceRoot = new double[len];
				sentenceDecision = new double[len][2][this.decisionValency][2];

				this.cacheModel(s.wordIdxInDic, chCount, rootCount, decCount,
						sentenceChild, sentenceRoot, sentenceDecision);
				///
				if(ischeck){}
				else {
					if (this.iteration > 1) {
						for (int si = 0; si < len; si++) {
							if (this.DEBUGE1 == 1) {
								for (int sj = 0; sj < len; sj++) {
									for (int sk = 0; sk < this.childValency; sk++) {
										sentenceChild[si][sj][sk] = this.childPyStcsSpec_val[i][si][sj][sk];
									}
								}
							}
							if (this.DEBUGE3 == 1) {
								for (int sl = 0; sl < 2; sl++) {
									for (int sm = 0; sm < this.decisionValency; sm++) {
										for (int sn = 0; sn < 2; sn++) {
											sentenceDecision[si][sl][sm][sn] = this.decisionPyStcsSpec_val[i][si][sl][sm][sn];
										}
									}
								}
							}
						}
					}
				}
				////

				sd.cacheModelFromOutside(sentenceRoot, sentenceChild, sentenceDecision);
				int[] parser = new int[s.getLength()];
				double score = CKYParser.parseSentence(sd, parser, null, null, universalRule, regulationValue);
				llh += score;
				acc += s.accWords(parser);
				wordNum += s.getLength();
				allValidTrees.put(depins.postags, parser);

				count++;
				if ((count - times * 1000) > 0) {
					toc = System.currentTimeMillis();
					//System.out.println("Iteration 1000 times! with total:" + count + "time consuming:" + (toc - tic));
					tic = toc;
					times++;

				}
			}

			System.out.println("Debuge! Validation set accuracy performance(10):\t" + (double) acc / wordNum);

			System.out.println("Validation data llh(10):\t" + llh);
			acc2File("accValidation" + String.valueOf(accIdx) + ".txt", String.valueOf((double) acc / wordNum));
			acc2File("llhValidation" + String.valueOf(accIdx) + ".txt", String.valueOf(llh));
		}
		if(0==1){
			acc = 0;
			wordNum = 0;
			llh = 0.0;
			count = 0;
			times = 0;
			for(Sentence s : c.validStcs){
				// Using the CYK parsing function in the library,
				// We need to prepare the parameters first!


				NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
				DepInstance depins = s.tran2DepIns();
				DepSentenceDist sd = new DepSentenceDist(depins, nonMap);
				//	sd.cacheModelFromOutside(root, child, decision);
				int len = s.getLength();
				sentenceChild = new double[len][len][this.childValency];
				sentenceRoot = new double[len];
				sentenceDecision = new double[len][2][this.decisionValency][2];


				this.cacheModel(s.wordIdxInDic, chCount, rootCount, decCount,
						sentenceChild, sentenceRoot, sentenceDecision);


				sd.cacheModelFromOutside(sentenceRoot, sentenceChild, sentenceDecision);
				int[] parser = new int[s.getLength()];
				double score = CKYParser.parseSentence(sd, parser, null, null, universalRule, regulationValue);

				llh += score;
				acc += s.accWords(parser);
				wordNum += s.getLength();
				//Actually we do not use the tag information

				count ++;
				if((count - times * 1000) > 0){
					toc = System.currentTimeMillis();
					//System.out.println("Iteration 1000 times! with total:" + count + "time consuming:" + (toc - tic));
					tic = toc;
					times++;

				}
			}
			System.out.println("Debuge! Validation accuracy performance(all):\t" + ( double ) acc / wordNum);

			System.out.println("Validation data llh(all):\t" + llh);
			acc2File("accValidationall" + String.valueOf(accIdx) + ".txt", String.valueOf(( double ) acc / wordNum));
			acc2File("llhValidationall" + String.valueOf(accIdx) + ".txt", String.valueOf(llh));
		}



		double accur = (double) acc/wordNum;

	}


	public void ViterbiEStep(boolean isAlternate) throws Exception{
		double llh = 0.0;
		
		int count = 0;
		int times = 0;
		
		long tic, toc;
		tic = System.currentTimeMillis();
		int acc = 0, wordNum = 0;//accurate edge & word number
		


		double[][][] sentenceChild;
		double[] sentenceRoot;
		double[][][][] sentenceDecision;
		//if(EMiter != 1)
		//	addNoise2Model(childCountForComp, rootCountForComp, decisionCountForComp,
			//	childCountForCompModel, rootCountForCompModel, decisionCountForCompModel, 2);
		
		if((isAlternate) || iteration == 1)
			System.out.println("DMV");
		else
			System.out.println("ANN");
		System.out.println("Test Scale!");
		
		double[][][][] chCount = null, decCount = null;
		double[] rootCount = null;
		if(DEBUGE1 == 0 || isAlternate ||  iteration == 1)
			chCount = this.childCountForComp;
		else{
			//chCount = childCountForCompModel;
			chCount = this.childPy;
			//System.out.println("use child stuff from python!");
		}
		if(DEBUGE2 == 0 || isAlternate ||  iteration == 1)
			rootCount = this.rootCountForComp;
		else{
			//rootCount = rootCountForCompModel;
			rootCount = this.rootPy;
			//System.out.println("use root stuff from NN!");
		}
		if(DEBUGE3 == 0 || isAlternate ||  iteration == 1)
			decCount = this.decisionCountForComp;
		else{
			//decCount = decisionCountForCompModel;
			decCount = this.decisionPy;
			//System.out.println("use decision stuff from NN!");
		}
		if(this.iteration > 1){
			for(int i = this.onlineStsSplit[this.onlineIdx][0]; i < this.onlineStsSplit[this.onlineIdx][1]; i++){
				// Using the CYK parsing function in the library,
				// We need to prepare the parameters first!
				Sentence s = c.stcsWithLengthLessThanTen.get(i);
				
				NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
				DepInstance depins = s.tran2DepIns();
				DepSentenceDist sd = new DepSentenceDist(depins, nonMap);
			
				int len = s.getLength();
				sentenceChild = new double[len][len][this.childValency];
				sentenceRoot = new double[len];
				sentenceDecision = new double[len][2][this.decisionValency][2];
	
				this.cacheModel(s.wordIdxInDic, chCount, rootCount, decCount, 
						sentenceChild, sentenceRoot, sentenceDecision);			
				
				
				sd.cacheModelFromOutside(sentenceRoot, sentenceChild, sentenceDecision);
				int[] parser = new int[s.getLength()];
				double score = CKYParser.parseSentence(sd, parser, null, null, universalRule, regulationValue);//10.7
				Pair<int[], Integer> temp = new Pair<>(parser, i);
				this.allTrees.put(depins.postags, temp);
	
				llh += score;
				acc += s.accWords(parser);
				wordNum += s.getLength();
	
				count ++;
				if((count - times * 1000) > 0){
					toc = System.currentTimeMillis();
					//System.out.println("Iteration 1000 times! with total:" + count + "time consuming:" + (toc - tic));
					tic = toc;
					times++;
					
				}
			}
		}else{
			for(int i=0; i<c.stcsWithLengthLessThanTen.size(); i++){
				Sentence s = c.stcsWithLengthLessThanTen.get(i);
				NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
				DepInstance depins = s.tran2DepIns();
				DepSentenceDist sd = new DepSentenceDist(depins, nonMap);
			
				int len = s.getLength();
				sentenceChild = new double[len][len][this.childValency];
				sentenceRoot = new double[len];
				sentenceDecision = new double[len][2][this.decisionValency][2];
	
				this.cacheModel(s.wordIdxInDic, chCount, rootCount, decCount, 
						sentenceChild, sentenceRoot, sentenceDecision);			
				
				
				sd.cacheModelFromOutside(sentenceRoot, sentenceChild, sentenceDecision);
				int[] parser = new int[s.getLength()];
				double score = CKYParser.parseSentence(sd, parser, null, null, universalRule, regulationValue);//10.7
				Pair<int[], Integer> temp= new Pair<>(parser, i);
				allTrees.put(depins.postags, temp);
	
				llh += score;
				acc += s.accWords(parser);
				wordNum += s.getLength();
	
				count ++;
				if((count - times * 1000) > 0){
					toc = System.currentTimeMillis();
					//System.out.println("Iteration 1000 times! with total:" + count + "time consuming:" + (toc - tic));
					tic = toc;
					times++;
					
				}
			}
			this.onlineIdx--;
		}
		
		for(Sentence s : c.stcsWithLengthLessThanTen){
			NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
			DepInstance depins = s.tran2DepIns();
			DepSentenceDist sd = new DepSentenceDist(depins, nonMap);
		
			int len = s.getLength();
			sentenceChild = new double[len][len][this.childValency];
			sentenceRoot = new double[len];
			sentenceDecision = new double[len][2][this.decisionValency][2];

			this.cacheModel(s.wordIdxInDic, chCount, rootCount, decCount, 
					sentenceChild, sentenceRoot, sentenceDecision);			
			
			
			sd.cacheModelFromOutside(sentenceRoot, sentenceChild, sentenceDecision);
			int[] parser = new int[s.getLength()];
			double score = CKYParser.parseSentence(sd, parser, null, null, universalRule, regulationValue);//10.7
//			allTreesForCount.put(depins.postags, parser);
			allTreesForCount.add(new Pair(depins.postags, parser));

			llh += score;
			acc += s.accWords(parser);
			wordNum += s.getLength();

			count ++;
			if((count - times * 1000) > 0){
				toc = System.currentTimeMillis();
				//System.out.println("Iteration 1000 times! with total:" + count + "time consuming:" + (toc - tic));
				tic = toc;
				times++;
				
			}
		}
		
		if(this.onlineIdx < this.onlineNum - 1)
			this.onlineIdx ++;
		else{
			this.onlineIdx = 0;
			if(isShuffle)
				trainStsShuffle();
		}
		
		
		if(0 == 0){//this.onlineIdx == 0

			
			
			acc = 0;
			wordNum = 0;
			llh = 0.0;
			count = 0;
			times = 0;
	
			for(Sentence s : c.testStcsLessThanTen){
				// Using the CYK parsing function in the library,
				// We need to prepare the parameters first!
				
				NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
				DepInstance depins = s.tran2DepIns();
				DepSentenceDist sd = new DepSentenceDist(depins, nonMap);
			//	sd.cacheModelFromOutside(root, child, decision);
				int len = s.getLength();
				sentenceChild = new double[len][len][this.childValency];
				sentenceRoot = new double[len];
				sentenceDecision = new double[len][2][this.decisionValency][2];
	
				this.cacheModel(s.wordIdxInDic, chCount, rootCount, decCount, 
						sentenceChild, sentenceRoot, sentenceDecision);
				
				sd.cacheModelFromOutside(sentenceRoot, sentenceChild, sentenceDecision);
				int[] parser = new int[s.getLength()];
				double score = CKYParser.parseSentence(sd, parser, null, null, universalRule, regulationValue);
				//this.allTrees.put(depins.postags, parser);
				llh += score;
				acc += s.accWords(parser);
				wordNum += s.getLength();
				
				count ++;
				if((count - times * 1000) > 0){
					toc = System.currentTimeMillis();
					//System.out.println("Iteration 1000 times! with total:" + count + "time consuming:" + (toc - tic));
					tic = toc;
					times++;
					
				}
			}
			
			System.out.println("Debuge! Testing accuracy performance(10):\t" + ( double ) acc / wordNum);
			acc2File("acc" + String.valueOf(accIdx) + ".txt", String.valueOf(( double ) acc / wordNum));
			System.out.println("Testing data llh(10):\t" + llh);	
			
			acc = 0;
			wordNum = 0;
			llh = 0.0;
			count = 0;
			times = 0;
			for(Sentence s : c.testStcs){
				// Using the CYK parsing function in the library,
				// We need to prepare the parameters first!
				
				
				NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
				DepInstance depins = s.tran2DepIns();
				DepSentenceDist sd = new DepSentenceDist(depins, nonMap);
			//	sd.cacheModelFromOutside(root, child, decision);
				int len = s.getLength();
				sentenceChild = new double[len][len][this.childValency];
				sentenceRoot = new double[len];
				sentenceDecision = new double[len][2][this.decisionValency][2];
	
				
				this.cacheModel(s.wordIdxInDic, chCount, rootCount, decCount, 
						sentenceChild, sentenceRoot, sentenceDecision);			
				
				
				sd.cacheModelFromOutside(sentenceRoot, sentenceChild, sentenceDecision);
				int[] parser = new int[s.getLength()];
				double score = CKYParser.parseSentence(sd, parser, null, null, universalRule, regulationValue);
				//allTrees.put(depins.postags, parser);
	
				llh += score;
				acc += s.accWords(parser);
				wordNum += s.getLength();
	
				count ++;
				if((count - times * 1000) > 0){
					toc = System.currentTimeMillis();
					//System.out.println("Iteration 1000 times! with total:" + count + "time consuming:" + (toc - tic));
					tic = toc;
					times++;
					
				}
			}
			System.out.println("Debuge! Testing accuracy performance(all):\t" + ( double ) acc / wordNum);
		        acc2File("accall" + String.valueOf(accIdx) + ".txt", String.valueOf(( double ) acc / wordNum));	
			System.out.println("Testing data llh(all):\t" + llh);
			
			acc = 0;
			wordNum = 0;
			llh = 0.0;
			count = 0;
			times = 0;
	
			for(Sentence s : c.validStcsLessThanTen){
				// Using the CYK parsing function in the library,
				// We need to prepare the parameters first!
				
				NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
				DepInstance depins = s.tran2DepIns();
				DepSentenceDist sd = new DepSentenceDist(depins, nonMap);
			//	sd.cacheModelFromOutside(root, child, decision);
				int len = s.getLength();
				sentenceChild = new double[len][len][this.childValency];
				sentenceRoot = new double[len];
				sentenceDecision = new double[len][2][this.decisionValency][2];
	
				this.cacheModel(s.wordIdxInDic, chCount, rootCount, decCount, 
						sentenceChild, sentenceRoot, sentenceDecision);
				
				sd.cacheModelFromOutside(sentenceRoot, sentenceChild, sentenceDecision);
				int[] parser = new int[s.getLength()];
				double score = CKYParser.parseSentence(sd, parser, null, null, universalRule, regulationValue);
				llh += score;
				acc += s.accWords(parser);
				wordNum += s.getLength();
				allValidTrees.put(depins.postags, parser);
				
				count ++;
				if((count - times * 1000) > 0){
					toc = System.currentTimeMillis();
					//System.out.println("Iteration 1000 times! with total:" + count + "time consuming:" + (toc - tic));
					tic = toc;
					times++;
					
				}
			}
			
			System.out.println("Debuge! Validation set accuracy performance(10):\t" + ( double ) acc / wordNum);
			
			System.out.println("Validation data llh(10):\t" + llh);	
		        acc2File("accValidation" + String.valueOf(accIdx) + ".txt", String.valueOf(( double ) acc / wordNum));
			acc2File("llhValidation" + String.valueOf(accIdx) + ".txt", String.valueOf(llh));	
			acc = 0;
			wordNum = 0;
			llh = 0.0;
			count = 0;
			times = 0;
			for(Sentence s : c.validStcs){
				// Using the CYK parsing function in the library,
				// We need to prepare the parameters first!
				
				
				NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
				DepInstance depins = s.tran2DepIns();
				DepSentenceDist sd = new DepSentenceDist(depins, nonMap);
			//	sd.cacheModelFromOutside(root, child, decision);
				int len = s.getLength();
				sentenceChild = new double[len][len][this.childValency];
				sentenceRoot = new double[len];
				sentenceDecision = new double[len][2][this.decisionValency][2];
	
				
				this.cacheModel(s.wordIdxInDic, chCount, rootCount, decCount, 
						sentenceChild, sentenceRoot, sentenceDecision);			
				
				
				sd.cacheModelFromOutside(sentenceRoot, sentenceChild, sentenceDecision);
				int[] parser = new int[s.getLength()];
				double score = CKYParser.parseSentence(sd, parser, null, null, universalRule, regulationValue);
	
				llh += score;
				acc += s.accWords(parser);
				wordNum += s.getLength();
	                              //Actually we do not use the tag information
				
				count ++;
				if((count - times * 1000) > 0){
					toc = System.currentTimeMillis();
					//System.out.println("Iteration 1000 times! with total:" + count + "time consuming:" + (toc - tic));
					tic = toc;
					times++;
					
				}
			}
			System.out.println("Debuge! Validation accuracy performance(all):\t" + ( double ) acc / wordNum);
			
			System.out.println("Validation data llh(all):\t" + llh);
                        acc2File("accValidationall" + String.valueOf(accIdx) + ".txt", String.valueOf(( double ) acc / wordNum));
                        acc2File("llhValidationall" + String.valueOf(accIdx) + ".txt", String.valueOf(llh));
		}


		
		double accur = (double) acc/wordNum;

	}
	
private void trainStsShuffle() {
		long seed = System.nanoTime();
		Collections.shuffle(c.stcsWithLengthLessThanTen, new Random(seed));
	}

	public void sentenceOnlineProcess(int onlineBatch, int onlineWinSize, double onlineEta, int accIdx){
		this.onlineBatch = onlineBatch;
		this.onlineWinSize = onlineWinSize;
		this.onlineEta = onlineEta;
		this.onlineIdx = 0;
		this.accIdx = accIdx;
		this.onlineNum = (int)Math.ceil(c.stcsWithLengthLessThanTen.size()/((double)this.onlineBatch));
		this.onlineStsSplit = new int[onlineNum][2];
		for(int i = 0; i < onlineNum; i ++){
			this.onlineStsSplit[i][0] = i * this.onlineBatch;
		    this.onlineStsSplit[i][1] = (i + 1) * this.onlineBatch > c.stcsWithLengthLessThanTen.size()? c.stcsWithLengthLessThanTen.size() : (i + 1) * this.onlineBatch;
		}
		for(int i = 0; i < this.onlineNum; i++){
			for(int j = 0; j < 2; j++)
			System.out.println("this.onlineStsSplit:\t" + i +"\t"+ j +"\t"+ this.onlineStsSplit[i][j]);
		}
		
		System.out.println("this.onlineBatch:\t" + this.onlineBatch);
		System.out.println("this.onlineEta:\t" + this.onlineEta);
	}
	public void acc2File(String fileName, String content) {
	        try {
	            FileWriter writer = new FileWriter(fileName, true);
	            writer.write(content);
	            writer.write("\n");
	            writer.close();
	        } catch (IOException e) {
	            e.printStackTrace();
	        }
	}
	
	public void addNoise2Model(double[][][][] child, double[] root, double[][][][] decision,
			double[][][][] childModel, double[] rootModel, double[][][][] decisionModel, double eps){
		double[][][][] childDiff = new double[child.length][child[0].length][child[0][0].length][child[0][0][0].length];
		double[] rootDiff = new double[root.length];
		double[][][][] decisionDiff = new double[decision.length][decision[0].length][decision[0][0].length][decision[0][0][0].length];
		
		for(int i = 0; i < child.length; i++)
			for(int j = 0; j < child[0].length; j++)
				for(int p = 0; p < child[0][0].length; p++)
					for(int q = 0; q < child[0][0][0].length; q++){
						childDiff[i][j][p][q] = Math.exp(child[i][j][p][q]) -  Math.exp(childModel[i][j][p][q]);
						child[i][j][p][q] = eps * childDiff[i][j][p][q] + Math.exp(childModel[i][j][p][q]);
					}
		
		for(int i = 0; i < rootDiff.length; i++){
			rootDiff[i] = Math.exp(root[i]) - Math.exp(rootModel[i]);
			root[i] = eps * rootDiff[i] + Math.exp(rootModel[i]);
		}
		
		for(int i = 0; i < decision.length; i++)
			for(int j = 0; j < decision[0].length; j++)
				for(int p = 0; p < decision[0][0].length; p++)
					for(int q = 0; q < decision[0][0][0].length; q++){
						decisionDiff[i][j][p][q] = Math.exp(decision[i][j][p][q]) - Math.exp(decisionModel[i][j][p][q]);
						decision[i][j][p][q] = eps * decisionDiff[i][j][p][q] + Math.exp(decisionModel[i][j][p][q]);
						
					}
		
		this.normPara(0);
		
		
	}
	 

	public double ViterbiMStepOnceOnline(HashMap<int[], int[]> allTrees, boolean normalizedGredient, double rateChoose) throws Exception{
       return 0;

	}
	
	public SimpleMatrix rowNorm(SimpleMatrix mat){
		int row = mat.numRows();
		int col = mat.numCols();
		double scale = 0.8;
		SimpleMatrix result = new SimpleMatrix(row, col);
		SimpleMatrix normMat = new SimpleMatrix(row, 1);
		for(int i = 0; i < row; i++){
			for(int j = 0; j < col; j++){
				normMat.set(i, 0, normMat.get(i, 0) + mat.get(i, j) * mat.get(i, j)); 
			}
		}
		
		for(int i = 0; i < row; i++){
			for(int j = 0; j < col; j++){
				result.set(i, j, (mat.get(i, j) * scale) / (Math.sqrt(normMat.get(i, 0))));
			}
		}

		
		return result;
	}
	
	/**
	 * Now we have a parse tree, we need to get information 
	 * 		from this parse tree, say, childCount, decisionCount,...
	 * modified from Kewei Tu's mycode.ViterbiEM
	 * 
	 * 
	 * @param postags
	 */
	public void countDependencies(int[] parse, int[] postags, double[][][][] child, 
			double[] root, double[][][][] decision, NonterminalMap nonMap) {
		int[] posTags = postags;
		int numWords = posTags.length;
		int[][] histogram = new int[numWords][2];

		// Cycle *left-to-right* through all words in the sentence
		int[] headValences = new int[numWords];
		for (int i = 0; i < numWords; i++) {
			int head = parse[i] - 1;

			// Update lambda_root
			if (head == -1)
				root[posTags[i]]++;
			else {
				if (i < head) { // i is to the left of head
					// Fill in left child counts and update histogram for use in
					// setting decision counts
					child[posTags[i]][posTags[head]][LEFT][headValences[head]]++;
					if (headValences[head] < nonMap.childValency - 1)
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
				child[posTags[i]][posTags[head]][RIGHT][headValences[head]]++;
				if (headValences[head] < nonMap.childValency - 1)
					headValences[head]++;
				histogram[head][RIGHT]++;
			}
		}

		processHistogram(child, root, decision, nonMap, histogram, postags);
	}

	/**
	 * adapted from MaxLikelihoodEstimator.processHistogram
	 */
	public static void processHistogram(double[][][][] child, double[] root, double[][][][] decision,
			NonterminalMap nonMap, int[][] histogram, int[] postags) {
		for (int i = 0; i < postags.length; i++) {
			int headPosNum = postags[i];
			for (int dir = 0; dir < 2; dir++) {
				int current = histogram[i][dir];

				int maxValence = Math.min(current,
						nonMap.decisionValency - 1);
				decision[headPosNum][dir][maxValence][END]++;
				if (current > 0) {
					for (int prevCont = 0; prevCont < maxValence; prevCont++)
						decision[headPosNum][dir][prevCont][CONT]++;
					decision[headPosNum][dir][maxValence][CONT] += (current - maxValence);
				}
			}
		}
	}
	
	private void countDependenciesPairSoft(DepSentenceDist oneStce, HashMap<Integer, Double> rootMap,
			HashMap<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Double> child,
			HashMap<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Double> decision, NonterminalMap nonMap) throws Exception {
		// TODO Auto-generated method stub
		int[] posTags = oneStce.depInst.postags;//??
		int numWords = posTags.length;
		//tootMap	
		for(int i = 0; i < numWords; i++){
			double stcroot = Math.exp(oneStce.getRoot1Posterior(i));
			if(!rootMap.containsKey(posTags[i]))
				rootMap.put(posTags[i], stcroot);
			else
				rootMap.put(posTags[i], rootMap.get(posTags[i]) + stcroot); 
		}
		//child
		for(int h = 0; h < numWords; h++){
			for(int a = 0; a < numWords; a++){
				//for(int dir = 0; dir < 2; dir++){
					for(int val = 0; val < this.childValency; val++){
						int dir;
						if(h < a){
							dir = 1;
						}else{
							if(h > a){
								dir = 0;
							}else{
								continue;
							}
						}
						Pair<Integer, Integer> pair1 = new Pair<>(posTags[h], posTags[a]);
						Pair<Integer, Integer> pair2 = new Pair<>(dir, val);
						Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> pair = new Pair<>(pair1, pair2);
						double stcchild = Math.exp(oneStce.getChild1Posterior(a, h, val));
			//			if(Double.isNaN(stcchild)){
			//				throw new Exception();
			//			}
						if(!child.containsKey(pair))
							child.put(pair, stcchild);
						else
							child.put(pair, child.get(pair) + stcchild);				
					}
				//}
			}
		}
		//decision
		for(int h = 0; h < numWords; h++){
			for(int dir = 0; dir < 2; dir++){
				for(int val = 0; val < this.decisionValency; val++){
					for(int stop = 0; stop < 2; stop++){
						Pair<Integer, Integer> pair1 = new Pair<>(posTags[h], dir);//??ok
						Pair<Integer, Integer> pair2 = new Pair<>(val, stop);
						Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> pair = new Pair<>(pair1, pair2);
						double stcdecision = Math.exp(oneStce.getDecision1Posterior(h, dir, val, stop));
						if(!decision.containsKey(pair))
							decision.put(pair, stcdecision);
						else
							decision.put(pair, decision.get(pair) + stcdecision);	
					}
				}
			}
		}

	}

	public void countDependenciesPair(int[] parse, int[] postags, HashMap<Integer, Integer> rootMap, 
			HashMap<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Integer> child,
			HashMap<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>,Integer> decision, NonterminalMap nonMap) {
		int[] posTags = postags;
		int numWords = posTags.length;
		int[][] histogram = new int[numWords][2];

		// Cycle *left-to-right* through all words in the sentence
		int[] headValences = new int[numWords];
		for (int i = 0; i < numWords; i++) {
			int head = parse[i] - 1;
			
			// Update lambda_root
			if (head == -1){
				//root[posTags[i]]++;
				if(!rootMap.containsKey(posTags[i]))
					rootMap.put(posTags[i], 1);
				else
					rootMap.put(posTags[i], rootMap.get(posTags[i]) + 1);
			}
			else {
				if (i < head) { // i is to the left of head
					// Fill in left child counts and update histogram for use in
					// setting decision counts
					Pair<Integer, Integer> pair1 = new Pair<>(posTags[head], posTags[i]);
					Pair<Integer, Integer> pair2 = new Pair<>(LEFT, headValences[head]);
					Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> pair = new Pair<>(pair1, pair2);
					if(!child.containsKey(pair))
						child.put(pair, 1);
					else
						child.put(pair, child.get(pair) + 1);
					
					
					if (headValences[head] < nonMap.childValency - 1)
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
				Pair<Integer, Integer> pair1 = new Pair<>(posTags[head], posTags[i]);
				Pair<Integer, Integer> pair2 = new Pair<>(RIGHT, headValences[head]);
				Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> pair = new Pair<>(pair1, pair2);
				if(!child.containsKey(pair))
					child.put(pair, 1);
				else
					child.put(pair, child.get(pair) + 1);			
				
				if (headValences[head] < nonMap.childValency - 1)
					headValences[head]++;
				histogram[head][RIGHT]++;
			}
		}

		processHistogramPair(decision, nonMap, histogram, postags);
	}
	public static void processHistogramPair(HashMap<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>,Integer> decision,
			NonterminalMap nonMap, int[][] histogram, int[] postags) {
		Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> test = new Pair<>(new Pair<>(5, 1), new Pair<>(2, 1));
		for (int i = 0; i < postags.length; i++) {
			int headPosNum = postags[i];
			for (int dir = 0; dir < 2; dir++) {
				int current = histogram[i][dir];

				int maxValence = Math.min(current,
						nonMap.decisionValency - 1);
				Pair<Integer, Integer> pair1 = new Pair<>(headPosNum, dir);
				Pair<Integer, Integer> pair2 = new Pair<>(maxValence, END);
				Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> pair = new Pair<>(pair1, pair2);
				

				if(!decision.containsKey(pair))
					decision.put(pair, 1);
				else
					decision.put(pair, decision.get(pair) + 1);
				
				if (current > 0) {
					for (int prevCont = 0; prevCont < maxValence; prevCont++){
						//decision[headPosNum][dir][prevCont][CONT]++;
						Pair<Integer, Integer> pair3 = new Pair<>(prevCont, CONT);
						Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> newpair = new Pair<>(pair1, pair3);

						if(!decision.containsKey(newpair))
							decision.put(newpair, 1);
						else
							decision.put(newpair, decision.get(newpair) + 1);
					}
					//decision[headPosNum][dir][maxValence][CONT] += (current - maxValence);
					
					Pair<Integer, Integer> pair4 = new Pair<>(maxValence, CONT);
					Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> newPair2 = new Pair<>(pair1, pair4);

					
					if(current - maxValence != 0){
						if(!decision.containsKey(newPair2))
							decision.put(newPair2, (current - maxValence));
						else
							decision.put(newPair2, decision.get(newPair2) + (current - maxValence));
					}
				}
			}
		}
	}

//	public void countDependenciesChdWithDesPair(int[] parse, int[] postags, HashMap<-, Integer> rootMap,
//			HashMap<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Integer> child,
//			HashMap<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>,Integer> decision,
//			HashMap<Pair<Pair<Integer, Pair<Integer, Integer>>, Pair<Integer, Integer>>,Integer> decisionWithChd,
//			NonterminalMap nonMap) {
//		int[] posTags = postags;
//		int numWords = posTags.length;
//		int[][] histogram = new int[numWords][2];
//		int[][][] histogramWithChd = new int[numWords][2][numWords];//Added
//		// Cycle *left-to-right* through all words in the sentence
//		int[] headValences = new int[numWords];
//		for (int i = 0; i < numWords; i++) {
//			int head = parse[i] - 1;
//
//			// Update lambda_root
//			if (head == -1){
//				//root[posTags[i]]++;
//				if(!rootMap.containsKey(posTags[i]))
//					rootMap.put(posTags[i], 1);
//				else
//					rootMap.put(posTags[i], rootMap.get(posTags[i]) + 1);
//			}
//			else {
//				if (i < head) { // i is to the left of head
//					// Fill in left child counts and update histogram for use in
//					// setting decision counts
//					Pair<Integer, Integer> pair1 = new Pair<>(head, i);
//					Pair<Integer, Integer> pair2 = new Pair<>(LEFT, headValences[head]);
//					Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> pair = new Pair<>(pair1, pair2);
//					if(!child.containsKey(pair))
//						child.put(pair, 1);
//					else
//						child.put(pair, child.get(pair) + 1);
//
//
//					if (headValences[head] < nonMap.childValency - 1)
//						headValences[head]++;
//					histogram[head][LEFT]++;
//					histogramWithChd[head][LEFT][i]++;//Added
//				}
//			}
//		}
//
//		// Cycle *right-to-left* through all words in the sentence
//		Arrays.fill(headValences, 0);
//		for (int i = numWords - 1; i >= 0; i--) {
//			int head = parse[i] - 1;
//			if (i > head && head != -1) {
//				// Fill in right child counts and update histogram for use in
//				// setting decision counts
//				Pair<Integer, Integer> pair1 = new Pair<>(head, i);
//				Pair<Integer, Integer> pair2 = new Pair<>(RIGHT, headValences[head]);
//				Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> pair = new Pair<>(pair1, pair2);
//				if(!child.containsKey(pair))
//					child.put(pair, 1);
//				else
//					child.put(pair, child.get(pair) + 1);
//
//				if (headValences[head] < nonMap.childValency - 1)
//					headValences[head]++;
//				histogram[head][RIGHT]++;
//				histogramWithChd[head][RIGHT][i]++;//Added
//			}
//		}
//
//		processHistogramChdWithDesPair(decision, decisionWithChd, child, nonMap, histogram, histogramWithChd, postags);
//	}

	/**
	 * adapted from MaxLikelihoodEstimator.processHistogram
	 */
	public static void processHistogramChdWithDesPair(HashMap<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>,Integer> decision,
			HashMap<Pair<Pair<Integer, Pair<Integer, Integer>>, Pair<Integer, Integer>>,Integer> decisionWithChd,
			HashMap<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>, Integer> child,
			NonterminalMap nonMap, int[][] histogram, int[][][] histogramWithChd, int[] postags) {
		Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> test = new Pair<>(new Pair<>(5, 1), new Pair<>(2, 1));
		for (int i = 0; i < postags.length; i++) {
			int headPosNum = postags[i];
			for (int dir = 0; dir < 2; dir++) {
				int current = histogram[i][dir];

				int maxValence = Math.min(current,
						nonMap.decisionValency - 1);
				Pair<Integer, Integer> pair1 = new Pair<>(/*headPosNum*/ i, dir);// save its index, not value.
				Pair<Integer, Integer> pair2 = new Pair<>(maxValence, END);
				Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> pair = new Pair<>(pair1, pair2);
				Pair<Pair<Integer, Pair<Integer, Integer>>, Pair<Integer, Integer>> newAddChdPair =
						new Pair<>(new Pair<>(0, pair1), pair2);// 0 denotes nothing. Do not care this part.

				if(!decision.containsKey(newAddChdPair))
					decisionWithChd.put(newAddChdPair, 1);
				else
					decisionWithChd.put(newAddChdPair, decisionWithChd.get(newAddChdPair) + 1);

				if(!decision.containsKey(pair))
					decision.put(pair, 1);
				else
					decision.put(pair, decision.get(pair) + 1);
				
				if (current > 0) {
				//	int leftChd = -1;// left child in dic
					int leftChdIndex = 0;// left child index
				//	int rightChd = -1;// right child in dic
					int rightChdIndex = postags.length - 1;// right child index
					for (int prevCont = 0; prevCont < maxValence; prevCont++){
						//decision[headPosNum][dir][prevCont][CONT]++;
						Pair<Integer, Integer> pair3 = new Pair<>(prevCont, CONT);
						Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> newpair = new Pair<>(pair1, pair3);

						if(dir == 0){
							//leftChdIndex = i + leftChdIndex;
							while(histogramWithChd[i][dir][leftChdIndex] == 0 ){
								leftChdIndex += 1;
							}
//							Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> chd = 
//									new Pair<>(new Pair<>(i, leftChdIndex), new Pair<>(dir, prevCont));							
							
							histogramWithChd[i][dir][leftChdIndex] = 0;//After find the child, delete it from the histogram.
							//leftChd = postags[leftChdIndex];
							Pair<Pair<Integer, Pair<Integer, Integer>>, Pair<Integer, Integer>> newAddLeftChdPair =
									new Pair<>(new Pair<>(leftChdIndex, pair1), pair3);// save its index, not value.
							if(!decisionWithChd.containsKey(newAddLeftChdPair))
								decisionWithChd.put(newAddLeftChdPair, 1);
							else
								decisionWithChd.put(newAddLeftChdPair, decisionWithChd.get(newAddLeftChdPair) + 1);
						}
						else{
							//rightChdIndex = i + rightChdIndex;
							while(histogramWithChd[i][dir][rightChdIndex] == 0 && rightChdIndex > i){
								rightChdIndex -= 1;
							}
//							Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> chd = 
//									new Pair<>(new Pair<>(i, rightChdIndex), new Pair<>(dir, prevCont));
							
							histogramWithChd[i][dir][rightChdIndex] = 0;//After find the child, delete it from the histogram.
							//rightChd = postags[rightChdIndex];
							Pair<Pair<Integer, Pair<Integer, Integer>>, Pair<Integer, Integer>> newAddRightChdPair =
									new Pair<>(new Pair<>(rightChdIndex, pair1), pair3);// save its index, not value.
							
							if(!decisionWithChd.containsKey(newAddRightChdPair))
								decisionWithChd.put(newAddRightChdPair, 1);
							else
								decisionWithChd.put(newAddRightChdPair, decisionWithChd.get(newAddRightChdPair) + 1);
						}
						
						
						
						if(!decision.containsKey(newpair))
							decision.put(newpair, 1);
						else
							decision.put(newpair, decision.get(newpair) + 1);
					}
					//decision[headPosNum][dir][maxValence][CONT] += (current - maxValence);
					
					Pair<Integer, Integer> pair4 = new Pair<>(maxValence, CONT);
					Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> newPair2 = new Pair<>(pair1, pair4);

					int diff = current - maxValence;
					if(current - maxValence != 0){
						if(dir == 0){
							
							while(diff != 0){
								
								while(histogramWithChd[i][dir][leftChdIndex] == 0 && leftChdIndex < i){
									leftChdIndex++;
								}
								histogramWithChd[i][dir][leftChdIndex] = 0;//set it to zero
							//	leftChd = postags[leftChdIndex];
								Pair<Pair<Integer, Pair<Integer, Integer>>, Pair<Integer, Integer>> newAddLeftChdPair =
										new Pair<>(new Pair<>(leftChdIndex, pair1), pair4);// index, not value
								if(!decisionWithChd.containsKey(newAddLeftChdPair))
									decisionWithChd.put(newAddLeftChdPair, 1);
								else
									decisionWithChd.put(newAddLeftChdPair, decisionWithChd.get(newAddLeftChdPair) + 1);
								diff--;
							}
						}
						else{
							while(diff != 0){
								
								while(histogramWithChd[i][dir][rightChdIndex] == 0 && rightChdIndex > i){
									rightChdIndex--;
								}
								histogramWithChd[i][dir][rightChdIndex] = 0;//set it to zero
							//	rightChd = postags[rightChdIndex];
								Pair<Pair<Integer, Pair<Integer, Integer>>, Pair<Integer, Integer>> newAddRightChdPair =
										new Pair<>(new Pair<>(rightChdIndex, pair1), pair4);//Index, not value.
								if(!decisionWithChd.containsKey(newAddRightChdPair))
									decisionWithChd.put(newAddRightChdPair, 1);
								else
									decisionWithChd.put(newAddRightChdPair, decisionWithChd.get(newAddRightChdPair) + 1);


								diff--;
							}
							
						}
						
						
						if(!decision.containsKey(newPair2))
							decision.put(newPair2, (current - maxValence));
						else
							decision.put(newPair2, decision.get(newPair2) + (current - maxValence));
					}
				}
			}
		}
	}
	public SimpleMatrix getUnusualVec(int dim){
		SimpleMatrix vec = new SimpleMatrix(dim, 1);
		 Random generator = new Random();
		for(int i = 0; i < dim; i++){
		   
		    double g = 0;
		    while(true){
		    	g = generator.nextGaussian();
		    	if(Math.abs(g) < 1)
		    		break;
		    }
		    //double g = (generator.nextGaussian()) * 0.1;//0.3
		    //g = (Math.random() - 0.5) * 1;//0.01
			vec.set(i, 0, g * 0.1);
		}
		
		return vec;
	}


	public void updateValencyVectorElementInHashmap(ArrayList<SimpleMatrix> updatedWord2Vec, 
			HashMap<Integer, SimpleMatrix> changedHm, int label, boolean normalizedGredient, double rateChoose, int pairNum){
		for(Map.Entry<Integer, SimpleMatrix> element : changedHm.entrySet()){
			int name = element.getKey();
			SimpleMatrix mat = element.getValue();
			
			
			try{
				//double rate = this.inverseNormOfPara(mat, label, normalizedGredient, rateChoose);
				if(updatedWord2Vec.size() <= name){
					updatedWord2Vec.add(mat.scale(0.001));
					System.out.println(updatedWord2Vec.size());
				}else
					updatedWord2Vec.set(name, updatedWord2Vec.get(name).plus(mat.scale(/*0.1,1*/this.valencyRate*rateChoose*(1.0/pairNum))));                                                   //valency
			}
			catch(NullPointerException e){
				System.out.println("eeeee");
				throw new NullPointerException("Null pointer!");
			}
		}
		//
		//assert(updatedWord2Vec.size() <= 3) : "set of array is bigger than 3:" + updatedWord2Vec.size();
	}	
	public void updateVectorElementInHashmap(HashMap<String, SimpleMatrix> updatedWord2Vec,
			HashMap<String, SimpleMatrix> changedHm, int label, boolean normalizedGredient, double rateChoose, int pairNum) throws Exception{
		for(Map.Entry<String, SimpleMatrix> element : changedHm.entrySet()){
			String name = element.getKey();
			SimpleMatrix mat = element.getValue();
			
			if(Double.isNaN(mat.get(0, 0))){
				System.out.println("NAN in updateVectorElementInHashmap");
				//return;
					throw new Exception();
			}
			
			double norm = mat.normF();
			updatedWord2Vec.put(name, updatedWord2Vec.get(name).plus(mat.scale(/*0.1*/this.wordRate*rateChoose * (1.0 / pairNum))));

		}
		
	}
	
	public void changeElementInHashmap(HashMap<String, SimpleMatrix> storedHm, HashMap<String, SimpleMatrix> changedHm, double num){
		
		for(Map.Entry<String, SimpleMatrix> grad : changedHm.entrySet()){
			String name = grad.getKey();
			SimpleMatrix mat = grad.getValue().scale(num);
			if(!storedHm.containsKey(name))
				storedHm.put(name, mat);
			else
				storedHm.put(name, mat.plus(storedHm.get(name)));
		}
		
	}
	

	public void normPara(double smooth){
		double[][][] childCountForCompDen = new double[dicSize][2][this.childValency];// no use
		double rootCountForCompDen = 0.0;
		double[][][] decisionCountForCompDen = new double[dicSize][2][decisionValency];
		
		
		for(int c = 0; c < dicSize; c++){
			for(int p = 0; p < dicSize; p++){
				for(int d = 0; d < 2; d++){
					for(int v = 0; v < this.childValency; v++){

						childCountForCompDen[p][d][v] += childCountForComp[c][p][d][v];//

					}
				}			
			}
		}
		
		for(int r = 0; r < dicSize;r++)
			rootCountForCompDen += rootCountForComp[r];
		
		
		for(int p = 0; p < dicSize; p++){
			
			for(int d = 0; d < 2; d++){
				for(int v = 0; v < decisionValency; v++){
					for(int s = 0; s < 2; s++){
						decisionCountForCompDen[p][d][v] += decisionCountForComp[p][d][v][s];
					}
				}
				
			}
		}
		// Normalization!!!!
		for(int c = 0; c < dicSize; c++){
			for(int p = 0; p < dicSize; p++){
				for(int d = 0; d < 2; d++){
					for(int v = 0; v < this.childValency; v++){
						if(childCountForComp[c][p][d][v] > 0)
							childCountForComp[c][p][d][v] = Math.log((childCountForComp[c][p][d][v] + smooth) 
									/ (childCountForCompDen[p][d][v] + smooth * this.dicSize));
						else if(smooth < 1e-30)
							childCountForComp[c][p][d][v] = Double.NEGATIVE_INFINITY;
						else
							childCountForComp[c][p][d][v] = Math.log(smooth / (childCountForCompDen[p][d][v] + smooth * this.dicSize));
						
					}
				}			
			}
		}
		
		for(int r = 0; r < dicSize;r++)
			if(rootCountForComp[r] > 0)
				rootCountForComp[r] = Math.log((rootCountForComp[r] + smooth) / (rootCountForCompDen + smooth * this.dicSize));
			else if(smooth < 1e-30)
				rootCountForComp[r] = Double.NEGATIVE_INFINITY;
			else
				rootCountForComp[r] = Math.log(smooth / (rootCountForCompDen + smooth * this.dicSize));
				
		for(int p = 0; p < dicSize; p++){
			
			for(int d = 0; d < 2; d++){
				for(int v = 0; v < decisionValency; v++){
					for(int s = 0; s < 2; s++){
						if(decisionCountForComp[p][d][v][s] > 0)
							decisionCountForComp[p][d][v][s] = Math.log((decisionCountForComp[p][d][v][s] + smooth) 
									/ (decisionCountForCompDen[p][d][v] + smooth * 2)) ;
						else if(smooth < 1e-30)
							decisionCountForComp[p][d][v][s] = Double.NEGATIVE_INFINITY;
						else
							decisionCountForComp[p][d][v][s] = Math.log(smooth / (decisionCountForCompDen[p][d][v] + smooth * this.dicSize));
					}
				}
				
			}
		}
		
	}
	
	public void kmInit(int pascalIdx) throws UnsupportedEncodingException, FileNotFoundException, IOException{
		
		childCountForComp = new double[dicSize][dicSize][2][this.childValency];// no use
		rootCountForComp = new double[dicSize];
		decisionCountForComp = new double[dicSize][2][decisionValency][2];
		String corpusParams = "";
//		if(pascalIdx==0)
//			corpusParams = "data/corpus-params.txt";
//		else
		corpusParams = "data/pascal-corpus-params/corpus-params-"+String.valueOf(pascalIdx)+".txt";
		DepCorpus corpus = new DepCorpus(corpusParams, 1, 10, Integer.MAX_VALUE);
		DepProbMatrix model = new DepProbMatrix(corpus,this.decisionValency, this.childValency);
		KleinManningInitializer.initNoah(model);
		
		int len = corpus.tagAlphabet.size();
		for(int c = 0; c < len; c++){
			String chd = null;
			
			try{
				chd = corpus.tagAlphabet.index2feat.get(c);
			}catch(IndexOutOfBoundsException e){
				throw new IndexOutOfBoundsException();
			}
			int chdIndexInDic = Integer.valueOf(chd);//this.c.dic.get(chd);//here we use string index directly
		//	this.c.dic.remove(chd);
			for(int p = 0; p < len; p++){
				String par = null;
				par = corpus.tagAlphabet.index2feat.get(p);
				
				int parIndexInDic; 
				try{
					parIndexInDic =	Integer.valueOf(par);//this.c.dic.get(par);
				}catch(NullPointerException e){
					throw new NullPointerException();
				}
				
				for(int d = 0; d < 2; d++){
					for(int v = 0; v < this.childValency; v++){
						childCountForComp[chdIndexInDic][parIndexInDic][d][v] = model.child[c][p][d][v];
						
					}
				}
			}
		}
		
		for(int c = 0; c < len/*this.dicSize*/; c++){
			String chd = corpus.tagAlphabet.index2feat.get(c);
			int chdIndexInDic = Integer.valueOf(chd);//this.c.dic.get(chd);
			rootCountForComp[chdIndexInDic] = model.root[c]; 
		}
		
		for(int c = 0; c < len /*this.dicSize*/; c++){
			String chd = corpus.tagAlphabet.index2feat.get(c);
			int chdIndexInDic = Integer.valueOf(chd);//this.c.dic.get(chd);
			for(int d = 0; d < 2; d++){
				for(int v = 0; v < this.decisionValency; v++){
					for(int con = 0; con < 2; con++){
						decisionCountForComp[chdIndexInDic][d][v][con] = model.decision[c][d][v][con]; 
					}
				}
			}
		}
	}
	
	/**
	 *  Gold tree initialization
	 */
	public void goldInit(){
		childCountForComp = new double[dicSize][dicSize][2][this.childValency];// no use
		rootCountForComp = new double[dicSize];
		decisionCountForComp = new double[dicSize][2][decisionValency][2];

		
		HashMap<int[], int[]> allGoldTrees = new HashMap<>();
		for(Sentence s : c.stcsWithLengthLessThanTen){
			DepInstance depins = s.tran2DepIns();
			int[] gold = new int[s.goldTree.size()];
			for(int i = 0; i < gold.length; i++){
				gold[i] = s.goldTree.get(i);
			}
			allGoldTrees.put(depins.postags,  gold);
		}
	
		for(Map.Entry<int[], int[]> entry : /*this.allTrees*/allGoldTrees.entrySet()){
			int[] w = entry.getKey();
			int[] p = entry.getValue();
				
			NonterminalMap nonMapPri = new NonterminalMap(this.decisionValency, this.childValency);
			this.countDependencies(p, w, childCountForComp, rootCountForComp, decisionCountForComp, nonMapPri);
		}
		
	}
	public void DMV_Mstep(){
		boolean mstepAllTree = true;
		double mstepEta = 0.1;
		double[][][][] oldChildCountForComp = new double[dicSize][dicSize][2][this.childValency];// no use
		double[] oldRootCountForComp = new double[dicSize];
		double[][][][] oldDecisionCountForComp = new double[dicSize][2][decisionValency][2];
		if(!mstepAllTree){
			oldChildCountForComp  = childCountForComp.clone();
			oldRootCountForComp = rootCountForComp.clone();
			oldDecisionCountForComp = decisionCountForComp.clone();
		}
		
		childCountForComp = new double[dicSize][dicSize][2][this.childValency];// no use
		rootCountForComp = new double[dicSize];
		decisionCountForComp = new double[dicSize][2][decisionValency][2];
		
//		for(Map.Entry<int[], int[]> entry : this.allTreesForCount.entrySet()){//original it is allTrees. 10.27
//			int[] w = entry.getKey();
//			int[] p = entry.getValue();
		for(int i = 0; i < this.allTreesForCount.size(); i ++){//original it is allTrees. 10.27
			int[] w = this.allTreesForCount.get(i).a;
			int[] p = this.allTreesForCount.get(i).b;
			NonterminalMap nonMapPri = new NonterminalMap(this.decisionValency, this.childValency);
			this.countDependencies(p, w, childCountForComp, rootCountForComp, decisionCountForComp, nonMapPri);
		}
		
		if (!this.childCountForCompList.isEmpty())
			this.childCountForCompList.remove(0);
		this.childCountForCompList.add(this.childCountForComp);
		if (!this.decisionCountForCompList.isEmpty())
			this.decisionCountForCompList.remove(0);
		this.decisionCountForCompList.add(this.decisionCountForComp);
		if (!this.rootCountForCompList.isEmpty())
			this.rootCountForCompList.remove(0);
		this.rootCountForCompList.add(this.rootCountForComp);
		
		this.childCountForComp = new double[dicSize][dicSize][2][this.childValency];// inittilation is 0
		rootCountForComp = new double[dicSize];
		decisionCountForComp = new double[dicSize][2][decisionValency][2];
		double eta = 1;
		for(int num = 0; num < this.childCountForCompList.size(); num++){
			if(num == this.childCountForCompList.size() - 1){
				eta = 1;
				for(int i1 = 0; i1 < this.childCountForCompList.size() - 1; i1++){
					eta = eta - Math.pow(this.onlineEta,(i1 + 1));
				}
			}
			else
				eta = Math.pow(this.onlineEta,(this.childCountForCompList.size() - num -1));
			
			for(int i = 0; i < dicSize; i++){
				for(int j = 0; j < dicSize; j++){
					for(int k = 0; k < 2; k ++){
						for(int l = 0; l < this.childValency;l ++){
//								int p = -1;
//								int c = sd.depInst.words[i];
//								Pair<Integer, Integer> rule = new Pair<Integer, Integer>(Integer.valueOf(p) ,Integer.valueOf(c));
//								if(universalRule.containsValue(rule))
								this.childCountForComp[i][j][k][l] = this.childCountForCompList.get(num)[i][j][k][l] * eta; 
						}
					}
				}
			}
			for(int i = 0; i < dicSize; i++){
				for(int j = 0; j < 2; j++){
					for(int k = 0; k < decisionValency; k ++){
						for(int l = 0; l < 2;l ++){
							this.decisionCountForComp[i][j][k][l] = this.decisionCountForCompList.get(num)[i][j][k][l] * eta; 
						}
					}
				}
			}
			for(int i = 0; i < dicSize; i ++){
				this.rootCountForComp[i] = this.rootCountForCompList.get(num)[i] * eta;
			}
			DMVaddPrior();
		}
		
		if(!mstepAllTree){
			DMVaddmstepEta(mstepEta, oldChildCountForComp, oldRootCountForComp, oldDecisionCountForComp);
		}
		
	}
	private void DMVaddmstepEta(double mstepEta, double[][][][] oldChildCountForComp, double[] oldRootCountForComp,
			double[][][][] oldDecisionCountForComp) {
		for(int i = 0; i < dicSize; i++){
			for(int j = 0; j < dicSize; j++){
				for(int k = 0; k < 2; k ++){
					for(int l = 0; l < this.childValency;l ++){
							childCountForComp[i][j][k][l] = childCountForComp[i][j][k][l] * mstepEta + oldChildCountForComp[i][j][k][l] *(1-mstepEta); 
					}
				}
			}
		}
		for(int p = 0; p < dicSize; p++){
			for(int d = 0; d < 2; d++){
				for(int v = 0; v < this.decisionValency; v++){
					for(int s = 0; s < 2; s++){
						decisionCountForComp[p][d][v][s] = decisionCountForComp[p][d][v][s] * mstepEta + oldDecisionCountForComp[p][d][v][s] *(1-mstepEta);
					}
				}
			}
		}
		for(int i = 0; i < dicSize; i ++){
				this.rootCountForComp[i] = rootCountForComp[i]  * mstepEta + oldRootCountForComp[i] *(1-mstepEta);
		}
	}
	public void DMVaddPrior(){
		for(int i = 0; i < dicSize; i++){
			for(int j = 0; j < dicSize; j++){
				for(int k = 0; k < 2; k ++){
					for(int l = 0; l < this.childValency;l ++){
							int p = j;
							int c = i;
							Pair<Integer, Integer> rule = new Pair<Integer, Integer>(Integer.valueOf(p) ,Integer.valueOf(c));
							if(universalRule.containsValue(rule))
								this.childCountForComp[i][j][k][l] += prior -1; 
					}
				}
			}
		}
		for(int i = 0; i < dicSize; i ++){//par root is -1
			int p = -1;
			int c = i;
			Pair<Integer, Integer> rule = new Pair<Integer, Integer>(Integer.valueOf(p) ,Integer.valueOf(c));
			if(universalRule.containsValue(rule))
				this.rootCountForComp[i] += prior - 1;
		}
	}
	public void NeuDMVaddPrior(){
		for(int i = 0; i < dicSize; i++){
			for(int j = 0; j < dicSize; j++){
				for(int k = 0; k < 2; k ++){
					for(int l = 0; l < this.childValency;l ++){
							int p = j;
							int c = i;
							Pair<Integer, Integer> rule = new Pair<Integer, Integer>(Integer.valueOf(p) ,Integer.valueOf(c));
							if(universalRule.containsValue(rule))
								this.childCountForComp1[i][j][k][l] += prior -1; 
					}
				}
			}
		}
		for(int i = 0; i < dicSize; i ++){//par root is -1
			int p = -1;
			int c = i;
			Pair<Integer, Integer> rule = new Pair<Integer, Integer>(Integer.valueOf(p) ,Integer.valueOf(c));
			if(universalRule.containsValue(rule))
				this.rootCountForComp1[i] += prior - 1;
		}
	}
	public void randomInit(/*double[][][][] modelChild,
			double[] modelRoot, double[][][][] modelDecision*/){
		childCountForComp = new double[dicSize][dicSize][2][this.childValency];// no use
		rootCountForComp = new double[dicSize];
		decisionCountForComp = new double[dicSize][2][decisionValency][2];
		for (int c = 0; c < this.dicSize; c++) {
			this.rootCountForComp[c] = Math.random();
			for (int p = 0; p < this.dicSize; p++) {
				for(int dir = 0; dir < 2; dir++){
					for (int v = 0; v < this.childValency; v++) {
						//modelChild[c][p][dir][v] = Math.random();
						this.childCountForComp[c][p][dir][v] = Math.random();
					}
				}
			}
			for (int dir = 0; dir < 2; dir++) {
				for (int v = 0; v < this.decisionValency; v++) {
					for (int choice = 0; choice < 2; choice++) {
						//modelDecision[c][dir][v][choice] = Math.random();
						this.decisionCountForComp[c][dir][v][choice] = Math.random();
					}
				}
				
			}
		}
	}
	
	public void uniformInit(){
		childCountForComp = new double[dicSize][dicSize][2][this.childValency];// no use
		rootCountForComp = new double[dicSize];
		decisionCountForComp = new double[dicSize][2][decisionValency][2];
		for (int c = 0; c < this.dicSize; c++) {
			this.rootCountForComp[c] = 1;
			for (int p = 0; p < this.dicSize; p++) {
				for(int dir = 0; dir < 2; dir++){
					for (int v = 0; v < this.childValency; v++) {
						this.childCountForComp[c][p][dir][v] = 1;
					}
				}
			}
			for (int dir = 0; dir < 2; dir++) {
				for (int v = 0; v < this.decisionValency; v++) {
					for (int choice = 0; choice < 2; choice++) {
						this.decisionCountForComp[c][dir][v][choice] = 1;
					}
				}
				
			}
		}
	}
	
	
	public void reInit2Para(){
		assert(this.weightInit != null) : "Init weight is null!";
		this.weight = new SimpleMatrix(this.weightInit);
		this.decisionWeight = new SimpleMatrix(this.decisionWeightInit);
		this.rootWeight = new SimpleMatrix(this.rootWeightInit);
		

		
		this.leftDirVec = new SimpleMatrix(this.leftDirVecInit);
		this.rightDirVec = new SimpleMatrix(this.rightDirVecInit);
		this.stopVec = new SimpleMatrix(this.stopVecInit);
		this.continueVec = new SimpleMatrix(this.continueVecInit);
		
		this.valencyVecs = new ArrayList<>();
		for(SimpleMatrix mat : this.valencyVecsInit){
			this.valencyVecs.add(mat);
		}
	}
	
	public void rePara2Init(){
		this.weightInit = new SimpleMatrix(this.weight);
		this.decisionWeightInit = new SimpleMatrix(this.decisionWeight);
		this.rootWeightInit = new SimpleMatrix(this.rootWeight);
		this.leftDirVecInit = new SimpleMatrix(this.leftDirVec);
		this.rightDirVecInit = new SimpleMatrix(this.rightDirVec);
		this.stopVecInit = new SimpleMatrix(this.stopVec);
		this.continueVecInit = new SimpleMatrix(this.continueVec);
		
		this.valencyVecsInit = new ArrayList<>();
		for(SimpleMatrix mat : this.valencyVecs){
			this.valencyVecsInit.add(mat);
		}
	}	
	public void cacheModel(ArrayList<Integer> words, double[][][][] modelChild, double[] modelRoot, double[][][][] modelDecision,
			double[][][] sentenceChild, double[] sentenceRoot, double[][][][] sentenceDesicion) {// s.wordIdxInDic
		// modelChild    c p d v // sentenceChild c p v
		// modelDecision p d v s // sentenceDesicion p d v s


		for (int c = 0; c < words.size(); c++) {
			int ctag = words.get(c);
			sentenceRoot[c] = modelRoot[ctag];
			if(modelChild!=null) {
				for (int p = 0; p < words.size(); p++) {
					if (c == p) continue;
					int ptag = words.get(p);
					int dir = (c < p ? LEFT : RIGHT);
					for (int v = 0; v < this.childValency; v++) {
						sentenceChild[c][p][v] = modelChild[ctag][ptag][dir][v];
					}
				}
			}
			if(modelDecision!=null) {
				for (int dir = 0; dir < 2; dir++) {
					for (int v = 0; v < this.decisionValency; v++) {
						for (int choice = 0; choice < 2; choice++) {
							sentenceDesicion[c][dir][v][choice] = modelDecision[ctag][dir][v][choice];
						}
					}

				}
			}
		}
	}
	


	public void afterAnnealingInit(){
		childCountForComp = new double[dicSize][dicSize][2][this.childValency];// no use
		rootCountForComp = new double[dicSize];
		decisionCountForComp = new double[dicSize][2][decisionValency][2];

		
		HashMap<int[], int[]> allAfterAnnealingTrees = new HashMap<>();
		for(Sentence s : c.stcs_afterAnnealing){
			DepInstance depins = s.tran2DepIns();
			int[] gold = new int[s.goldTree.size()];
			for(int i = 0; i < gold.length; i++){
				gold[i] = s.goldTree.get(i);
			}
			allAfterAnnealingTrees.put(depins.postags,  gold);
		}
	
		for(Map.Entry<int[], int[]> entry : /*this.allTrees*/allAfterAnnealingTrees.entrySet()){
			int[] w = entry.getKey();
			int[] p = entry.getValue();
				
			NonterminalMap nonMapPri = new NonterminalMap(this.decisionValency, this.childValency);
			this.countDependencies(p, w, childCountForComp, rootCountForComp, decisionCountForComp, nonMapPri);
		}
		
	}

	
    public void SoftEStep(double Anneling_sigma) throws Exception{
    	//evaluate Step and //update model
    	this.iteration++;
		System.out.println("=============Now iteration(Soft):\t" + this.iteration );
		System.out.println("Anneling_sigma:\t" + Anneling_sigma );
    	SoftEStepValuateAndUpdateModel(false);
   
    	// count-MStep
		((DepModel) model).exponentiateParameters(1 / (1 - Anneling_sigma));//use 1 it is no change
		
		corpusEStep(counts, sentenceDists, stats);//>0    ?????
		//ChildBackoff??
		System.out.println("ChildBackoff:\t" + ChildBackoff );
		if(this.ChildBackoff > 0){
			counts.addChildBackoff(this.ChildBackoff, this.dicSize);
		}
		
		counts.logNormalize();
		
		//updateModel
    	//
    }
	private void SoftEStepValuateAndUpdateModel(boolean isAlternate) throws Exception {//counts is 0  ???????
		// TODO Auto-generated method stub
		double llh = 0.0;
		
		int count = 0;
		int times = 0;
		
		long tic, toc;
		tic = System.currentTimeMillis();
		int acc = 0, wordNum = 0;//accurate edge & word number
		
		//updateModel(counts.child, counts.root, counts.decision);
		//value process
		double[][][][] chCount = null, decCount = null;
		double[] rootCount = null;
		if(DEBUGE1 == 0 || isAlternate ||  iteration == 1 || this.lastIsViterbi)
			chCount = counts.child;
		else{
			//chCount = childCountForCompModel;
			chCount = this.childPy;//Here is childPy means that there should be at least one time for viterbi em step, only initStep then annealingStep is denied. 
			System.out.println("use child stuff from python!");
		}
		if(DEBUGE2 == 0 || isAlternate ||  iteration == 1 || this.lastIsViterbi)
			rootCount = counts.root;
		else{
			//rootCount = rootCountForCompModel;
			rootCount = this.rootPy;
			System.out.println("use root stuff from NN!");
		}
		if(DEBUGE3 == 0 || isAlternate ||  iteration == 1 || this.lastIsViterbi)
			decCount = counts.decision;
		else{
			//decCount = decisionCountForCompModel;
			decCount = this.decisionPy;
			System.out.println("use decision stuff from NN!");
		}
		//updateModel process
		updateModel(chCount, rootCount, decCount);//      may be childOff!!!!!?????
		
		double[][][] sentenceChild;//init is -infit??   sentenceChild[0][0][0][0] = 0 /??????
		double[] sentenceRoot;
		double[][][][] sentenceDecision;
		for(Sentence s : c.stcsWithLengthLessThanTen){
			NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
			DepInstance depins = s.tran2DepIns();
			DepSentenceDist sd = new DepSentenceDist(depins, nonMap);
		//	sd.cacheModelFromOutside(root, child, decision);
			int len = s.getLength();
			sentenceChild = new double[len][len][this.childValency];
			sentenceRoot = new double[len];
			sentenceDecision = new double[len][2][this.decisionValency][2];
            
			
			this.cacheModel(s.wordIdxInDic, chCount, rootCount, decCount, sentenceChild, sentenceRoot, sentenceDecision);


			sd.cacheModelFromOutside(sentenceRoot, sentenceChild, sentenceDecision);
			int[] parser = new int[s.getLength()];
			double score = CKYParser.parseSentence(sd, parser, null, null, universalRule, regulationValue);
			//allTrees.put(depins.postags, parser);//do not need to update allTrees
	/*		assert(score >= 0): "sentence:\t" + Arrays.toString(s.goldTree.toArray()) 
				+ "CYK result:\t" + Arrays.toString(parser) + "score:\t" + score;
			if(score <= 1e-20)
				score = 1e-20;*/
			llh += score;
			acc += s.accWords(parser);
			wordNum += s.getLength();
			//System.out.println("Now accuracy:\t" + (double) acc / wordNum);
		//	System.out.println("length:" + parser.length + " elements:" + Arrays.toString(parser));
		//	System.out.println("gold parser:\t" + Arrays.toString(s.goldTree.toArray()));
		//	countDependencies(parser, depins.postags, childCount, rootCount, decisionCount, nonMap);                                    //Actually we do not use the tag information
			
			count ++;
			if((count - times * 1000) > 0){
				toc = System.currentTimeMillis();
				System.out.println("Iteration 1000 times! with total:" + count + "time consuming:" + (toc - tic));
				tic = toc;
				times++;
				
			}
			
		}
		System.out.println("Debuge! Training accuracy performance:\t" + ( double ) acc / wordNum);		
		//System.out.println("Training data llh:\t" + llh);
		acc = 0;
		wordNum = 0;
		llh = 0.0;
		count = 0;
		times = 0;
		for(Sentence s : c.testStcsLessThanTen){
			// Using the CYK parsing function in the library,
			// We need to prepare the parameters first!
			
			NonterminalMap nonMap = new NonterminalMap(this.decisionValency, this.childValency);
			DepInstance depins = s.tran2DepIns();
			DepSentenceDist sd = new DepSentenceDist(depins, nonMap);
		//	sd.cacheModelFromOutside(root, child, decision);
			int len = s.getLength();
			sentenceChild = new double[len][len][this.childValency];
			sentenceRoot = new double[len];
			sentenceDecision = new double[len][2][this.decisionValency][2];
//			if(DEBUGE1 == 1 && DEBUGE2 == 1 && DEBUGE3 == 1){
//				this.cacheModel(s.wordIdxInDic, /*childCountForComp*/ childCountForCompModel, rootCountForCompModel, decisionCountForCompModel, sentenceChild, sentenceRoot, sentenceDecision);
//			}
			this.cacheModel(s.wordIdxInDic, ((DepModel)model).params.child, ((DepModel)model).params.root, ((DepModel)model).params.decision, sentenceChild, sentenceRoot, sentenceDecision);
			sd.cacheModelFromOutside(sentenceRoot, sentenceChild, sentenceDecision);
			int[] parser = new int[s.getLength()];
			double score = CKYParser.parseSentence(sd, parser, null, null, universalRule, regulationValue);
			llh += score;
			if(score > 0){
				System.out.println("Score bigger than 0");
			}
			acc += s.accWords(parser);
			wordNum += s.getLength();
			
			count ++;
			if((count - times * 1000) > 0){
				toc = System.currentTimeMillis();
				System.out.println("Iteration 1000 times! with total:" + count + "time consuming:" + (toc - tic));
				tic = toc;
				times++;
				
			}
		}
		
		System.out.println("Debuge! Testing accuracy performance:\t" + ( double ) acc / wordNum);
		
		System.out.println("Testing data llh:\t" + llh);	
	}
	
	
	public void setLastIsViterbiTrue(){
		this.lastIsViterbi = true;
		//template
		this.allTrees.clear();
	}
	public void setLastIsViterbiFalse(){
		this.lastIsViterbi = false;
	}
	public boolean returnLastIsViterbi(){
		return this.lastIsViterbi;
	}
	
	public List<Integer> dic2Tag(){
		List<Integer> dicIdx2tagIdx = new ArrayList<>();
		for(int i = 0; i < this.c.dic.size(); i++){
			dicIdx2tagIdx.add(0);
		}
		
		int dicIdx = 0;
		int tagIdx = 0;
		for(Map.Entry<Pair<String,String>, Integer> entry : this.c.dic.entrySet()){
			dicIdx = entry.getValue();
			tagIdx = this.c.tag2idx.get(entry.getKey().b);
			dicIdx2tagIdx.set(dicIdx, Integer.valueOf(tagIdx));
		}
		return dicIdx2tagIdx;
	}
	public List<String> dic2WordStr(){
		List<String> dicIdx2WordStr = new ArrayList<>();
		for(int i = 0; i < this.c.dic.size(); i++){
			dicIdx2WordStr.add("");
		
		}
		int dicIdx = 0;
		String wordStr = "";
		for(Map.Entry<Pair<String,String>, Integer> entry : this.c.dic.entrySet()){
			dicIdx = entry.getValue();
			wordStr = entry.getKey().a;
			dicIdx2WordStr.set(dicIdx, wordStr);
		}
		return dicIdx2WordStr;
	}

	public Integer get_nb_classes(){
		return this.c.dic.size();
	}
	public Integer get_nb_classes_tag(){
		return this.c.tag2idx.size();
	}
	/**
	 *  Save weight matrix to file, so it can be easy for debugging. 
	 * @param sm
	 * @param fileName
	 * @throws IOException
	 */
	public void toFile(SimpleMatrix sm, String fileName) throws IOException{
		BufferedWriter outputBw = new BufferedWriter(new FileWriter(fileName));
		outputBw.write(sm.toString());
		outputBw.close();
	}
	public void setRegulationValue(double regulationValue1) {
		this.regulationValue = regulationValue1;
		
	}
	public void setIsEarlyStop(boolean b) {
		this.isEarlyStopping = b;
	}

	void saveSentenceAndParsing(int accIdx, ArrayList<Pair<int[], int[]>> allTreesForCount, ArrayList<Sentence> stcsWithLengthLessThanTen) throws IOException {
//		String corpusParams = "data/pascal-corpus-params/corpus-params-"+String.valueOf(pascalIdx)+".txt";
		String corpusParams = "data/lang"+String.valueOf(pascalIdx)+"_tagCorrect";

		String file_init = System.getProperty("user.dir") + "/temp/" + "train_parsing" + String.valueOf(accIdx) +  ".txt";
//		BufferedWriter file_init_w = new BufferedWriter(new FileWriter(file_init));
//		for (int i = 0; i < stcsWithLengthLessThanTen.size(); i++) { // allTreesForCount is used for one kind of online em: for mstep of ROOT
//			Sentence s = stcsWithLengthLessThanTen.get(i);
//			int len = s.getLength();
//
//		}
//		file_init_w.close();
		int indexWord, indexPOS, indexParent, sts=0;
		indexWord = 1;
		indexPOS = 4;
		indexParent = 6;
		boolean parsed = true;

		Pattern whitespace = Pattern.compile("\\s+");
		Scanner rf = new Scanner(new BufferedReader(new FileReader(corpusParams)));  // inf
		FileWriter wf = new FileWriter(file_init);  // outf


		while (rf.hasNext()) {
			// read one sentence
			ArrayList<String> words = new ArrayList<String>();
			ArrayList<String> tokens = new ArrayList<String>();
			ArrayList<Integer> parents = new ArrayList<Integer>();
			ArrayList<String[]> lines = new ArrayList<String[]>();
			String ln = rf.nextLine();
			int[] new_p = allTreesForCount.get(sts).b;
			int wordidx_in_stc = 0;
			while (ln.trim().length() != 0) {//one sentence
				String[] info = whitespace.split(ln);
				words.add(info[indexWord]);
				tokens.add(info[indexPOS]);
				if (parsed) {
					try {
						parents.add(Integer.parseInt(info[indexParent]));
					} catch(Exception o) {
						System.out.print(info[indexParent]);
					}
				}

				info[indexParent] = String.valueOf(new_p[wordidx_in_stc]);
				lines.add(info);
				if (rf.hasNext())
					ln = rf.nextLine();
				else
					break;
				wordidx_in_stc++;
			}

			// process one sts
//			for (int i = 0; i < words.size();) {
//				if (false) { //isPunctuation(tokens.get(i))
//					System.out.println("Punctuation!!!(should not exit): " + tokens.get(i));
//					if (parsed)
//						if (parents.contains(i + 1) || parents.get(i) == -1) {
//							System.out.println("May need manual correction: "
//									+ tokens.get(i) + "\n" + words + "\n"
//									+ tokens + "\n" + parents + "\n");
//						}
//
//					words.remove(i);
//					tokens.remove(i);
//					lines.remove(i);
//					if (parsed) {
//						int ip = parents.remove(i);
//						if (ip > i + 1)
//							ip--;
//						for (int j = 0; j < parents.size(); j++) {
//							int p = parents.get(j);
//							if (p > i + 1)
//								parents.set(j, p - 1);
//							else if (p == i + 1)
//								parents.set(j, ip);
//							lines.get(j)[indexParent] = Integer
//									.toString(parents.get(j));
//						}
//					}
//				} else
//					i++;
//			}
//

			Sentence s = stcsWithLengthLessThanTen.get(sts);
			if (words.size() != s.getLength()) {
				System.out.println("Error!!. Sentence not equal here.");
				continue;
			}


			// write
			for (String[] line : lines) {
				wf.write(line[0]);
				for (int i = 1; i < line.length; i++) {
					wf.write("\t" + line[i]);
				}
				wf.write("\n");
			}
			wf.write("\n");
			sts += 1;


		}
		rf.close();
		wf.close();
	}

	public void saveCountForComp() throws IOException {
		String tempfilechd = System.getProperty("user.dir") + "/temp/" + "ChdCountForComp" + String.valueOf(accIdx) +  ".txt";
		String tempfiledec = System.getProperty("user.dir") + "/temp/" + "DecCountForComp" + String.valueOf(accIdx) +  ".txt";
		String tempfileroot = System.getProperty("user.dir") + "/temp/" + "RootCountForComp" + String.valueOf(accIdx) +  ".txt";
		
		BufferedWriter outputBwchd = new BufferedWriter(new FileWriter(tempfilechd));
		for(int c = 0; c < dicSize; c++){
		  for(int p = 0; p < dicSize; p++){
			for(int d = 0; d < 2; d++){
				for(int v = 0; v < this.childValency; v++){
					    outputBwchd.write(String.valueOf(this.childCountForComp[c][p][d][v])+"\t");
				    }
			    }			
		    }
	    }
		outputBwchd.close();
		
		BufferedWriter outputBwdec = new BufferedWriter(new FileWriter(tempfiledec));
		for(int p = 0; p < dicSize; p++){
			for(int d = 0; d < 2; d++){
				for(int v = 0; v < decisionValency; v++){
					for(int s = 0; s < 2; s++){
						outputBwdec.write(String.valueOf(this.decisionCountForComp[p][d][v][s])+"\t");
					    }
				    }
				
			 }
	    }
		outputBwdec.close();
		
		BufferedWriter outputBwroot = new BufferedWriter(new FileWriter(tempfileroot));
		for(int r = 0; r < dicSize;r++)
			outputBwroot.write(String.valueOf(this.rootCountForComp[r])+"\t");
		outputBwroot.close();
		

        }

}


