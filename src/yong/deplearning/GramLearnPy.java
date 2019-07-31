package yong.deplearning;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.regex.Pattern;

import org.ejml.simple.SimpleMatrix;






//import py4j.g;
import py4j.GatewayServer;
import depparsing.data.DepInstance;
import depparsing.decoding.CKYParser;
import depparsing.model.DepModel;
import depparsing.model.DepSentenceDist;
import depparsing.model.NonterminalMap;
import depparsing.util.Pair;
import depparsing.model.DepModel.UpdateType;
import depparsing.model.DepProbMatrix;
import dic.Corpus;
import dic.Sentence;
import yong.depmodel.*;


public class GramLearnPy {
	private static AnnealingModelSoftPy model;
	private static int wordf;
	private static int onlineBatch;
	private static double onlineEta;
	private static int onlineWinSize;
	private static int accIdx;
	private static int pre_acc_idx;
	private static double universalValue;
	private static double  regulationValue;
	private static int corpusIdx;
	private static int initType;
	private static int stsLimitNum;
	private static int pascalIdx;
	
	private static long newTime;
	private static long oldTime;
	
	public static int corpusType;
	public static final int COMPACT = 0; // <word,POS,parent>
	public static final int PASCAL = 1;
	public static final int CONLL = 2;
	public static boolean parsed;
	
	public static String recurrentData;

	private static double prior;

        private static GatewayServer server;

	public static void main(String[] args) throws Exception{
	
	GramLearnPy app = new GramLearnPy();
	int port = Integer.parseInt(args[0]);//23331  200 5779 1 0 4 5779
	wordf = Integer.parseInt(args[1]);//10000000;
	onlineBatch = Integer.parseInt(args[2]);//5000;
	onlineWinSize = 1;//Integer.parseInt(args[3]);//3;//no use, real win size is decided by this.onlineNum = (int)Math.ceil(c.stcsWithLengthLessThanTen.size()/((double)this.onlineBatch));
	onlineEta = 0;//Double.valueOf((args[4]));//0;
	accIdx = Integer.valueOf((args[3]));
	corpusIdx = Integer.valueOf((args[4]));
	universalValue = 0;
	regulationValue = 0;//Double.valueOf((args[7]));//1e-2
	initType = Integer.valueOf((args[6]));//1 is km, 2 is good init.4 is random init. 5 is uniform init
	pre_acc_idx = Integer.valueOf((args[7]));
	stsLimitNum = Integer.valueOf((args[5]));
	pascalIdx = Integer.valueOf((args[8]));
	prior = 1;
	System.out.println("prior: "+ prior);
	recurrentData = "";//args[4];
	//director optimize likelihood 
	// app is now the gateway.entry_point
	server = new GatewayServer(app, port);
	server.start();
	}

	public void serverShutdown(){
    	    server.shutdown();
        }
	
	public static void paraSetting() throws Exception {
		newTime = 0;
		oldTime = 0;

		// TODO Auto-generated method stub
		final int decisionValency = 2;
		final int childValency = 2;
		
		final int negSetSize = 2;
		final int wordDim = 5;
		final int valencyDim = 5;
		final int tagDim = 5;
		final int dirDim = 5;
		
		double lambda = 0;
		double lambda_l1 = 0;
		double wRate = 0.04;
		double dirRate = 0.02;
		double wordRate = 0.02;
		double valencyRate = 0.02;
		//int wordf = 200;//10000000
		
		double backoff = 0;
		
		double sigma0 = 1;
		double sigmak = -0.003;
		double sigmae = 0;
		System.out.println("RELU activation function in use.");
		System.out.println("Add Regularization:\t" + lambda);
		System.out.println("weight rate:\t" + wRate);
		System.out.println("direction rate:\t" + dirRate);
		System.out.println("word rate:\t" + wordRate);
		System.out.println("valency rate:\t" + valencyRate);
		System.out.println("word frequency:\t" + wordf);
		System.out.println("sigma0:\t" + sigma0);
		System.out.println("sigmak:\t" + sigmak);
		System.out.println("sigmae:\t" + sigmae);
		Corpus c = new Corpus(wordDim, tagDim, valencyDim, dirDim,0);//here is no business
		//About init, if use good init, just change AnnealingModelSoftPy.py AnnealingEM--funcyion, just delete comments of the following two lines. You can also make some changes in Estep(if(iteration == 5)
		//About train sts number, it is controlled by (155line) stsLimitNum --- int stsLimitNum = onlineBatch, so just change onlineBatch if not onlineEM. 
		String testData = "";
		String validData = "";
		String[] langs = {"arabic", "basque", "czech", "danish", "dutch", "portuguese", "slovene", "swedish", "ctb9.0"};
//		String[] f_langs = {"arabic", "basque", "czech", "danish", "portuguese", "slovene", "swedish"};  // course: "dutch", "slovene"
		if(pascalIdx==0) {
			testData = "wsj-inf_23_dep_all";
			validData = "wsj-inf_22_dep";
			System.out.println("using english of wsj data");
		}else{
//			testData = "pascal-pos/"+langs[pascalIdx-1]+"/test-all_conll";//"wsj-inf_23_dep";
//			validData = "pascal-pos/"+langs[pascalIdx-1]+"/dev_conll";//"wsj-inf_22_dep";
			testData = "pascal-pos/"+langs[pascalIdx-1]+"/test_conll";//"wsj-inf_23_dep";
			validData = "pascal-pos/"+langs[pascalIdx-1]+"/dev_conll";//"wsj-inf_22_dep";
			System.out.println("using pascal data: "+langs[pascalIdx-1]);
		}
		String anotherdata = "BillpAll2_noCorrectSts_" + String.valueOf(corpusIdx);//"BillpAll2_noCorrectSts_" + String.valueOf(corpusIdx);
		boolean isRecurrent = false;
		String data = "";
		String dataAfterAnnealing = "";
		if(isRecurrent){
			data = dataAfterAnnealing = recurrentData;
		}else{
			if(pascalIdx==0){
				data = "wsj-inf_2-21_dep_filter_10";
				dataAfterAnnealing = "wsj-inf_2-21_dep_filter_10_init";//"annealingInstanceswsj_23_0.01";////annealing/annealing_viterbi_annealingInstanceswsj_23_0.01_10
			}else {
//				data = "pascal-pos/"+langs[pascalIdx-1]+"/train_conll";  //FOR DRBUG HANWJ
//				dataAfterAnnealing = "pascal-pos/"+langs[pascalIdx-1]+"/init_conll";  //FOR DRBUG HANWJ
				data = "pascal-pos/"+langs[pascalIdx-1]+"/train_conll";  //FOR DRBUG HANWJ
				dataAfterAnnealing = "pascal-pos/"+langs[pascalIdx-1]+"/train_hdpdep_init_conll";  //FOR DRBUG HANWJ
			}
		}
		
		c.countFreqFromFile("data/" + data);//dicNUm
		c.countFreqFromFile("data/" + testData);
		c.countFreqFromFile("data/" + validData);
		c.genDic2Tag();//dic2tag not need actually
		c.genTag2Idx();//tag2idx
		c.deleteUNKforDic(wordf);//dic
		//dic process before here.
		//c.getWord2VecFromFile("glove.50.txt");//glove.50.txt
		c.readFromFile("data/" + data);
		int stcsNum = c.stcs.size();
		int stcsAllNum = 0;
		if(anotherdata != "")
			c.readAnotherFromFile("data/" + anotherdata);
		stcsAllNum = c.stcs.size();
		//when we use different corpus, such as bllip, annealing init should be changed!! --hwj
//		if(pascalIdx==0){
		    c.readAfterAnnealingFromFile("data/" + dataAfterAnnealing);//
//		}
		c.readValidDataFromFile("data/" + validData);
		c.readTestDataFromFile("data/" + testData);
		c.saveAllWord2File("dic" + String.valueOf(accIdx) + ".txt");
		c.index2Dic();//idx2dic
		
		//int stsLimitNum = stsLimitNum;//5779;//?? -9.18
		c.reduceSentencesNum(10, stsLimitNum);
		//c.reduceSentences(10);//stcsWithLengthLessThanTen is what to train, it may not be length 10! or all the sts <= 10
		int stcsWithLengthLessThanTenNum = c.stcsWithLengthLessThanTen.size();
		System.out.println("c.stcsWithLengthLessThanTen.size():\t" + c.stcsWithLengthLessThanTen.size());
		c.reduceTestSentences(10);
		System.out.println("c.testStcsLessThanTen.size():\t" + c.testStcsLessThanTen.size());
		c.reduceValidSentences(10);
		System.out.println("c.validStcsLessThanTen.size():\t" + c.validStcsLessThanTen.size());

		boolean isAllKminit = true;//we use good init
		String File = "";
		if(isAllKminit){
			if(anotherdata != ""){
				File = "data/" + "forkminit";
				String f1 = "data/" + data;
				String f2 = "data/" + anotherdata;//"data/BillpAll2_noCorrectSts_" + String.valueOf(corpusIdx);
				mergedFile(f1, f2, File);
			}else
			File = "data/" + data;//20000";//"wsj-inf_2-21_dep_filter_10";//dataj-inf_23_dep_defined
		}else{
			File = "data/" + data;
		}
		String newFile = "data/lang"+String.valueOf(pascalIdx)+"_tagCorrect";
		String dicFile = "dic.txt";//no use? so when we draw a picture and run multi thred, we do not need give a different name.  
		tagCorrectInFile(File, newFile,c);//, dicFile
		String tagSts2File = "data/forTag2Vec/lang"+String.valueOf(pascalIdx)+"_tagSts";
		prepareTagSts2File(File, tagSts2File,c);
		System.out.println("PrepareAbstractSts2File done.");
		System.out.println("PrepareTagSts2File done.");

		String abstractSts2File = "data/forWord2Vec/lang"+String.valueOf(pascalIdx)+"_abstractSts";
		prepareAbstractSts2File(File, abstractSts2File,c);

		String abstractSts2File_test = "data/forWord2Vec/lang"+String.valueOf(pascalIdx)+"_abstractSts_test";
		prepareAbstractSts2File("data/"+testData, abstractSts2File_test,c);

		String abstractSts2File_val = "data/forWord2Vec/lang"+String.valueOf(pascalIdx)+"_abstractSts_dev";
		prepareAbstractSts2File("data/"+validData, abstractSts2File_val,c);
        //--------for wordseq-----------
		String abstractSts2File1 = "data/forWord2Vec/lang"+String.valueOf(pascalIdx)+"_wordSts";
		prepareWordSts2File(File, abstractSts2File1,c);
		String abstractSts2File_test1 = "data/forWord2Vec/lang"+String.valueOf(pascalIdx)+"_wordSts_test";
		prepareWordSts2File("data/"+testData, abstractSts2File_test1,c);
		String abstractSts2File_val1 = "data/forWord2Vec/lang"+String.valueOf(pascalIdx)+"_wordSts_dev";
		prepareWordSts2File("data/"+validData, abstractSts2File_val1,c);

		SimpleMatrix weight = new SimpleMatrix(wordDim + tagDim, wordDim + tagDim + valencyDim + dirDim);
		SimpleMatrix weight2 = new SimpleMatrix(wordDim + tagDim, wordDim + tagDim);

		
		Random generator = new Random(2);
		double scale = 0.01;                                                                                                                      //scale = 1; initialization used.
		for(int i = 0; i < wordDim + tagDim; i++){

			for(int j = 0; j < wordDim + tagDim + valencyDim + dirDim; j++){
			//	Random r = new Random();
			//	double g = r.nextGaussian() + 1.0;
			//	System.out.println(g + " ");

			    double g = 0;
			    while(true){
			    	g = generator.nextGaussian();
			    	if(Math.abs(g) < 1)
			    		break;
			    }

				//TODO
				//double g = Math.random() / 1000;
				weight.set(i, j, g * 1);

			}
		}
		for(int i = 0; i < wordDim + tagDim; i++){

			for(int j = 0; j < wordDim + tagDim; j++){
			//	Random r = new Random();
			//	double g = r.nextGaussian() + 1.0;
			//	System.out.println(g + " ");

			    double g = 0;
			    while(true){
			    	g = generator.nextGaussian();
			    	if(Math.abs(g) < 1)
			    		break;
			    }

				//TODO
				//double g = Math.random() / 1000;
				weight2.set(i, j, g * 1);

			}
		}
		//weight = rowNorm(weight);   //debuge
		SimpleMatrix rootWeight = new SimpleMatrix(wordDim + tagDim, wordDim);
		for(int i = 0; i < wordDim + tagDim; i++){

			for(int j = 0; j < wordDim; j++){
			//	Random r = new Random();
			//	double g = r.nextGaussian() + 1.0;
			//	System.out.println(g + " ");
			   // Random generator = new Random(1);
				double g = 0;
			    while(true){
			    	g = generator.nextGaussian();
			    	if(Math.abs(g) < 1)
			    		break;
			    }
				//TODO
			//	double g = Math.random() / 1000;
				rootWeight.set(i, j, g * 1);

			}
		}



		SimpleMatrix decisionWeight = new SimpleMatrix(dirDim, wordDim + tagDim + valencyDim + dirDim);

		for(int i = 0; i < dirDim; i++){

			for(int j = 0; j < wordDim + tagDim + valencyDim + dirDim; j++){
			//	Random r = new Random();
			//	double g = r.nextGaussian() + 1.0;
			//	System.out.println(g + " ");
			//    Random generator = new Random(1);
				double g = 0;
			    while(true){
			    	g = generator.nextGaussian();
			    	if(Math.abs(g) < 1)
			    		break;
			    }
				//TODO
			//	double g = Math.random() / 1000;
				decisionWeight.set(i, j, g * 1);
			}
		}
		SimpleMatrix stopVec = new SimpleMatrix(dirDim, 1);         //Do not give a initialization of stopvec and contvec.
//		stopVec = getUnusualVec(dirDim);
		SimpleMatrix contVec = new SimpleMatrix(dirDim, 1);
//		contVec = getUnusualVec(dirDim);
	
	
	
	
		double lll = 0.0;
		double priorLll = 0.0;
		
		int debugIter = 1;
		
		for(int EMiter = 1; EMiter <= debugIter; EMiter++){
			//TODO Toy example testing
			//stopVec.set(0, 0, 5);
			//contVec.set(0, 0, 6);
			int rateChoose = 0;
			boolean normalizedGredient = false;//'false' is better.
			rateChoose = EMiter - 1;
			
		//	ModelSoft model = new ModelSoft(valancySize, negSetSize,  weight, rootWeight, decisionWeight, stopVec, contVec,
		//			wordDim, tagDim, valencyDim, dirDim, c, /*0.00001*/lambda);//negSetSize is not used
			DepProbMatrix counts = new DepProbMatrix(c.dic.size(), decisionValency, childValency);
			DepModel parmeter = new DepModel(counts, null, decisionValency, childValency, 0, null);
			model = new AnnealingModelSoftPy(decisionValency, childValency, negSetSize,  weight, weight2, rootWeight, decisionWeight, stopVec, contVec, wordDim, tagDim, 
					valencyDim, dirDim, c, /*0.00001*/lambda, lambda_l1, parmeter, sigma0, sigmak, sigmae, backoff, prior, pascalIdx);//null is model.negSetSize is not used
			
			
			model.rateSetting(wRate, dirRate, wordRate, valencyRate);
			//model.ViterbiEStep(); 
			model.AnnealingEM(normalizedGredient, rateChoose, EMiter, false, initType, pre_acc_idx);
		}
		model.sentenceOnlineProcess(onlineBatch , onlineWinSize, onlineEta, accIdx);
		model.setRegulationValue(regulationValue);
	}

	public static void saveCountForComp() throws IOException{
		System.out.println("saveCountForComp begin:");
		model.saveCountForComp();
		System.out.println("saveCountForComp end");
	}

	public static void EStep() throws Exception{
//		System.out.println("receive predicted rules");
		System.out.println("Enter function");
		newTime = System.currentTimeMillis();
		System.out.println("One iteration of EM Time:\t" + (newTime - oldTime));
		oldTime = newTime;
		//universalize
		if (universalValue > 1e-7)
			model.universalization(universalValue);
		
		
		model.ViterbiEStep();
		long eFinishTime = System.currentTimeMillis();
		System.out.println("E-Step time consuming:\t" + (eFinishTime - oldTime));
		System.out.println("Outside function");
	}

	public static void EStep(String train_chd, String train_dec, String val_chd, String val_dec, String test_chd, String test_dec, String test_all_chd, String test_all_dec) throws Exception{
		System.out.println("receive predicted rules");
		System.out.println("Enter function");
		newTime = System.currentTimeMillis();
		System.out.println("One iteration of EM Time:\t" + (newTime - oldTime));
		oldTime = newTime;
		//universalize
		if (universalValue > 1e-7)
			model.universalization(universalValue);
		model.ViterbiEStep(true, train_chd,  train_dec,  val_chd,  val_dec,  test_chd,  test_dec, test_all_chd, test_all_dec);
		long eFinishTime = System.currentTimeMillis();
		System.out.println("E-Step time consuming:\t" + (eFinishTime - oldTime));
		System.out.println("Outside function");
	}

	public static void SoftEStep(double Anneling_sigma) throws Exception{
		System.out.println("Enter function");
		model.SoftEStep(Anneling_sigma);
		System.out.println("Outside function");
	}
	public static void test() throws Exception{
		System.out.println(model.dirDim);
	}
	
	public static int getMaxVal(){
		return model.getMaxX();
	}
//	public static ArrayList<String> printTags(){
//		return model.tranTags();
//	}
	
	public static void setChdAndDesPy(ArrayList<ArrayList<Double>> chdPy, ArrayList<ArrayList<Double>> decisionPy){
		model.setChdAndDesPy(chdPy, decisionPy);
		
	}

//	public static void setChdPy(ArrayList<ArrayList<Double>> chdPy){
//		model.setChdPy(chdPy);
//	}
//	public static void setRootPy(ArrayList<ArrayList<Double>> rootPy){
//		model.setRootPy(rootPy);
//	}
//	public static void setDecisionPy(ArrayList<ArrayList<Double>> decisionPy){
//		model.setDecisionPy(decisionPy);
//	}
	
//	public static List<List<Integer>> MStep() {
//		model.DMV_Mstep();
//		model.normPara(1);
//		System.out.println("Sending data to python part!");
//		return model.ChdMStep();
//
//		//return chd;
//	}
//	public static List<Integer[][][][]> MStepCountTable() {
//		model.DMV_Mstep();
//		model.normPara(1);
//		System.out.println("Sending data to python part!");
//		return model.ChdMStepCountTable();
//
//		//return chd;
//	}
	/**
	 * child and decision txt output
	 * @return
	 * @throws IOException 
	 */
	public static void MStepTxt() throws IOException {
		model.DMV_Mstep();
		model.normPara(1);
		System.out.println("save data to in file for python!");
		System.out.println(System.getProperty("user.dir"));
		String tempfilechd = System.getProperty("user.dir") + "/temp/" + "chdTemp" + String.valueOf(accIdx) +  ".txt";
		String tempfiledec = System.getProperty("user.dir") + "/temp/" + "decTemp" + String.valueOf(accIdx) + ".txt";
		model.MStepTxt(tempfilechd, tempfiledec);
	}
	
//	public static List<List<List<Integer>>> chdAndDecisionMStep(){
//		model.DMV_Mstep();
//		model.normPara(1);
//		System.out.println("Sending data to python part!");
//		return model.ChdAndDesMStep();
//	}
	
	public static List<Integer> dic2Tag(){
		return model.dic2Tag();
	}
	public static List<String> dic2WordStr(){
		return model.dic2WordStr();
	}
	public static Integer nb_classes(){
		return model.get_nb_classes();
	}
	public static Integer nb_classes_tag(){
		return model.get_nb_classes_tag();
	}
//	public static List<List<Double>> chdAndDecisionSoftMStep() throws Exception{
//		return model.ChdAndDecSoftMStep();
//	}
	public static double getValidPerc(){
		return model.getPerc();
	}
	public static List<List<Double>> SoftMStep() throws Exception{
		return model.ChdSoftMStep();
	}
	public static List<List<Integer>> MStep_root(){
		return model.getRoot();
	}
	public static List<List<Integer>> MStep_decision(){
		return model.getDecision();
	}
	public static List<List<Double>> SoftMStep_root(){
		return model.getSoftRoot();
	}
	public static List<List<Double>> SoftMStep_decision(){
		return model.getSoftDecision();
	}
	public static List<Integer[][][][]> MStep_decisionCountTable(){
		return model.getDecisionCountTable();
	}
	public static Integer[] MStep_rootCountTable(){
		return model.getRootCountTable();
	}
	public Integer[][][][] getValidChdCountTable(){
		return model.getValidChdCountTable();
	}
	
	public Integer[][][][] getValidDecisionCountTable(){
		return model.getValidDecisionCountTable();
	}
	public void setLastIsViterbiTrue(){
		model.setLastIsViterbiTrue();//complete changed!!
	}
	public void setLastIsViterbiFalse(){
		model.setLastIsViterbiFalse();
	}
	public void setIsEarlyStop(Boolean a){
		model.setIsEarlyStop(a);
		System.out.println("IsEarlyStop:" + a);
	}
	
	public Integer returnLastIsViterbi(){
		if(model.returnLastIsViterbi())
		    return Integer.valueOf(1);
		else
			return Integer.valueOf(0);
	}
//	public void Viterbi2Annealing() throws Exception{
//		model.Viterbi2Annealing();
//	}
//	public void Init2Annealing() throws Exception{
//		model.Init2Annealing();
//	}
	public Double getOnlineEta(){
		return onlineEta;
	}

	public static boolean tagCorrectInFile(String File, String newFile, Corpus c) throws IOException{// String dicFile,
		BufferedReader inputReader = new BufferedReader(new FileReader(File));
		@SuppressWarnings("resource")
//		BufferedWriter bw = new BufferedWriter(new FileWriter(newFile));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(newFile), "UTF-8"));
		int count = 0;
		String line = inputReader.readLine();
		while(line!=null){
			ArrayList<String> lineList = new ArrayList<String>();
			while (line != null && !line.equals("") && !line.startsWith("*")) {
	//			lineList.add(line.split("\\s+"));  ArrayList<String[]> lineList = new ArrayList<String[]>();
				//process here!
				String[] elementInLineList = line.split("\\s+");
				String word = new String();
				String tag = new String();
				String head = new String();
				if(elementInLineList.length ==7){
					word = elementInLineList [1];
					tag = elementInLineList[4];
					head = elementInLineList[6];
				}
				//if(c.dic.containsKey(word.toLowerCase())){
					Integer dicIdx = 0;
					if(c.dic.containsKey(new Pair<String, String>(word.toLowerCase(), tag))){
						dicIdx = c.dic.get(new Pair<String, String>(word.toLowerCase(), tag));
					}else{
						assert(c.dic.containsKey(new Pair<String, String>(tag, tag)));
						dicIdx = c.dic.get(new Pair<String, String>(tag, tag));
					}
					tag = Integer.toString(dicIdx);//word.toLowerCase();
				//}
				String newline = "x" + "\t"
						+word+"\t"
						+"x"+ "\t"
						+"x"+ "\t"
						+tag+"\t"
						+"x"+ "\t"
						+head+"\t"; 
				//
				lineList.add(newline);
				line = inputReader.readLine();
			}
			int length = lineList.size();
			if (length == 0) {
				inputReader.close();//if it comes here, means it is wrong!
				return false;
			}
			line = inputReader.readLine();
			//write
			count++;
			for(int i = 0; i<length; i++){
				bw.write(lineList.get(i));
				bw.newLine();
			}
			bw.newLine();
		}

		inputReader.close();
		bw.close();
		System.out.println("# of length-nobigger-than-"+"  dependency instances: " + count);
		return true;
//      Process with sentence
//		while(line!=null){
//			ArrayList<String> lineList = new ArrayList<String>();
//			while (line != null && !line.equals("") && !line.startsWith("*")) {
//	//			lineList.add(line.split("\\s+"));
//				lineList.add(line);
//				line = inputReader.readLine();
//			}
//			int length = lineList.size();
//			if (length == 0) {
//				inputReader.close();//if it comes here, means it is wrong!
//				return false;
//			}
//			line = inputReader.readLine();
//			if(length > sizeLimit) continue;
//			//write
//			count++;
//			for(int i = 0; i<length; i++){
//				bw.write(lineList.get(i));
//				bw.newLine();
//			}
//			bw.newLine();
//		}
	}
	public static boolean prepareAbstractSts2File(String File, String stsFile, Corpus c) throws IOException{//newline += dic+"\t";//here is pair<> index
		BufferedReader inputReader = new BufferedReader(new FileReader(File));
		BufferedWriter bw = new BufferedWriter(new FileWriter(stsFile));
		int count = 0;
		String line = inputReader.readLine();
		while(line!=null){
			String newline = "";
			while (line != null && !line.equals("") && !line.startsWith("*")) {
				//process here!
				String[] elementInLineList = line.split("\\s+");
				String word = new String();
				String tag = new String();
				String head = new String();
				if(elementInLineList.length ==7){
					word = elementInLineList [1];
					tag = elementInLineList[4];
					head = elementInLineList[6];
				}
				//if(c.dic.containsKey(word.toLowerCase())){
					Integer dicIdx = 0;
					if(c.dic.containsKey(new Pair<String, String>(word.toLowerCase(), tag))){
						dicIdx = c.dic.get(new Pair<String, String>(word.toLowerCase(), tag));
					}else{
						assert(c.dic.containsKey(new Pair<String, String>(tag, tag)));
						dicIdx = c.dic.get(new Pair<String, String>(tag, tag));
					}
					String dic = Integer.toString(dicIdx);//word.toLowerCase();
				//}
				
				newline += dic+"\t";//here is pair<> index
				line = inputReader.readLine();
			}
			line = inputReader.readLine();
			//write
			count++;
			bw.write(newline);
			bw.newLine();
		}

		inputReader.close();
		bw.close();
		System.out.println("# of length-nobigger-than-"+"  dependency instances: " + count);
		return true;
	}
	public static boolean prepareDicSts2File(String File, String stsFile, Corpus c) throws IOException{//newline += dicWord+"\t";//here is dic word
		BufferedReader inputReader = new BufferedReader(new FileReader(File));
		BufferedWriter bw = new BufferedWriter(new FileWriter(stsFile));
		int count = 0;
		String line = inputReader.readLine();
		while(line!=null){
			String newline = "";
			while (line != null && !line.equals("") && !line.startsWith("*")) {
				//process here!
				String[] elementInLineList = line.split("\\s+");
				String word = new String();
				String tag = new String();
				String head = new String();
				if(elementInLineList.length ==7){
					word = elementInLineList [1];
					tag = elementInLineList[4];
					head = elementInLineList[6];
				}
				//if(c.dic.containsKey(word.toLowerCase())){
				    String dicWord = "";
					
					if(c.dic.containsKey(new Pair<String, String>(word.toLowerCase(), tag))){
						dicWord = word.toLowerCase();
					}else{
						assert(c.dic.containsKey(new Pair<String, String>(tag, tag)));
						dicWord = tag;
					}
					
				//}
				
				newline += dicWord+"\t";//here is dic word
				line = inputReader.readLine();
			}
			line = inputReader.readLine();
			//write
			count++;
			bw.write(newline);
			bw.newLine();
		}

		inputReader.close();
		bw.close();
		System.out.println("# of length-nobigger-than-"+"  dependency instances: " + count);
		return true;
	}
	public static boolean prepareWordSts2File(String File, String stsFile, Corpus c) throws IOException{//newline += word.toLowerCase()+"\t";//here is word
		BufferedReader inputReader = new BufferedReader(new FileReader(File));
		BufferedWriter bw = new BufferedWriter(new FileWriter(stsFile));
		int count = 0;
		String line = inputReader.readLine();
		while(line!=null){
			String newline = "";
			while (line != null && !line.equals("") && !line.startsWith("*")) {
				//process here!
				String[] elementInLineList = line.split("\\s+");
				String word = new String();
				String tag = new String();
				String head = new String();
				if(elementInLineList.length ==7){
					word = elementInLineList [1];
					tag = elementInLineList[4];
					head = elementInLineList[6];
				}
				
				newline += word.toLowerCase()+"\t";//here is word
				line = inputReader.readLine();
			}
			line = inputReader.readLine();
			//write
			count++;
			bw.write(newline);
			bw.newLine();
		}

		inputReader.close();
		bw.close();
		System.out.println("# of length-nobigger-than-"+"  dependency instances: " + count);
		return true;
	}
	public static boolean prepareTagSts2File(String File, String stsFile, Corpus c) throws IOException{//newline += Integer.toString(tagIdx)+"\t";//here is tag
		BufferedReader inputReader = new BufferedReader(new FileReader(File));
		BufferedWriter bw = new BufferedWriter(new FileWriter(stsFile));
		int count = 0;
		String line = inputReader.readLine();
		while(line!=null){
			String newline = "";
			while (line != null && !line.equals("") && !line.startsWith("*")) {
				//process here!
				String[] elementInLineList = line.split("\\s+");
				String word = new String();
				String tag = new String();
				String head = new String();
				if(elementInLineList.length ==7){
					word = elementInLineList [1];
					tag = elementInLineList[4];
					head = elementInLineList[6];
				}
				int tagIdx = 0;
				tagIdx = c.tag2idx.get(tag);
				newline += Integer.toString(tagIdx)+"\t";//here is tag
				line = inputReader.readLine();
			}
			line = inputReader.readLine();
			//write
			count++;
			bw.write(newline);
			bw.newLine();
		}

		inputReader.close();
		bw.close();
		System.out.println("# of length-nobigger-than-"+"  dependency instances: " + count);
		return true;
	}
	
	private static void mergedFile(String in1, String in2, String out) throws IOException {

		corpusType = CONLL;
		parsed = false;
		
		int sts = 0;
		int indexWord, indexPOS, indexParent;
		switch (corpusType) {
		default:
		case COMPACT:
			indexWord = 0;
			indexPOS = 1;
			indexParent = 2;
			break;
		case PASCAL:
			indexWord = 1;
			indexPOS = 5;
			indexParent = 7;
			break;
		case CONLL:
			indexWord = 1;
			indexPOS = 4;
			indexParent = 6;
			break;
		}
		Pattern whitespace = Pattern.compile("\\s+");
		Scanner sf1 = new Scanner(new BufferedReader(new FileReader(in1)));
		Scanner sf2 = new Scanner(new BufferedReader(new FileReader(in2)));
		FileWriter fw = new FileWriter(out);
		HashSet<String> POSs = new HashSet<String>();
//		HashSet<String> POSsAUX = new HashSet<String>();
//		HashSet<String> POSsAUXG = new HashSet<String>();
		while (sf1.hasNext()) {
			// read one sentence
			ArrayList<String> words = new ArrayList<String>();
			ArrayList<String> tokens = new ArrayList<String>();
			ArrayList<Integer> parents = new ArrayList<Integer>();
			ArrayList<String[]> lines = new ArrayList<String[]>();
			String ln = sf1.nextLine();
			while (ln.trim().length() != 0) {//one sentence
				String[] info = whitespace.split(ln);
				words.add(info[indexWord]);
				tokens.add(info[indexPOS]);
				if (parsed)
					parents.add(Integer.parseInt(info[indexParent]));
				lines.add(info);
				if (sf1.hasNext())
					ln = sf1.nextLine();
				else
					break;
			}

			// process
			POSs.addAll(tokens);
			// write
			for (String[] line : lines) {
				fw.write(line[0]);
				for (int i = 1; i < line.length; i++) {
					fw.write("\t" + line[i]);
				}
				fw.write("\n");
			}
			fw.write("\n");
			sts += 1;

		}
		
		while (sf2.hasNext()) {
			// read one sentence
			ArrayList<String> words = new ArrayList<String>();
			ArrayList<String> tokens = new ArrayList<String>();
			ArrayList<Integer> parents = new ArrayList<Integer>();
			ArrayList<String[]> lines = new ArrayList<String[]>();
			String ln = sf2.nextLine();
			while (ln.trim().length() != 0) {//one sentence
				String[] info = whitespace.split(ln);
				words.add(info[indexWord]);
				tokens.add(info[indexPOS]);
				if (parsed)
					parents.add(Integer.parseInt(info[indexParent]));
				lines.add(info);
				if (sf2.hasNext())
					ln = sf2.nextLine();
				else
					break;
			}

			// process
			POSs.addAll(tokens);
			// write
			for (String[] line : lines) {
				fw.write(line[0]);
				for (int i = 1; i < line.length; i++) {
					fw.write("\t" + line[i]);
				}
				fw.write("\n");
			}
			fw.write("\n");
			sts += 1;

		}
		sf1.close();
		sf2.close();
		fw.close();
		
		System.out.println(sts + "\t");
		
		
	}

}
