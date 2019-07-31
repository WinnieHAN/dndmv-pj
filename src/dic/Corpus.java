package dic;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;
import org.kohsuke.args4j.CmdLineException;

import depparsing.util.Pair;

public class Corpus {
	public final String unk = "UNK";
	public ArrayList<Sentence> stcs;
	public ArrayList<Sentence> stcs_afterAnnealing;
	public ArrayList<Sentence> stcsWithLengthLessThanTen;
	public ArrayList<Sentence> testStcs;
	public ArrayList<Sentence> validStcs;
	public ArrayList<Sentence> validStcsLessThanTen;
	public ArrayList<Sentence> testStcsLessThanTen;
	private int size;
	
	public HashMap<Pair<String, String>, Integer> dic; // This is what we need, <word, tag>
	public HashMap<Pair<String, String> , Integer> dicNum;
	public HashMap<Pair<String,String>, String> dic2tag;//dic is Pair<word, tag>
	public HashMap<Integer, Pair<String, String>> idx2dic;

	public HashMap<String, SimpleMatrix> tag2vec;
	public HashMap<String, SimpleMatrix> word2vec;

	public HashMap<String, SimpleMatrix> tag2vecInit;
	public HashMap<String, SimpleMatrix> word2vecInit;
	
	public SimpleMatrix rootVec;//
	
	public ArrayList<String> negDic;
	public HashMap<String, Integer> tag2idx;
	
	public ArrayList<Sentence> baggingStcsWithLengthLessThanTen;
	
	public final int wordDim;
	public final int tagDim;
	public final int valencyDim;
	public final int dirDim;
	public Corpus(int wordDim, int tagDim, int valencyDim, int dirDim, int valNum){
		this.stcs = new ArrayList<>();
		this.wordDim = wordDim;
		this.tagDim = tagDim;
		this.valencyDim = valencyDim;
		this.dirDim = dirDim;
		this.tag2vec = new HashMap<>();
		

	}
	
	public void samplingSentences(){
		baggingStcsWithLengthLessThanTen = new ArrayList<>();
		int len = this.stcsWithLengthLessThanTen.size();
		int index = 0;
		while(index != (0.9 * len)){
			int randInt = (int) (Math.random() * len);
			this.baggingStcsWithLengthLessThanTen.add(this.stcsWithLengthLessThanTen.get(randInt));
			index++;
		}
		
	}
	public ArrayList<Sentence> reduceStceOutside(int len){
		ArrayList<Sentence> stces = new ArrayList<Sentence>();
		for(Sentence s : this.stcs){
			if(s.wordSeq.size() <= len){
				stcsWithLengthLessThanTen.add(s);
				
			}
			
		}
		return stces;
	}
	
//	public void addAddtionalWordVecs(){
//		for(Map.Entry<String, Integer> word : this.dic.entrySet()){
//			
//			if(!this.word2vec.containsKey(word.getKey())){
//				this.word2vec.put(word.getKey(), this.getUnusualVec(wordDim));
//			}
//			
//		}
//	}
	public void reduceSentencesNum(int len, int num){
		if(this.stcsWithLengthLessThanTen == null)
			stcsWithLengthLessThanTen = new ArrayList<>();
		int counts = 0;
		for(Sentence s : this.stcs){
			if(s.wordSeq.size() <= len & counts < num){
				stcsWithLengthLessThanTen.add(s);
				counts += 1;
			}
			
		}
		System.out.println("After Done Reduction, Sentence Size:\t" + stcsWithLengthLessThanTen.size());
	}
	
	public void reduceSentences(int len){
		if(this.stcsWithLengthLessThanTen == null)
			stcsWithLengthLessThanTen = new ArrayList<>();
		for(Sentence s : this.stcs){
			if(s.wordSeq.size() <= len){
				stcsWithLengthLessThanTen.add(s);
				
			}
			
		}
		System.out.println("After Done Reduction, Sentence Size:\t" + stcsWithLengthLessThanTen.size());
	}
	public void reduceValidSentences(int len){
		if(this.validStcsLessThanTen == null)
			validStcsLessThanTen = new ArrayList<>();
		for(Sentence s : this.validStcs){
			if(s.wordSeq.size() <= len){
				validStcsLessThanTen.add(s);
				
			}
			
		}
		System.out.println("After Done Reduction, Sentence Size:\t" + validStcsLessThanTen.size());
	}
	
	public void reduceTestSentences(int len){
		if(this.testStcsLessThanTen == null)
			testStcsLessThanTen = new ArrayList<>();
		for(Sentence s : this.testStcs){
			if(s.wordSeq.size() <= len){
				testStcsLessThanTen.add(s);
				
			}
			
		}
		System.out.println("After Done Reduction, Sentence Size:\t" + testStcsLessThanTen.size());
	}
	
	
	public void setTagVec(HashMap<String, SimpleMatrix> tagvec){
		this.tag2vec = tagvec;
		
	}
	
	

	/**
	 *  renew tag vector after update
	 * @param
	 */
	public void renewTagVec(HashMap<String, SimpleMatrix> newtag2vec){
		int vecSize = this.tag2vec.size();
		for(Map.Entry<String, SimpleMatrix> entry : newtag2vec.entrySet()){
			
			this.tag2vec.put(entry.getKey(), entry.getValue());
			System.out.println("testing the hashmap:\t" + (vecSize == this.tag2vec.size()));
		}
		
	}	
	
	/**
	 *  renew word vector after update
	 * @param newW2vec
	 */
	public void renewWordVec(HashMap<String, SimpleMatrix> newW2vec){
		int vecSize = this.word2vec.size();
		for(Map.Entry<String, SimpleMatrix> entry : newW2vec.entrySet()){
			
			this.word2vec.put(entry.getKey(), entry.getValue());
			System.out.println("testing the hashmap:\t" + (vecSize == this.word2vec.size()));
		}
		
	}
	

//	/**
//	 * Generate negtive dictionary from dictionary of our corpus
//	 */
//	public void getNegDic(){
//		
//		double allProb = 0.0;
//		ArrayList<Pair<String, Double>> arrDic = new ArrayList<>();
//		for(Map.Entry<String, Integer> insWord : dic.entrySet()){
//			allProb += Math.pow(insWord.getValue(), 0.75); 
//			arrDic.add(new Pair<String, Double>(insWord.getKey(), Math.pow((double) (insWord.getValue()), 0.75)));
//		}
//
//		int dicSize = dic.size();
//		int segSize = dicSize * 10;//Put the probabilities to a box ----Word2Vec
//		negDic = new ArrayList<>();
//		
//		double scope = 1.0 / segSize;
//		//all noralize to 1
//		for(int idx = 0; idx < dicSize; idx++){
//			
//			Pair<String, Double> wordPair = arrDic.get(idx);
//			if(idx == 0)
//				wordPair.b = wordPair.b / allProb;
//			else
//				wordPair.b = wordPair.b / allProb + arrDic.get(idx - 1).b;
//		}
//		int wordInx = 0;
//		for(int idx = 0; idx < segSize; idx++){
//			if(idx * scope < arrDic.get(wordInx).b)
//				negDic.add(arrDic.get(wordInx).a);
//			else{
//				wordInx++;
//				negDic.add(arrDic.get(wordInx).a);
//			}
//				
//		}
//	}
	
	public void index2Dic() throws Exception{
		if(idx2dic == null)
			idx2dic = new HashMap<Integer, Pair<String, String>>(dic.size());
		for(Map.Entry<Pair<String, String>, Integer> word : dic.entrySet()){
			if(idx2dic.containsKey(word.getValue()))
				throw new Exception("dic has duplicated Strings");
			else
				idx2dic.put(word.getValue(), word.getKey());
		}
		
		
	}
	/**
	 *  During reading the input file, we build the dictionary...
	 *  if wordname is in the dictionary with common words, we return true.
	 *  otherwise, we return false. witch means that we will use the tag to 
	 *  represente the words.
	 * @param wordname
	 */
	public boolean checkDic(String wordname){
		if(dic == null)
			assert(false): "need call deleteUNKforDic function first!";
		if(!dic.containsKey(wordname))
			return false;
		else
			return true;
	}
	
	public boolean checkDic(Pair<String, String> wordname){
		if(dic == null)
			assert(false): "need call deleteUNKforDic function first!";
		if(!dic.containsKey(wordname))
			return false;
		else
			return true;
	}
	
	/**
	 *  During reading the input file, we build the dictionary sum...which have word-tag in dictionary.
	 * @param wordname
	 */
	public void buildDicNum(String wordname, String tag){
		Pair<String, String> p = new Pair<>(wordname, tag);
		if(dicNum == null)
			dicNum = new HashMap<>();
		if(!dicNum.containsKey(new Pair<String, String>(wordname, tag)))
			dicNum.put(p, 1);
		else
			dicNum.put(p, dicNum.get(p) + 1);
	}
	
	/**
	 *  load a word2vec dictionary.
	 * @param word2vecFileName
	 * @throws IOException
	 */
	public void getWord2VecFromFile(String word2vecFileName) throws IOException{
		
		this.word2vec = new HashMap<>();
		BufferedReader vecBr = new BufferedReader(new FileReader(word2vecFileName));
		String line = null;
		while( (line = vecBr.readLine()) != null){
			String[] toks = line.split(" ");
			String wordname = toks[0];

			//String[] vec = toks[1].split(" ");

			if(dic.containsKey(wordname)){
				SimpleMatrix wordVec = new SimpleMatrix(this.wordDim, 1);
				for(int i = 1; i < wordDim + 1; i++){
					if(toks[i].length() != 0)
						wordVec.set(i - 1, 0, Double.parseDouble(toks[i]));
	
				}
				word2vec.put(wordname, wordVec.scale(0.1));
			}
		}
		
		System.out.println(word2vec.size() + " vs " + this.dic.size());

		word2vec.put("$", getUnusualVec(tagDim));
		tag2vec.put("$", getUnusualVec(tagDim));
		
		this.rootVec = new SimpleMatrix(this.wordDim, 1);
		for(int row = 0; row < this.wordDim; row++){
			this.rootVec.set(row, 0, 1.0);
		}
		
		vecBr.close();
	}
	
	public void saveAllWord2File(String outputFileName) throws IOException{
		BufferedWriter bw = new BufferedWriter(new FileWriter(outputFileName));
		
		for(Map.Entry<Pair<String, String>, Integer> word : dic.entrySet()){
			bw.write(word.getKey().a + "\t" + word.getKey().b + "\t"+ String.valueOf((int)word.getValue()) + "\n"); // hanwj 10.9
                 //bw.write(word.getKey().a + "\t" + word.getKey().b + "\n");
		//	System.out.println(word.getKey());
		}
		bw.close();
	}
	
	
	public void countFreqFromFile(String filename) throws IOException{
		BufferedReader br = new BufferedReader(new FileReader(filename));
		
		String line;
		while((line = br.readLine()) != null){

			if(line.length() != 0){
				String name = null;
				String[] wordAttr =line.split("\t");
				name = wordAttr[1].toLowerCase();
				this.buildDicNum(name, wordAttr[4]);
				
			}
				
		}
		
		
		br.close();
	}
	
	
	public void readAnotherFromFile(String filename) throws IOException{
		//if(this.tag2idx == null)	this.tag2idx = new HashMap<>();
		
		//this.stcs = new ArrayList<>();
		BufferedReader br = new BufferedReader(new FileReader(filename));
		int count = 0;
		String line;
		Sentence s = null;
		ArrayList<Integer> chd2par = null;
		
		while((line = br.readLine()) != null){
			
			if(line.length() == 0 || count == 0){//process a sts
				count++;
				if(s != null){
					for(int i = 0; i < chd2par.size(); i++){
						s.addGoldTree(chd2par.get(i), i);
					}
					
					this.stcs.add(s);
					size++;
				}
				s = new Sentence(count - 1);
				chd2par = new ArrayList<>();
			
				
			}
			if(line.length() != 0){//process a word
				
				String name = null, tag = null;
				int parIdx = 0;
				String[] wordAttr =line.split("\t");
				
				name = wordAttr[1].toLowerCase();//name is 1
				tag = wordAttr[4];//tag is 4
				
				Pair<String, String> modelName = new Pair<String, String>(name, tag);
				modelName = (checkDic(modelName) == true ? modelName : new Pair<String, String>(tag, tag));
				//this.word2tag.put(modelName, tag);//?

				parIdx = Integer.parseInt(wordAttr[6]);//parent is 6
				
				if(this.dic.containsKey(modelName)){
					s.addWord(modelName, this.dic.get(modelName));
				}
				else{
					System.out.println("ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"+ modelName.a + "\t" + modelName.b);
				}

				chd2par.add(parIdx);
				
			}
				
		}
		
		
		br.close();
	}
	
	public void readFromFile(String filename) throws IOException{
		//if(this.tag2idx == null)	this.tag2idx = new HashMap<>();
		
		this.stcs = new ArrayList<>();
		BufferedReader br = new BufferedReader(new FileReader(filename));
		int count = 0;
		String line;
		Sentence s = null;
		ArrayList<Integer> chd2par = null;
		
		while((line = br.readLine()) != null){
			
			if(line.length() == 0 || count == 0){//process a sts
				count++;
				if(s != null){
					for(int i = 0; i < chd2par.size(); i++){
						s.addGoldTree(chd2par.get(i), i);
					}
					
					this.stcs.add(s);
					size++;
				}
				s = new Sentence(count - 1);
				chd2par = new ArrayList<>();
			
				
			}
			if(line.length() != 0){//process a word
				
				String name = null, tag = null;
				int parIdx = 0;
				String[] wordAttr =line.split("\t");
				
				name = wordAttr[1].toLowerCase();//name is 1
				tag = wordAttr[4];//tag is 4
				
				Pair<String, String> modelName = new Pair<String, String>(name, tag);
				modelName = (checkDic(modelName) == true ? modelName : new Pair<String, String>(tag, tag));
				//this.word2tag.put(modelName, tag);//?

				parIdx = Integer.parseInt(wordAttr[6]);//parent is 6
				

				s.addWord(modelName, this.dic.get(modelName));

				chd2par.add(parIdx);
				
			}
				
		}
		
		
		br.close();
	}
	
	public void readTestDataFromFile(String filename) throws IOException{
		
		
		this.testStcs = new ArrayList<>();
		BufferedReader br = new BufferedReader(new FileReader(filename));
		int count = 0;
		String line;
		Sentence s = null;
		ArrayList<Integer> chd2par = null;
		
		while((line = br.readLine()) != null){
			
			if(line.length() == 0 || count == 0){
				count++;
				if(s != null){
					for(int i = 0; i < chd2par.size(); i++){
						s.addGoldTree(chd2par.get(i), i);
					}
					
					this.testStcs.add(s);
					size++;
				}
				s = new Sentence(count - 1);
				chd2par = new ArrayList<>();
				
			}
			if(line.length() != 0){
				
				String name = null, tag = null;
				int parIdx = 0;
				String[] wordAttr =line.split("\t");
				
				name = wordAttr[1].toLowerCase();
				tag = wordAttr[4];

				Pair<String, String> modelName = new Pair<String, String>(name, tag);
				modelName = (checkDic(modelName) == true ? modelName : new Pair<String, String>(tag, tag));
				//this.word2tag.put(modelName, tag);//?

				parIdx = Integer.parseInt(wordAttr[6]);//parent is 6
				

				s.addWord(modelName, this.dic.get(modelName));

				chd2par.add(parIdx);					

				
			}
				
		}
		
		
		br.close();
	}
	
	
	public void readValidDataFromFile(String filename) throws IOException{

		this.validStcs = new ArrayList<>();
		BufferedReader br = new BufferedReader(new FileReader(filename));
		int count = 0;
		String line;
		Sentence s = null;
		ArrayList<Integer> chd2par = null;
	
		while((line = br.readLine()) != null){
			
			if(line.length() == 0 || count == 0){
				count++;
				if(s != null){
					for(int i = 0; i < chd2par.size(); i++){
						s.addGoldTree(chd2par.get(i), i);
					}
					
					this.validStcs.add(s);
					size++;
				}
				s = new Sentence(count - 1);
				chd2par = new ArrayList<>();
			
				
			}
			if(line.length() != 0){
			
				String name = null, tag = null;
				int parIdx = 0;
				String[] wordAttr =line.split("\t");
				
				name = wordAttr[1].toLowerCase();
				tag = wordAttr[4];
				
				Pair<String, String> modelName = new Pair<String, String>(name, tag);
				modelName = (checkDic(modelName) == true ? modelName : new Pair<String, String>(tag, tag));
				//this.word2tag.put(modelName, tag);//?

				parIdx = Integer.parseInt(wordAttr[6]);//parent is 6
				

				s.addWord(modelName, this.dic.get(modelName));

				chd2par.add(parIdx);
				
			}
				
		}
		
		
		br.close();
	}
	
	public void readAfterAnnealingFromFile(String filename) throws IOException{

		this.stcs_afterAnnealing = new ArrayList<>();
		BufferedReader br = new BufferedReader(new FileReader(filename));
		int count = 0;
		String line;
		Sentence s = null;
		ArrayList<Integer> chd2par = null;
		
		while((line = br.readLine()) != null){
			
			if(line.length() == 0 || count == 0){
				count++;
				if(s != null){
					for(int i = 0; i < chd2par.size(); i++){
						s.addGoldTree(chd2par.get(i), i);
					}
					
					this.stcs_afterAnnealing.add(s);
					//size++;
				}
				s = new Sentence(count - 1);
				chd2par = new ArrayList<>();
			
				
			}
			if(line.length() != 0){
			
				String name = null, tag = null;
				int parIdx = 0;
				String[] wordAttr =line.split("\t");
				name = wordAttr[1].toLowerCase();
				tag = wordAttr[4];

				Pair<String, String> modelName = new Pair<String, String>(name, tag);
				modelName = (checkDic(modelName) == true ? modelName : new Pair<String, String>(tag, tag));
				//this.word2tag.put(modelName, tag);//?

				parIdx = Integer.parseInt(wordAttr[6]);//parent is 6
				

				s.addWord(modelName, this.dic.get(modelName));
//				s.addWord(modelName, 1);

				chd2par.add(parIdx);
				
			}
				
		}
		
		
		br.close();
	}
	public void deleteUNKforDic(int wordf){
		int count = 0;
		int deletedCount = 0;
		if(this.dicNum.size() == 0)
			assert(false): "program terminates";
		for(Map.Entry<Pair<String, String>, Integer> entry :  this.dicNum.entrySet()){

			if(dic == null)
				dic = new HashMap<>();
			if(entry.getValue() < wordf){
				count++;
				if(!this.dic.containsKey(new Pair<String, String>(entry.getKey().b, entry.getKey().b))){
					this.dic.put(new Pair<String, String>(entry.getKey().b, entry.getKey().b), deletedCount++);
				}
			}
			else{
				if( !this.dic.containsKey(entry.getKey()) )
					this.dic.put(entry.getKey(), deletedCount++);
			}
				
		}
		//this.dic.put(unk, deletedCount++);
		System.out.println("Dictionary size:" + this.dic.size());
		System.out.println("word frequences less than " + wordf + " is " + count);
	}
	
	//TODO this is just for running the entire code
	/**
	 * 
	 * @return
	 */
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
		   // double g = (Math.random() - 0.5) * 1;// * 0.1
			vec.set(i, 0, g * 0.1);
		}
		
		return vec;
	}
	public void para2Init(){

		
		
		if(word2vecInit == null)	word2vecInit = new HashMap<>();
		assert(this.tag2vec != null) : "tag2vec should be set before set the init vec";
		if(tag2vecInit == null)	tag2vecInit = new HashMap<>();
		assert(this.word2vec != null) : "word2vec should be set before set the init vec";
		for(Map.Entry<String, SimpleMatrix> t2v : tag2vec.entrySet()){
			this.tag2vecInit.put(t2v.getKey(), t2v.getValue());
			
		}
		
		for(Map.Entry<String, SimpleMatrix> w2v : word2vec.entrySet()){
			this.word2vecInit.put(w2v.getKey(), w2v.getValue());
		}
		
	}
	
	public void init2Para(){
		if(word2vec == null)	word2vec = new HashMap<>();
		assert(this.tag2vecInit != null) : "tag2vecInit should be set before set the init vec";
		if(tag2vec == null)	tag2vec = new HashMap<>();
		assert(this.word2vecInit != null) : "word2vecInit should be set before set the init vec";
		this.word2vec.clear();
		this.tag2vec.clear();
		for(Map.Entry<String, SimpleMatrix> t2v : tag2vecInit.entrySet()){
			this.tag2vec.put(t2v.getKey(), t2v.getValue());
			
		}
		
		for(Map.Entry<String, SimpleMatrix> w2v : word2vecInit.entrySet()){
			this.word2vec.put(w2v.getKey(), w2v.getValue());
		}
	}
	
	public void genDic2Tag(){
		//HashMap<String, Integer> word2num = new HashMap<>();
		if(this.dic2tag == null)
			this.dic2tag = new HashMap<>();//this.dic2tag is <Pair<word, tag>, tag>
		for(Map.Entry<Pair<String, String> , Integer> p : this.dicNum.entrySet()){
			if(!this.dic2tag.containsKey(p.getKey())){//p.getKey().a is word, while p.getKey().b is tag.
				this.dic2tag.put(p.getKey(), p.getKey().b);
			}
			
		}
		System.out.println("dic2tag size:" + this.dic2tag.size());
		
	}
	public void genTag2Idx(){
		int idx = 0;
		if(this.tag2idx == null){
			this.tag2idx = new HashMap<>();
		}
		else{
			idx = this.tag2idx.size();
		}
		for(Map.Entry<Pair<String, String> , Integer> p : this.dicNum.entrySet()){
			if(!this.tag2idx.containsKey(p.getKey().b)){
				this.tag2idx.put(p.getKey().b, idx);
				idx++;
			}
		}
		System.out.println("tag2idx size:" + this.tag2idx.size());
	}
	
}
