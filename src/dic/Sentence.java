package dic;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import nn.Net;
import nn.Net2;
import nn.Net2Soft;
import nn.Node;

import org.ejml.simple.SimpleMatrix;

import depparsing.data.DepInstance;
import depparsing.util.GrammarMath;
import depparsing.util.Pair;

public class Sentence {
	private int length;
	public  SimpleMatrix rootVec;//TODO  this should be final
	public final ArrayList<String> wordSeq;// hope it not used
	public ArrayList<Integer> goldTree;
	public ArrayList<Integer> wordIdxInDic;
	private int sentenceIndex;
	
	
	public Sentence(int stcIndex){
		length = 0;
		this.wordSeq = new ArrayList<String>();
		this.goldTree = new ArrayList<Integer>();
		this.wordIdxInDic = new ArrayList<Integer>();
		this.sentenceIndex = stcIndex;
		
	}
	
	public Sentence(int len, ArrayList<String> wordSeq, ArrayList<Integer> goldTree){
		this.length = len;
		this.wordSeq = wordSeq;
		this.goldTree = new ArrayList<>(length);
		this.goldTree = goldTree;
		
		
	}
	public void setRootVec(SimpleMatrix vec){
		for(int i = 0; i < vec.numRows(); i++){
			vec.set(i, 0, 1);
		}
		
		this.rootVec = vec;
		
	}
	public int getLength(){
		return this.length;
	}
	
	public void addWord(String wordname, int wIdxInDic){
		this.length++;
		this.wordSeq.add(wordname);
		this.wordIdxInDic.add(wIdxInDic);
	}
	public void addWord(Pair<String, String> wordname, int wIdxInDic){
		this.length++;
		this.wordSeq.add(wordname.a);//here just use dic. Hope we will not use it then.
		this.wordIdxInDic.add(wIdxInDic);
	}
	/**
	 * set a parent index to a child
	 * @param prt
	 * @param cld
	 */
	public void addGoldTree(int prt, int cld){
		if(this.goldTree.size() == 0){
			for(int i = 0; i < this.length; i++){
				this.goldTree.add( -1 );
			}
		}
		this.goldTree.set(cld, prt);
	}
	
	public DepInstance tran2DepIns(){
		int[] words = new int[this.length];
		int[] tags = new int[this.length];
		int[] parents = new int[this.length];
		
		for(int i = 0; i < this.length; i++){
			words[i] = this.wordIdxInDic.get(i);
			tags[i] = this.wordIdxInDic.get(i);
			parents[i] = this.goldTree.get(i);
			
		}
		DepInstance depIns = new DepInstance(words, tags, parents, this.sentenceIndex);
		return depIns;
	}
	
	public void rootNNForward(double[] root, SimpleMatrix initWeight,
			int negSetSize, ArrayList<String> negDic, HashMap<String, SimpleMatrix> word2vec, 
			HashMap<String, SimpleMatrix> tag2vec, HashMap<String, String> word2tag,
			HashMap<String, Integer> str2idx) throws Exception{
		
		for(int chdIdx = 0; chdIdx < this.wordSeq.size(); chdIdx ++){
			
			
			String name = this.wordSeq.get(chdIdx);
			String tag = word2tag.get(name);
			Word chd = new Word(chdIdx, name, name, tag, word2vec.get(name), tag2vec.get(tag));
			
			Node chdNode = new Node(chd, true, false);
			SimpleMatrix rootvec = new SimpleMatrix(initWeight.numRows(), 1);
			
			this.setRootVec(rootvec);
			Net2 net = new Net2(this.rootVec, chdNode, initWeight, negSetSize);
			net.genNegSet(negDic, word2tag, word2vec, tag2vec);
			root[chdIdx] = net.outputProb;
			
			double allScore = GrammarMath.LOGZERO;
/*
			for(Map.Entry<String, SimpleMatrix> otherChdEntry : word2vec.entrySet()){
				String otherName = otherChdEntry.getKey();
				String otherTag = word2tag.get(otherName);
				Word otherChd = new Word(str2idx.get(otherName), otherName, otherName, otherTag,
						otherChdEntry.getValue(), tag2vec.get(otherTag));
				Node otherChdNode = new Node(otherChd, true, false);
				
				Net2 netother = new Net2(this.rootVec, otherChdNode, initWeight, negSetSize);
				allScore = GrammarMath.log_sum_exp(allScore, netother.outputProb);

			}
			root[chdIdx] = root[chdIdx] - allScore;
	*/		
		/*	if(Double.isNaN(root[chdIdx])){
				System.exit(0);
				//throw new Exception();
				
			}*/
			assert(!Double.isNaN(root[chdIdx])):
				name + "\t tag:\t" + tag + "\t allscore:\t" + allScore;
		}		
		
	}
	
	public void rootNNForwardSoft(double[] root, SimpleMatrix initWeight,
			int negSetSize, ArrayList<String> negDic, HashMap<String, SimpleMatrix> word2vec, 
			HashMap<String, SimpleMatrix> tag2vec, HashMap<String, String> word2tag,
			HashMap<String, Integer> str2idx) throws Exception{
		
		for(int chdIdx = 0; chdIdx < this.wordSeq.size(); chdIdx ++){
			
			
			String name = this.wordSeq.get(chdIdx);
			String tag = word2tag.get(name);
			Word chd = new Word(chdIdx, name, name, tag, word2vec.get(name), tag2vec.get(tag));
			
			Node chdNode = new Node(chd, true, false);
			SimpleMatrix rootvec = new SimpleMatrix(initWeight.numRows(), 1);
			
			this.setRootVec(rootvec);//
			Net2Soft net = new Net2Soft(this.rootVec, chdNode, initWeight, negSetSize);
			net.genNegSet(str2idx, word2tag, word2vec, tag2vec);
			root[chdIdx] = net.softmaxFunction(net.outputProb);
	//		root[chdIdx]  = Math.log(root[chdIdx] );//
			double allScore = GrammarMath.LOGZERO;
/*
			for(Map.Entry<String, SimpleMatrix> otherChdEntry : word2vec.entrySet()){
				String otherName = otherChdEntry.getKey();
				String otherTag = word2tag.get(otherName);
				Word otherChd = new Word(str2idx.get(otherName), otherName, otherName, otherTag,
						otherChdEntry.getValue(), tag2vec.get(otherTag));
				Node otherChdNode = new Node(otherChd, true, false);
				
				Net2 netother = new Net2(this.rootVec, otherChdNode, initWeight, negSetSize);
				allScore = GrammarMath.log_sum_exp(allScore, netother.outputProb);

			}
			root[chdIdx] = root[chdIdx] - allScore;
	*/		
		/*	if(Double.isNaN(root[chdIdx])){
				System.exit(0);
				//throw new Exception();
				
			}*/
			assert(!Double.isNaN(root[chdIdx])):
				name + "\t tag:\t" + tag + "\t allscore:\t" + allScore;
		}		
		
	}

	public void nnForwardSentence(double[][][] child, int valancySize, ArrayList<SimpleMatrix> valencyVecList,
			SimpleMatrix initWeight,  SimpleMatrix leftVec, SimpleMatrix rightVec, int negSetSize, ArrayList<String> negDic,
			HashMap<String, SimpleMatrix> word2vec, HashMap<String, SimpleMatrix> tag2vec, HashMap<String, String> word2tag,
			HashMap<String, Integer> str2idx) throws Exception{
		for(int parIdx = 0; parIdx < this.wordSeq.size(); parIdx ++){
			
			String parname = this.wordSeq.get(parIdx);
			String partag = word2tag.get(parname);
			Word par = new Word(str2idx.get(parname), parname, parname, partag, word2vec.get(parname), tag2vec.get(partag));
			
			Node prtNode = new Node(par, true, false);
			
			for(int chdIdx = 0; chdIdx < this.wordSeq.size(); chdIdx ++){
				
				String chdname = this.wordSeq.get(chdIdx);
				String chdtag = word2tag.get(chdname);

				Word chd = new Word(str2idx.get(chdname), chdname, chdname, chdtag, word2vec.get(chdname), tag2vec.get(chdtag));
				
				Node chdNode = new Node(chd, true, false);
				
		//		SimpleMatrix dirVec = (dir == 0) ? leftDirVec : rightDirVec;
				if( ! this.wordSeq.get(parIdx).equals(this.wordSeq.get(chdIdx)) ){
					int dir = (parIdx < chdIdx) ? 1 : 0;
					SimpleMatrix dirVec = (dir == 1) ? rightVec : leftVec;
					for(int i = 0; i < valancySize; i++){
						
						
						Net2 net = new Net2(prtNode, chdNode, initWeight, valencyVecList.get(i), dirVec,
								valancySize, i, negSetSize);
						//net.genNegSet(negDic, word2tag, word2vec, tag2vec);
						child[parIdx][chdIdx][i] = net.outputProb;
						double allScore = GrammarMath.LOGZERO;
						int index = 0;
						
					/*	for(Map.Entry<String, SimpleMatrix> otherChdEntry : word2vec.entrySet()){
							String otherName = otherChdEntry.getKey();
							String otherTag = word2tag.get(otherName);
							Word otherChd = new Word(str2idx.get(otherChdEntry.getKey()), otherName, otherName, otherTag,
									otherChdEntry.getValue(), tag2vec.get(otherTag));
							Node otherChdNode = new Node(otherChd, true, false);
							Net2 netother = new Net2(prtNode, otherChdNode, initWeight, valencyVecList.get(i), valancySize, i, negSetSize);
							//allScore += Math.exp(netother.outputProb);
							allScore = GrammarMath.log_sum_exp(allScore, netother.outputProb);
							index ++;
							
						}
						
						child[parIdx][chdIdx][i] = child[parIdx][chdIdx][i] - allScore;
						*/
						assert(!Double.isNaN(child[chdIdx][chdIdx][i])):
							chdname + "\t chdtag:\t" + chdtag;
					}
					
					
				}
				
			}
			
		}
		
	}
	
	public void nnForwardSentenceSoft(double[][][] child, int valancySize, ArrayList<SimpleMatrix> valencyVecList,
			SimpleMatrix initWeight,  SimpleMatrix leftVec, SimpleMatrix rightVec, int negSetSize, ArrayList<String> negDic,
			HashMap<String, SimpleMatrix> word2vec, HashMap<String, SimpleMatrix> tag2vec, HashMap<String, String> word2tag,
			HashMap<String, Integer> str2idx) throws Exception{
		for(int parIdx = 0; parIdx < this.wordSeq.size(); parIdx ++){
			
			String parname = this.wordSeq.get(parIdx);
			String partag = word2tag.get(parname);
			Word par = new Word(str2idx.get(parname), parname, parname, partag, word2vec.get(parname), tag2vec.get(partag));
			
			Node prtNode = new Node(par, true, false);
			
			for(int chdIdx = 0; chdIdx < this.wordSeq.size(); chdIdx ++){
				
				String chdname = this.wordSeq.get(chdIdx);
				String chdtag = word2tag.get(chdname);

				Word chd = new Word(str2idx.get(chdname), chdname, chdname, chdtag, word2vec.get(chdname), tag2vec.get(chdtag));
				
				Node chdNode = new Node(chd, true, false);
				
		//		SimpleMatrix dirVec = (dir == 0) ? leftDirVec : rightDirVec;
				if( ! this.wordSeq.get(parIdx).equals(this.wordSeq.get(chdIdx)) ){
					int dir = (parIdx < chdIdx) ? 1 : 0;
					SimpleMatrix dirVec = (dir == 1) ? rightVec : leftVec;
					for(int i = 0; i < valancySize; i++){
						
						
						Net2Soft net = new Net2Soft(prtNode, chdNode, initWeight, valencyVecList.get(i), dirVec,
								valancySize, i, negSetSize);
						net.genNegSet(str2idx, word2tag, word2vec, tag2vec);
						child[chdIdx][parIdx][i] = net.softmaxFunction(net.outputProb);
				//		child[parIdx][chdIdx][i] = Math.log(child[parIdx][chdIdx][i]);//
						double allScore = GrammarMath.LOGZERO;
						int index = 0;
						
//						for(Map.Entry<String, SimpleMatrix> otherChdEntry : word2vec.entrySet()){
//							String otherName = otherChdEntry.getKey();
//							String otherTag = word2tag.get(otherName);
//							Word otherChd = new Word(str2idx.get(otherChdEntry.getKey()), otherName, otherName, otherTag,
//									otherChdEntry.getValue(), tag2vec.get(otherTag));
//							Node otherChdNode = new Node(otherChd, true, false);
//							Net2 netother = new Net2(prtNode, otherChdNode, initWeight, valencyVecList.get(i), valancySize, i, negSetSize);
//							//allScore += Math.exp(netother.outputProb);
//							allScore = GrammarMath.log_sum_exp(allScore, netother.outputProb);
//							index ++;
//							
//						}
//						
//						child[parIdx][chdIdx][i] = child[parIdx][chdIdx][i] - allScore;
//						
						assert(!Double.isNaN(child[chdIdx][chdIdx][i])):
							chdname + "\t chdtag:\t" + chdtag;
					}
					
					
				}
				
			}
			
		}
		
	}
	
	public int accWords(int[] parser){
		
		int count = 0;
		for(int i = 0; i < this.length; i++){
			if(this.goldTree.get(i) == parser[i])
				count++;
		}
		return count;
	}
	
	public void nnForwardDecision(double[][][][] decision, 
			SimpleMatrix weight, ArrayList<SimpleMatrix> valencyVecs, SimpleMatrix dirLeftVec, SimpleMatrix dirRightVec,
			SimpleMatrix stopVec, SimpleMatrix continueVec,
			HashMap<String, SimpleMatrix> word2vec, HashMap<String, SimpleMatrix> tag2vec, HashMap<String, String> word2tag,
			int valencySize){
		
		for(int parIdx = 0; parIdx < this.wordSeq.size(); parIdx++){
			//Node prtNode = new Node(this.wordSeq.get(parIdx), true, false);
			String parname = this.wordSeq.get(parIdx);
			String partag = word2tag.get(parname);
			Word par = new Word(parIdx, parname, parname, partag, word2vec.get(parname), tag2vec.get(partag));
			Node prtNode = new Node(par, true, false);
			for(int dir = 0; dir < 2; dir++){
				
				for(int v = 0; v < valencySize; v++){
					//for(int dc = 0; dc < 2; dc++){
					SimpleMatrix dirVec = (dir == 0) ? dirLeftVec : dirRightVec;
					
					Net net = new Net(prtNode, weight,
							valencyVecs.get(v), dirVec, 
							valencySize, v, dir == 0, stopVec, continueVec, true);

					decision[parIdx][dir][v][1] = net.forward();
					decision[parIdx][dir][v][0] = 1 - decision[parIdx][dir][v][1];
					
					if(decision[parIdx][dir][v][0] < 1e-20)
						decision[parIdx][dir][v][0] = 1e-20;
					
					if(decision[parIdx][dir][v][1] < 1e-20)
						decision[parIdx][dir][v][1] = 1e-20;					
					
					decision[parIdx][dir][v][0] = Math.log(decision[parIdx][dir][v][0]);
					decision[parIdx][dir][v][1] = Math.log(decision[parIdx][dir][v][1]);
					assert(!Double.isNaN(decision[parIdx][dir][v][0] )):
						parname + "\t prttag:\t" + partag;
					//}
				}
				
				
			}
			
		}
	}
	
}
