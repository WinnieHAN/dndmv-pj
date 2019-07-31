package depparsing.data;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;

import util.CountAlphabet;
import data.Corpus;
import data.InstanceList;
import data.WordInstance;

/**
 * Contains a dependency parsing corpus:
 * Extends the base corpus by adding parent information
 * @author javg
 */
public class DepCorpus extends Corpus {
	public CountAlphabet<String> tagAlphabet;
	
	public DepCorpus(String corpusParams) throws UnsupportedEncodingException, FileNotFoundException, IOException {
		this(corpusParams, 0, Integer.MAX_VALUE, Integer.MAX_VALUE);
	}
	
	public DepCorpus(String corpusParams, int minSentenceSize, int maxSentenceSize, int maxNumberOfSentences)
	throws UnsupportedEncodingException, FileNotFoundException, IOException {
		super(corpusParams, minSentenceSize, maxSentenceSize, maxNumberOfSentences);	
	}

	@Override
	public void initStructures(String corpusParams) {
		super.initStructures(corpusParams);
		this.tagAlphabet = new CountAlphabet<String>();	
	}
	
	public void freezeAlphabetsCounts(){
		super.freezeAlphabetsCounts();
		this.tagAlphabet.setStopCounts(true);
	}

	public int getNrTags(){
		return tagAlphabet.size();
	}
	
	public String[] getTagStrings(int[] tagIds) {
		String[] tags = new String[tagIds.length];
		for (int i = 0; i < tags.length; i++) {
			tags[i] = tagAlphabet.index2feat.get(tagIds[i]);
		}
		return tags;	
	}
	
	public String[] getAllTagsStrings() {
		String[] tags= new String[getNrTags()];
		for (int i = 0; i < tags.length; i++) {
			tags[i]=tagAlphabet.index2feat.get(i);
		}
		return tags;
	}
	
	@Override 
	public InstanceList readInstanceList(String name, String fileName,
			String readerType, boolean lowercase, 
			int minSentenceLenght, int maxSentenceLenght, int maxNumberOfSentences,
			CountAlphabet<String> fullVocab, int minWordOccurs) throws UnsupportedEncodingException, FileNotFoundException, IOException{
		System.out.println("Calling readInstanceList " + name + " " + fileName);
		if(readerType.equalsIgnoreCase("conll-data")){
			return DepInstanceList.readFromConll(name, fileName, 
					this.wordAlphabet,this.tagAlphabet, 
					lowercase,minSentenceLenght,maxSentenceLenght
					,fullVocab,minWordOccurs);
		}else{
			System.out.println("Unknow reader type");
			System.exit(-1);
		}
		return null;
	}

	public void printCorpusStats(){
		super.printCorpusStats();
		System.out.println("Number of tags: " + this.getNrTags());
	}

	public String printTags(InstanceList list ,int[][] predictions){
		StringBuffer sb = new StringBuffer();
		int i = 0;
		for(WordInstance inst: list.instanceList){
			int[] words = inst.words;
			int[] tags = predictions[i];
			for (int j = 0; j < tags.length; j++) {
				String tagS = "noTag:"+j;
				if(tags[j] != -1){
					tagS = tagAlphabet.index2feat.get(tags[j]);
				}
				sb.append(tagS+"\t"
						+wordAlphabet.index2feat.get(words[j])+"\n");
			}
			sb.append("\n");
			i++;
		}
		return sb.toString();
	}
	
	public boolean isPunctuation(int tagID) {
		return tagAlphabet.index2feat.get(tagID).compareTo("punct") == 0;
	}
	
	public static void main(String[] args) throws UnsupportedEncodingException, FileNotFoundException, IOException {
			DepCorpus c = new DepCorpus(args[0]);
			int i = 0;
			for (WordInstance inst : c.trainInstances.instanceList) {
				DepInstance inst2 = (DepInstance) inst;
				System.out.print(inst.getSentence(c));
				System.out.print(inst2.getTagsStrings(c));
				if(i > 100) break;
				i++;
			}
		}
	
}
