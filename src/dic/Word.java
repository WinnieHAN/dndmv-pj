package dic;

import org.ejml.simple.SimpleMatrix;

import depparsing.util.Pair;

public class Word {
	private int index;
	public final String fullName;
	public String wordName;
	public final String tagName;
	public final SimpleMatrix wordVec;
	public final SimpleMatrix tagVec;
	public final int dim;
	
	public Word(Word w){
		this.index = w.getInx();
		this.wordName = w.wordName;
		this.fullName = w.fullName;
		this.tagName = w.tagName;
		this.wordVec = w.wordVec;
		this.dim = w.dim;
		this.tagVec = w.tagVec;
	}
	public Word(int idx, String fullname, String word, String tag, 
			SimpleMatrix vec, SimpleMatrix tagv){
		this.index = idx;
		this.fullName = fullname;
		this.wordName = word;
		this.tagName = tag;
		this.wordVec = vec;
		this.dim = vec.numRows();
		this.tagVec = tagv;
	}
	
	public Word(String wordname, String tag, SimpleMatrix vec, SimpleMatrix tagv){
		this.fullName = "NOT_VISUALABLE";
		this.wordName = wordname;
		this.wordVec = vec;
		this.dim = vec.numRows();
		this.tagVec = tagv;
		this.tagName = tag;
	}
	
	public String getWordName(){
		return this.wordName;
	}
	public void setWordName(String wordname){
		this.wordName = wordname;
	}

	public int getInx(){
		return this.index;
	}

	@Override
	public boolean equals(Object obj) {
		if (obj.getClass() == getClass()) {
			@SuppressWarnings("unchecked")
			Word p2 = (Word) obj;
			return this.wordName.equals(p2.wordName);
		}
		return false;
	}

	@Override
	public int hashCode() {
		return this.wordName.hashCode();
	}
}
