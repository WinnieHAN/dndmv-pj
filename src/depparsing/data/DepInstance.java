package depparsing.data;

import data.Corpus;
import data.WordInstance;

public class DepInstance extends WordInstance {
	
	public final int numWords;
	public final int postags[];
	public final int parents[];
	
	public DepInstance(int[] words, int[] tags, int[] parents, int instanceNumber) {
		super(words, instanceNumber);
		assert words.length == tags.length;
		assert words.length == parents.length;
		this.postags = tags;
		this.parents = parents;
		this.numWords = postags.length;
	}
	
	@Override
	public String toString() {
		return super.toString() 
		+ util.ArrayPrinting.intArrayToString(postags, null,"sentence tags")
		+ util.ArrayPrinting.intArrayToString(parents, null, "sentence parents")+ "\n";
	}

	@Override
	public String toString(Corpus c) {
		return super.toString(c) 
		+ util.ArrayPrinting.intArrayToString(postags, ((DepCorpus)c).getTagStrings(postags),"sentence tags") + "\n";
	}

	public String getTagsStrings(DepCorpus c){
		String tagsS[] = c.getTagStrings(postags);
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < tagsS.length; i++) {
			sb.append(tagsS[i]+ " ");
		}
		sb.append("\n");
		return sb.toString();
	}
}
