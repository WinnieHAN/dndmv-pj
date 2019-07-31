package depparsing.constraints;

import depparsing.data.DepCorpus;

public enum PCType {
	WORD, TAG;
	String id2string(DepCorpus c, int id){
		switch (this) {
		case TAG:
			return (String) c.tagAlphabet.index2feat.get(id);
		case WORD:
			return (String) c.wordAlphabet.index2feat.get(id);
		default:
			throw new RuntimeException("unknown type of PCTAG");
		}
	}
}