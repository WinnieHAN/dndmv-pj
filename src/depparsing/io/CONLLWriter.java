package depparsing.io;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.IOException;

import data.InstanceList;
import depparsing.data.DepCorpus;
import depparsing.data.DepInstance;

public class CONLLWriter {

	private final BufferedWriter writer;
	private final DepCorpus corpus;
	
    public CONLLWriter(String filename, DepCorpus corpus) throws IOException {
    	writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename),"UTF8"));
    	this.corpus = corpus;
    }
    
    public void writeNext(DepInstance towrite) throws IOException {

    	for(int i = 0; i < towrite.parents.length; i++) {
    		int nextID = i + 1;
    		String tag = (String) corpus.tagAlphabet.index2feat.get(towrite.postags[i]);
    		writer.append(nextID + "\t" + corpus.wordAlphabet.index2feat.get(towrite.words[i]) 
    				+ "\t" + corpus.wordAlphabet.index2feat.get(towrite.words[i])  + "\t" +
    				tag + "\t" + tag + "\t_\t" + towrite.parents[i] +
    				"\t_\t_\t_\n");
    	}
    	
    	writer.append("\n");
    	writer.flush();
    }
    
    public void writeLine(String towrite) throws IOException {
    	writer.append(towrite + "\n");
    	writer.flush();
    }
    
	/**
	 * Outputs the parses as the sentences without changing the original Instance List
	 */
	public static void printConll(int[][] parses, String outfile, InstanceList il, DepCorpus corpus)
	throws IOException {
		CONLLWriter writer = new CONLLWriter(outfile, corpus);
		
		for(int i = 0; i < il.instanceList.size(); i++) {
			DepInstance depInst = (DepInstance)il.instanceList.get(i);
			int[] old = depInst.parents.clone();
			replaceDependencies(depInst, parses[i]);
			writer.writeNext(depInst);
			replaceDependencies(depInst, old);
		}
	}

    private static void replaceDependencies(DepInstance depInst, int[] newdeps) {
    		System.arraycopy(newdeps, 0, depInst.parents, 0, depInst.parents.length);
    }
}
