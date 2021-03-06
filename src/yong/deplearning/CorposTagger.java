package yong.deplearning;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;
import java.util.regex.Pattern;

import depparsing.util.Pair;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.List;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

public class CorposTagger {
	/**
	 * @param args
	 * @throws IOException
	 */
	//public static HashMap<String, Integer> AUXG;
//	public static HashMap<Pair<String, String>, String> wordCorrect;//<pair<word, tag>,ctag>
	public static HashMap<Pair<String, String>, String> sWordCorrect;//<pair<word, tag>,ctag>
	public static ArrayList<HashSet<String>> w2t;
	public static void main(String[] args) throws IOException {
		
//		AUXG = new HashMap<String, Integer>();
//		BufferedReader w = new BufferedReader(new FileReader("bllip_out/AUX_word"));
//		String line =  w.readLine();
//		String[] words = line.split("\\s+");
//		for( int i = 0; i < words.length; i++){
//			AUXG.put(words[i], i);
//		}
		
//		wordCorrect = new HashMap<Pair<String, String>, String>();
//		BufferedReader w = new BufferedReader(new FileReader("bllip_out/WORD2TAG1"));
//        String line = "";
//        int i = 0;
//		while((line =  w.readLine()) != null){
//        	String[] lineParts = line.split("\\s+");
//        	wordCorrect.put(new Pair<String,String> (lineParts[0], "AUX"), lineParts[1]);
//        }
		
		sWordCorrect = new HashMap<Pair<String, String>, String>();
		String specialTag = "'d";//"had do DO Need 've 'd need Have Do NEED HAVE have Had HAD"
		String[] specialTags = specialTag.split("\\s+");
		for (int i1 = 0; i1 < specialTags.length; i1++){
			sWordCorrect.put(new Pair<String,String> (specialTags[i1], "AUX"), "_");
		}
		
//		w2t = new ArrayList<HashSet<String>>(AUXG.size());
//		for (int i = 0; i < AUXG.size(); i ++){
//			w2t.add(new HashSet<String>());
//		}
		
		corpusType = CONLL;
		parsed = false;
		//removePunctuation("bllip_out/sp1/SpecialWordSentence", "bllip_out/bllipCorpus_all_remove8.txt", "bllip_out/sp1/SpecialWordSentence_TagC");//"bllip_out/bllipCorpus_init.txt"//"bllip_out/bllipCorpus_all_remove.txt"
		
		//removePunctuation("bllip_out/wsj-inf_2-21_dep", "bllip_out/wsj-inf_2-21_dep1");
		// File root = new File("pascal");
		// assert root.isDirectory();
		// String[] dirs = root.list();
		// for (int i = 0; i < dirs.length; i++) {
		// File dir = new File(root, dirs[i]);
		// assert dir.isDirectory();
		// String[] files = dir.list();
		// for (int j = 0; j < files.length; j++) {
		// if (files[j].endsWith("cleaned"))
		// removeColumnInCorpus(dir.getPath() + "/" + files[j], 5);
		// }
		// }

		// switchColumnInCorpus("pascal/dutch/train", 3, 4);
		



	    
//	    MaxentTagger tagger = new MaxentTagger("models/english-bidirectional-distsim.tagger");
//	    List<List<HasWord>> sentences = MaxentTagger.tokenizeText(new BufferedReader(new FileReader("bllip_out/sp1/SpecialWordSentence_TagC0")));
//	    System.out.println(sentences.size());
//	    int[] a = SentenceCompare("bllip_out/sp1/SpecialWordSentence_TagC0", sentences);
//	    
//	    String initFile = "bllip_out/sp1/SpecialWordSentence_TagC";
//	    int fileID = 0;
//	    while(a[1] == 0){
//	    	newfile(initFile, fileID, a[0]);
//	    	fileID ++;
//		    List<List<HasWord>> sentences1 = MaxentTagger.tokenizeText(new BufferedReader(new FileReader(initFile + String.valueOf(fileID))));
//		    System.out.println(sentences1.size());
//		    a = SentenceCompare(initFile+String.valueOf(fileID), sentences1);
//		    
//	    	
//	    }
//	    System.out.println("........Over!");
		
		
		
//		MaxentTagger tagger = new MaxentTagger("models/english-bidirectional-distsim.tagger");
//	    String in = "bllip_out/SpecialWordSentence2";
//	    String out = "bllip_out/SpecialWordSentence2_out";
//	    compareTwoSentence(tagger, in, out, "bllip_out/000");//AUX to other tags.
		
//		String in1 = "bllip_out/NoSpecialWordSentence2";
//		String in2 = "bllip_out/SpecialWordSentence2_out";
//		String out = "bllip_out/BillpAll2";
//		combine2File( in1,  in2,  out);
		
//		MaxentTagger tagger = new MaxentTagger("models/english-bidirectional-distsim.tagger");
//		String in = "bllip_out/BillpAll2";
//		String out = "bllip_out/BillpAll2_noCorrectSts";//"bllip_out/BillpAll2_tagCorrect";
//		toTagCorrect(tagger, in, out, "bllip_out/000");
		
//		removePunctuation("bllip_out/BillpAll2_tagCorrect","bllip_out/BillpAll2_tagCorrect_remove", "bllip_out/0");
		int stsnum = 100000;
		stsReduce("bllip_out/BillpAll2_noCorrectSts","data/BillpAll2_noCorrectSts_"+String.valueOf(stsnum), "bllip_out/0", stsnum);
	}
	
	public static void stsReduce(String in, String out, String stsout, int stsnum)
			throws IOException {
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
		Scanner sf = new Scanner(new BufferedReader(new FileReader(in)));
		FileWriter fw = new FileWriter(out);
		@SuppressWarnings("resource")
		FileWriter stsfw = new FileWriter(stsout);
		HashSet<String> POSs = new HashSet<String>();
		HashSet<String> POSsAUX = new HashSet<String>();
		HashSet<String> POSsAUXG = new HashSet<String>();
		while (sf.hasNext()) {
			// read one sentence
			ArrayList<String> words = new ArrayList<String>();
			ArrayList<String> tokens = new ArrayList<String>();
			ArrayList<Integer> parents = new ArrayList<Integer>();
			ArrayList<String[]> lines = new ArrayList<String[]>();
			String ln = sf.nextLine();
			while (ln.trim().length() != 0) {//one sentence
				String[] info = whitespace.split(ln);
				words.add(info[indexWord]);
				tokens.add(info[indexPOS]);
				if (parsed)
					parents.add(Integer.parseInt(info[indexParent]));
				lines.add(info);
				if (sf.hasNext())
					ln = sf.nextLine();
				else
					break;
			}

			// process
			for (int i = 0; i < words.size();) {
				if (isPunctuation(tokens.get(i))) {
					if (parsed)
						if (parents.contains(i + 1) || parents.get(i) == -1) {
							System.out.println("May need manual correction: "
									+ tokens.get(i) + "\n" + words + "\n"
									+ tokens + "\n" + parents + "\n");
						}

					words.remove(i);
					tokens.remove(i);
					lines.remove(i);
					if (parsed) {
						int ip = parents.remove(i);
						if (ip > i + 1)
							ip--;
						for (int j = 0; j < parents.size(); j++) {
							int p = parents.get(j);
							if (p > i + 1)
								parents.set(j, p - 1);
							else if (p == i + 1)
								parents.set(j, ip);
							lines.get(j)[indexParent] = Integer
									.toString(parents.get(j));
						}
					}
				} else
					i++;
			}
			

			
			
			if (words.size() == 0) {
				System.out.println("empty sentence.");
				continue;
			}
			if (words.size() > 10) {
				continue;
			}
			if (words.size() > 100)
				System.out.println(words.size());

			POSs.addAll(tokens);
            
			if(sts >= stsnum){//number to reduce
				continue;
			}

			
			// write
			
			
			for (String[] line : lines) {
				fw.write(line[0]);
				for (int i = 1; i < line.length; i++) {
					if (corpusType == PASCAL && i == 5)
						continue;
					fw.write("\t" + line[i]);
				}
				fw.write("\n");
			}
			fw.write("\n");
			sts += 1;
		}
		sf.close();
		fw.close();
		stsfw.close();

		System.out.println(POSs + "\t" + POSs.size());
		System.out.println(sts + "\t");
		

	}
	
	private static void toTagCorrect(MaxentTagger tagger, String in, String out, String string) throws IOException {
		int wsts = 0;//write sentence = output sts
		int allsts = 0;// input sts
		int allw = 0;// all words
		int diffw = 0;// #words with different tag 
		int nocorrectsts = 0;// sts which tags are all the same
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
		Scanner sf = new Scanner(new BufferedReader(new FileReader(in)));
		FileWriter fw = new FileWriter(out);
		@SuppressWarnings("resource")
		
		HashSet<String> POSs = new HashSet<String>();
		HashSet<String> POSsAUX = new HashSet<String>();
		HashSet<String> POSsAUXG = new HashSet<String>();
		while (sf.hasNext()) {
			// read one sentence
			ArrayList<String> words = new ArrayList<String>();
			ArrayList<String> tokens = new ArrayList<String>();
			ArrayList<Integer> parents = new ArrayList<Integer>();
			ArrayList<String[]> lines = new ArrayList<String[]>();
			String ln = sf.nextLine();
			while (ln.trim().length() != 0) {//one sentence
				String[] info = whitespace.split(ln);
				words.add(info[indexWord]);
				tokens.add(info[indexPOS]);
				if (parsed)
					parents.add(Integer.parseInt(info[indexParent]));
				lines.add(info);
				if (sf.hasNext())
					ln = sf.nextLine();
				else
					break;
			}
            allsts++;
            allw += words.size();
			// process
			
			boolean isOK = true;
			boolean noCorrect = true;
			String[] a = new String[words.size()];
			for (int i = 0; i < words.size(); i++){
				a[i] = words.get(i);
			}
		    List<HasWord> sent = Sentence.toWordList(a);
		    List<TaggedWord> taggedSent = tagger.tagSentence(sent);
		    for (int i = 0; i < words.size(); i++){
		    	if(!taggedSent.get(i).word().equals(words.get(i)))
		    		isOK = false;
		    }
		    if(!isOK)
		    	continue;
		    else{
		    	for (int i = 0; i < words.size(); i++){
		    		if(!tokens.get(i).equals(taggedSent.get(i).tag())){
		    			tokens.set(i, taggedSent.get(i).tag());
		    			lines.get(i)[indexPOS] = taggedSent.get(i).tag();
		    			diffw ++;
		    			noCorrect = false;
		    		}
		    	}
		    }
		    if(noCorrect)  nocorrectsts++;
		    if(!noCorrect) continue; 

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
			wsts += 1;
			
		}
		sf.close();
		fw.close();
		

		System.out.println(wsts + "\t");
		System.out.println(allsts + "\t");
		System.out.println(allw + "\t");
		System.out.println(diffw + "\t");
		System.out.println(nocorrectsts + "\t");
		
	}

	private static void combine2File(String in1, String in2, String out) throws IOException{
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
		HashSet<String> POSsAUX = new HashSet<String>();
		HashSet<String> POSsAUXG = new HashSet<String>();
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
	
    private static void compareTwoSentence(MaxentTagger tagger,String in, String out, String stsout) throws IOException{
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
		Scanner sf = new Scanner(new BufferedReader(new FileReader(in)));
		FileWriter fw = new FileWriter(out);
		@SuppressWarnings("resource")
		FileWriter stsfw = new FileWriter(stsout);
		HashSet<String> POSs = new HashSet<String>();
		HashSet<String> POSsAUX = new HashSet<String>();
		HashSet<String> POSsAUXG = new HashSet<String>();
		while (sf.hasNext()) {
			// read one sentence
			ArrayList<String> words = new ArrayList<String>();
			ArrayList<String> tokens = new ArrayList<String>();
			ArrayList<Integer> parents = new ArrayList<Integer>();
			ArrayList<String[]> lines = new ArrayList<String[]>();
			String ln = sf.nextLine();
			while (ln.trim().length() != 0) {//one sentence
				String[] info = whitespace.split(ln);
				words.add(info[indexWord]);
				tokens.add(info[indexPOS]);
				if (parsed)
					parents.add(Integer.parseInt(info[indexParent]));
				lines.add(info);
				if (sf.hasNext())
					ln = sf.nextLine();
				else
					break;
			}

			// process
			
			boolean isOK = true;
			String[] a = new String[words.size()];
			for (int i = 0; i < words.size(); i++){
				a[i] = words.get(i);
			}
		    List<HasWord> sent = Sentence.toWordList(a);
		    List<TaggedWord> taggedSent = tagger.tagSentence(sent);
		    for (int i = 0; i < words.size(); i++){
		    	if(!taggedSent.get(i).word().equals(words.get(i)))
		    		isOK = false;
		    }
		    if(!isOK)
		    	continue;
		    else{
		    	for (int i = 0; i < words.size(); i++){
		    		if(tokens.get(i).equals("AUX")){
		    			tokens.set(i, taggedSent.get(i).tag());
		    			lines.get(i)[indexPOS] = taggedSent.get(i).tag();
		    		}
		    	}
		    }


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
			
//			for (String[] line : lines) {
//				stsfw.write(line[0]);
//				for (int i = 1; i < line.length; i++) {
//					if (corpusType == PASCAL && i == 5)
//						continue;
//					stsfw.write("\t" + line[i]);
//				}
//				stsfw.write("\n");
//			}
//			stsfw.write("\n");
//			sts += 1;
		}
		sf.close();
		fw.close();
		stsfw.close();

		//System.out.println(POSs + "\t" + POSs.size());
		//System.out.println(POSsAUXG + "\t" + POSsAUXG.size());
		//System.out.println(POSsAUX + "\t" + POSsAUX.size());
		System.out.println(sts + "\t");
		
		//System.out.println(AUXG);
		//System.out.println(w2t);
		
		
    }
	private static void newfile(String init, int fileID, int idxDelete) throws IOException {
		String old = init + String.valueOf(fileID);
		String now = init + String.valueOf(fileID + 1);
		Scanner sf = new Scanner(new BufferedReader(new FileReader(old)));
		FileWriter fw = new FileWriter(now);
		int idx = -1;
		while(sf.hasNext()){
			idx = idx + 1;
			String b = sf.nextLine();
			if(idx == idxDelete){
				System.out.println(b);
				
				continue;
			}
			fw.write(b);
			
		}
		sf.close();
		fw.close();
	}

	private static int[] SentenceCompare(String in, List<List<HasWord>> sentences) throws FileNotFoundException {
		Pattern whitespace = Pattern.compile("\\s+");
		Scanner sf = new Scanner(new BufferedReader(new FileReader(in)));
		int idx = 0;
		boolean allRight = true;
		while(sf.hasNext()){
			String line = sf.nextLine();
			System.out.println(line);
//			while (ln.trim().length() != 0) {//one sentence
			String[] info = whitespace.split(line);
			String[] s = new String[info.length];
			for (int i = 0; i < s.length; i ++){
				s[i] = "";
			}
			for (int i = 0; i < info.length; i ++){
				s[i] = sentences.get(idx).get(i).toString();
				if(!s[i].equals(info[i])){
					allRight = false;
					break;
//					System.out.print("?????????"+ idx +"\t(");
//					System.out.print(info[i] + "\t:");
//					System.out.print(s[i] + ")\t");
					
				}
				
			}
			if(allRight == false) break;
			idx++;
		}
		sf.close();
		int[] re = new int[2];
		re[0] = idx;
		re[1] = (allRight)? 1 : 0;
		return re;
	}

	public static int corpusType;
	public static final int COMPACT = 0; // <word,POS,parent>
	public static final int PASCAL = 1;
	public static final int CONLL = 2;
	public static boolean parsed;

	/**
	 * Remove punctuations in the dependency treebank
	 * 
	 * @param in
	 * @param out
	 * @throws IOException
	 */
	public static void removePunctuation(String in, String out, String stsout)
			throws IOException {
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
		Scanner sf = new Scanner(new BufferedReader(new FileReader(in)));
		FileWriter fw = new FileWriter(out);
		@SuppressWarnings("resource")
		FileWriter stsfw = new FileWriter(stsout);
		HashSet<String> POSs = new HashSet<String>();
		HashSet<String> POSsAUX = new HashSet<String>();
		HashSet<String> POSsAUXG = new HashSet<String>();
		while (sf.hasNext()) {
			// read one sentence
			ArrayList<String> words = new ArrayList<String>();
			ArrayList<String> tokens = new ArrayList<String>();
			ArrayList<Integer> parents = new ArrayList<Integer>();
			ArrayList<String[]> lines = new ArrayList<String[]>();
			String ln = sf.nextLine();
			while (ln.trim().length() != 0) {//one sentence
				String[] info = whitespace.split(ln);
				words.add(info[indexWord]);
				tokens.add(info[indexPOS]);
				if (parsed)
					parents.add(Integer.parseInt(info[indexParent]));
				lines.add(info);
				if (sf.hasNext())
					ln = sf.nextLine();
				else
					break;
			}

			// process
			for (int i = 0; i < words.size();) {
				if (isPunctuation(tokens.get(i))) {
					if (parsed)
						if (parents.contains(i + 1) || parents.get(i) == -1) {
							System.out.println("May need manual correction: "
									+ tokens.get(i) + "\n" + words + "\n"
									+ tokens + "\n" + parents + "\n");
						}

					words.remove(i);
					tokens.remove(i);
					lines.remove(i);
					if (parsed) {
						int ip = parents.remove(i);
						if (ip > i + 1)
							ip--;
						for (int j = 0; j < parents.size(); j++) {
							int p = parents.get(j);
							if (p > i + 1)
								parents.set(j, p - 1);
							else if (p == i + 1)
								parents.set(j, ip);
							lines.get(j)[indexParent] = Integer
									.toString(parents.get(j));
						}
					}
				} else
					i++;
			}
			
			for (int i = 0; i < words.size(); i++){
				if(tokens.get(i).contains("PRP#") && !tokens.get(i).equals("PRP#")){//PRP$
					tokens.set(i, "PRP");//PRP$
					lines.get(i)[indexPOS] = "PRP";
				}
			}
			

			for (int i = 0; i < words.size(); i++){
				if(tokens.get(i).contains("PRP-PLE")){//PRP$
					tokens.set(i, "PRP");//PRP$
					lines.get(i)[indexPOS] = "PRP";
				}
			}

			for (int i = 0; i < words.size(); i++){
				if(tokens.get(i).contains("PRP-DEI")){//PRP$
					tokens.set(i, "PRP");//PRP$
					lines.get(i)[indexPOS] = "PRP";
				}
			}
//			boolean isAUXG = false;
			for (int i = 0; i < words.size(); i++){
				if(tokens.get(i).equals("AUX") ){
					//System.out.println(words.get(i) + "\t"+ tokens.get(i));
					POSsAUX.add(words.get(i));
//					isAUXG = true;
				}
			}
//		if (!isAUXG) continue;
			
			for (int i = 0; i < words.size(); i++){
				if(tokens.get(i).equals("AUXG")){
					//System.out.println(words.get(i) + "\t"+ tokens.get(i));
					POSsAUXG.add(words.get(i));
					
				}
				
			}
			
//			boolean isS = false;
//			for (int i = 0; i < words.size(); i++){
//				if(sWordCorrect.containsKey(new Pair<String,String> (words.get(i), "AUX")) ){
//					isS = true;
//				}
//			}
//		    if (!isS) continue;
			
			
//			for (int i = 0; i < words.size(); i++){
//				Pair<String, String> a = new Pair<String, String>(words.get(i), tokens.get(i));
//				if(wordCorrect.containsKey(a)){
//					tokens.set(i, wordCorrect.get(a));
//					lines.get(i)[indexPOS] = wordCorrect.get(a);
//				}
//			}
			
			if (words.size() == 0) {
				System.out.println("empty sentence.");
				continue;
			}
			if (words.size() > 10) {
				continue;
			}
			if (words.size() > 100)
				System.out.println(words.size());

			POSs.addAll(tokens);

//			for (int i = 0; i < words.size(); i++){
//				if(AUXG.containsKey(words.get(i))){
//				    HashSet<String> a = new HashSet<String>();
//				    a = w2t.get(AUXG.get(words.get(i)));
//					a.add(tokens.get(i));
//		            w2t.set(AUXG.get(words.get(i)), a);
//				}
//				
//			}
			
			// write
			
			
			for (String[] line : lines) {
				fw.write(line[0]);
				for (int i = 1; i < line.length; i++) {
					if (corpusType == PASCAL && i == 5)
						continue;
					fw.write("\t" + line[i]);
				}
				fw.write("\n");
			}
			fw.write("\n");
			sts += 1;
			
//			for (String[] line : lines) {
//				stsfw.write(line[0]);
//				for (int i = 1; i < line.length; i++) {
//					if (corpusType == PASCAL && i == 5)
//						continue;
//					stsfw.write("\t" + line[i]);
//				}
//				stsfw.write("\n");
//			}
//			stsfw.write("\n");
//			sts += 1;

//   Conell to a just-sentences type.			
//			for (String[] line : lines) {
//				stsfw.write(line[indexWord] + "\t");
//			}
//			stsfw.write("."+"\n");
//			sts += 1;
		}
		sf.close();
		fw.close();
		stsfw.close();

		System.out.println(POSs + "\t" + POSs.size());
		System.out.println(POSsAUXG + "\t" + POSsAUXG.size());
		System.out.println(POSsAUX + "\t" + POSsAUX.size());
		System.out.println(sts + "\t");
		
		//System.out.println(AUXG);
		//System.out.println(w2t);
		
		
//		String w = "";
//		for(int i = 0; i < AUXG.size(); i++){
//			for(Entry<String, Integer> entry: AUXG.entrySet()){
//			    if(Integer.valueOf(i).equals(entry.getValue()))
//			    		 w = entry.getKey();
//
//			}
//			System.out.println(w + "\t" +w2t.get(i));
//		}
	}

	protected static final String[] punctuationPOSs = { ",", ".", ":", "``",
			"''", "-LRB-", "-RRB-", "#", "LS", "SYM", "-NONE-", "``#28" };

	protected static boolean isPunctuation(String string) {
		switch (corpusType) {
		default:
		case COMPACT:
			return Arrays.asList(punctuationPOSs).contains(string);
		case PASCAL:
			return Arrays.asList(punctuationPOSs).contains(string);//string.equals(".");
		}
	}

	/**
	 * @param filename
	 * @param col
	 *            0-based
	 * @throws IOException
	 */
	public static void removeColumnInCorpus(String filename, int col)
			throws IOException {
		Pattern whitespace = Pattern.compile("\\s+");
		File f = new File(filename);
		File f2 = new File(filename + ".bak");
		Scanner sf = new Scanner(new BufferedReader(new FileReader(f)));
		FileWriter fw = new FileWriter(f2);
		while (sf.hasNext()) {
			// read
			ArrayList<ArrayList<String>> lines = new ArrayList<ArrayList<String>>();
			String ln = sf.nextLine();
			while (ln.trim().length() != 0) {
				String[] info = whitespace.split(ln);
				ArrayList<String> a = new ArrayList<String>(Arrays.asList(info));
				a.remove(col);
				lines.add(a);
				if (sf.hasNext())
					ln = sf.nextLine();
				else
					break;
			}

			// write
			for (ArrayList<String> line : lines) {
				fw.write(line.get(0));
				for (int i = 1; i < line.size(); i++)
					fw.write("\t" + line.get(i));
				fw.write("\n");
			}
			fw.write("\n");
		}
		sf.close();
		fw.close();

		f.delete();
		f2.renameTo(f);
	}

	/**
	 * @param filename
	 * @param col
	 *            0-based
	 * @throws IOException
	 */
	public static void switchColumnInCorpus(String filename, int col1, int col2)
			throws IOException {
		Pattern whitespace = Pattern.compile("\\s+");
		File f = new File(filename);
		File f2 = new File(filename + ".bak");
		Scanner sf = new Scanner(new BufferedReader(new FileReader(f)));
		FileWriter fw = new FileWriter(f2);
		while (sf.hasNext()) {
			// read
			ArrayList<ArrayList<String>> lines = new ArrayList<ArrayList<String>>();
			String ln = sf.nextLine();
			while (ln.trim().length() != 0) {
				String[] info = whitespace.split(ln);
				ArrayList<String> a = new ArrayList<String>(Arrays.asList(info));
				String tmp = a.get(col1);
				a.set(col1, a.get(col2));
				a.set(col2, tmp);
				lines.add(a);
				if (sf.hasNext())
					ln = sf.nextLine();
				else
					break;
			}

			// write
			for (ArrayList<String> line : lines) {
				fw.write(line.get(0));
				for (int i = 1; i < line.size(); i++)
					fw.write("\t" + line.get(i));
				fw.write("\n");
			}
			fw.write("\n");
		}
		sf.close();
		fw.close();

		f.delete();
		f2.renameTo(f);
	}
}
