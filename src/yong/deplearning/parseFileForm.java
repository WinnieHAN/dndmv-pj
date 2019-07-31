package yong.deplearning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URLDecoder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Scanner;
import java.util.regex.Pattern;


import java.util.Random;



public class parseFileForm {
	/**
	 * @param args
	 * @throws IOException
	 */
	
	public static int corpusType;
	public static final int COMPACT = 0; // <word,POS,parent>
	public static final int PASCAL = 1;
	public static final int CONLL = 2;
	public static boolean parsed;

	public static void main(String[] args) throws IOException {
		
		corpusType = CONLL;
		parsed = true;
		int isfast2wsj = Integer.valueOf(args[3]);
        //sts from conll form to fast dep form
		System.out.println("args[0]  " + args[0]);
		//@SuppressWarnings("deprecation")
		//String decodedPath = URLDecoder.decode(args[0]);
		//System.out.println("decodedPath  " + decodedPath);
		
		String in = args[0];
		String out = args[1];
		if(isfast2wsj == 0){
			String depsF = out + "/deps_english";
			String wordsF = out + "/words_english";
			String posesF = out + "/poses_english";
			sts2rbg(in, depsF, wordsF, posesF);
		}else{
			//return fast dep form to conll form
			String fast = args[0];
			String wsj = args[1];
			String conll = args[2];
			rbg2conll(fast, wsj, conll);
		}
	}
	
	public static void rbg2conll(String fast, String wsj, String conll)
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
		Pattern rung = Pattern.compile("-");
		Pattern whitespace = Pattern.compile("\\s+");
		Scanner fastf = new Scanner(new BufferedReader(new FileReader(fast)));
		@SuppressWarnings("resource")
		Scanner wsjf = new Scanner(new BufferedReader(new FileReader(wsj)));
		FileWriter conllf = new FileWriter(conll);
		HashSet<String> POSs = new HashSet<String>();
		while (fastf.hasNext()) {
			// read one sentence
			ArrayList<String> words = new ArrayList<String>();
			ArrayList<String> tokens = new ArrayList<String>();
			ArrayList<Integer> parents = new ArrayList<Integer>();
			ArrayList<String[]> lines = new ArrayList<String[]>();
			String ln = fastf.nextLine();
			while (ln.trim().length() != 0) {//one sentence
				String[] info = whitespace.split(ln);
//				words.add(info[indexWord]);
//				tokens.add(info[indexPOS]);
//				if (parsed)
//					parents.add(Integer.parseInt(info[indexParent]));
				lines.add(info);
				if (fastf.hasNext())
					ln = fastf.nextLine();
				else
					break;
			}
			//read another file sts.
			ArrayList<String> words_wsj = new ArrayList<String>();
			ArrayList<String> tokens_wsj = new ArrayList<String>();
			ArrayList<Integer> parents_wsj = new ArrayList<Integer>();
			ArrayList<String[]> lines_wsj = new ArrayList<String[]>();
			String ln_wsj = wsjf.nextLine();
			while (ln_wsj.trim().length() != 0) {//one sentence
				String[] info_wsj = whitespace.split(ln_wsj);
				words_wsj.add(info_wsj[indexWord]);
				tokens_wsj.add(info_wsj[indexPOS]);
				if (parsed)
					parents_wsj.add(Integer.parseInt(info_wsj[indexParent]));
				lines_wsj.add(info_wsj);
				if (wsjf.hasNext())
					ln_wsj = wsjf.nextLine();
				else
					break;
			}

//			// process
//			for (int i = 0; i < words.size();) {
//				if (isPunctuation(tokens.get(i))) {
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
			

			
			
//			if (words.size() == 0) {
//				System.out.println("empty sentence.");
//				continue;
//			}
//			if (words.size() > 100)
//				System.out.println(words.size());
//
//			POSs.addAll(tokens);
			
			// write
			String[] tr = lines.get(3);
			String[] tr1 = new String[lines.get(3).length - 1];
			
			for(int i = 1; i < tr.length; i ++){
				String[] pc = rung.split(tr[i]);
				int p = Integer.valueOf(pc[0]);
				int c = Integer.valueOf(pc[1]);
				//System.out.println("==============" + String.valueOf(p) + "==========" + String.valueOf(c));
				if(p == (tr.length - 1)){
					p = 0;
				}else{
					p = p + 1;
				}
				//System.out.println("==============" + String.valueOf(p) + "==========" + String.valueOf(c) + "====" );
				tr1[c] = String.valueOf(p);
			}
			
			int idxk = 0;
			for (String[] line : lines_wsj) {
				conllf.write(line[0]);
				for (int i = 1; i < line.length - 1; i++) {
					if (corpusType == PASCAL && i == 5)
						continue;
					conllf.write("\t" + line[i]);
				}
				conllf.write("\t" + tr1[idxk]);
				idxk++;
				conllf.write("\n");
			}
			conllf.write("\n");
			
			sts += 1;
		}
		
		fastf.close();
		conllf.close();

		System.out.println(POSs + "\t" + POSs.size());
		System.out.println(sts + "\t");
		

	}
	
	public static void sts2rbg(String in, String depsF, String wordsF,String posesF)
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
		FileWriter depf = new FileWriter(depsF);
		FileWriter wordf = new FileWriter(wordsF);
		FileWriter posef = new FileWriter(posesF);
		HashSet<String> POSs = new HashSet<String>();
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
			// un-wsj may do not need the process
//			for (int i = 0; i < words.size();) {
//				if (isPunctuation(tokens.get(i))) {
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
//			if (words.size() == 0) {
//				System.out.println("empty sentence.");
//				continue;
//			}
//			if (words.size() > 10) {
//				continue;
//			}
//			if (words.size() > 100)
//				System.out.println(words.size());
//
//			POSs.addAll(tokens);
            

			
			// write
			
			for(String[] line : lines){
				wordf.write(line[indexWord] + " ");
			}
			wordf.write("#" + "\n");
			
			for(String[] line : lines){
				posef.write(line[indexPOS] + " ");
			}
			posef.write("#" + "\n");
			
			int idx = 0;
			int p = -1; 
			for(String[] line : lines){
				p = Integer.valueOf(line[indexParent]);
				if(p < 0.5){
					p = lines.size();
				}else{
					p = p - 1;
				}
				assert(p < -0.5);
				depf.write(String.valueOf(p) + "-" + String.valueOf(idx) + " ");
				idx++;
			}
			depf.write("\n");
			
			
			sts += 1;
		}
		sf.close();
		depf.close();
		wordf.close();
		posef.close();

		System.out.println(POSs + "\t" + POSs.size());
		System.out.println(sts + "\t");
		

	}
	protected static boolean isPunctuation(String string) {
		switch (corpusType) {
		default:
		case COMPACT:
			return Arrays.asList(punctuationPOSs).contains(string);
		case PASCAL:
			return Arrays.asList(punctuationPOSs).contains(string);//string.equals(".");
		}
	}
	
	protected static final String[] punctuationPOSs = { ",", ".", ":", "``",
			"''", "-LRB-", "-RRB-", "#", "LS", "SYM", "-NONE-", "``#28" };



}
