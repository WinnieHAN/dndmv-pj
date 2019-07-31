package yong.deplearning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Scanner;
import java.util.regex.Pattern;
import java.io.File;

public class combinefile {
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
		String metas = args[0];
		String newf = args[1];
		combinefile(metas, newf);

	}
	

	public static void combinefile(String metas,String newf) throws IOException{
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
		
		FileWriter out = new FileWriter(newf, true);
		HashSet<String> POSs = new HashSet<String>();
		File root = new File(metas);
		File[] files = root.listFiles();
		for(File file: files){
			String meta = file.getAbsolutePath();
			Scanner conllin = new Scanner(new BufferedReader(new FileReader(meta)));
			while (conllin.hasNext()) {
				// read one sentence
				ArrayList<String> words = new ArrayList<String>();
				ArrayList<String> tokens = new ArrayList<String>();
				ArrayList<Integer> parents = new ArrayList<Integer>();
				ArrayList<String[]> lines = new ArrayList<String[]>();
				String ln = conllin.nextLine();
				while (ln.trim().length() != 0) {//one sentence
					String[] info = whitespace.split(ln);
					words.add(info[indexWord]);
					tokens.add(info[indexPOS]);
					if (parsed)
						parents.add(Integer.parseInt(info[indexParent]));
					lines.add(info);
					if (conllin.hasNext())
						ln = conllin.nextLine();
					else
						break;
				}
	
				POSs.addAll(tokens);
				if (words.size() > 10) {
					continue;
				}
				// write
				for (String[] line : lines) {
					out.write(line[0]);
					for (int i = 1; i < line.length; i++) {
						out.write("\t" + line[i]);
					}
					out.write("\n");
				}
				out.write("\n");

				sts += 1;
			}
			conllin.close();
		}
		out.close();
		System.out.println(String.valueOf(sts));
	}
	
	public static void conll2mit(String conllinf, String mitoutf) throws IOException{
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
		Scanner conllin = new Scanner(new BufferedReader(new FileReader(conllinf)));
		FileWriter mitout = new FileWriter(mitoutf);
		HashSet<String> POSs = new HashSet<String>();
		while (conllin.hasNext()) {
			// read one sentence
			ArrayList<String> words = new ArrayList<String>();
			ArrayList<String> tokens = new ArrayList<String>();
			ArrayList<Integer> parents = new ArrayList<Integer>();
			ArrayList<String[]> lines = new ArrayList<String[]>();
			String ln = conllin.nextLine();
			while (ln.trim().length() != 0) {//one sentence
				String[] info = whitespace.split(ln);
				words.add(info[indexWord]);
				tokens.add(info[indexPOS]);
				if (parsed)
					parents.add(Integer.parseInt(info[indexParent]));
				lines.add(info);
				if (conllin.hasNext())
					ln = conllin.nextLine();
				else
					break;
			}

			POSs.addAll(tokens);
			if (words.size() > 10) {
				continue;
			}
			// write
			mitout.write(lines.get(0)[indexWord]);
			for (int i = 1; i < lines.size(); i++) {
				mitout.write("\t" + lines.get(i)[indexWord]);
			}
			mitout.write("\n");
			
			mitout.write(lines.get(0)[indexPOS]);
			for (int i = 1; i < lines.size(); i++) {
				mitout.write("\t" + lines.get(i)[indexPOS]);
			}
			mitout.write("\n");
			
			mitout.write(lines.get(0)[indexParent]);
			for (int i = 1; i < lines.size(); i++) {
				mitout.write("\t" + lines.get(i)[indexParent]);
			}
			mitout.write("\n");

			mitout.write("\n");
			sts += 1;
		}
		
		conllin.close();
		mitout.close();

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

