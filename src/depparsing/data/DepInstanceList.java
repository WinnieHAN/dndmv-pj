package depparsing.data;

import gnu.trove.TIntArrayList;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.regex.Pattern;

import util.Alphabet;
import util.CountAlphabet;
import util.InputOutput;
import data.InstanceList;

public class DepInstanceList extends InstanceList{
	
	// This is useful to compare when merging two instance lists to see if they are from the same alphabet.
	// Also to print the sentences with names instead of numbers.
	Alphabet<String> tagsAlphabet;
	
	public DepInstanceList(String name) {
		super(name);
	}
	
	public static DepInstanceList readFromConll(String name, String fileName, Alphabet<String> words, 
			Alphabet<String> tags,
			boolean lowercase,
			int minSentenceLength,int maxSentenceLenght, 
			CountAlphabet<String> fullAlphabet, int minWordOccurs)
	throws UnsupportedEncodingException, FileNotFoundException, IOException{	
		
		if(minWordOccurs > 0) System.out.println("Warning: Ignoring parameter minWordOccurs for dependency parsing");
		
		DepInstanceList il = new DepInstanceList(name);
		il.wordsAlphabet = words;
		il.tagsAlphabet = tags;
		il.name = name;
		BufferedReader reader = InputOutput.openReader(fileName);
		
		Pattern whitespace = Pattern.compile("\\s+");
		TIntArrayList wordsList  =new TIntArrayList();
		TIntArrayList posList  =new TIntArrayList();
		TIntArrayList parentsList  =new TIntArrayList();
		String line = reader.readLine();
		while(line != null) {
			if(!line.matches("\\s*")){
				String[] info = whitespace.split(line);
				String word = normalize(info[1]);
				if(lowercase){
					word = word.toLowerCase();
				}
				wordsList.add(words.lookupObject(word));
				posList.add(tags.lookupObject(info[4]));
				parentsList.add(Integer.parseInt(info[6]));
			}
			else { // Case of end of sentence
				int sentenceSize = wordsList.size();
				if(sentenceSize >= minSentenceLength && sentenceSize <= maxSentenceLenght){
					addDepInst(il, wordsList, posList, parentsList);
				}
				wordsList.clear();
				posList.clear();
				parentsList.clear();
			}
			line = reader.readLine();
		}
		// Add final dependency instance
		// (need this in case file ends without a trailing newline)
		if(wordsList.size() > 0)
			addDepInst(il, wordsList, posList, parentsList);
		
		return il;
	}
	
	private static void addDepInst(DepInstanceList il, TIntArrayList wordsList, TIntArrayList posList, TIntArrayList parentsList)
	{
		il.add(new DepInstance(wordsList.toNativeArray(),posList.toNativeArray(),parentsList.toNativeArray(),il.instanceList.size()));
		il.maxInstanceSize = Math.max(il.maxInstanceSize,wordsList.size());
	}
}
