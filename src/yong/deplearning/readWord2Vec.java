package yong.deplearning;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;

public class readWord2Vec {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		readWord2Vec rwv = new readWord2Vec();
		//String vecfile = "C:\\Users\\Administrator\\Downloads\\Compressed\\GoogleNews-vectors-negative300.bin_2\\word2vec.txt";
		String vecfile = "D:\\saved code\\glove.6B\\glove.6B.50d.txt";
		String dicfile = "dicAll.txt";
		String outputfile = "glove.50..txt";
		rwv.readfile(vecfile, dicfile, outputfile);
	}

	public void readfile(String vecfile, String dicfile, String outputfile) throws IOException{
		BufferedReader vecBr = new BufferedReader(new FileReader(vecfile));
		BufferedReader dicBr = new BufferedReader(new FileReader(dicfile));
		
		BufferedWriter outputBw = new BufferedWriter(new FileWriter(outputfile));
		
		String vecLine = null;
		String dicLine = null;
		HashSet<String> dic = new HashSet<String>();
		while( (dicLine = dicBr.readLine()) != null){
			dic.add(dicLine);
			System.out.println("Reading vecs:" + dicLine);
		}
		
		while( (vecLine = vecBr.readLine()) != null){
			String[] words = vecLine.split(" ");
		//	System.out.println(words[0]);
			if(dic.contains(words[0].split("\t")[0])){
				outputBw.write(vecLine + "\n");
			}
				
		}
		
		outputBw.close();
		vecBr.close();
		dicBr.close();
	}
}
