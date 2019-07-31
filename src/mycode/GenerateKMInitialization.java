package mycode;

import static depparsing.globals.Constants.LEFT;
import static depparsing.globals.Constants.RIGHT;

import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.UnsupportedEncodingException;

import depparsing.data.DepCorpus;
import depparsing.model.DepProbMatrix;
import depparsing.model.KleinManningInitializer;

public class GenerateKMInitialization {

	/**
	 * @param args
	 * @throws IOException
	 * @throws FileNotFoundException
	 * @throws UnsupportedEncodingException
	 */
	public static void main(String[] args) throws UnsupportedEncodingException,
			FileNotFoundException, IOException {
		String corpusParams = "data/corpus-params.txt";
		DepCorpus corpus = new DepCorpus(corpusParams, 1, 10, Integer.MAX_VALUE);
		DepProbMatrix model = new DepProbMatrix(corpus, 2, 1);
		KleinManningInitializer.initNoah(model);

		FileWriter fw = new FileWriter("wsj10-km-init");
		String[] sortedTags = corpus.getAllTagsStrings();
		for (String tag : sortedTags) {
			int posNum = corpus.tagAlphabet.lookupObject(tag);
			fw.write("root(\"" + tag + "\") " + model.root[posNum] + "\n");

			for (int dir = 0; dir < 2; dir++) {
				for (int choice = 0; choice < 2; choice++) {
					for (int v = 0; v < model.nontermMap.decisionValency; v++) {
						fw.write("contend(" + (2 - choice) + ",\"" + tag
								+ "\"," + (dir + 1) * 100 + "," + (v + 1)
								+ ") " + model.decision[posNum][dir][v][choice]
								+ "\n");
					}
				}
			}

			assert model.nontermMap.childValency == 1;
			for (int i = 0; i < model.numTags; i++) {
				fw.write("dep(\"" + tag + "\",\""
						+ corpus.tagAlphabet.index2feat.get(i) + "\"," + 100
						+ ") " + model.child[i][posNum][LEFT][0] + "\n");
				fw.write("dep(\"" + tag + "\",\""
						+ corpus.tagAlphabet.index2feat.get(i) + "\"," + 200
						+ ") " + model.child[i][posNum][RIGHT][0] + "\n");
			}
		}
		fw.close();
	}

}
