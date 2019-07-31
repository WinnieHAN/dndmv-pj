package mycode;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.regex.Pattern;

public class PascalCompetitionUtil {
	public static void main(String[] args) throws IOException {
		String dir = "pascal/danish/";
		mergeTest(dir + "testparsetest", dir + "danish_cdt_test.unlabelled",
				dir + "tmp");
	}

	public static void mergeTest(String fnResult, String fnOriginal,
			String fnOutput) throws IOException {
		Scanner result = new Scanner(new BufferedReader(
				new FileReader(fnResult)));
		Scanner original = new Scanner(new BufferedReader(new FileReader(
				fnOriginal)));
		FileWriter fw = new FileWriter(fnOutput);
		Pattern whitespace = Pattern.compile("\\s+");
		while (original.hasNext()) {
			ArrayList<String[]> olines = new ArrayList<String[]>();
			String oln = original.nextLine();
			while (oln.trim().length() != 0) {
				String[] os = whitespace.split(oln);
				olines.add(os);
				if (original.hasNext())
					oln = original.nextLine();
				else
					break;
			}

			ArrayList<String[]> rlines = new ArrayList<String[]>();
			String rln = result.nextLine();
			while (rln.trim().length() != 0) {
				String[] rs = whitespace.split(rln);
				rlines.add(rs);
				if (result.hasNext())
					rln = result.nextLine();
				else
					break;
			}

			int j = 0;
			for (int i = 0; i < olines.size(); i++) {
				if (!rlines.isEmpty()
						&& olines.get(i)[4].equals(rlines.get(j)[4])) {
					if (!olines.get(i)[1].equals(rlines.get(j)[1]))
						System.out
								.println("Warning: different word: "
										+ olines.get(i)[1] + " vs. "
										+ rlines.get(j)[1]);
					j++;
					if (j == rlines.size())
						break;
				} else {
					for (int k = 0; k < rlines.size(); k++) {
						int index = Integer.parseInt(rlines.get(k)[6]);
						if (index - 1 >= i)
							rlines.get(k)[6] = Integer.toString(index + 1);
					}
				}
			}

			j = 0;
			for (int i = 0; i < olines.size(); i++) {
				if (!rlines.isEmpty()
						&& olines.get(i)[4].equals(rlines.get(j)[4])) {
					olines.get(i)[7] = rlines.get(j)[6];
					j++;
					if (j == rlines.size())
						break;
				}
			}

			for (String[] line : olines) {
				fw.write(line[0]);
				for (int i = 1; i < line.length; i++) {
					fw.write("\t" + line[i]);
				}
				fw.write("\n");
			}
			fw.write("\n");
		}
		fw.close();
		result.close();
		original.close();
	}
}
