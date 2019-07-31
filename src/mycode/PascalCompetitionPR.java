package mycode;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import java.util.regex.Pattern;

public class PascalCompetitionPR {

	/**
	 * @param args
	 *            0. corpus folder <br>
	 *            1. max-len <br>
	 *            2. init <br>
	 *            3. valence: [2/1, 2/2, 3/3] <br>
	 *            4. child-backoff <br>
	 *            5. PR type: S, AS <br>
	 *            6. PR strength <br>
	 *            7. sigma <br>
	 */
	public static void main(String[] args) throws Exception {
		Random r = new Random();
		String filename = Integer.toString(r.nextInt());

		String root = "pascal/" + args[0] + "/";
		String defaultArgs = "-corpus-params " + root + "corpus-params.txt "
				+ "-num-em-iters 100 -trainingType 9 -constrain-root ";

		ArrayList<String> maxlens = parseArgSet(args[1]);
		ArrayList<String> inits = parseArgSet(args[2]);
		ArrayList<String> valences = parseArgSet(args[3]);
		ArrayList<String> childBackoffs = parseArgSet(args[4]);
		ArrayList<String> prTypes = parseArgSet(args[5]);
		ArrayList<String> prStrengths = parseArgSet(args[6]);
		ArrayList<String> sigmas = parseArgSet(args[7]);

		int num = maxlens.size() * inits.size() * valences.size()
				* childBackoffs.size() * prTypes.size() * prStrengths.size()
				* sigmas.size();
		Process[] processes = new Process[num];
		int index = 0;
		ArrayList<String[]> allArgs = new ArrayList<String[]>();
		for (String maxlen : maxlens) {
			for (String init : inits) {
				for (String valence : valences) {
					int dvalence = 0, cvalence = 0;
					if (valence.equals("2/1")) {
						dvalence = 2;
						cvalence = 1;
					} else if (valence.equals("2/2")) {
						dvalence = 2;
						cvalence = 2;
					} else if (valence.equals("3/3")) {
						dvalence = 3;
						cvalence = 3;
					} else if (valence.equals("4/4")) {
						dvalence = 4;
						cvalence = 4;
					}
					for (String childBackoff : childBackoffs) {
						for (String prType : prTypes) {
							String pr = "";
							if (prType.equals("AS"))
								pr = " -use-fernando-constraints ";
							for (String prStrength : prStrengths) {
								for (String sigma : sigmas) {
									String[] curArgs = { maxlen, init,
											"(" + valence + ")", childBackoff,
											prType, prStrength, sigma };
									allArgs.add(curArgs);
									System.out.println("Starting: "
											+ Arrays.toString(curArgs));
									String logfile = root + filename + "."
											+ index + ".log";
									String tmpfile = root + filename + "."
											+ index + ".tmp";
									String cmd = "java -server -d64 -Xmx4g -cp pr2.jar:lib/* depparsing.programs.RunModel "
											+ defaultArgs
											+ " -max-sentence-size "
											+ maxlen
											+ " -model-init "
											+ init
											+ " -dvalency "
											+ dvalence
											+ " -cvalency "
											+ cvalence
											+ " -child-backoff "
											+ childBackoff
											+ pr
											+ " -constraint-strength "
											+ prStrength
											+ " -sigma "
											+ sigma
											+ " -tmpfile " + tmpfile;
									processes[index] = Runtime.getRuntime()
											.exec(cmd);
									FileOutputStream of = new FileOutputStream(
											logfile);
									new StreamGobbler(
											processes[index].getInputStream(),
											of).start();
									new StreamGobbler(
											processes[index].getErrorStream(),
											of).start();
									index++;
								}
							}
						}
					}
				}
			}
		}
		assert index == num;

		for (int i = 0; i < num; i++) {
			String[] a = allArgs.get(i);
			for (int j = 0; j < a.length; j++)
				System.out.print(a[j] + "\t");
			int ret = processes[i].waitFor();
			// System.out.println(ret);
			String tmpfile = root + filename + "." + i + ".tmp";
			File f = new File(tmpfile);
			Scanner sf = new Scanner(new BufferedReader(new FileReader(f)));
			for (int j = 0; j < 6; j++)
				System.out.print(sf.next() + "\t");
			System.out.println("");
			sf.close();
			f.delete();
		}
	}

	private static ArrayList<String> parseArgSet(String s) {
		ArrayList<String> re = new ArrayList<String>();
		if (s.startsWith("{")) {
			String[] args = Pattern.compile("[\\{\\},]").split(s.substring(1));
			re.addAll(Arrays.asList(args));
		} else {
			re.add(s);
		}
		return re;
	}

	static class StreamGobbler extends Thread {
		InputStream is;
		OutputStream os;

		StreamGobbler(InputStream is, OutputStream redirect) {
			this.is = is;
			this.os = redirect;
		}

		public void run() {
			try {
				PrintWriter pw = null;
				if (os != null)
					pw = new PrintWriter(os);

				InputStreamReader isr = new InputStreamReader(is);
				BufferedReader br = new BufferedReader(isr);
				String line = null;
				while ((line = br.readLine()) != null) {
					if (pw != null)
						pw.println(line);
					else
						System.out.println(line);
				}
				if (pw != null)
					pw.flush();
			} catch (IOException ioe) {
				ioe.printStackTrace();
			}
		}
	}
}
