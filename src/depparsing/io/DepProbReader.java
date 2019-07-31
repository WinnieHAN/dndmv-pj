package depparsing.io;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import depparsing.model.DepProbMatrix;

public class DepProbReader {

	private DepProbReader() {
	}
	
	public static DepProbMatrix readFromFile(String filename)
	throws IOException, ClassNotFoundException {
		ObjectInputStream instream = new ObjectInputStream(new FileInputStream(filename));
		DepProbMatrix matrix = (DepProbMatrix) instream.readObject();
		instream.close();
		
		return matrix;
	}
}
