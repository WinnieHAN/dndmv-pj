package depparsing.io;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import depparsing.model.DepProbMatrix;

public class DepProbWriter {
	
	private DepProbWriter() {
	}
	
	public static void writeToFile(String filename, DepProbMatrix matrix) throws IOException {
		ObjectOutputStream outstream = new ObjectOutputStream(new FileOutputStream(filename));
	    outstream.writeObject(matrix);
	    outstream.close();
	}
	
}
