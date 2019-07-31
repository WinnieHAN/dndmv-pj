package nn;

import org.ejml.simple.SimpleMatrix;
import dic.Word;
public class Node {

	public final Word word;
	private boolean isLeaf = false;
	private boolean isStop = false;
	public final int dimension;

	private Node nextNode;
	
	//Input layer
	public Node(Word word, boolean isLeaf,
			boolean isStop){
		this.word = word;
		this.isLeaf = isLeaf;
		this.isStop = isStop;
		this.dimension = word.dim;

	}
	

	
	public void LinkNode(Node nextnode){
		this.nextNode = nextnode;
	}
	
	
	
}
