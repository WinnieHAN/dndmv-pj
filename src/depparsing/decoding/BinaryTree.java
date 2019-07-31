package depparsing.decoding;

import java.util.ArrayList;

class TooManyChildrenException extends RuntimeException {
	private final static long serialVersionUID = -1L;

	public TooManyChildrenException(String message) {
		super(message);
	}
}

public class BinaryTree extends Tree {

	// Constructs a node with no children.
	public BinaryTree(String node) {
		super(node);
	}

	// Constructs a tree with children.
	public BinaryTree(String node, ArrayList<Tree> children) {
		super(node, children);
		if(children.size() > 2) {
			throw new TooManyChildrenException("Nodes in binary trees can take at most 2 children.");
		}
	}

	// Adds the given child tree to this node's children.
	public BinaryTree addChild(BinaryTree child) {
		if(_children.size() < 2) {
			_children.add(child);
		} else {
			throw new TooManyChildrenException("Nodes in binary trees can take at most 2 children.");
		}
		return child;
	}
	
	// Prints a vertical string representation of this tree.
	public void printBinaryTree() {
		String tree = printBinaryTreeHelper(this, "");
		tree = tree.substring(3);
		System.out.println(tree);
	}
	
	// Helper method for vertical printing;
	// allows a base string to be passed along as an accumulator.
	private String printBinaryTreeHelper(Tree t, String start) {
		String ls = "", rs = "";
		
		ArrayList<Tree> kids = t.getChildren();
		int numkids = kids.size();
		if(numkids == 2){
			ls = printBinaryTreeHelper(kids.get(1), start + "|    ");
			rs = printBinaryTreeHelper(kids.get(0), start + "     ");
			return "|_ " + t.getNode() + "\n" + start + ls + "\n" + start + rs;
		} else if(numkids == 1) {
			rs = printBinaryTreeHelper(kids.get(0), start + "     ");
			return "|_ " + t.getNode() + "\n" + start + rs;
		} else {
			return "|_ " + t.getNode();
		}
	}
}
