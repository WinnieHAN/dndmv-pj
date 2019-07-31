package depparsing.decoding;

import java.util.ArrayList;
import java.util.ListIterator;

public class Tree {
  
  protected String _node;
  protected ArrayList<Tree> _children;
  
  // Constructs a node with no children.
  public Tree(String node) {
    _node = node;
    _children = new ArrayList<Tree>();
  }
 
  // Constructs a tree with children.
  public Tree(String node, ArrayList<Tree> children) {
    this(node);
    _children = children;
  }
  
  // Returns this tree's node value.
  public String getNode() {
    return _node;
  }
  
  // Returns this node's child trees.
  public ArrayList<Tree> getChildren() {
    return _children;
  }
  
  // Returns this tree's depth.
  public int getDepth() {
	  int depth = 1;
	  
	  ListIterator<Tree> childIter = _children.listIterator();
	  while(childIter.hasNext()) {
		depth = Math.max(depth, 1 + childIter.next().getDepth());  
	  }
	  
	  return depth;
  }
  
  // Returns the max number of children of any node in this tree.
  public int getMaxBranching() {
	  int maxBranching = _children.size();
	  
	  ListIterator<Tree> childIter = _children.listIterator();
	  while(childIter.hasNext()) {
		maxBranching = Math.max(maxBranching, childIter.next().getMaxBranching());  
	  }
	  
	  return maxBranching;
  }
  
  // Returns the length of the longest string in the tree.
  public int getMaxNodeSize() {
	  int maxNodeSize = _node.length();
	  
	  ListIterator<Tree> childIter = _children.listIterator();
	  while(childIter.hasNext()) {
		maxNodeSize = Math.max(maxNodeSize, childIter.next().getMaxNodeSize());  
	  }
	  
	  return maxNodeSize;
  }
  
  // Adds the given child tree to this node's children.
  public Tree addChild(Tree child) {
	  _children.add(child);
	  return child;
  }
  
  // Returns a parenthesized Lisp-style string representation of this tree.
  public String toString() {
    return toStringHelper("(" + _node, _node);
  }
  
  // Helper method for toString.
  // (Allows for omission of space character in front of root.)
  private String toStringHelper(String withChildren, String woChildren) {
    String tree = "";
    String end = "";
    if(_children.size() > 0) {
      tree += withChildren;
      end = ")";
    }
    else {
      tree += woChildren;
    }
    
    ListIterator<Tree> childIter = _children.listIterator();
    while(childIter.hasNext()) {
      Tree child = childIter.next();
      String childNode = child.getNode();
      tree += child.toStringHelper(" (" + childNode, " " + childNode);
    }
    
    return tree + end;
  }
  
}