package depparsing.model;

import static depparsing.globals.Constants.CHILD;
import static depparsing.globals.Constants.CHOICE;
import static depparsing.globals.Constants.CONT;
import static depparsing.globals.Constants.END;
import static depparsing.globals.Constants.LEFT;
import static depparsing.globals.Constants.RIGHT;

public class NonterminalMap {
	
	public final int numNontermTypes;
	public final int decisionValency;
	public final int childValency;
	public final int maxValency;
	
	private final int[] indicies;
	
	public NonterminalMap(int decisionValency, int childValency) {
		// Nonterminal types (not counting S)
		// end/continue: R0, ..., R(maxValency - 1), L0, ..., L(maxValency - 1)
		// child: Rc0, ..., Rc(maxValency - 1), Lc0, ..., Lc(maxValency - 1)
		this.maxValency = Math.max(decisionValency, childValency);
		this.numNontermTypes = 4*maxValency;
		
		// Valency of 1 -> no distinctions
		//            2 -> kids vs no kids
		//            3 -> 2 or more kids vs 1 kid vs no kids
		//            etc.
		this.decisionValency = decisionValency;
		this.childValency = childValency;
		
		this.indicies = new int[]{getNontermIndex(LEFT, CHOICE), getNontermIndex(LEFT, CHILD),
				getNontermIndex(RIGHT, CHOICE), getNontermIndex(RIGHT, CHILD)};
	}
	
	/**
	 * Returns the valence-0 index corresponding to the parameters.
	 * (Add 1 to this index to get the corresponding valence-1 index, and so on.)
	 */
	public int getNontermIndex(int leftOrRight, int choiceOrChild) {
		// To set an ordering on nonterminals, let
		// end/continue: L0, ..., L(maxValency - 1) map to 0, ..., maxValency - 1
		// child: Lc0, ..., Lc(childValency - 1) map to maxValency, ..., 2*maxValency - 1
		// (and similarly on the right)
		
		// 00->0, 01->m, 10->2m, 11->3m
		return (leftOrRight*2 + (1 - choiceOrChild))*maxValency; 
	}
	
	public String direction2String(int dir) {
		if(dir == LEFT) return "L";
		if(dir == RIGHT) return "R";
		assert(false) : "Called direction2String on an invalid direction: " + dir;
		return "";
	}
	
	public String choice2String(int choice) {
		if(choice == END) return "end";
		if(choice == CONT) return "continue";
		assert(false) : "Called choice2String on an invalid choice: " + choice;
		return "";
	}
	
	public String ruleType2String(int ruleType) {
		if(ruleType == CHILD) return "c";
		if(ruleType == CHOICE) return "";
		assert(false) : "Called ruleType2String on an invalid ruleType: " + ruleType;
		return "";
	}
	
	public String nontermIndex2String(int index) {
		String str = "";
		
		assert(index >= 0 && index < numNontermTypes) : "Called valenceIndex2String on an invalid index: " + index;
		
		if(isLeftChoiceIndex(index)) {
			str = direction2String(LEFT) + ruleType2String(CHOICE);
		} else if(isLeftChildIndex(index)) {
			str = direction2String(LEFT) + ruleType2String(CHILD);
			index -= maxValency;
		} else if(isRightChoiceIndex(index)) {
			str = direction2String(RIGHT) + ruleType2String(CHOICE);
			index -= 2*maxValency;
		} else {
			str = direction2String(RIGHT) + ruleType2String(CHILD);
			index -= 3*maxValency;
		}
		
		str += index;
		
		return str;
	}
	
	public boolean isUnaryIndex(int index) {
		return isLeftChoiceIndex(index) || isRightChoiceIndex(index);
	}
	
	public boolean isLeftChoiceIndex(int index) {	
		return (index >= indicies[0] && index < indicies[1]);
	}

	public boolean isRightChoiceIndex(int index) {
		return (index >= indicies[2] && index < indicies[3]);
	}
	
	public boolean isBinaryIndex(int index) {
		return isLeftChildIndex(index) || isRightChildIndex(index);
	}
	
	public boolean isLeftChildIndex(int index) {
		return (index >= indicies[1] && index < indicies[2]);
	}
	
	public boolean isRightChildIndex(int index) {
		return (index >= indicies[3] && index < indicies[3] + maxValency);
	}
}
