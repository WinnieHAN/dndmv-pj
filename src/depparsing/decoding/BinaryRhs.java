package depparsing.decoding;

public class BinaryRhs extends RuleRhs {

	public final int C;
	public final int split;
	public final int childpos;
	
	public BinaryRhs(int h, int s, int b, int c, int pos) {
		super(h, b);
		C = c;
		split = s;
		childpos = pos;
	}
}
