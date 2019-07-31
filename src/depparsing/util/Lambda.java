package depparsing.util;

public class Lambda {

	public static interface ZeroVoid {
		public void call();
	}

	public static interface OneVoid<P> {
		public void call(P p);
	}

	public static interface TwoVoid<P1,P2> {
		public void call(P1 p1, P2 p2);
	}

	public static interface ThreeVoid<P1,P2,P3> {
		public void call(P1 p1, P2 p2, P3 p3);
	}

	public static interface FourVoid<P1,P2,P3,P4> {
		public void call(P1 p1, P2 p2, P3 p3, P4 p4);
	}

	public static interface FiveVoid<P1,P2,P3,P4,P5> {
		public void call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5);
	}
	
	public static interface Zero<T> {
		public T call();
	}

	public static interface One<T,P> {
		public T call(P p);
	}

	public static interface Two<T,P1,P2> {
		public T call(P1 p1, P2 p2);
	}

	public static interface Three<T,P1,P2,P3> {
		public T call(P1 p1, P2 p2, P3 p3);
	}

	public static interface Four<T,P1,P2,P3,P4> {
		public T call(P1 p1, P2 p2, P3 p3, P4 p4);
	}

	public static interface Five<T,P1,P2,P3,P4,P5> {
		public T call(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5);
	}
}
