package depparsing.util;

import util.LogSummer;
import gnu.trove.TIntDoubleHashMap;

public final class SparseArray {
	TIntDoubleHashMap[] array;
	
	// If the value is not contained in the array, return default value
	double defaultValue = Double.NEGATIVE_INFINITY;
	
	public SparseArray(int xSize, int ySize){
		array = (TIntDoubleHashMap[]) new TIntDoubleHashMap[xSize];
		for(int i =0; i < xSize; i++){
			array[i] = new TIntDoubleHashMap();
		}
	}

	public SparseArray(int xSize, int ySize, double defaultV){
		this(xSize,ySize);
		defaultValue = defaultV;
	}

	public void clear() {
		for(int i = 0; i < array.length; i++)
			array[i].clear();
	}

	public void set(int x, int y, double value){
		array[x].put(y, value);
	}

	public void increment(int x, int y){
		array[x].put(y, 1+get(x,y));
	}

	
	public double get(int x,int y){
		TIntDoubleHashMap v = array[x];
		if(v.containsKey(y)){
			return v.get(y);
		}else{
			return defaultValue;
		}
	}
	
	public void logIncrement(int x, int y, double value) {
		this.set(x, y, LogSummer.sum(this.get(x, y), value));
	}
	
	public int size(){
		int sum = 0;
		for(int i = 0; i < array.length;i++){
			sum += array[i].size();
		}
		return sum;
	}
}
