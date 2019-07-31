package depparsing.util;

import java.util.Arrays;
import java.util.Hashtable;
import java.util.Iterator;

import util.LogSummer;

public class HashUtils {

	public static void incrementHash(Hashtable<Integer,Double> hash, Integer key, Double increment) {
		if(hash.containsKey(key))
			hash.put(key, LogSummer.sum(hash.get(key), increment));
		else hash.put(key, increment);
	}

	public static int[] convertToArray(Hashtable<Integer, Double> hash) {
		Iterator<Integer> iter = hash.keySet().iterator();
		int[] array = new int[hash.size()];
		for(int i = 0; i < array.length; i++)
			array[i] = iter.next();
		
		return array;
	}
	
	public static int[] sortHash(Hashtable<Integer, Double> hash) {
		int[] array = convertToArray(hash);
		Arrays.sort(array);
		return array;
	}
	
}
