package depparsing.learning.stats;

import gnu.trove.TDoubleArrayList;
import gnu.trove.TIntArrayList;

import java.util.ArrayList;

import learning.EM;
import learning.stats.TrainStats;
import util.LogSummer;
import data.Corpus;
import depparsing.constraints.ConstraintEnumerator;
import depparsing.constraints.PCType;
import depparsing.data.DepCorpus;
import depparsing.data.DepInstance;
import depparsing.model.DepModel;
import depparsing.model.DepSentenceDist;
import depparsing.util.SparseArray;

public class L1LMaxStats extends TrainStats<DepModel, DepSentenceDist> {
	
	class MaxMatrix {
		ConstraintEnumerator cstraints;
		TIntArrayList counts;
		TDoubleArrayList vals;
		public MaxMatrix(Corpus c, PCType child, PCType parent, boolean useRoot, boolean useDirection){
			cstraints = new ConstraintEnumerator((DepCorpus) c,child, parent, useRoot, useDirection);
			counts = new TIntArrayList();
			vals = new TDoubleArrayList();
		}

		private void addmax(int index, double posterior){
			if (index < 0) return;
			counts.ensureCapacity(index+1);
			vals.ensureCapacity(index+1);
			while(counts.size()<index+1) counts.add(0);
			while(vals.size()<index+1) vals.add(0);
			counts.set(index, counts.get(index)+1);
			vals.set(index, Math.max(posterior,vals.get(index)));						
		}
		
		public void updateEdge(DepInstance di, int child, int parent, double posterior){
			int index = cstraints.edge2cid(di, child, parent);
			addmax(index,posterior);
		}
		
		public void updateRoot(DepInstance di, int child, double posterior){
			int index = cstraints.root2cid(di, child);
			addmax(index,posterior);
		}
		
		public double l1normMax(){
			double sum = 0;
			for (int i = 0; i < vals.size(); i++) {
				if (counts.get(i) < minOccurences) continue;
				sum += 1;
			}
			return sum;
		}
		
		public double l1norm(){
			double sum = 0;
			for (int i = 0; i < vals.size(); i++) {
				if (counts.get(i) < minOccurences) continue;
				sum += vals.get(i);
			}
			return sum;
		}
		
		public double l2l1norm(){
			TIntArrayList[] idsPerChild = cstraints.cidAsMatrix();
			double result = 0;
			for (int child = 0; child < idsPerChild.length; child++) {
				double sum = 0;
				for (int i = 0; i < idsPerChild[child].size(); i++) {
					int id = idsPerChild[child].get(i);
					if(counts.get(id) < minOccurences) continue;
					sum += vals.get(id);
				}
				result += sum*sum;
			}
			return Math.sqrt(result);
		}
		
		public double l2l1normMax(){
			TIntArrayList[] idsPerChild = cstraints.cidAsMatrix();
			double result = 0;
			for (int child = 0; child < idsPerChild.length; child++) {
				double sum = 0;
				for (int i = 0; i < idsPerChild[child].size(); i++) {
					int id = idsPerChild[child].get(i);
					if(counts.get(id) < minOccurences) continue;
					sum += 1;
				}
				result += sum*sum;
			}
			return Math.sqrt(result);
		}
		
		/**
		 * gives the number of children that have some parent with at least minOccurences
		 * @return
		 */
		public int activeChildren(){
			TIntArrayList[] idsPerChild = cstraints.cidAsMatrix();
			int result = 0;
			for (int child = 0; child < idsPerChild.length; child++) {
				boolean done = false;
				double tmp = 0;
				for (int i = 0; i < idsPerChild[child].size(); i++) {
					int id = idsPerChild[child].get(i);
					if(counts.get(id) < minOccurences) continue;
					if (!done) result+=1;
					done = true;
					tmp+=vals.get(id);
					//break;
				}
//				System.out.println(cstraints.getParentType()+"->"+cstraints.getChildType()+": "+tmp);
			}
			System.out.println("---");
			return result;
		}

		public String name() {
			return "Child "+cstraints.getChildType()+
			" Parent "+cstraints.getParentType()+" "+(cstraints.useRoot?"useRoot":"no-Root")+
			" "+(cstraints.useDirection?"useDirection":"no-Direciont");
		}

		
	}
	
	ArrayList<MaxMatrix> measurements;
	
	// posteriors as array so that we can print them as a square. 
	SparseArray childPosParentPosPosteriors;

	int minOccurences = 10;

	public L1LMaxStats(String minOccurrences) {
		this.minOccurences = Integer.parseInt(minOccurrences);
		System.out.println("minword-" + minOccurrences);
	}

	public String getPrefix(){
		return "L1LMax Stats\n";
	}

	@Override
	public void eStepStart(DepModel model,EM em){              
		Corpus c = model.corpus;
		measurements = new ArrayList<MaxMatrix>();
//		measurements.add(new MaxMatrix(c,PCType.TAG, PCType.TAG,false,false));
//		measurements.add(new MaxMatrix(c,PCType.WORD, PCType.TAG,false,false));
//		measurements.add(new MaxMatrix(c,PCType.TAG, PCType.WORD,false,false));
//		measurements.add(new MaxMatrix(c,PCType.WORD, PCType.WORD,false,false));
		measurements.add(new MaxMatrix(c,PCType.TAG, PCType.TAG,true,false));
		measurements.add(new MaxMatrix(c,PCType.WORD, PCType.TAG,true,false));
		measurements.add(new MaxMatrix(c,PCType.TAG, PCType.WORD,true,false));
		measurements.add(new MaxMatrix(c,PCType.WORD, PCType.WORD,true,false));
//		measurements.add(new MaxMatrix(c,PCType.TAG, PCType.TAG,false,true));
//		measurements.add(new MaxMatrix(c,PCType.WORD, PCType.TAG,false,true));
//		measurements.add(new MaxMatrix(c,PCType.TAG, PCType.WORD,false,true));
//		measurements.add(new MaxMatrix(c,PCType.WORD, PCType.WORD,false,true));
//		measurements.add(new MaxMatrix(c,PCType.TAG, PCType.TAG,true,true));
//		measurements.add(new MaxMatrix(c,PCType.WORD, PCType.TAG,true,true));
//		measurements.add(new MaxMatrix(c,PCType.TAG, PCType.WORD,true,true));
//		measurements.add(new MaxMatrix(c,PCType.WORD, PCType.WORD,true,true));
		childPosParentPosPosteriors = new SparseArray(model.corpus.getNrTags(), model.corpus.getNrTags());
	}

	@Override
	public void eStepSentenceEnd(DepModel model, EM em, DepSentenceDist sd) {
		int numWords = sd.depInst.numWords;

		for(int childPosition = 0; childPosition < numWords; childPosition++) {
			double rootP = sd.getRootPosterior(childPosition);
			rootP = Math.exp(rootP);
			// FIXME get rid of mysum
			double mysum = rootP;
			for(MaxMatrix measurement: measurements){
				measurement.updateRoot(sd.depInst, childPosition, rootP);
			}
			for(int parentPosition = 0; parentPosition < numWords; parentPosition++) {
				if(childPosition == parentPosition) continue;
				
				double posterior = Double.NEGATIVE_INFINITY;
				for(int v = 0; v < sd.nontermMap.childValency; v++)
					posterior = LogSummer.sum(posterior, sd.getChildPosterior(childPosition, parentPosition, v));
				
				// work in log space for childPosPos array
				double oldM = childPosParentPosPosteriors.get(sd.depInst.postags[childPosition],sd.depInst.postags[parentPosition]);
				oldM = Math.max(posterior,oldM);
				childPosParentPosPosteriors.set(sd.depInst.postags[childPosition], sd.depInst.postags[parentPosition], oldM);
				// exp space for the measurements
				posterior = Math.exp(posterior);
				mysum+=posterior;
				for(MaxMatrix measurement: measurements){
					measurement.updateEdge(sd.depInst, childPosition, parentPosition, posterior);
				}
			}
			if (Math.abs(mysum-1)>0.001) System.out.println("Sum not 1: "+mysum);
		}
	}

	@Override
	public String printEndEStep(final DepModel model,EM em){
		StringBuffer s = new StringBuffer(); 
		s.append(printMaxMatrices(model));
		int iter = em.getCurrentIterationNumber();
		for (MaxMatrix measurement: measurements){
			String prefix = "Iter "+iter+" Average "+measurement.name();
			// compute l1 of measurement
			double l1norm = measurement.l1norm();
			double l1normMax = measurement.l1normMax();
			double numC = measurement.activeChildren();
			s.append(prefix+" L1Lmax"+String.format(" %.2f",measurement.l1norm()));
			s.append(String.format(" (%.2f / %.2f)\n",l1norm/numC,l1normMax/numC));
			// compute l2/l1 of measurement
			double l2l1norm = measurement.l2l1norm();
			double l2l1normMax = measurement.l2l1normMax();
			s.append(prefix+" L2L1Lmax"+String.format(" %.2f",measurement.l2l1norm()));
			s.append(String.format(" (%.2f / %.2f)\n",l2l1norm/Math.sqrt(numC),
													  l2l1normMax/Math.sqrt(numC)));
		}
		s.append("\n");
		return s.toString();
	}

	/**
	 * Print the matrix (child tag / parent tag) with the corresponding max values.
	 */
	public String printMaxMatrices(DepModel model){
		StringBuffer s = new StringBuffer();
		s.append("c/p\t");
		int nrPosTags = model.corpus.getNrTags();
		for(int parentTag = 0; parentTag < nrPosTags; parentTag++){
			s.append(model.corpus.tagAlphabet.index2feat.get(parentTag)+"\t");
		}
		s.append("\n");
		for(int childTag = 0; childTag < nrPosTags; childTag++){
			s.append(model.corpus.tagAlphabet.index2feat.get(childTag)+"\t");
			for(int parentTag = 0; parentTag < nrPosTags; parentTag++){
				s.append(String.format("%.2f",Math.exp(childPosParentPosPosteriors.get(childTag, parentTag))) + "\t");
			}
			s.append("\n");
		}
		return s.toString();
	}

//	private void printL1LmaxHelper(int bound1, 
//			SparseArray parentPosMaxes, SparseArray parentWordMaxes, 
//			SparseArray parentPosCounts, SparseArray parentWordCounts,
//			StringBuffer s, String itemType, int iter) {
//		double totalCountedChildItems = 0;
//		double totalL1ChildItemParentTag = 0;
//		double totalL1ChildItemParentWord = 0;
//		double totalL2L1ChildItemParentTag = 0;
//		double totalL2L1ChildItemParentWord = 0;
//		for(int childItem = 0; childItem < bound1; childItem++) {
//			totalCountedChildItems++;
//			double childItemParentTagSum = 0;       
//			for(int parentTag = 0; parentTag < nrPosTags; parentTag++) {
//				if (parentPosCounts.get(childItem,parentTag) < minOccurences) continue;
//				double max = Math.exp(parentPosMaxes.get(childItem,parentTag));
//				childItemParentTagSum += max;
//			}
//			totalL1ChildItemParentTag += childItemParentTagSum;
//			totalL2L1ChildItemParentTag += childItemParentTagSum*childItemParentTagSum;
//
//			double childItemParentWordSum = 0;      
//			for(int parentWord = 0; parentWord < nrWordTypes; parentWord++) {
//				if (parentWordCounts.get(childItem,parentWord) < minOccurences) continue;
//				double max = Math.exp(parentWordMaxes.get(childItem,parentWord));
//				childItemParentWordSum += max;
//			}
//			totalL1ChildItemParentWord += childItemParentWordSum;
//			totalL2L1ChildItemParentWord += childItemParentWordSum*childItemParentWordSum;
//
//		}
//
//		double[] counts = new double[]{totalL1ChildItemParentTag, parentPosMaxes.size(), totalL1ChildItemParentWord, parentWordMaxes.size(),
//				totalL2L1ChildItemParentTag, parentPosMaxes.size(), totalL2L1ChildItemParentWord, parentWordMaxes.size()};
//		ArrayMath.internalCombine(counts, 1/(double)totalCountedChildItems, CommonOps.multiply);
//
//		s.append("\nIter " + iter +  " Average Child" + itemType + "ParentTag L1LMax " +
//				String.format("%.2f", counts[0]) + " / " + 
//				String.format("%.2f", counts[1]));
//		s.append("\nIter " + iter +  " Average Child" + itemType + "ParentWord L1LMax " +
//				String.format("%.1f", counts[2])+ " / " +
//				String.format("%.1f", counts[3]));
//		s.append("\nIter " + iter +  " Average Child" + itemType + "ParentTag L2L1LMax " +
//				String.format("%.2f", Math.sqrt(counts[4])) + " / " + 
//				String.format("%.2f", counts[5]));
//		s.append("\nIter " + iter +  " Average Child" + itemType + "ParentWord L2L1LMax " +
//				String.format("%.1f", Math.sqrt(counts[6]))+ " / " +
//				String.format("%.1f", counts[7]));
//		s.append("\n");
//	}
}
