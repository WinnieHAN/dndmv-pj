package depparsing.learning.stats;

import java.util.Hashtable;

import learning.EM;
import learning.stats.TrainStats;
import util.ArrayMath;
import util.LogSummer;
import depparsing.data.DepCorpus;
import depparsing.model.DepModel;
import depparsing.model.DepSentenceDist;
import depparsing.util.HashUtils;
import depparsing.util.SparseArray;

public class ParseStats extends TrainStats<DepModel,DepSentenceDist> {

        // Accuracy by POS
        SparseArray childPosParentPosCorrect;
        SparseArray childPosParentPosError;
        
        // Distance and location histograms
        Hashtable<Integer, Double> edgeLengthHistogram;
        Hashtable<Integer, Double> rootLocationHistogram;
        
        int nrWordTypes;
        int nrPosTags;
        
        
        public String getPrefix(){
                return "ParseStats \n";
        }
        
        @Override
        public void eStepStart(DepModel model, EM em){
                nrWordTypes=model.corpus.getNrWordTypes();
                nrPosTags=model.corpus.getNrTags();
                childPosParentPosCorrect = new SparseArray(nrPosTags, nrPosTags);
                childPosParentPosError = new SparseArray(nrPosTags, nrPosTags);
                
                edgeLengthHistogram = new Hashtable<Integer, Double>();
                rootLocationHistogram = new Hashtable<Integer, Double>();
        }
        
        @Override
        public void eStepSentenceEnd(DepModel model,EM em,DepSentenceDist sd){
                int[] wordTokens = sd.depInst.words;
                int[] tags = sd.depInst.postags;
                
                for(int childPosition = 0; childPosition < wordTokens.length; childPosition++) {
                        for(int parentPosition = 0; parentPosition < wordTokens.length; parentPosition++) {
                                if(childPosition == parentPosition) continue;
                                
                                // Change L1Lmax for tag-tag, tag-word, word-word, and word-tag
                                
                                int childTag = tags[childPosition];
                                int parentTag = tags[parentPosition];
                                //int childWord = wordTokens[childPosition];
                                //int parentWord = wordTokens[parentPosition];
                                double posterior = Double.NEGATIVE_INFINITY;
                				for(int v = 0; v < sd.nontermMap.childValency; v++)
                					posterior = LogSummer.sum(posterior, sd.getChildPosterior(childPosition, parentPosition, v));
                				                                                                
                                // Accuracy by POS
                                if(sd.depInst.parents[childPosition] == parentPosition)
                                        childPosParentPosCorrect.logIncrement(childTag, parentTag, posterior);
                                else childPosParentPosError.logIncrement(childTag, parentTag, posterior);
                                
                                // Edge lengths
                                int separation = parentPosition - childPosition;
                                HashUtils.incrementHash(edgeLengthHistogram, separation, posterior);
                        }
                }
                
                // Root location
                for(int rootLoc = 0; rootLoc < wordTokens.length; rootLoc++)
                        HashUtils.incrementHash(rootLocationHistogram, rootLoc, sd.getRootPosterior(rootLoc));
                
        }
        
        public void printAccuracyByPOS(StringBuffer s, DepCorpus c) {
                // If making an error an average of once every 20 sentences, print this error
                double errCutoff = c.getNrOfTrainingSentences()/20;
                s.append("\n\nMost frequent tag-tag expected # of errors, accuracy for this tag-tag combination\n");
                for (int childPos = 0; childPos < nrPosTags; childPos++) {
                        for (int parentPos = 0; parentPos < nrPosTags; parentPos++) {
                                double correct = Math.exp(childPosParentPosCorrect.get(childPos, parentPos));
                                double error = Math.exp(childPosParentPosError.get(childPos, parentPos));
                                double accuracy = 100*correct / (correct + error);
                                if(error > errCutoff)
                                        s.append(c.tagAlphabet.index2feat.get(parentPos) + " -> " + c.tagAlphabet.index2feat.get(childPos) + "\t" + error + 
                                                        ",\t" + accuracy + "\n");
                        }               
                }
        }
        
        public void printEdgeLengths(StringBuffer s) {
                // Print edgeLengthHistogram
                s.append("\n\nExpected edge length histogram\n");
                s.append("Lengths\t\tPercentage\n");
                
                // Sort bins in increasing order, exponentiate values, and normalize
                int[] sortedBins = HashUtils.sortHash(edgeLengthHistogram);
                double[] expectations = expAndNormalize(edgeLengthHistogram, sortedBins);

                // Print values
                String dir = "left";
                for(int i = 0; i < sortedBins.length; i++) {
                        double percentage = 100*expectations[i];
                        double length = Math.abs(sortedBins[i]);
                        if(sortedBins[i] > 0)
                                dir = "right";
                        s.append(dir + " " + length + "\t" + percentage + "\n");
                }
        }
        
        public void printRootLocations(StringBuffer s) {
                // Print rootLocationHistogram
                s.append("\n\nExpected root location histogram\n");
                s.append("Locations\t\tPercentage\n");
                
                // Sort bins in increasing order, exponentiate values, and normalize
                int[] sortedBins = HashUtils.sortHash(rootLocationHistogram);
                double[] expectations = expAndNormalize(rootLocationHistogram, sortedBins);

                // Print values
                for(int i = 0; i < sortedBins.length; i++) {
                        double percentage = 100*expectations[i];
                        double location = Math.abs(sortedBins[i]);
                        s.append(location + "\t\t" + percentage + "\n");
                }               
        }
        
        private static double[] expAndNormalize(Hashtable<Integer, Double> hash, int[] sortedBins) {
                double[] expectations = new double[hash.size()];
                for(int i = 0; i < sortedBins.length; i++)
                        expectations[i] = Math.exp(hash.get(sortedBins[i]));
                
                double norm = ArrayMath.sum(expectations);

                for(int i = 0; i < sortedBins.length; i++)
                        expectations[i] /= norm;
                
                return expectations;
        }
        
        @Override
        public String printEndEStep(final DepModel model,EM em){
                StringBuffer s = new StringBuffer(); 
                
                printAccuracyByPOS(s,model.corpus);
                printEdgeLengths(s);
                printRootLocations(s);
                
                return s.toString();
        }
}