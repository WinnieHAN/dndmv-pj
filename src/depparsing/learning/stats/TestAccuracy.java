package depparsing.learning.stats;

import learning.EM;
import learning.stats.TrainStats;
import data.InstanceList;
import depparsing.decoding.CKYParser;
import depparsing.model.DepModel;
import depparsing.model.DepProbMatrix;
import depparsing.model.DepSentenceDist;

//Computes the accuracy after the M-Step
public class TestAccuracy extends TrainStats<DepModel,DepSentenceDist> {

        int printEvery;
        
        public TestAccuracy(String printEvery) {
                this.printEvery = Integer.parseInt(printEvery);
        }
        
        
        @Override
        public String getPrefix() {
                return "TestACC:";
        }
        
        public String printEndMStep(DepProbMatrix model, EM em){
        	DepProbMatrix smooth = new DepProbMatrix(model.corpus, model.nontermMap.decisionValency, model.nontermMap.childValency);
        	smooth.copyFrom(model);
        	smooth.backoff(-1e3);
        	smooth.logNormalize();
            if(em.getCurrentIterationNumber() % printEvery == 0){
            	StringBuffer sb = new StringBuffer();
            	//hanwj10.15
//            	for (InstanceList testData:model.corpus.testInstances){
//            		double[] accuracy = CKYParser.computeAccuracy(smooth, testData.instanceList, null);
//            		sb.append("Iter " + em.getCurrentIterationNumber() +" "+testData.name
//            		    + " direct accuracy " + accuracy[0]   
//            		    + " undirect accuracy " + accuracy[1] + "\n");
//            	}
            	return sb.toString().replaceAll("\n", "\n"+getPrefix());
            }
            else return "";
        }
        
}