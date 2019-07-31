package depparsing.learning.stats;

import depparsing.decoding.CKYParser;
import depparsing.model.DepModel;
import depparsing.model.DepSentenceDist;
import learning.EM;
import learning.stats.TrainStats;

//Computes the accuracy after the M-Step
public class AccuracyStats extends TrainStats<DepModel,DepSentenceDist> {

        int printEvery;
        
        public AccuracyStats(String printEvery) {
                this.printEvery = Integer.parseInt(printEvery);
        }
        
        int[][] parses;
        public void emStart(DepModel model, EM em){
                parses = new int[model.corpus.getNrOfTrainingSentences()][];
        }
        
        @Override
        public String getPrefix() {
                return "ACCUR";
        }
        //hanwj10.15
        public String printEndMStep(DepModel model, EM em){
//                if(em.getCurrentIterationNumber() % printEvery == 0){
//                        double[] accuracy = CKYParser.computeAccuracy(model.params, model.corpus.trainInstances.instanceList, parses);
//                        return "Iter " + em.getCurrentIterationNumber() 
//                        + " direct accuracy " + accuracy[0] 
//                        + " undirect accuracy " + accuracy[1]
//                        + "\n";
//                }
                return "";
        }
        
}