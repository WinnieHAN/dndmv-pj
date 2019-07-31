import os, sys, gensim,logging
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        print(self.dirname)
        for fname in os.listdir(self.dirname):
            print(fname)
            for line in open(os.path.join(self.dirname, fname)):
                print(line.split())
                yield line.split()
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = MySentences("/home/hanwj/Java Workspace/DepGrammarNN4/data/forWord2Vec") # a memory-friendly iterator
#print(sentences)
vecIter = int(sys.argv[1])
model = gensim.models.Word2Vec(sentences, size=50, min_count=1,max_vocab_size=None,  iter=vecIter)
#model.save('/home/hanwj/Java Workspace/DepGrammarNN1/data/mymodel')
word2VecFile = open('/home/hanwj/Java Workspace/DepGrammarNN4/data/word2Vec_d50_iter' + str(vecIter) + '.txt', "w")
for i in range(99)+range(100,107):
    JohnSnow = len(model[str(i)])
    print(i)
for i in range(99)+range(100,107):#range(94) + range(95,98):
    word2VecFile.write(str(i) + '\t')
    #print(len(model[str(i)]))
    for j in range(len(model[str(i)])):
        word2VecFile.write(str(model[str(i)][j]) + '\t')
    word2VecFile.write('\n')
word2VecFile.close()
