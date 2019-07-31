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
sentences = MySentences("/home/hanwj/Java Workspace/DepGrammarNN4/data/forTag2Vec") # a memory-friendly iterator
#print(sentences)
vecIter = int(sys.argv[1])
model = gensim.models.Word2Vec(sentences, size=10, min_count=1,max_vocab_size=None,  iter=vecIter)
#model.save('/home/hanwj/Java Workspace/DepGrammarNN1/data/mymodel')
word2VecFile = open('/home/hanwj/Java Workspace/DepGrammarNN4/data/tag2Vec_d10_iter' + str(vecIter) + '.txt', "w")
print('======================Finished building vec.txt!!===================================')
for i in range(34):#tag 34 has not been train
    word2VecFile.write(str(i) + '\t')
    #print(len(model[str(i)]))
    for j in range(len(model[str(i)])):
        word2VecFile.write(str(model[str(i)][j]) + '\t')
    word2VecFile.write('\n')
word2VecFile.close()

#valid
#model.most_similar(positive=['woman', 'king'], negative=['man'])
#model.doesnt_match("breakfast cereal dinner lunch".split())
#model.similarity('woman', 'man')
#model['computer']
