import codecs, os, sys
import numpy as np
from utils import *
import xlsxwriter

in_stc = os.path.join('../wsj_idx168', 'stc')
stc_emb = os.path.join('../wsj_idx168', 'stc_represent')
dic = os.path.join('../wsj_idx168', 'dic168.txt')
stc_w = os.path.join('../wsj_idx168', 'sentences_words')  # write
stc_2d_emb_txt = os.path.join('../wsj_idx168', 'tsne_Y')
stc_2d_emb_xlsx = os.path.join('../wsj_idx168', 'tsne_Y_xlsx.xlsx')


if __name__ == "__main__":
    print ("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    # print ("Running example on 2,500 MNIST digits...")
#     X = Math.loadtxt("mnist2500_X.txt");
#     labels = Math.loadtxt("mnist2500_labels.txt");
#     parent = np.random.randn(8, 10)

    # stcs_emb = np.loadtxt(stc_emb)

    X = np.loadtxt(stc_emb)
#     labels = [0,1,0,4,0,2,1,1]

#     Plot.scatter(Y[:,0], Y[:,1], 20, labels);
#     Plot.show();

    dic = open(dic).read().rstrip("\n").split("\n")
    dic2pairlist = [[j for j in i.rstrip("\t").split("\t")] for i in dic]
    dic2pair = {}
    for i in dic2pairlist:
        dic2pair[int(i[2])] = i[0]
    print(dic2pair)


    stcs = codecs.open(in_stc, 'r', encoding='utf8').read().rstrip('\n').split('\n')
    sent_w = [[dic2pair[int(j)] for j in i.rstrip('\t').split('\t')] for i in stcs]

    a = ['_'.join(i) for i in sent_w]
    b = '\n'.join(a)
    wr = open(stc_w, 'w')
    wr.write(b)
    wr.close()

    Y = tsne(X, 2, 10, 25.0)  # 5.0

    np.savetxt(stc_2d_emb_txt, Y)

    workbook = xlsxwriter.Workbook(stc_2d_emb_xlsx)  #
    worksheet = workbook.add_worksheet()

    for i in range(501):
        worksheet.write(i, 0, i)
        worksheet.write(i, 1, a[i])
        worksheet.write(i, 2, Y[i][0])
        worksheet.write(i, 3, Y[i][1])

    workbook.close()