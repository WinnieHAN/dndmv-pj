import subprocess
from multiprocessing import Pool
import os
import time
import numpy as np
import sys
#sys.path.insert(0, '/home/hanwj/lndmv_split')
#from pyFile.runProgram import run_program_main


port = np.array([22411+x for x in range(243)])
acc_idx = np.array([x for x in range(243)])
pre_acc_idx = np.array([100000 for x in range(243)]+[100000 for x in range(243)])
initType = np.array([2 for x in range(243)] + [1 for x in range(243)])  #1 is km, 2 is good init.4 is random init. 5 is uniform init
idx =np.array([x for x in range(243)])
# java
wf = np.array([100000])
wf = wf.repeat(243)
onlineBatch = np.array([100000])
onlineBatch = onlineBatch.repeat(243)
corpusIdx = np.array([0])  # 184161
corpusIdx = corpusIdx.repeat(243)
stsLimitNum = np.array([100000])  # 189940  54944
stsLimitNum = stsLimitNum.repeat(243)
pascalIdx = np.array([i for i in range(1, 10)])
#"wsj_english"  "arabic", "basque", "czech", "danish", "dutch", "portuguese", "slovene", "swedish", "ctb9.0"}

# python
chd_lr = np.array([0.01])
chd_lr = chd_lr.repeat(81)

chd_dropout_p = np.array([0.5])
chd_dropout_p = chd_dropout_p.repeat(243)

chd_head_lstm_dim = np.array([10])
# chd_head_lstm_dim = np.tile(chd_head_lstm_dim, 3)
chd_head_lstm_dim = chd_head_lstm_dim.repeat(27)

chd_direct_dim = np.array([10])
# chd_direct_dim = np.tile(chd_direct_dim, 9)
chd_direct_dim = chd_direct_dim.repeat(90)

chd_lstm_hidden_dim = np.array([10])
# chd_lstm_hidden_dim = np.tile(chd_lstm_hidden_dim, 27)
chd_lstm_hidden_dim = chd_lstm_hidden_dim.repeat(30)

chd_softmax_layer_dim = np.array([20])
# chd_softmax_layer_dim = np.tile(chd_softmax_layer_dim, 81)
chd_softmax_layer_dim = chd_softmax_layer_dim.repeat(30)


def strfind(a,b):
    return a[a.index(b):].split(' ')[1]

def Thread(arg):
    cmd = arg
    # print(arg)
    fname = "0.log"
    file = open(fname, 'w')
    subprocess.call(cmd, shell=True, stdout=file)

def main():
    arglist = []
    st = int(sys.argv[1])
    print(st)
    end = int(sys.argv[2])
    print(end)
    for i in range(st, end):
        jcmd = str(port[i]) + " " + str(wf[i]) + " "+\
               str(onlineBatch[i])+" " + str(acc_idx[i]) +" "+str(corpusIdx[i])+" "+ str(stsLimitNum[i]) + " "+\
               str(initType[i]) + " "+str(pre_acc_idx[i]) + " " + str(pascalIdx[i]) + " -idx " + str(idx[i]) + " -javaOrPy java"
        pcmd = " --port " + str(port[i]) + " --acc_idx " + str(acc_idx[i]) + \
               " --chd_head_lstm_dim "+ str(chd_head_lstm_dim[i]) + " --chd_direct_dim " + str(chd_direct_dim[i])+\
               " --chd_lstm_hidden_dim " + str(chd_lstm_hidden_dim[i]) + " --chd_softmax_layer_dim " + str(chd_softmax_layer_dim[i])+\
               " --chd_dropout_p " + str(chd_dropout_p[i]) + " --chd_lr " + str(chd_lr[i]) +\
               " --pascal_idx " + str(pascalIdx[i]) + " -idx " + str(idx[i]) + " -javaOrPy py"
        cmd_all = "python pyFile/runProgram.py " + jcmd + " SPLITTAG " + pcmd
        arglist.append(cmd_all)

    p = Pool(1)#20
    p.map(Thread, arglist, chunksize=1)
    p.close()
    p.join()

if __name__ == '__main__':
    main()
