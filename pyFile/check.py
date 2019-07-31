import timeit
import numpy as np
from keras.utils import np_utils
import sys
from utils import splitArr
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
from keras.callbacks import EarlyStopping


def predictBatchMStep(childprob, decisionprob, NN_child, NN_decision, predicted_chd, predicted_dec, sentences_words, sentence_lens, sentences_posSeq, valency_size): # sentences_posSeq is the same with sentences_words
    wf = open(predicted_chd, 'w')
    data = np.array([(sts_index, h, h_index, c, chd_index, 1 if h_index > chd_index else 0,
                      1 if h_index < chd_index else 0, valence)
                     for sts_index in range(len(sentence_lens))
                     for h_index, h in enumerate(sentences_posSeq[sts_index])
                     for chd_index, c in enumerate(sentences_posSeq[sts_index])
                     for valence in range(valency_size) if not h_index == chd_index])
    decData = np.array(
        [(sts_index, h, h_index, c, -1, d, 1-d, valence)
         for sts_index in range(len(sentence_lens))
         for h_index, h in enumerate(sentences_posSeq[sts_index])
         for c in [0, 1]
         for d in [0, 1]
         for valence in range(valency_size)])
    evalDataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)  # batch_size=10
    evalDecisionDataLoader = torch.utils.data.DataLoader(decData, batch_size=1, shuffle=False)

    for ii, data in enumerate(evalDataLoader):
        for k in range(len(data) - 1, 0, -1):  # bubble sort
            for j in range(0, k):
                if sentence_lens[data[j, 0]] < sentence_lens[data[j + 1, 0]]:
                    data[j], data[j + 1] = data[j + 1].clone(), data[j].clone()

        sentence_index = data[:, 0]
        sts_temp = [sentences_words[idx] for idx in sentence_index]
        sentences_temp = []
        sentences_temp_len = [len(idx) for idx in sts_temp]
        for idx in range(len(sts_temp)):
            sentences_temp = sentences_temp + sts_temp[idx]

        input_pos = data[:, 1]
        direction_left = data[:, 5]
        direction_right = data[:, 6]
        valence = data[:, 7]

        output_pos = data[:, 3]

        for i in range(len(sentence_index)):
            idx = data[i, 0]
            h_idx = data[i, 2]
            h = data[i, 1]
            c = data[i, 3]
            c_idx = data[i, 4]
            dir_right = direction_right[i]
            val = valence[i]
            # pred_prob = predict_output_pos_logp[i]  # i
            pred_prob = childprob[c][h][dir_right][val]
            wf.write(str(idx) + '\t' + str(c_idx) + '\t' + str(h_idx) + '\t' + str(val) + '\t' + str(pred_prob) + '\n')

    wf.close()
    wf = open(predicted_dec, 'w')
    for ii, data in enumerate(evalDecisionDataLoader):
        for k in range(len(data) - 1, 0, -1):  # bubble sort
            for j in range(0, k):
                if sentence_lens[data[j, 0]] < sentence_lens[data[j + 1, 0]]:
                    data[j], data[j + 1] = data[j + 1].clone(), data[j].clone()

        sentence_index = data[:, 0]
        sts_temp = [sentences_words[idx] for idx in sentence_index]
        sentences_temp = []
        sentences_temp_len = [len(idx) for idx in sts_temp]
        for idx in range(len(sts_temp)):
            sentences_temp = sentences_temp + sts_temp[idx]
        input_pos = data[:, 1]
        direction_left = data[:, 5]
        direction_right = data[:, 6]
        valence = data[:, 7]
        output_pos = data[:, 3]

        for i in range(len(sentence_index)):
            idx = data[i, 0]
            h_idx = data[i, 2]
            c_idx = data[i, 4]
            h = data[i, 1]
            c = data[i, 3]
            # dir_left = direction_left[i]
            dir_right = direction_right[i]
            val = valence[i]
            pred_prob = decisionprob[c][h][dir_right][val]
            # pred_prob = predict_output_pos_logp[i]
            wf.write(str(idx) + '\t' + str(h_idx) + '\t' + str(dir_right) + '\t' + str(val) + '\t'+ str(c) + '\t' + str(pred_prob) + '\n')
    wf.close()

def checkpredictBatchMStep(chdin, decin, predicted_chd, predicted_dec, sentences_words):
    chd_sts = open(chdin).read().rstrip("\n").split("\n")  # idx c p v prob
    chd_sts_w = [[i for i in j.rstrip("\t").split("\t")] for j in chd_sts]
    childprob = torch.zeros(35, 35, 2, 2) # c p d v
    # wf = open(predicted_chd, 'w')
    for i in range(len(chd_sts)):
        for j in range(5):
            idx = int(chd_sts_w[i][0])
            c_idx= int(chd_sts_w[i][1])
            p_idx= int(chd_sts_w[i][2])
            v= int(chd_sts_w[i][3])
            prob= float(chd_sts_w[i][4])
            d = 0 if p_idx>c_idx else 1
            c = sentences_words[idx][c_idx]
            p = sentences_words[idx][p_idx]
            if (childprob[c][p][d][v] == 0):
                childprob[c][p][d][v] = prob
            else:
                if np.abs(childprob[c][p][d][v] - prob) > 0.0001:
                    print('error!!', childprob[c][p][d][v], '   ', prob)

    # wf.close()

    dec_sts = open(decin).read().rstrip("\n").split("\n")  # idx p d v s prob
    dec_sts_w = [[i for i in j.rstrip("\t").split("\t")] for j in dec_sts]
    decisionprob = torch.zeros(35, 2, 2, 2) #  p d v s
    # wf = open(predicted_dec, 'w')
    for i in range(len(dec_sts)):
        for j in range(6):
            idx = int(dec_sts_w[i][0])
            p_idx = int(dec_sts_w[i][1])
            d = int(dec_sts_w[i][2])
            v = int(dec_sts_w[i][3])
            s = int(dec_sts_w[i][4])
            prob = float(dec_sts_w[i][5])
            p = sentences_words[idx][p_idx]
            if (decisionprob[p][d][v][s] == 0):
                decisionprob[p][d][v][s] = prob
            else:
                if np.abs(decisionprob[p][d][v][s] - prob) > 0.0001:
                    print('error!!', decisionprob[p][d][v][s] , '   ', prob)

    writeprob(childprob, decisionprob, predicted_chd, predicted_dec, True)
    # wf.close()


def writeprob(childprob, decisionprob, predicted_chd, predicted_dec, ispdvs):
    # childprob = torch.zeros(35,35,2,2) # c p d v  ->
    # decisionprob = torch.transpose(decisionprob, 0, 1) # s p d v -> p d v s

    wf = open(predicted_chd, 'w')
    size = [35, 35, 2, 2]
    for i in range(int(size[0])):
        for j in range(int(size[1])):
            for k in range(int(size[2])):
                for l in range(int(size[3])):
                    wf.write(str(i)+'\t'+str(j)+'\t'+str(k)+'\t'+str(l)+'\t'+str(childprob[i][j][k][l])+'\n')

    wf.close()

    wf = open(predicted_dec, 'w')
    if ispdvs:
        size = [35, 2,2,2]
    else:
        size = [2, 35, 2, 2]
    for i in range(int(size[0])):
        for j in range(int(size[1])):
            for k in range(int(size[2])):
                for l in range(int(size[3])):
                    if ispdvs:
                        wf.write(str(i) + '\t' + str(j) + '\t' + str(k) + '\t' + str(l) + '\t' + str(
                        decisionprob[i][j][k][l]) + '\n')
                    else:
                        wf.write(str(j) + '\t' + str(k) + '\t' + str(l) + '\t' + str(i) + '\t' + str(
                        decisionprob[i][j][k][l]) + '\n')
    wf.close()


def childAndDecisionANNMstep_torch(NN_child=None, NN_decision=None, epouches=1, batch_size_nn=10, child_rule_str='rule_0.txt', dec_rule_str='rule_0.txt',
                                   sentences_words_train=None, sentence_lens_train=None, dic2Tag=None, nb_classes=None, valency_size=2):# trian and predict
    arr_child = np.loadtxt(child_rule_str,dtype='int')
    # arr_child = arr_chd[splitArr(arr_chd, 0)]
    arr_decision = np.loadtxt(dec_rule_str,dtype='int')
    # arr_decision = arr_dec[splitArr(arr_dec, 1)]
    batch_size_nn = 1  # !!!hanwj
    epouches = 1 # !!!hanwj

    # optimizers_child = torch.optim.SGD(filter(lambda p: p.requires_grad, NN_child.parameters()), lr = 0.01, weight_decay=1e-4)
    # optimizers_decision = torch.optim.SGD(filter(lambda p: p.requires_grad, NN_decision.parameters()), lr = 0.01, weight_decay=1e-4)
    # loss_func_child = torch.nn.CrossEntropyLoss()
    # loss_func_decision = torch.nn.CrossEntropyLoss()
    trainChddataloader = torch.utils.data.DataLoader(arr_child,batch_size = batch_size_nn, shuffle=True)  # change it to dataloader of tree_lstm with same length sts
    trainDecDataloader = torch.utils.data.DataLoader(arr_decision, batch_size=batch_size_nn, shuffle=True)

    childcount = torch.zeros(35, 35, 2, 2)  # c p d v
    decisioncount = torch.zeros(2, 35, 2, 2)  # s p d v


    # train chd nn
    print("begin training, training data shape:")
    running_loss = 0.0
    count = 0
    for iter in range(epouches):
        print("Epouch: " + str(iter) + "\tof Epouches " + str(epouches))
        for i, data in enumerate(trainChddataloader):
            # NN_child.zero_grad()
            # optimizers_child.zero_grad()
            for k in range(len(data) - 1, 0, -1): # bubble sort
                for j in range(0, k):
                    if sentence_lens_train[data[j,0]] < sentence_lens_train[data[j + 1,0]]:
                        data[j], data[j+1] = data[j+1].clone(), data[j].clone()  # ??
            sentence_index = data[:,0]
            sts_temp = [sentences_words_train[idx] for idx in sentence_index]
            sentences_temp_len = [len(idx) for idx in sts_temp]
            sentences_temp = []
            for idx in range(len(sts_temp)):
                sentences_temp = sentences_temp + sts_temp[idx]
            input_pos = data[:, 2]
            direction_left = data[:, 4]
            direction_right = data[:, 5]
            valence = data[:, 6]
            label = data[:, 3]

            childcount[label[0]][input_pos[0]][direction_right[0]][valence[0]] += 1
            # label = autograd.Variable(torch.from_numpy((autograd.Variable(data[:, 3])).data.numpy())) # ??
            # pred_y = NN_child.forward_chd_train(sentences=sentences_temp, # ?????
            #                                     sentences_len=sentences_temp_len, h=input_pos,
            #                                     direction_left=direction_left, direction_right=direction_right, v=valence)
            # loss = loss_func_child(pred_y, label)
            # loss.backward()
            # optimizers_child.step()
            # running_loss += loss.data[0]
            # count += len(data)
            # if i % 1000 == 999:
            #     print('[%d, %5d] child loss:%.3f' % (iter + 1, i + 1, running_loss / count))
            #     running_loss = 0
            #     count = 0

    # train dec nn
    running_loss = 0.0
    count = 0
    for iter in range(epouches):
        print("Epouch: " + str(iter) + "\tof Epouches " + str(epouches))
        for i, data in enumerate(trainDecDataloader):
            # NN_decision.zero_grad()
            # optimizers_decision.zero_grad()
            for k in range(len(data) - 1, 0, -1): # bubble sort
                for j in range(0, k):
                    if sentence_lens_train[data[j,0]] < sentence_lens_train[data[j + 1,0]]:
                        data[j], data[j+1] = data[j+1].clone(), data[j].clone()  # ??
            sentence_index = data[:,0]
            sts_temp = [sentences_words_train[idx] for idx in sentence_index]
            sentences_temp_len = [len(idx) for idx in sts_temp]
            sentences_temp = []
            for idx in range(len(sts_temp)):
                sentences_temp = sentences_temp + sts_temp[idx]
            input_pos = data[:, 2]
            direction_left = data[:, 4]
            direction_right = data[:, 5]
            valence = data[:, 6]

            label = data[:, 3]
            decisioncount[label[0]][input_pos[0]][direction_right[0]][valence[0]] += 1

            # label = autograd.Variable(torch.from_numpy((autograd.Variable(data[:, 3])).data.numpy())) # ??
            # pred_y = NN_decision.forward_chd_train(sentences=sentences_temp, # ?????
            #                                     sentences_len=sentences_temp_len, h=input_pos,
            #                                     direction_left=direction_left, direction_right=direction_right, v=valence)
            # loss = loss_func_decision(pred_y, label)
            # loss.backward()
            # optimizers_decision.step()
            # running_loss += loss.data[0]
            # count += len(data)
            # if i % 1000 == 999:
            #     print('[%d, %5d] child decision loss :%.3f' % (iter + 1, i + 1, running_loss / count))
            #     running_loss = 0
            #     count = 0
    childprob, decisionprob = normPara(childcount, decisioncount, smooth=1e-4)
    return childprob, decisionprob
####################################################################
def normPara(childcount, decisioncount, smooth):

    childcomp = torch.squeeze(torch.sum(childcount, 0), 0)
    decisioncomp = torch.squeeze(torch.sum(decisioncount, 0))

    childprob = torch.zeros(35,35,2,2) # c p d v
    decisionprob = torch.zeros(2,35,2,2) # s p d v

    size = childcount.size()
    for i in range(int(size[0])):
        for j in range(int(size[1])):
            for k in range(int(size[2])):
                for l in range(int(size[3])):
                    if childcount[i][j][k][l] > 0:
                        childprob[i][j][k][l] = (float(childcount[i][j][k][l])+smooth)/(float(childcomp[j][k][l])+smooth*size[0])
                    else:
                        childprob[i][j][k][l] = (smooth)/(float(childcomp[j][k][l])+smooth*size[0])



    size = decisioncount.size()
    for i in range(int(size[0])):
        for j in range(int(size[1])):
            for k in range(int(size[2])):
                for l in range(int(size[3])):
                    if decisioncount[i][j][k][l] > 0:
                        decisionprob[i][j][k][l] = (float(decisioncount[i][j][k][l])+smooth)/(float(decisioncomp[j][k][l])+smooth*2)
                    else:
                        decisionprob[i][j][k][l] = smooth/(float(decisioncomp[j][k][l])+smooth*2)



    return childprob, decisionprob
######################SOFT################################################
def childAndDecisionANNSoftMstep3():
    print("no code")

