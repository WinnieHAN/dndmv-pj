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


def predictBatchMStep(NN_child, NN_decision, predicted_chd, predicted_dec, sentences_words, sentence_lens, sentences_posSeq, valency_size): # sentences_posSeq is the same with sentences_words
    NN_child.eval()
    NN_decision.eval()
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
    evalDataLoader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=False)  # batch_size=10
    evalDecisionDataLoader = torch.utils.data.DataLoader(decData, batch_size=10, shuffle=False)

    for ii, data in enumerate(evalDataLoader):
        for k in range(len(data) - 1, 0, -1):  # bubble sort
            for j in range(0, k):
                if sentence_lens[data[j, 0]] < sentence_lens[data[j + 1, 0]]:
                    data[j], data[j + 1] = data[j + 1].clone(), data[j].clone()

        sentence_index = data[:, 0]

        sts_temp = [sentences_posSeq[idx] for idx in sentence_index]
        sentences_temp_len = [len(idx) for idx in sts_temp]
        sentences_temp = []
        for idx in range(len(sts_temp)):
            sentences_temp = sentences_temp + sts_temp[idx]

        w_sts_temp = [sentences_words[idx] for idx in sentence_index]
        w_sentences_temp = []
        for idx in range(len(w_sts_temp)):
            w_sentences_temp = w_sentences_temp + sts_temp[idx]

        input_pos = data[:, 1]
        direction_left = data[:, 5]
        direction_right = data[:, 6]
        valence = data[:, 7]
        output_pos = data[:, 3]

        pred_y = NN_child.forwardChd(sentences=sentences_temp,  # sentences_len = [3,2] not [2, 3] sequence of decreasing
                                     w_sentences=w_sentences_temp,
                                     sentences_len=sentences_temp_len, h=input_pos,
                                     direction_left=direction_left, direction_right=direction_right, v=valence)
        predict_output_pos_logp = pred_y.data.numpy()[[kk for kk in range(len(sentence_index))], output_pos.numpy()]
        for i in range(len(sentence_index)):
            idx = data[i, 0]
            h_idx = data[i, 2]
            h = data[i, 1]
            c = data[i, 3]
            c_idx = data[i, 4]
            dir_right = direction_right[i]
            val = valence[i]
            pred_prob = predict_output_pos_logp[i]  # i
            wf.write(str(idx) + '\t' + str(c_idx) + '\t' + str(h_idx) + '\t' + str(val) + '\t' + str(pred_prob) + '\n')

    wf.close()
    wf = open(predicted_dec, 'w')
    for ii, data in enumerate(evalDecisionDataLoader):
        for k in range(len(data) - 1, 0, -1):  # bubble sort
            for j in range(0, k):
                if sentence_lens[data[j, 0]] < sentence_lens[data[j + 1, 0]]:
                    data[j], data[j + 1] = data[j + 1].clone(), data[j].clone()

        sentence_index = data[:, 0]

        sts_temp = [sentences_posSeq[idx] for idx in sentence_index]
        sentences_temp_len = [len(idx) for idx in sts_temp]
        sentences_temp = []
        for idx in range(len(sts_temp)):
            sentences_temp = sentences_temp + sts_temp[idx]

        w_sts_temp = [sentences_words[idx] for idx in sentence_index]
        w_sentences_temp = []
        for idx in range(len(w_sts_temp)):
            w_sentences_temp = w_sentences_temp + sts_temp[idx]

        input_pos = data[:, 1]
        direction_left = data[:, 5]
        direction_right = data[:, 6]
        valence = data[:, 7]
        output_pos = data[:, 3]
        pred_y = NN_decision.forwardChd(sentences=sentences_temp,
                                 # sentences_len = [3,2] not [2, 3] sequence of decreasing
                                        w_sentences=w_sentences_temp,
                                 sentences_len=sentences_temp_len, h=input_pos,
                                 direction_left=direction_left, direction_right=direction_right, v=valence)
        predict_output_pos_logp = pred_y.data.numpy()[[kk for kk in range(len(sentence_index))], output_pos.numpy()]  #pred_y.data.numpy()[:, output_pos.numpy()][0]
        for i in range(len(sentence_index)):
            idx = data[i, 0]
            h_idx = data[i, 2]
            c = data[i, 3]
            # dir_left = direction_left[i]
            dir_right = direction_right[i]
            val = valence[i]
            pred_prob = predict_output_pos_logp[i]
            wf.write(str(idx) + '\t' + str(h_idx) + '\t'  + str(dir_right) + '\t' + str(val) + '\t'+ str(c) + '\t' + str(pred_prob) + '\n')
    wf.close()

def childAndDecisionANNMstep_torch(NN_child=None, NN_decision=None, nn_epouches=1, batch_size_nn=10, child_rule_str='rule_0.txt', dec_rule_str='rule_0.txt',
                                   sentences_words_train=None, sentence_lens_train=None, sentences_tags_train=None, dic2Tag=None, nb_classes=None, valency_size=2, chd_nn=1, dec_nn=1,
                                   chd_lr=0.01, dec_lr=0.01):# trian and predict
    arr_child = np.loadtxt(child_rule_str,dtype='int')
    # arr_child = arr_chd[splitArr(arr_chd, 0)]
    arr_decision = np.loadtxt(dec_rule_str,dtype='int')
    # arr_decision = arr_dec[splitArr(arr_dec, 1)]
    NN_child.train(True)
    NN_decision.train(True)

    optimizers_child = torch.optim.SGD(filter(lambda p: p.requires_grad, NN_child.parameters()), lr=chd_lr, weight_decay=1e-4)
    optimizers_decision = torch.optim.SGD(filter(lambda p: p.requires_grad, NN_decision.parameters()), lr=dec_lr, weight_decay=1e-4)
    loss_func_child = torch.nn.CrossEntropyLoss()
    loss_func_decision = torch.nn.CrossEntropyLoss()
    trainChddataloader = torch.utils.data.DataLoader(arr_child,batch_size = batch_size_nn, shuffle=True)  # change it to dataloader of tree_lstm with same length sts
    trainDecDataloader = torch.utils.data.DataLoader(arr_decision, batch_size=batch_size_nn, shuffle=True)

    # train chd nn
    if chd_nn==1:
        print("begin training, training data shape:")
        running_loss = 0.0
        count = 0
        for iter in range(nn_epouches):
            print("Epouch: " + str(iter) + "\tof Epouches " + str(nn_epouches))
            for i, data in enumerate(trainChddataloader):
                NN_child.zero_grad()
                optimizers_child.zero_grad()
                for k in range(len(data) - 1, 0, -1): # bubble sort
                    for j in range(0, k):
                        if sentence_lens_train[data[j,0]] < sentence_lens_train[data[j + 1,0]]:
                            data[j], data[j+1] = data[j+1].clone(), data[j].clone()  # ??
                sentence_index = data[:,0]

                sts_temp = [sentences_tags_train[idx] for idx in sentence_index]
                sentences_temp_len = [len(idx) for idx in sts_temp]
                sentences_temp = []
                for idx in range(len(sts_temp)):
                    sentences_temp = sentences_temp + sts_temp[idx]

                w_sts_temp = [sentences_words_train[idx] for idx in sentence_index]
                w_sentences_temp = []
                for idx in range(len(w_sts_temp)):
                    w_sentences_temp = w_sentences_temp + sts_temp[idx]

                input_pos = data[:, 2]
                direction_left = data[:, 4]
                direction_right = data[:, 5]
                valence = data[:, 6]
                label = autograd.Variable(torch.from_numpy((autograd.Variable(data[:, 3])).data.numpy())) # ??
                predy_extroloss = NN_child.forward_chd_train(sentences=sentences_temp, # ?????
                                                             w_sentences=w_sentences_temp,
                                                    sentences_len=sentences_temp_len, h=input_pos,
                                                    direction_left=direction_left, direction_right=direction_right, v=valence)
                loss = loss_func_child(predy_extroloss[0], label) if isinstance(predy_extroloss, tuple) else loss_func_child(predy_extroloss, label)
                loss = loss + predy_extroloss[1] if isinstance(predy_extroloss, tuple) else loss
                loss.backward()
                optimizers_child.step()
                running_loss += loss.data[0]
                count += len(data)
                if i % 1000 == 999:
                    print('[%d, %5d] child loss:%.3f' % (iter + 1, i + 1, running_loss / count))
                    running_loss = 0
                    count = 0

    # train dec nn
    if dec_nn==1:
        running_loss = 0.0
        count = 0
        for iter in range(nn_epouches):
            print("Epouch: " + str(iter) + "\tof Epouches " + str(nn_epouches))
            for i, data in enumerate(trainDecDataloader):
                NN_decision.zero_grad()
                optimizers_decision.zero_grad()
                for k in range(len(data) - 1, 0, -1): # bubble sort
                    for j in range(0, k):
                        if sentence_lens_train[data[j,0]] < sentence_lens_train[data[j + 1,0]]:
                            data[j], data[j+1] = data[j+1].clone(), data[j].clone()  # ??
                sentence_index = data[:,0]

                sts_temp = [sentences_tags_train[idx] for idx in sentence_index]
                sentences_temp_len = [len(idx) for idx in sts_temp]
                sentences_temp = []
                for idx in range(len(sts_temp)):
                    sentences_temp = sentences_temp + sts_temp[idx]

                w_sts_temp = [sentences_words_train[idx] for idx in sentence_index]
                w_sentences_temp = []
                for idx in range(len(w_sts_temp)):
                    w_sentences_temp = w_sentences_temp + sts_temp[idx]

                input_pos = data[:, 2]
                direction_left = data[:, 4]
                direction_right = data[:, 5]
                valence = data[:, 6]
                label = autograd.Variable(torch.from_numpy((autograd.Variable(data[:, 3])).data.numpy())) # ??
                predy_extroloss = NN_decision.forward_chd_train(sentences=sentences_temp, # ?????
                                                                w_sentences=w_sentences_temp,
                                                    sentences_len=sentences_temp_len, h=input_pos,
                                                    direction_left=direction_left, direction_right=direction_right, v=valence)
                loss = loss_func_decision(predy_extroloss[0], label)
                loss = loss + predy_extroloss[1] if predy_extroloss[1] else loss
                loss.backward()
                optimizers_decision.step()
                running_loss += loss.data[0]
                count += len(data)
                if i % 1000 == 999:
                    print('[%d, %5d] decision loss :%.3f' % (iter + 1, i + 1, running_loss / count))
                    running_loss = 0
                    count = 0
####################################################################


def childAndDecisionANNMstep3(model, isCountTable, onlineEta, dic2Tag, isEarlyStop, res, res_decision,
                              res_val, res_decision_val, iterationOfEM, gramRun, nb_classes, nb_epoch, valenceSize, gateway, batch_size_nn):
    # global res
    # global res_decision
    # print("model.get_weights()[0][0] before:	" + model.get_weights()[0][0])
    rr = model.get_weights()
    print("model.get_weights()[0][0][0] before:", rr[0][0][0])
    print("model.get_weights()[1][0][0] before:", rr[1][0][0])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    #			0 p 1tag 2
    chd_tr_X = [[], [], [], [], [], []]

    chdSampleWeight = []

    desSampleWeight = []

    chd_tr_Y_child = []

    chd_tr_Y_decision = []

    start = timeit.default_timer()
    #   res = p * * dir val iscon child conandstop
    #   chd_tr_X = 0p 1left 2right 3val 4iscon 5ptag
    # print("res", res[0][0][0])
    # for i in range(len(res[0][0][0][0])):
    # 	print(res[0][0][0][0][i])
    # print("range(len(res))", range(len(res)))
    sum1 = 0

    if (isCountTable):
        for num in range(len(res)):
            if (num == len(res) - 1):
                eta = 1
                for i1 in range(len(res) - 1):
                    eta = eta - (onlineEta ^ (i1 + 1))
            else:
                eta = onlineEta ^ (len(res) - num - 1)

            for i in range(len(res[num])):  # c
                for j in range(len(res[num][0])):  # p
                    for k in range(len(res[num][0][0])):  # dir
                        for l in range(len(res[num][0][0][0])):  # val
                            sum1 = sum1 + res[num][i][j][k][l]
                            for countNum in range(res[num][i][j][k][l]):
                                chd_tr_X[0].append([j])
                                if (k < 0.5):
                                    chd_tr_X[1].append([1])
                                    chd_tr_X[2].append([0])
                                else:
                                    chd_tr_X[1].append([0])
                                    chd_tr_X[2].append([1])
                                chd_tr_X[3].append([l])
                                chd_tr_X[4].append([1])
                                chd_tr_X[5].append(dic2Tag[j])
                                chdSampleWeight.append(1 * eta)
                                desSampleWeight.append(1e-40)
                                chd_tr_Y_child.append(i)
                                chd_tr_Y_decision.append(0)
            # chd_tr_X = 0p 1left 2right 3val 4iscon 5ptag

            for i in range(len(res_decision[num])):  # p
                for j in range(len(res_decision[num][0])):  # dir
                    for k in range(len(res_decision[num][0][0])):  # val
                        for l in range(len(res_decision[num][0][0][0])):  # stoporcon
                            sum1 = sum1 + res_decision[num][i][j][k][l]
                            for countNum in range(res_decision[num][i][j][k][l]):
                                chd_tr_X[0].append([i])
                                if (k < 0.5):
                                    chd_tr_X[1].append([1])
                                    chd_tr_X[2].append([0])
                                else:
                                    chd_tr_X[1].append([0])
                                    chd_tr_X[2].append([1])
                                chd_tr_X[3].append([k])
                                chd_tr_X[4].append([0])
                                chd_tr_X[5].append(dic2Tag[j])
                                chdSampleWeight.append(1e-40)
                                desSampleWeight.append(1 * eta)
                                chd_tr_Y_child.append(0)
                                chd_tr_Y_decision.append(l)

        # valid data
        if (isEarlyStop):
            for i in range(len(res_val)):  # c
                for j in range(len(res_val[0])):  # p
                    for k in range(len(res_val[0][0])):  # dir
                        for l in range(len(res_val[0][0][0])):  # val
                            for countNum in range(res_val[i][j][k][l]):
                                chd_tr_X[0].append([j])
                                if (k < 0.5):
                                    chd_tr_X[1].append([1])
                                    chd_tr_X[2].append([0])
                                else:
                                    chd_tr_X[1].append([0])
                                    chd_tr_X[2].append([1])
                                chd_tr_X[3].append([l])
                                chd_tr_X[4].append([1])
                                chd_tr_X[5].append(dic2Tag[j])
                                chdSampleWeight.append(1)
                                desSampleWeight.append(1e-40)
                                chd_tr_Y_child.append(i)
                                chd_tr_Y_decision.append(0)
            # chd_tr_X = 0p 1left 2right 3val 4iscon 5ptag
            for i in range(len(res_decision_val)):  # p
                for j in range(len(res_decision_val[0])):  # dir
                    for k in range(len(res_decision_val[0][0])):  # val
                        for l in range(len(res_decision_val[0][0][0])):  # stoporcon
                            for countNum in range(res_decision_val[i][j][k][l]):
                                chd_tr_X[0].append([i])
                                if (k < 0.5):
                                    chd_tr_X[1].append([1])
                                    chd_tr_X[2].append([0])
                                else:
                                    chd_tr_X[1].append([0])
                                    chd_tr_X[2].append([1])
                                chd_tr_X[3].append([k])
                                chd_tr_X[4].append([0])
                                chd_tr_X[5].append(dic2Tag[j])
                                chdSampleWeight.append(1e-40)
                                desSampleWeight.append(1)
                                chd_tr_Y_child.append(0)
                                chd_tr_Y_decision.append(l)
    else:
        # 0c	1p	2dir	3val
        #   chd_tr_X = 0p 1left 2right 3val 4ischildrule 5ptag

        # sum1 = 10# !!!!!
        # print("res:",len(res))
        # print("shape(res):",np.shape(res))

        if (np.shape(res) == (4,)):
            res = [res]
        for i in range(len(res)):
            chd_tr_X[0].append([res[i][1]])
            if (res[i][2] == 0):
                chd_tr_X[1].append([1])  # isleft
                chd_tr_X[2].append([0])
            else:
                chd_tr_X[1].append([0])  # isleft
                chd_tr_X[2].append([1])
            chd_tr_X[3].append([res[i][3]])
            chd_tr_X[4].append([1])
            chd_tr_X[5].append([dic2Tag[int(res[i][1])]])
            chdSampleWeight.append(1)
            desSampleWeight.append(1e-40)
            chd_tr_Y_child.append(res[i][0])
            chd_tr_Y_decision.append(0)
        # 0p	1dir	2val	3stop
        # print("res[i]:",len(res_decision))
        if (np.shape(res_decision) == (4,)):
            res_decision = [res_decision]
        for i in range(len(res_decision)):
            chd_tr_X[0].append([res_decision[i][0]])
            if (res_decision[i][1] == 0):
                chd_tr_X[1].append([1])  # isleft
                chd_tr_X[2].append([0])
            else:
                chd_tr_X[1].append([0])  # isleft
                chd_tr_X[2].append([1])
            chd_tr_X[3].append([res_decision[i][2]])
            chd_tr_X[4].append([0])
            chd_tr_X[5].append([dic2Tag[int(res_decision[i][0])]])
            chdSampleWeight.append(1e-40)
            desSampleWeight.append(1)
            chd_tr_Y_child.append(0)
            chd_tr_Y_decision.append(res_decision[i][3])

        sum1 = sum1 + len(res)
        sum1 = sum1 + len(res_decision)
        if (iterationOfEM == 1):
            sum1 = 10

    stop = timeit.default_timer()
    #	print("running time:\t" + str(stop - start))
    #	print("Have prepared the training data!")

    # para init here!

    # onehot = gramRun.getMaxVal()

    #	print(len(chd_tr_X))

    X_train_0 = np.array(chd_tr_X[0])  # parent

    X_train_1 = np.array(chd_tr_X[1])  # left dir

    X_train_2 = np.array(chd_tr_X[2])  # right dir

    X_train_3 = np.array(chd_tr_X[3])  # valency

    X_train_4 = np.array(chd_tr_X[4])  # is Cont

    X_train_5 = np.array(chd_tr_X[5])  # parent tag

    y_train_child = np.array(chd_tr_Y_child)

    y_train_decision = np.array(chd_tr_Y_decision)

    X_test_0 = X_train_0

    X_test_1 = X_train_1

    X_test_2 = X_train_2

    X_test_3 = X_train_3

    X_test_4 = X_train_4

    X_test_5 = X_train_5

    Y_test_child = y_train_child  # np.array(chd_tr_Y)#[2, 1, 2])

    Y_test_decision = y_train_decision
    #	print(X_train_0.shape[0], 'train samples')
    #	print(X_test_0.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train_child = np_utils.to_categorical(y_train_child, nb_classes)
    y_train_decision = np_utils.to_categorical(y_train_decision, 2)
    # Y_test = np_utils.to_categorical(Y_test, nb_classes)
    chdW = np.array(chdSampleWeight)
    #	print(chdW.shape)
    desW = np.array(desSampleWeight)

    batch_size = batch_size_nn  # sum1
    print("batch_size", batch_size)
    # , shuffle='true'
    if (isEarlyStop):
        model.fit({'parent_input': X_train_0, 'parent_input_tag': X_train_5, 'dir_left_input': X_train_1,
                   'dir_right_input': X_train_2, 'val_input': X_train_3, 'cont_chd_input': X_train_4},

                  {'child_output': y_train_child, 'decision_output': y_train_decision},

                  epochs=nb_epoch, batch_size=batch_size, validation_split=(1 - gramRun.getValidPerc()),
                  callbacks=[early_stopping], sample_weight=[chdW,
                                                             desW])  # , sample_weight_mode = 'tempporal')  #validation_split=(1-gramRun.getValidPerc()), callbacks=[early_stopping],
    else:
        model.fit({'parent_input': X_train_0, 'parent_input_tag': X_train_5, 'dir_left_input': X_train_1,
                   'dir_right_input': X_train_2, 'val_input': X_train_3, 'cont_chd_input': X_train_4},

                  {'child_output': y_train_child, 'decision_output': y_train_decision},

                  epochs=nb_epoch, batch_size=batch_size, sample_weight=[chdW,
                                                                           desW])  # , sample_weight_mode = 'tempporal')  #validation_split=(1-gramRun.getValidPerc()), callbacks=[early_stopping],

    # print('Test score:', score[0])

    # print('Test accuracy:', score[1])

    # print(left.get_weights())

    X_0 = [[0], [0]]

    X_1 = [[0], [0]]

    X_2 = [[0], [0]]

    X_3 = [[0], [1]]
    X_4 = [[0], [1]]

    X_0 = []

    X_1 = []

    X_2 = []

    X_3 = []

    X_4 = []

    X_5 = []

    X_6 = []  # parent tag

    for i in range(0, nb_classes):

        for j in range(0, 2):  #

            for k in range(0, valenceSize):

                X_0 = X_0 + [[i]]  # parent

                X_6 = X_6 + [[dic2Tag[i]]]  # parent tag

                if j == 0:

                    X_1 = X_1 + [[1]]  # dir_chd

                    X_2 = X_2 + [[0]]  # valence_chd

                else:

                    X_1 = X_1 + [[0]]

                    X_2 = X_2 + [[1]]

                # X_3 = X_3 + [[j]]  #dir_des

                # X_4 = X_4 + [[k]]  #valence_des

                X_3 = X_3 + [[k]]

                X_5 = X_5 + [[1]]  # CONT

    probs = model.predict([np.array(X_0), np.array(X_6), np.array(X_1), np.array(X_2), np.array(X_3), np.array(X_5)])

    listSz = len(probs[0])

    #	print(listSz)

    java_list_chd = gateway.jvm.java.util.ArrayList()

    java_list_des = gateway.jvm.java.util.ArrayList()

    for i in range(0, listSz):
        # child
        java_pro = gateway.jvm.java.util.ArrayList()

        for j in range(0, nb_classes):
            java_pro.append(float(probs[0][i][j]))

        java_list_chd.append(java_pro)

        # decision

        java_pro_des = gateway.jvm.java.util.ArrayList()

        for j in range(0, 2):
            java_pro_des.append(float(probs[1][i][j]))

        java_list_des.append(java_pro_des)
    rr = model.get_weights()
    #	print("chd_tr_Y_child:", len(chd_tr_Y_child))
    print("model.get_weights()[0][0][0] after:", rr[0][0][0])
    print("model.get_weights()[1][0][0] after:", rr[1][0][0])
    # print("lr:", lr)
    gramRun.setChdAndDesPy(java_list_chd, java_list_des)



######################SOFT################################################
def childAndDecisionANNSoftMstep3():
    print("no code")

def predictSentence(NN_child, NN_decision, sentences_words, sentence_lens, sentences_posSeq, valency_size): # sentences_posSeq is the same with sentences_words
    NN_child.eval()          #, predicted_chd, predicted_dec#
    NN_decision.eval()
    data = np.array([(sts_index, h, h_index, c, chd_index, 1 if h_index > chd_index else 0,
                      1 if h_index < chd_index else 0, valence)
                     for sts_index in range(len(sentence_lens))
                     for h_index, h in enumerate([0, 2])
                     for chd_index, c in enumerate([0])
                     for valence in range(1) if not h_index == chd_index])

    evalDataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)  # batch_size=10
    wf = open('stc_represent', 'w')
    wf_stc = open('stc', 'w')
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

        sts_represetation = NN_child.forward_sts_represent(sentences=sentences_temp,  # sentences_len = [3,2] not [2, 3] sequence of decreasing
                                     sentences_len=sentences_temp_len, h=input_pos,
                                     direction_left=direction_left, direction_right=direction_right, v=valence)
        stc_r = torch.squeeze(sts_represetation).data.numpy()
        stc = sentences_temp
        line = '\t'.join([str(i) for i in stc_r])
        wf.write(line)
        wf.write('\n')
        line = '\t'.join([str(i) for i in stc])
        wf_stc.write(line)
        wf_stc.write('\n')
    wf.close()
    wf_stc.close()



