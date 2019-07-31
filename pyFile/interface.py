from __future__ import print_function
from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters
from model import *
from utils import *
# from check import *
from trainer import *
from clustering import *
import sys
import time
import numpy as np
import os
from config import parse_args
from torch_py.model import AttnLSTM
# from keras.models import load_model



def py_interface_main(args):
    print(args)
    port = int(args.port)  # 23330
    accIdx = str(args.acc_idx)
    pre_accIdx = str(args.pre_acc_idx)
    time.sleep(1)
    time.sleep(args.sleep)
    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port))
    nowPath = os.getcwd()
    np.random.seed()  # for reproducibility

    isParDivid = True  # word and tag divid, it is always true
    isParInit = False
    # isParWordInit = True  # when it is true, dim_parent_word should be 50
    # isParTagInit = True  # when it is true, dim_parent_tag should be 10
    isParWordInit = False  # when it is true, dim_parent_word should be 50
    isParTagInit = False  # when it is true, dim_parent_tag should be 10

    isParInitGlove = False  # when it is true, dim_parent_word should be 50
    isChdDivid = True  # # when it is true, dim_child_word should be 50, dim_child_tag should be 10
    isChdInit = False  # when it is true, chdWordInit and chdTagInit are all be true.
    isEarlyStop = False
    torch.set_num_threads(20)
    isManualTaging = args.is_manually_tagging
    if (isParInitGlove):
        wordVec = readVectorFromFile('glove.6B.50d.txt', 51)  # 51 - 1 is dim of par word init
    elif(isParWordInit):
        tempPath = os.path.join(nowPath, 'data/word2Vec_d50_iter1000.txt')
        wordVec = readVectorFromFile(tempPath, 51)  # print(wordVec[str(1)])
    if (isChdInit or isParTagInit):
        tempPath = os.path.join(nowPath, 'data/tag2Vec_d10_iter1000.txt')
        tagVec = readVectorFromFile(tempPath, 11)

    # torch.manual_seed(1)
    is_pytorch = True
    maxIter = args.epochs  # 13

    if is_pytorch:
        childValency = 2  # for soft
        decisionValency = 2  # for soft
        chd_head_dim = dec_head_dim = 10
        chd_head_lstm_dim = dec_head_lstm_dim = int(args.chd_head_lstm_dim)#10
        chd_valency_dim = dec_valency_dim = 5
        chd_direct_dim = dec_direct_dim = int(args.chd_direct_dim)#10
        chd_lstm_hidden_dim = dec_lstm_hidden_dim = int(args.chd_lstm_hidden_dim) #10
        chd_dropout_p = dec_dropout_p = float(args.chd_dropout_p)#0.5
        chd_softmax_layer_dim = dec_softmax_layer_dim = int(args.chd_softmax_layer_dim)#10
        chd_nn = 1
        dec_nn = 0
        chd_lr = float(args.chd_lr)#0.01
        dec_lr = 0.01

    else:
        dim_parent_word = int(args.dim2) * 5
        dim_parent_tag = int(args.dim2)
        dim_val = 5
        dim_child_word = 50  # #dim_child_word + dim_child_tag = dim2
        dim_child_tag = 10  # 1 * dim
        lr = float(args.lr)
        dim1 = int(args.dim1)  # 5*dim int(sys.args[6])
        dim2 = 60  # 60#10*dim#5*dim
        dim3 = 10  # 2*dim
        nb_epoch = 1  # 30
        valenceSize = 2  # for viterbi



    gramRun = gateway.entry_point
    gramRun.paraSetting()
    print('gramRun.paraSetting()')
    abstractSts2File = "data/forWord2Vec/lang"+str(args.pascal_idx)+"_wordSts"
    sentences_train = open(os.path.join(nowPath, abstractSts2File)).read().rstrip("\n").split("\n")
    w2i, i2w, sentences_words_train, sentence_lens_train = sentence_loader(sentences_train)

    abstractSts2File_dev = "data/forWord2Vec/lang"+str(args.pascal_idx)+"_wordSts_dev"
    sentences_dev = open(os.path.join(nowPath, abstractSts2File_dev)).read().rstrip("\n").split("\n")
    sentences_words_dev, sentence_lens_dev = sentences2id(sentences_dev, w2i, 11)

    abstractSts2File_test = "data/forWord2Vec/lang"+str(args.pascal_idx)+"_wordSts_test"
    sentences_test = open(os.path.join(nowPath, abstractSts2File_test)).read().rstrip("\n").split("\n")
    sentences_words_test, sentence_lens_test = sentences2id(sentences_test, w2i, 11)

    sentences_words_test_all, sentence_lens_test_all = sentences2id(sentences_test, w2i, np.inf)
    # print('max_test_sts_len:  ', max_test_sts_len)
    # ----
    abstractSts2File = "data/forWord2Vec/lang"+str(args.pascal_idx)+"_abstractSts"
    sentences_train = open(os.path.join(nowPath, abstractSts2File)).read().rstrip("\n").split("\n")
    sentences_tags_train = [[int(j) for j in i.rstrip("\t").split('\t')] for i in sentences_train]
    sentence_tags_lens_train = [len(j) for j in sentences_tags_train]
    abstractSts2File_dev = "data/forWord2Vec/lang"+str(args.pascal_idx)+"_abstractSts_dev"
    sentences_dev = open(os.path.join(nowPath, abstractSts2File_dev)).read().rstrip("\n").split("\n")
    sentences_tags_dev = [[int(j) for j in i.rstrip("\t").split('\t')] for i in sentences_dev if len(i.rstrip("\t").split('\t')) < 11]
    sentence_tags_lens_dev = [len(j) for j in sentences_tags_dev]

    abstractSts2File_test = "data/forWord2Vec/lang"+str(args.pascal_idx)+"_abstractSts_test"
    sentences_test = open(os.path.join(nowPath, abstractSts2File_test)).read().rstrip("\n").split("\n")
    sentences_tags_test = [[int(j) for j in i.rstrip("\t").split('\t')] for i in sentences_test if len(i.rstrip("\t").split('\t')) < 11]
    sentence_tags_lens_test = [len(j) for j in sentences_tags_test]

    sentences_tags_test_all = [[int(j) for j in i.rstrip("\t").split('\t')] for i in sentences_test]
    sentence_tags_lens_test_all = [len(j) for j in sentences_tags_test_all]


    iterationOfEM = 1
    sigma_0 = 0.99
    sigma_k = 0  # 0  -0.05 #MUST be <-0.05, or modelConversionV2S must be changed
    sigma_e = 0
    n_clusters = args.n_clusters
    nb_classes = gramRun.nb_classes()
    nb_classes_tag = gramRun.nb_classes_tag()  # 35#nb_classes + 1, when use wordf or not

    if not isManualTaging==0:
        dic2Tag = loadData(os.path.join(nowPath, 'temp/newDic2tag' + pre_accIdx + '.txt'))
        for idx in range(len(dic2Tag)):
            dic2Tag[idx] = int(dic2Tag[idx])
        nb_classes_tag = n_clusters
        print("after manual taging, nb_classes_tag:\t" + str(nb_classes_tag))
    else:
        dic2Tag = gramRun.dic2Tag()
    dic2WordStr = gramRun.dic2WordStr()
    print(dic2WordStr)
    # saveData(dic2WordStr, os.path.join(nowPath, 'temp/dic2WordStr' + accIdx + '.txt'))

    if is_pytorch:
        model = AttnLSTM(head_dic_size=nb_classes, head_dim=chd_head_dim, head_tag_lstm_size=nb_classes, head_word_lstm_size=len(w2i), head_lstm_dim=chd_head_lstm_dim, valency_size=childValency, valency_dim=chd_valency_dim, direct_size=2,
        direct_dim=chd_direct_dim, nhid=chd_direct_dim, nclass=nb_classes, lstm_hidden_dim=chd_lstm_hidden_dim, dropout_p=chd_dropout_p, max_length=10, softmax_layer_dim=chd_softmax_layer_dim)
        model_dec = AttnLSTM(head_dic_size=nb_classes, head_dim=dec_head_dim, head_tag_lstm_size=nb_classes, head_word_lstm_size=len(w2i), head_lstm_dim=dec_head_lstm_dim, valency_size=2, valency_dim=dec_valency_dim, direct_size=2,
                         direct_dim=dec_direct_dim, nhid=dec_direct_dim, nclass=2, lstm_hidden_dim=dec_lstm_hidden_dim, dropout_p=dec_dropout_p, max_length=10, softmax_layer_dim=dec_softmax_layer_dim)
    else:
        onlineEta = gramRun.getOnlineEta()
        gramRun.setIsEarlyStop(isEarlyStop)
        model = nnModel(dim_parent_word, nb_classes, dim_parent_tag, nb_classes_tag, dim1, dim_val, valenceSize, dim2,
                    isChdDivid, dim_child_word, dim_child_tag, isChdInit, wordVec, tagVec, dic2Tag, dim3, isParInit,
                    isParWordInit, isParInitGlove, dic2WordStr, isParTagInit, lr)
        model_dec = nnModel(dim_parent_word, 2, dim_parent_tag, nb_classes_tag, dim1, dim_val, valenceSize, dim2,
                    isChdDivid, dim_child_word, dim_child_tag, isChdInit, wordVec, tagVec, dic2Tag, dim3, isParInit,
                    isParWordInit, isParInitGlove, dic2WordStr, isParTagInit, lr)

    while (iterationOfEM != maxIter):
        print("========== iter : ", iterationOfEM, "==========")
        print("Port:\t", port)
        Anneling_sigma = sigma_0 + (iterationOfEM - 1) * sigma_k
        Anneling_sigma = max(sigma_e, Anneling_sigma)
        isCountTable = False
        print("SoftEM_sigma is : ", Anneling_sigma)
        if (Anneling_sigma > 0.95):  # Anneling_sigma > 0.95
            train_chd = os.path.join(nowPath, 'temp/predicted_train_chd' + accIdx + '.txt')
            train_dec = os.path.join(nowPath, 'temp/predicted_train_dec' + accIdx + '.txt')
            dev_chd = os.path.join(nowPath, 'temp/predicted_val_chd' + accIdx + '.txt')
            dev_dec = os.path.join(nowPath, 'temp/predicted_val_dec' + accIdx + '.txt')
            test_chd = os.path.join(nowPath, 'temp/predicted_test_chd' + accIdx + '.txt')
            test_dec = os.path.join(nowPath, 'temp/predicted_test_dec' + accIdx + '.txt')
            test_all_chd = os.path.join(nowPath, 'temp/predicted_test_all_chd' + accIdx + '.txt')
            test_all_dec = os.path.join(nowPath, 'temp/predicted_test_all_dec' + accIdx + '.txt')
            gramRun.EStep(train_chd, train_dec, dev_chd, dev_dec, test_chd, test_dec, test_all_chd, test_all_dec)
            gramRun.MStepTxt()

            if is_pytorch:
                # train nn
                childAndDecisionANNMstep_torch(NN_child=model, NN_decision=model_dec, nn_epouches=1, batch_size_nn=int(args.batch_size_nn),
                                               child_rule_str=os.path.join(nowPath, 'temp/chdTemp' + accIdx + '.txt'),
                                               dec_rule_str=os.path.join(nowPath, 'temp/decTemp' + accIdx + '.txt'),
                                               sentences_words_train=sentences_words_train,
                                               sentence_lens_train=sentence_lens_train,
                                               sentences_tags_train=sentences_tags_train,
                                               dic2Tag=dic2Tag, nb_classes=nb_classes,
                                               valency_size=childValency,
                                               chd_nn=chd_nn,
                                               dec_nn=dec_nn,
                                               chd_lr=chd_lr,
                                               dec_lr=dec_lr
                                               )
                # predict
                predictBatchMStep(NN_child=model, NN_decision=model_dec,
                                  predicted_chd=train_chd,
                                  predicted_dec=train_dec,
                                  sentences_words=sentences_words_train,
                                  sentence_lens=sentence_lens_train,
                                  sentences_posSeq=sentences_tags_train,
                                  valency_size=decisionValency)
                predictBatchMStep(NN_child=model, NN_decision=model_dec,
                                  predicted_chd=dev_chd,
                                  predicted_dec=dev_dec,
                                  sentences_words=sentences_words_dev,
                                  sentence_lens=sentence_lens_dev,
                                  sentences_posSeq=sentences_tags_dev,
                                  valency_size=decisionValency)
                predictBatchMStep(NN_child=model, NN_decision=model_dec,
                                  predicted_chd=test_chd,
                                  predicted_dec=test_dec,
                                  sentences_words=sentences_words_test,
                                  sentence_lens=sentence_lens_test,
                                  sentences_posSeq=sentences_tags_test,
                                  valency_size=decisionValency)
                predictBatchMStep(NN_child=model, NN_decision=model_dec,
                                  predicted_chd=test_all_chd,
                                  predicted_dec=test_all_dec,
                                  sentences_words=sentences_words_test_all,
                                  sentence_lens=sentence_lens_test_all,
                                  sentences_posSeq=sentences_tags_test_all,
                                  valency_size=decisionValency)

            else:
                res_val = []
                res_decision_val = []
                tempPath = os.path.join(nowPath, 'temp/chdTemp' + accIdx + '.txt')
                res = LoadTrainData(tempPath)
                tempPath = os.path.join(nowPath, 'temp/decTemp' + accIdx + '.txt')
                res_decision = LoadTrainData(tempPath)
                # childAndDecisionANNMstep3(model, isCountTable, onlineEta, dic2Tag, isEarlyStop, res, res_decision,
                #                           res_val, res_decision_val, iterationOfEM, gramRun, nb_classes, nb_epoch,
                #                           valenceSize, gateway, args.batch_size_nn)  # childANNMstep()

            gramRun.setLastIsViterbiTrue()  # allTrees.clear
        else:
            pass
            # soft em

        # save the model(python, no use), dic2tag(python, result of classify), dic(java, no use, just for sure), childCountForComp(java)
        if not is_pytorch:
            save_model(model, os.path.join(nowPath, 'temp/my_model' + accIdx + '.h5'), is_pytorch)
            saveData(get_newDic2tag(model.get_weights()[0], args.clustering_linkage, n_clusters), os.path.join(nowPath, 'temp/newDic2tag' + accIdx + '.txt')) # gramRun.saveDicAndIdx()
            gramRun.saveCountForComp()
        else:
            torch.save(model, os.path.join(nowPath, 'temp/my_model_chd_' + accIdx + '_iter_' + str(iterationOfEM) + '.h5'))
            torch.save(model_dec, os.path.join(nowPath, 'temp/my_model_dec_' + accIdx + '_iter_' + str(iterationOfEM) + '.h5'))
            # saveData(dic2WordStr, os.path.join(nowPath, 'temp/dic2WordStr' + accIdx + '.txt'))

        iterationOfEM = iterationOfEM + 1

    gramRun.serverShutdown()

if __name__ == "__main__":
    args = parse_args(type=1)  # i is pyorch
    py_interface_main(args)
