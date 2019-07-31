import os

print(os.getcwd())
nowPath = os.getcwd()
import numpy as np
np.random.seed()  # for reproducibility
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.utils import np_utils
from keras.layers import Dense, merge, LSTM, Input, Embedding
from keras.regularizers import l2  # , activity_l2
from keras.layers.advanced_activations import LeakyReLU
import timeit
from keras.layers import Input, LSTM, Dense, Reshape  # , TimeDistributedDense, merge,
from keras.layers.merge import *
from keras.layers.embeddings import Embedding
# from keras.layers import Merge
from keras.layers.core import RepeatVector
from keras.models import Model
from random import randint


def nnModel(dim_parent_word, nb_classes, dim_parent_tag, nb_classes_tag, dim1, dim_val, valenceSize, dim2, isChdDivid,
            dim_child_word, dim_child_tag, isChdInit, wordVec, tagVec, dic2Tag, dim3, isParInit, isParWordInit,
            isParInitGlove, dic2WordStr, isParTagInit, lr):  # dic2WordStr id used when using glove word vectors
    ######Model Build################

    parent_input = Input(shape=(1,), dtype='int32', name='parent_input')

    par_input = Embedding(embeddings_initializer="glorot_uniform", output_dim=dim_parent_word, input_dim=nb_classes, input_length=1)(
        parent_input)

    parent_input_tag = Input(shape=(1,), dtype='int32', name='parent_input_tag')

    par_input_tag = Embedding(embeddings_initializer="glorot_uniform", output_dim=dim_parent_tag, input_dim=nb_classes_tag, input_length=1)(parent_input_tag)

    dir_left_input = Input(shape=(1,), dtype='int32', name='dir_left_input')

    weight_l = np.empty([2, dim1])

    weight_l[0].fill(0)

    weight_l[1].fill(1)

    dir_l_input = Embedding(output_dim=dim1, input_dim=2, input_length=1, weights=[weight_l])(dir_left_input)

    dir_l_input.params = []

    dir_l_input.updates = []

    # dir_l_input = Flatten()(dir_l_input)

    dir_right_input = Input(shape=(1,), dtype='int32', name='dir_right_input')

    weight_r = np.empty([2, dim1])

    weight_r[0].fill(0)

    weight_r[1].fill(1)

    dir_r_input = Embedding(output_dim=dim1, input_dim=2, input_length=1, weights=[weight_r])(dir_right_input)

    dir_r_input.params = []

    dir_r_input.updates = []

    # dir_r_input = Flatten()(dir_r_input)

    val_input = Input(shape=(1,), dtype='int32', name='val_input')

    valency_input = Embedding(embeddings_initializer="glorot_uniform", output_dim=dim_val, input_dim=valenceSize, input_length=1)(
        val_input)

    cont_chd_input = Input(shape=(1,), dtype='int32', name='cont_chd_input')

    weight = np.empty([2, dim2])

    weight[0].fill(0)

    weight[1].fill(1)

    cont_chd_input_n = Embedding(output_dim=dim2, input_dim=2, input_length=1, weights=[weight],
                                 name='cont_chd_input_n')(
        cont_chd_input)

    # print(cont_chd_input_n.params)

    cont_chd_input_n.params = []

    cont_chd_input_n.updates = []

    # cont_chd_input_n.set_weights(weight)

    cont_chd_input_f = Flatten()(cont_chd_input_n)

    # begin merge

    child = concatenate([par_input, par_input_tag, valency_input])#merge([par_input, par_input_tag, valency_input], mode='concat')

    child = Flatten()(child)

    left = Dense(dim1, activation='relu')(child)

    right = Dense(dim1, activation='relu')(child)

    left = Reshape((1, dim1))(left)

    left = multiply([left, dir_l_input])#merge([left, dir_l_input], mode='mul')

    right = Reshape((1, dim1))(right)

    right = multiply([right, dir_r_input])#merge([right, dir_r_input], mode='mul')

    input_vec = add([left, right])  #merge([left, right], mode='sum')

    input_vec = Flatten()(input_vec)

    if (not isChdDivid):
        child = Dense(dim2, activation='relu')(input_vec)  # child dim

        child = multiply([child, cont_chd_input_f])#merge([child, cont_chd_input_f], mode='mul')  # final representation of child  #cont_chd_input_f is not necessily needed.

        child_loss = Dense(nb_classes, activation='softmax', name='child_output')(child)
    else:
        ###child word and tag###
        child_word = Dense(dim_child_word, activation='relu')(input_vec)

        child_tag = Dense(dim_child_tag, activation='relu')(input_vec)

        if (isChdInit):
            weight_child_word = np.zeros((nb_classes, dim_child_word))
            for i in range(len(weight_child_word)):
                if (wordVec.has_key(str(i))):
                    weight_child_word[i] = wordVec[str(i)]
                else:
                    weight_child_word[i] = np.random.randn(dim_child_word)
            weight_child_word = np.array(zip(*weight_child_word))  # ?
            weight_child_word_bias = np.zeros((nb_classes,))
            child_word = Dense(nb_classes, activation='linear', weights=[weight_child_word, weight_child_word_bias])(
                child_word)
            weight_child_tag = np.zeros((nb_classes_tag, dim_child_tag))
            for i in range(len(weight_child_tag)):
                if (tagVec.has_key(str(i))):
                    weight_child_tag[i] = tagVec[str(i)]
                else:
                    weight_child_tag[i] = np.random.randn(dim_child_tag)
            weight_child_tag = np.array(zip(*weight_child_tag))
            weight_child_tag_bias = np.zeros((nb_classes_tag,))
            child_tag = Dense(nb_classes_tag, activation='linear', weights=[weight_child_tag, weight_child_tag_bias])(
                child_tag)

        else:
            child_word = Dense(nb_classes, activation='linear')(child_word)  # init weight is here!
            child_tag = Dense(nb_classes_tag, activation='linear')(child_tag)  # init weight is here!

        weight_tag2word = np.zeros((nb_classes_tag, nb_classes))  # np.empty([2, dim1])
        weight_tag2word_bias = np.zeros((nb_classes,))
        # process
        for i in range(nb_classes):
            weight_tag2word[dic2Tag[i]][i] = 1

        child_tag = Dense(nb_classes, activation='linear', weights=[weight_tag2word, weight_tag2word_bias])(
            child_tag)  # matrixTag2Word*child_tag

        child_tag.params = []  # parameter here []

        child_tag.updates = []

        child = add([child_word, child_tag])  # merge([child_word, child_tag], mode='sum')

        weight_child_loss = np.zeros((nb_classes, nb_classes))
        weight_child_loss_bias = np.zeros((nb_classes,))
        # process
        for i in range(nb_classes):
            for j in range(nb_classes):
                if (i == j):
                    weight_child_loss[i][j] = 1

        child_loss = Dense(nb_classes, activation='softmax', name='child_output',
                           weights=[weight_child_loss, weight_child_loss_bias])(child)

        child_loss.params = []  # parameter here is [1, 0; 0, 1]

        child_loss.updates = []
    ###



    decision = Dense(dim3, activation='relu')(input_vec)

    # decision = Dense(50, activation='relu')(decision)

    decision_loss = Dense(2, activation='softmax', name='decision_output')(decision)

    model = Model(inputs=[parent_input, parent_input_tag, dir_left_input, dir_right_input, val_input, cont_chd_input],
                  outputs=[child_loss, decision_loss])

    if (isParInit):
        # initialize
        model_weights = model.get_weights()
        print('model_weights', len(model_weights))
        print('model_weights[0]', len(model_weights[0]))
        print('model_weights[0][0]', len(model_weights[0][0]))
        print('model_weights[1]', len(model_weights[1]))
        print('model_weights[1][0]', len(model_weights[1][0]))
        if (isParWordInit):
            parPosition = 0
            #	if(iterationOfEM > 1):
            #		parPosition = 1

            for k in range(len(model_weights)):
                if (len(model_weights[k]) == nb_classes):
                    parPosition = k
                    break
            print('parPosition:\t', parPosition)
            par_input_weight = model_weights[parPosition]  # (input_dim, output_dim)
            print('nb_classes', nb_classes)
            print('len(par_input_weight)', len(par_input_weight))
            #   use glove6B.50d.txt
            if (isParInitGlove):
                print('dic2WordStr\n', dic2WordStr)
                for i in range(len(par_input_weight)):
                    if (wordVec.has_key(dic2WordStr[i])):
                        print('dic2WordStr[i]', dic2WordStr[i])
                        par_input_weight[i] = wordVec[dic2WordStr[i]]
                        #   use my word3Vec.txt
            else:
                for i in range(len(par_input_weight)):
                    if (wordVec.has_key(str(i))):
                        par_input_weight[i] = wordVec[str(i)]

            model_weights[parPosition] = par_input_weight
        if (isParTagInit):
            parTagPosition = 0
            for k in range(len(model_weights)):
                if (len(model_weights[k]) == nb_classes_tag):
                    parTagPosition = k
                    break
            if (nb_classes == nb_classes_tag):
                parTagPosition = 1
            print('parTagPosition:\t', parTagPosition)
            par_tag_input_weight = model_weights[parTagPosition]  # (input_dim, output_dim)
            print('nb_classes_tag', nb_classes_tag)
            print('len(par_tag_input_weight)', len(par_tag_input_weight))

            for i in range(len(par_tag_input_weight)):
                if (tagVec.has_key(str(i))):
                    par_tag_input_weight[i] = tagVec[str(i)]
            model_weights[parTagPosition] = par_tag_input_weight

        model.set_weights(model_weights)

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)  # 5e-2 0.5 1e-6

    model.compile(loss={'child_output': 'categorical_crossentropy', 'decision_output': 'categorical_crossentropy'},
                  optimizer=sgd, sample_weight_mode="None")
    return model