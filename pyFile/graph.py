from __future__ import print_function
from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters
def childAndDecisionANNMstep3():

	global res
	global res_decision
	#print("model.get_weights()[0][0] before:	" + model.get_weights()[0][0])
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
	#print("res", res[0][0][0])
	# for i in range(len(res[0][0][0][0])):
	# 	print(res[0][0][0][0][i])
	# print("range(len(res))", range(len(res)))
	sum1 = 0

	if(isCountTable):
		for num in range(len(res)):
			if(num == len(res) - 1):
				eta = 1
				for i1 in range(len(res) - 1):
					eta = eta - (onlineEta^(i1 + 1))
			else:
				eta = onlineEta^(len(res) - num - 1)

			for i in range(len(res[num])):#c
				for j in range(len(res[num][0])):#p
					for k in range(len(res[num][0][0])):#dir
						for l in range(len(res[num][0][0][0])):#val
							sum1 = sum1 + res[num][i][j][k][l]
							for countNum in range(res[num][i][j][k][l]):
								chd_tr_X[0].append([j])
								if(k<0.5):
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
			#chd_tr_X = 0p 1left 2right 3val 4iscon 5ptag 

			for i in range(len(res_decision[num])):#p
				for j in range(len(res_decision[num][0])):#dir
					for k in range(len(res_decision[num][0][0])):#val
						for l in range(len(res_decision[num][0][0][0])):#stoporcon
							sum1 = sum1 + res_decision[num][i][j][k][l]
							for countNum in range(res_decision[num][i][j][k][l]):
								chd_tr_X[0].append([i])
								if(k<0.5):
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

		#valid data
		if(isEarlyStop):
			for i in range(len(res_val)):#c
				for j in range(len(res_val[0])):#p
					for k in range(len(res_val[0][0])):#dir
						for l in range(len(res_val[0][0][0])):#val
							for countNum in range(res_val[i][j][k][l]):
								chd_tr_X[0].append([j])
								if(k<0.5):
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
			#chd_tr_X = 0p 1left 2right 3val 4iscon 5ptag 
			for i in range(len(res_decision_val)):#p
				for j in range(len(res_decision_val[0])):#dir
					for k in range(len(res_decision_val[0][0])):#val
						for l in range(len(res_decision_val[0][0][0])):#stoporcon
							for countNum in range(res_decision_val[i][j][k][l]):
								chd_tr_X[0].append([i])
								if(k<0.5):
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

		#sum1 = 10# !!!!!
		#print("res:",len(res))
		#print("shape(res):",np.shape(res))

		if(np.shape(res) == (4,)):
			res = [res]
		for i in range(len(res)):
			chd_tr_X[0].append([res[i][1]])
			if(res[i][2] == 0):
				chd_tr_X[1].append([1])#isleft
				chd_tr_X[2].append([0])
			else:
				chd_tr_X[1].append([0])#isleft
				chd_tr_X[2].append([1])
			chd_tr_X[3].append([res[i][3]])
			chd_tr_X[4].append([1])
			chd_tr_X[5].append([dic2Tag[int(res[i][1])]])
			chdSampleWeight.append(1)
			desSampleWeight.append(1e-40)
			chd_tr_Y_child.append(res[i][0])
			chd_tr_Y_decision.append(0)
			#0p	1dir	2val	3stop
		#print("res[i]:",len(res_decision))
		if(np.shape(res_decision) == (4,)):
			res_decision = [res_decision]
		for i in range(len(res_decision)):
			chd_tr_X[0].append([res_decision[i][0]])
			if(res_decision[i][1] == 0):
				chd_tr_X[1].append([1])#isleft
				chd_tr_X[2].append([0])
			else:
				chd_tr_X[1].append([0])#isleft
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
		if(iterationOfEM == 1):
			sum1 = 10

	stop = timeit.default_timer()
#	print("running time:\t" + str(stop - start))
#	print("Have prepared the training data!")

	# para init here!

	onehot = gramRun.getMaxVal()

#	print(len(chd_tr_X))

	X_train_0 = np.array(chd_tr_X[0]) #parent 

	X_train_1 = np.array(chd_tr_X[1]) #left dir

	X_train_2 = np.array(chd_tr_X[2]) #right dir

	X_train_3 = np.array(chd_tr_X[3]) #valency

	X_train_4 = np.array(chd_tr_X[4]) #is Cont

	X_train_5 = np.array(chd_tr_X[5]) #parent tag

	y_train_child = np.array(chd_tr_Y_child)

	y_train_decision = np.array(chd_tr_Y_decision)

	X_test_0 = X_train_0

	X_test_1 = X_train_1

	X_test_2 = X_train_2

	X_test_3 = X_train_3

	X_test_4 = X_train_4

	X_test_5 = X_train_5

	Y_test_child = y_train_child#np.array(chd_tr_Y)#[2, 1, 2])

	Y_test_decision = y_train_decision
#	print(X_train_0.shape[0], 'train samples')
#	print(X_test_0.shape[0], 'test samples')
	# convert class vectors to binary class matrices
	y_train_child = np_utils.to_categorical(y_train_child, nb_classes)
	y_train_decision = np_utils.to_categorical(y_train_decision, 2)
	#Y_test = np_utils.to_categorical(Y_test, nb_classes)
	chdW = np.array(chdSampleWeight)
#	print(chdW.shape)
	desW = np.array(desSampleWeight)


	batch_size = int(sys.argv[4])#sum1
	print("batch_size",batch_size)
	#, shuffle='true'
	if(isEarlyStop):
		model.fit({'parent_input': X_train_0,'parent_input_tag': X_train_5, 'dir_left_input' : X_train_1, 'dir_right_input' : X_train_2, 'val_input' : X_train_3, 'cont_chd_input' : X_train_4}, 

		{'child_output': y_train_child, 'decision_output': y_train_decision},

		nb_epoch=nb_epoch, batch_size=batch_size,validation_split=(1-gramRun.getValidPerc()), callbacks=[early_stopping],  sample_weight = [chdW, desW])#, sample_weight_mode = 'tempporal')  #validation_split=(1-gramRun.getValidPerc()), callbacks=[early_stopping],
	else:
		model.fit({'parent_input': X_train_0,'parent_input_tag': X_train_5, 'dir_left_input' : X_train_1, 'dir_right_input' : X_train_2, 'val_input' : X_train_3, 'cont_chd_input' : X_train_4}, 

		{'child_output': y_train_child, 'decision_output': y_train_decision},

		nb_epoch=nb_epoch, batch_size=batch_size,  sample_weight = [chdW, desW])#, sample_weight_mode = 'tempporal')  #validation_split=(1-gramRun.getValidPerc()), callbacks=[early_stopping],



	#print('Test score:', score[0])

	#print('Test accuracy:', score[1])

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
	
	X_6 = []#parent tag


	for i in range(0, nb_classes):

		for j in range(0, 2):#

			for k in range(0, valenceSize):

				X_0 = X_0 + [[i]]  #parent

				X_6 = X_6 + [[dic2Tag[i]]] # parent tag

				if j == 0:

					X_1 = X_1 + [[1]]  #dir_chd

					X_2 = X_2 + [[0]]  #valence_chd

				else:

					X_1 = X_1 + [[0]]

					X_2 = X_2 + [[1]]

				#X_3 = X_3 + [[j]]  #dir_des

				#X_4 = X_4 + [[k]]  #valence_des

				X_3 = X_3 + [[k]]

				X_5 = X_5 + [[1]]  #CONT

	probs = model.predict([np.array(X_0), np.array(X_6), np.array(X_1), np.array(X_2), np.array(X_3), np.array(X_5)])

	listSz = len(probs[0])

#	print(listSz)

	java_list_chd = gateway.jvm.java.util.ArrayList()

	java_list_des = gateway.jvm.java.util.ArrayList()

	for i in range(0, listSz):
#child
		java_pro = gateway.jvm.java.util.ArrayList()

		for j in range(0, nb_classes):
			java_pro.append(float(probs[0][i][j]))

		java_list_chd.append(java_pro)

#decision

		java_pro_des = gateway.jvm.java.util.ArrayList()

		for j in range(0, 2):

			java_pro_des.append(float(probs[1][i][j]))

		java_list_des.append(java_pro_des)
	rr = model.get_weights()
#	print("chd_tr_Y_child:", len(chd_tr_Y_child))
	print("model.get_weights()[0][0][0] after:", rr[0][0][0])
	print("model.get_weights()[1][0][0] after:", rr[1][0][0])
	print("lr:", lr)
	gramRun.setChdAndDesPy(java_list_chd, java_list_des)

######################SOFT################################################
def childAndDecisionANNSoftMstep3():
	print("no code")

###########################################################################

def readVectorFromFile(vecfile, line_length):
	vectorFile = open(vecfile , 'r')
	wordVec = {}
	for line in vectorFile:
		#print(line)
		a = line.split()
		vec = []
		if(line_length != len(a)):
			print('readVectorFromFile Error!')
		for i in range(len(a) - 1):
			vec += [float(a[i + 1])]

		wordVec[a[0]] = vec
	vectorFile.close()
	return wordVec
#########################################################################
import sys
port = int(sys.argv[1])#23330
accIdx = sys.argv[2]
gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port))


#gateway = JavaGateway()
#gateway = JavaGateway(GatewayParameters(auto_convert=True))
random = gateway.jvm.java.util.Random()
number1 = random.nextInt(5)
number2 = random.nextInt(10)
print(number1,number2)
import os
print (os.getcwd())
import numpy as np
np.random.seed()  # for reproducibility
from keras.layers.core import Flatten
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.utils import np_utils
from keras.layers import Dense, merge, LSTM, Input, Embedding
#from keras.layers.embeddings import Embedding
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import LeakyReLU
import timeit


from keras.layers import Input, Embedding, LSTM, Dense, merge, Reshape, TimeDistributedDense

from keras.layers import Merge

from keras.layers.core import RepeatVector

from keras.models import Model

from random import randint

from keras.callbacks import EarlyStopping



isVerb = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
isNoun = {"NN", "NNS", "NNP", "NNPS"}
isAdj = {"JJ", "JJR", "JJS"}

isParDivid = True # word and tag divid, it is always true
isParInit = False
isParWordInit = True # when it is true, dim_parent_word should be 50
isParTagInit = True  # when it is true, dim_parent_tag should be 10
isParInitGlove = False # when it is true, dim_parent_word should be 50
isChdDivid = True # # when it is true, dim_child_word should be 50, dim_child_tag should be 10
isChdInit = False # when it is true, chdWordInit and chdTagInit are all be true.
isEarlyStop = False

if(isParInitGlove):
	wordVec = readVectorFromFile('glove.6B.50d.txt', 51)#51 - 1 is dim of par word init
else:
	wordVec = readVectorFromFile('/public/home/tukewei/hanwj/Code/DepGrammarNN12/data/word2Vec_d50_iter1000.txt', 51)
if(isChdInit or isParTagInit):
	tagVec = readVectorFromFile('/public/home/tukewei/hanwj/Code/DepGrammarNN12/data/tag2Vec_d10_iter1000.txt', 11)

print(wordVec[str(1)])




nb_epoch = 1#30

#dim = 5	

dim_parent_word = int(sys.argv[6])*5
dim_parent_tag = int(sys.argv[6])
dim_val = 5	

dim1 = int(sys.argv[5])#5*dim int(sys.argv[6])
dim2 = 60#60#10*dim#5*dim

dim_child_word =50# int(sys.argv[6])*5#50#9 * dim#dim_child_word + dim_child_tag = dim2
dim_child_tag = 10#int(sys.argv[6])#1 * dim

dim3 = 10#2*dim

lr = float(sys.argv[3])

maxIter = 50


childValency = 2#for soft
decisionValency = 2#for soft
valenceSize = 2# for viterbi

gramRun = gateway.entry_point  
gramRun.paraSetting()
#tagList = gramRun.printTags()
iterationOfEM = 1
sigma_0 = 0.99
sigma_k = 0#0  -0.05 #MUST be <-0.05, or modelConversionV2S must be changed
sigma_e = 0

nb_classes = gramRun.nb_classes()#159  
print("nb_classes:\t" + str(nb_classes))
nb_classes_tag = gramRun.nb_classes_tag()#35#nb_classes + 1, when use wordf or not
print("nb_classes_tag:\t" + str(nb_classes_tag))

dic2Tag = []
dic2Tag = gramRun.dic2Tag()
dic2WordStr = []
dic2WordStr = gramRun.dic2WordStr()

onlineEta = gramRun.getOnlineEta()
gramRun.setIsEarlyStop(isEarlyStop)
######Model Build################

parent_input = Input(shape=(1,), dtype='int32', name='parent_input')

par_input = Embedding(output_dim=dim_parent_word, input_dim=nb_classes, init = 'glorot_uniform', input_length=1)(parent_input)


parent_input_tag = Input(shape=(1,), dtype='int32', name='parent_input_tag')

par_input_tag = Embedding(output_dim=dim_parent_tag, input_dim=nb_classes_tag, init = 'glorot_uniform', input_length=1)(parent_input_tag)


dir_left_input = Input(shape=(1,), dtype='int32', name='dir_left_input')

weight_l = np.empty([2, dim1])

weight_l[0].fill(0)

weight_l[1].fill(1)

dir_l_input = Embedding(output_dim=dim1, input_dim=2, input_length=1, weights = [weight_l])(dir_left_input)

dir_l_input.params = []

dir_l_input.updates = []

#dir_l_input = Flatten()(dir_l_input)

dir_right_input = Input(shape=(1,), dtype='int32', name='dir_right_input')

weight_r = np.empty([2, dim1])

weight_r[0].fill(0)

weight_r[1].fill(1)

dir_r_input = Embedding(output_dim=dim1, input_dim=2, input_length=1, weights = [weight_r])(dir_right_input)

dir_r_input.params = []

dir_r_input.updates = []

#dir_r_input = Flatten()(dir_r_input)

val_input = Input(shape=(1,), dtype='int32', name='val_input')

valency_input = Embedding(output_dim=dim_val, input_dim=valenceSize, init = 'glorot_uniform', input_length=1)(val_input)



cont_chd_input = Input(shape=(1,), dtype='int32', name='cont_chd_input')

weight = np.empty([2, dim2])

weight[0].fill(0)

weight[1].fill(1)

cont_chd_input_n = Embedding(output_dim=dim2, input_dim=2, input_length=1, weights = [weight], name = 'cont_chd_input_n')(cont_chd_input)

#print(cont_chd_input_n.params)

cont_chd_input_n.params = []

cont_chd_input_n.updates = []

#cont_chd_input_n.set_weights(weight)

cont_chd_input_f = Flatten()(cont_chd_input_n)

#begin merge

child = merge([par_input, par_input_tag, valency_input], mode='concat')

child = Flatten()(child)

left = Dense(dim1, activation='relu')(child)

right = Dense(dim1, activation='relu')(child)

left = Reshape((1, dim1))(left)

left = merge([left, dir_l_input], mode = 'mul')

right = Reshape((1, dim1))(right)

right = merge([right, dir_r_input], mode = 'mul')

input_vec = merge([left, right], mode = 'sum')

input_vec = Flatten()(input_vec)

if(not isChdDivid):
	child = Dense(dim2, activation='relu')(input_vec)#child dim

	child = merge([child, cont_chd_input_f], mode = 'mul')#final representation of child  #cont_chd_input_f is not necessily needed.

	child_loss = Dense(nb_classes, activation='softmax', name='child_output')(child)
else:
	###child word and tag###
	child_word = Dense(dim_child_word, activation='relu')(input_vec)

	child_tag = Dense(dim_child_tag, activation='relu')(input_vec)

	#child_word = merge([child_word, cont_chd_word_input_f], mode = 'mul')

	#child_tag = merge([child_tag, cont_chd_tag_input_f], mode = 'mul')
	if(isChdInit):
		weight_child_word = np.zeros((nb_classes, dim_child_word))
		for i in range(len(weight_child_word)):
			if(wordVec.has_key(str(i))):
				weight_child_word[i] = wordVec[str(i)]
			else:
				weight_child_word[i] = np.random.randn(dim_child_word)
		weight_child_word = np.array(zip(*weight_child_word))#?
		weight_child_word_bias = np.zeros((nb_classes,))
		child_word = Dense(nb_classes, activation='linear', weights = [weight_child_word, weight_child_word_bias])(child_word)
		weight_child_tag = np.zeros((nb_classes_tag, dim_child_tag))
		for i in range(len(weight_child_tag)):
			if(tagVec.has_key(str(i))):
				weight_child_tag[i] = tagVec[str(i)]
			else:
				weight_child_tag[i] = np.random.randn(dim_child_tag)
		weight_child_tag = np.array(zip(*weight_child_tag))
		weight_child_tag_bias = np.zeros((nb_classes_tag,))
		child_tag = Dense(nb_classes_tag, activation='linear', weights = [weight_child_tag, weight_child_tag_bias])(child_tag)

	else:
		child_word = Dense(nb_classes, activation='linear')(child_word)#init weight is here!
		child_tag = Dense(nb_classes_tag, activation='linear')(child_tag)#init weight is here!


	weight_tag2word = np.zeros((nb_classes_tag, nb_classes))#np.empty([2, dim1])
	weight_tag2word_bias = np.zeros((nb_classes,))
	#process
	for i in range(nb_classes):			
		weight_tag2word[dic2Tag[i]][i] = 1
	 

	child_tag = Dense(nb_classes, activation='linear', weights = [weight_tag2word, weight_tag2word_bias])(child_tag)#matrixTag2Word*child_tag

	child_tag.params = []#parameter here []

	child_tag.updates = []	

	child = merge([child_word, child_tag], mode = 'sum')

	weight_child_loss = np.zeros((nb_classes, nb_classes))
	weight_child_loss_bias = np.zeros((nb_classes,))
	#process
	for i in range(nb_classes):
		for j in range(nb_classes):
			if(i == j):
				weight_child_loss[i][j] = 1	

	child_loss = Dense(nb_classes, activation='softmax', name='child_output', weights = [weight_child_loss, weight_child_loss_bias])(child)

	child_loss.params = []#parameter here is [1, 0; 0, 1]

	child_loss.updates = []
###



#child_loss.params = []

#child_loss.updates = []

#child_loss = Reshape((1, nb_classes))(child_loss)

#decision = Flatten()(input_vec)

#child = Dense(50, activation='relu')(child)

decision = Dense(dim3, activation='relu')(input_vec)

#decision = Dense(50, activation='relu')(decision)

decision_loss = Dense(2, activation='softmax', name='decision_output')(decision)



model = Model(input=[parent_input, parent_input_tag, dir_left_input, dir_right_input, val_input, cont_chd_input], output=[child_loss, decision_loss])

if(isParInit):
	#initialize
	model_weights = model.get_weights()
	print('model_weights',len(model_weights))
	print('model_weights[0]', len(model_weights[0]))
	print('model_weights[0][0]', len(model_weights[0][0]))
	print('model_weights[1]', len(model_weights[1]))
	print('model_weights[1][0]', len(model_weights[1][0]))
	if(isParWordInit):
		parPosition = 0
	#	if(iterationOfEM > 1):
	#		parPosition = 1

		for k in range(len(model_weights)):
			if(len(model_weights[k]) == nb_classes):
				parPosition = k
				break
		print('parPosition:\t', parPosition)
		par_input_weight = model_weights[parPosition]#(input_dim, output_dim)
		print('nb_classes', nb_classes)
		print('len(par_input_weight)',len(par_input_weight))
	#   use glove6B.50d.txt
		if(isParInitGlove): 
			print('dic2WordStr\n', dic2WordStr)
			for i in range(len(par_input_weight)):
				if(wordVec.has_key(dic2WordStr[i])):
					 print('dic2WordStr[i]', dic2WordStr[i])
					 par_input_weight[i] = wordVec[dic2WordStr[i]]
	#   use my word3Vec.txt
		else:
			for i in range(len(par_input_weight)):
				if(wordVec.has_key(str(i))):
					 par_input_weight[i] = wordVec[str(i)]

		model_weights[parPosition] = par_input_weight
	if(isParTagInit):
		parTagPosition = 0
		for k in range(len(model_weights)):
			if(len(model_weights[k]) == nb_classes_tag):
				parTagPosition = k
				break
		if(nb_classes==nb_classes_tag):
			parTagPosition = 1
		print('parTagPosition:\t', parTagPosition)
		par_tag_input_weight = model_weights[parTagPosition]#(input_dim, output_dim)
		print('nb_classes_tag', nb_classes_tag)
		print('len(par_tag_input_weight)',len(par_tag_input_weight))
                
		for i in range(len(par_tag_input_weight)):
			if(tagVec.has_key(str(i))):
	 			par_tag_input_weight[i] = tagVec[str(i)]
		model_weights[parTagPosition] = par_tag_input_weight


	model.set_weights(model_weights)

sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)#5e-2 0.5 1e-6

model.compile(loss={'child_output': 'categorical_crossentropy', 'decision_output' : 'categorical_crossentropy'}, optimizer=sgd, sample_weight_mode="None")
##########Model build Finished!############
def LoadTrainData(filename):
    trEmisDt = np.loadtxt(filename)
    return trEmisDt.tolist()
###########################################

while(iterationOfEM != maxIter):
	
	print("========== iter : ", iterationOfEM, "==========")
	print("Port:\t",port)
	Anneling_sigma = sigma_0 + (iterationOfEM - 1) * sigma_k;
	Anneling_sigma  = max(sigma_e, Anneling_sigma);
	isCountTable = False
	print("SoftEM_sigma is : ", Anneling_sigma)
	if(Anneling_sigma > 0.95):#Anneling_sigma > 0.95
		gramRun.EStep()
		res = []
		res_decision = []
		res_val = []
		res_decision_val = []
		if(isCountTable):
			res = gramRun.MStepCountTable()
			res_decision = gramRun.MStep_decisionCountTable()
		else:
			gramRun.MStepTxt()
			res = LoadTrainData("/public/home/tukewei/hanwj/Code/DepGrammarNN12/temp/chdTemp" + accIdx + ".txt")
			res_decision = LoadTrainData("/public/home/tukewei/hanwj/Code/DepGrammarNN12/temp/decTemp" + accIdx + ".txt")
		if(isEarlyStop):
			res_val = gramRun.getValidChdCountTable()
			res_decision_val = gramRun.getValidDecisionCountTable()
		#res = gramRun.chdAndDecisionMStep()#gramRun.MStep()
		childAndDecisionANNMstep3()#childANNMstep()
		gramRun.setLastIsViterbiTrue()#allTrees.clear
	else:
		#Anneling_sigma = 0.1
		if(gramRun.returnLastIsViterbi() == 1):		
			gramRun.Viterbi2Annealing()#sentence and model 
			print('the last is viterb is TRUE')
		if(iterationOfEM == 1):
			gramRun.Init2Annealing()
		gramRun.SoftEStep(Anneling_sigma)#Valuate updateModel and countMstep
		res = []
		res_decision = []
		res = gramRun.SoftMStep()#gramRun.chdAndDecisionSoftMStep()##ANN and return  res
		res_decision = gramRun.SoftMStep_decision()
		childAndDecisionANNSoftMstep3()#childANNSoftMstep()
		gramRun.setLastIsViterbiFalse()
	#print("*******************************************************************************************")
	#print(len(model.get_weights()))
	#for i in range(len(model.get_weights())):
	#	print(str(i)+ ":  " + str(len(model.get_weights()[i][0])) + "  " + str(len(model.get_weights()[i][1]))) 
	if(iterationOfEM == 15):
		np.savetxt('/public/home/tukewei/hanwj/Code/DepGrammarNN12/2.txt', model.get_weights()[0])
	print('dic2Tag')
	print(dic2Tag)
        iterationOfEM = iterationOfEM + 1

