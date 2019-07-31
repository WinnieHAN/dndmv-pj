import time
import numpy as np
import os
from config import parse_args
from keras.models import load_model
from torch_py.model import AttnLSTM, lalala

def main1():
    args = parse_args(type=0)
    print(args)

if __name__ == "__main__":
    lalala()
    main1()

# from keras.layers import Input, Dense
# from keras.models import Model
#
#
# # This returns a tensor
# inputs = Input(shape=(784,))
# # a layer instance is callable on a tensor, and returns a tensor
# x = Dense(64, activation='relu', name='w1')(inputs)
# x = Dense(64, activation='relu', name='w2')(x)
# predictions = Dense(10, activation='softmax')(x)
#
# # This creates a model that includes
# # the Input layer and three Dense layers
# model = Model(inputs=inputs, outputs=predictions)
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# # print(model.get_weights())
#
# from time import time
# import numpy as np
# from sklearn.cluster import AgglomerativeClustering
# # X_red = np.zeros([30,20])
# # X_red[15:30,:] = 1
# X_red = model.get_weights()[4]
# for linkage in ('ward', 'average', 'complete'):
#     clustering = AgglomerativeClustering(linkage=linkage, n_clusters=2)
#     t0 = time()
#     clustering.fit(X_red)
#     print("%s : %.2fs" % (linkage, time() - t0))
# print(clustering.labels_)
#
#
# from keras.models import load_model
# model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# del model  # deletes the existing model
#
# # returns a compiled model
# # identical to the previous one
# model = load_model('my_model.h5')

