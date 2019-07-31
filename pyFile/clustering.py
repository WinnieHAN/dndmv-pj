from time import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering
# X_red = np.zeros([30,20])
# X_red[15:30,:] = 1


def clustering(X, linkage, n_clusters):
    clustering_model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)  # linkage = 'ward', 'average', 'complete'
    clustering_model.fit(X)
    return clustering_model.labels_


def get_newDic2tag(X, clustering_linkage, n_clusters):
    if clustering_linkage == 0:
        newDic2tag = clustering(X, linkage='ward',
                                n_clusters=n_clusters)  # # linkage = 'ward', 'average', 'complete'
    elif clustering_linkage == 1:
        newDic2tag = clustering(X, linkage='average',
                                n_clusters=n_clusters)  # # linkage = 'ward', 'average', 'complete'
    elif clustering_linkage == 2:
        newDic2tag = clustering(X, linkage='complete',
                                n_clusters=n_clusters)  # # linkage = 'ward', 'average', 'complete'
    return newDic2tag