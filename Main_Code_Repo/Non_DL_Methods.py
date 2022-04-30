import networkx
from matplotlib.cm import get_cmap
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
from community import community_louvain
from sklearn.cluster import SpectralClustering

#For saving results
data_dir = os.path.join((os.path.dirname(os.getcwd())), 'Data')


def Louvain_algorithm(g: networkx.Graph, seed = 0):
    """"Louvain's algorithm implementation for NetworkX graph"""
    partition_true = {node: i for i, cluster in enumerate(g.graph['partition']) for node in cluster}
    partition_predicted = community_louvain.best_partition(g, random_state=seed)

    return partition_true, partition_predicted

def Spectral_Clustering(g: networkx.Graph, seed = 0):
    """"Spectral clustering algorithm, assuming the input has ground truth clustering determined """
    k = len(g.graph['partition'])
    adj_mat = nx.to_numpy_matrix(g)
    sc = SpectralClustering(n_clusters= k, affinity='precomputed',random_state=seed)
    sc.fit(np.asarray(adj_mat))
    partition_true = {node: i for i, cluster in enumerate(g.graph['partition']) for node in cluster}
    partition_predicted = {node: label for node, label in zip(list(g.nodes), sc.labels_)}

    return partition_true, partition_predicted


