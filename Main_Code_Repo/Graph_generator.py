from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle



def SBM_generator(cluster_sizes, p = 0.5, q = 0.1, seed = 0):
    n = len(cluster_sizes)
    probl_matrix = np.eye(n) * p + (1 - np.eye(n)) * q  # probability matrix

    g = nx.stochastic_block_model(cluster_sizes, probl_matrix, seed=seed)
    return g
