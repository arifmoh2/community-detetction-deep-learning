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


def plot_clusters(g, show_labels = False):
    nx.draw(g, node_color=[colors[i] for i, nodes in enumerate(g.graph['partition']) for node in nodes],
            with_labels=show_labels)
    plt.show()



#For plotting
name = "Accent"
cmap = get_cmap(name)
colors = cmap.colors

#For saving results
data_dir = os.path.join((os.path.dirname(os.getcwd())), 'Data')