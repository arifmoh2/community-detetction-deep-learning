import networkx
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np


def graph_cluster_plot(g: networkx.Graph, partition:dict = None, title =  'Partition plot' ,seed = 0):
    pos = nx.spring_layout(g, seed=seed)
    if partition is None:
        partition = {node:i for i, cluster in enumerate(g.graph['partition']) for node in cluster}

    name = "Accent_r"
    cmap_len = len(np.unique(list(partition.values())))
    cmap = get_cmap(name, cmap_len)
    colors = cmap.colors

    nx.draw(g,
            pos,
            node_size=100,
            alpha=0.8,
            node_color=[colors[community] for node, community in partition.items()],
            with_labels=False,)
    plt.title(title)
    plt.show()

def clustering_comparison_plot(g: networkx.Graph, partition_true:dict,
                               partition_predicted:dict ,seed = 0, node_size = 100):

    pos = nx.spring_layout(g, seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # true color map
    name = "Accent_r"
    cmap_len = len(np.unique(list(partition_true.values())))
    cmap = get_cmap(name, cmap_len)
    colors = cmap.colors
    nx.draw(g,
            pos,
            node_size=node_size,
            alpha=0.8,
            node_color=[colors[community] for node, community in partition_true.items()],
            with_labels=False,
            ax=axes[0]
            )
    axes[0].set_title('True')

    # prediction color map
    cmap_len = len(np.unique(list(partition_predicted.values())))
    cmap = get_cmap(name, cmap_len)
    colors = cmap.colors
    nx.draw(g,
            pos,
            node_size=node_size,
            alpha=0.8,
            node_color=[colors[community] for node, community in partition_predicted.items()],
            with_labels=False,
            ax=axes[1]
            )
    axes[1].set_title('Predicted')
    plt.tight_layout()
    plt.show()





