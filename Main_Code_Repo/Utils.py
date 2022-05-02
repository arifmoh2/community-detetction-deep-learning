import pickle
from sklearn.metrics import normalized_mutual_info_score
import numpy as np


def pickle_object(file, save_location):
    """"Pickle and save object to a given location"""
    with open(save_location, 'wb') as handle:
        pickle.dump(file, handle, protocol=4)


def unpickle_object(file_location: object) -> object:
    """Load pickled object from a given location"""
    with open(file_location, 'rb') as handle:
        file = pickle.load(handle)
    return file


def NMI_score(partition_true, partition_predicted):
    """"Calculate the NMI score as defined in the sklearn, metric for clustering"""
    return normalized_mutual_info_score(np.array(list(partition_true.values())), np.array(list(partition_predicted.values())))
