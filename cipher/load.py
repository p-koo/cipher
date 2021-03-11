"""Load standard datasets."""

import h5py
import numpy as np


def single_task(filepath):
    """Load single task dataset

    Parameters
    ----------
    filepath : string
        String path to file.

    Returns
    -------
    x_train : training data, shape = (N, L, A)
    y_train : training labels, shape = (N, num_labels)
    x_valid : valid data, shape = (N, L, A)
    y_valid : valid labels, shape = (N, num_labels)
    x_test : test data, shape = (N, L, A)
    y_test : test labels, shape = (N, num_labels)

    """

    with h5py.File(filepath, "r") as dataset:
        x_train = np.array(dataset["x_train"]).astype(np.float32)
        y_train = np.array(dataset["y_train"]).astype(np.float32)
        x_valid = np.array(dataset["x_valid"]).astype(np.float32)
        y_valid = np.array(dataset["y_valid"]).astype(np.int32)
        x_test = np.array(dataset["x_test"]).astype(np.float32)
        y_test = np.array(dataset["y_test"]).astype(np.int32)

    return x_train, y_train, x_valid, y_valid, x_test, y_test


# TODO: def multi_task(filepath):
