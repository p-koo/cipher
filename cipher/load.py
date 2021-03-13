"""Load standard datasets."""

import h5py
import numpy as np


def standard_data(filepath, reverse_comp=False):
    """Load single task dataset.

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

    if reverse_comp:
        x_train_rc = x_train[:, ::-1, :][:, :, ::-1]
        x_valid_rc = x_valid[:, ::-1, :][:, :, ::-1]
        x_test_rc = x_test[:, ::-1, :][:, :, ::-1]

        # merge forward and reverse complement
        x_train = np.vstack([x_train, x_train_rc])
        x_valid = np.vstack([x_valid, x_valid_rc])
        x_test = np.vstack([x_test, x_test_rc])
        y_train = np.vstack([y_train, y_train])
        y_valid = np.vstack([y_valid, y_valid])
        y_test = np.vstack([y_test, y_test])

    return x_train, y_train, x_valid, y_valid, x_test, y_test


# TODO: def multi_task(filepath):
