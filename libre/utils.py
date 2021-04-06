"""Useful functions."""

import os
import numpy as np
import pandas as pd
import pkgutil

# TODO: this is the only dependency that requires a compiler. It does not ship a
# pre-compiled wheel. Perhaps we can write a python/numpy implementation?
from ushuffle import shuffle

def make_directory(dirpath, verbose=1):
    """make a directory.

    Parameters
    ----------
    dirpath : string
        String of path to directory.

    Returns
    -------


    Examples
    --------
    >>> dirpath = './results'
    >>> make_directory(dirpath)

    """
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
        print("making directory: " + dirpath)


def import_model(model_name):
    """Import a model from model_zoo.

    Parameters
    ----------
    model_name : string
        Name of model in model_zoo to import.

    Returns
    -------
        imported model

    Examples
    --------
    >>> model_name = 'deepbind'
    >>> model = import_model(model_name)

    """

    # Import model from the zoo as singular animal
    # Equivalent of `from model_zoo import model_name as animal` where model_name is evaluated at runtime
    animal = __import__(
        "cipher.model_zoo." + model_name, globals(), locals(), [model_name], 0
    )
    return animal



def list_model_zoo():
    """List models in cipher.model_zoo.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Examples
    --------
    >>> list_model_zoo()
    >>> 
    """

    # get names of files in model_zoo and filter out files that begin with "__"
    loader = pkgutil.get_loader(model_zoo)
    names = []
    for name in loader.contents():
        name = name.split('.')[0]
        if name[:2] != '__':
            names.append(name)
    
    # print to std out
    print('Model list: ')
    for name in names:
        print("    %s"%(name))  




def convert_one_hot(sequences, alphabet="ACGT") -> np.ndarray:
    """Convert flat array of sequences to one-hot representation.

    **Important**: all letters in `sequences` *must* be contained in `alphabet`, and
    all sequences must have the same length.

    Parameters
    ----------
    sequences : numpy.ndarray of strings
        The array of strings. Should be one-dimensional.
    alphabet : str
        The alphabet of the sequences.

    Returns
    -------
    Numpy array of sequences in one-hot representation. The shape of this array is
    `(len(sequences), len(sequences[0]), len(alphabet))`.

    Examples
    --------
    >>> one_hot(["TGCA"], alphabet="ACGT")
    array([[[0., 0., 0., 1.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.]]])
    """
    sequences = np.asanyarray(sequences)
    if sequences.ndim != 1:
        raise ValueError("array of sequences must be one-dimensional.")
    n_sequences = sequences.shape[0]
    sequence_len = len(sequences[0])

    # Unpack strings into 2D array, where each point has one character.
    s = np.zeros((n_sequences, sequence_len), dtype="U1")
    for i in range(n_sequences):
        s[i] = list(sequences[i])

    # Make an integer array from the string array.
    pre_onehot = np.zeros(s.shape, dtype=np.uint8)
    for i, letter in enumerate(alphabet):
        # do nothing on 0 because array is initialized with zeros.
        if i:
            pre_onehot[s == letter] = i

    # create one-hot representation
    n_classes = len(alphabet)
    return np.eye(n_classes)[pre_onehot]


def convert_onehot_to_sequence(one_hot, alphabet="ACGT"):
    """Convert DNA/RNA sequences from one-hot representation to
    string representation.

    Parameters
    ----------
    one_hot : <numpy.ndarray>
        one_hot encoded sequence with shape (N, L, A)
    alphabet : <str>
        DNA = 'ACGT'

    Returns
    -------
    sequences : <numpy.ndarray>
    A numpy vector of sequences in string representation.

    Example
    -------
    >>> one_hot = np.array(
            [[[1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.]],

            [[0., 0., 0., 1.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.]]]
                )
    >>> sequences = convert_onehot_to_sequence(one_hot)
    >>> sequences
    array([['A', 'A', 'G', 'A', 'C'],
       ['T', 'C', 'G', 'C', 'A']], dtype=object)
    """
    assert alphabet in ["ACGT", "ACGU"], "Enter a valid alphabet"

    # convert alphabet to dictionary
    alphabet_dict = {i: a for i, a in enumerate(list(alphabet))}

    # get indices of one-hot
    seq_indices = np.argmax(one_hot, axis=2)  # (N, L)

    # convert index to sequence
    sequences = []
    for seq_index in seq_indices:
        seq = pd.Series(seq_index).map(alphabet_dict)
        sequences.append(seq)
    return np.asarray(sequences)


def shuffle_onehot(one_hot, k=1):
    """Shuffle one-hot represented sequences while preserving k-let frequencies.

    Parameters
    ----------
    one_hot : np.ndarray
        One_hot encoded sequence with shape (N, L, A).
    k : int, optional
        k of k-let frequencies to preserve. For example, with k = 2, dinucleotide
        shuffle is performed. The default is k = 1 (i.e., single-nucleotide shuffle).

    Returns
    -------
    np.ndarray
        One-hot represented shuffled sequences, of the same shape as one_hot.

    Examples
    --------
    >>> seqs = ["ACGT", "GTCA"]
    >>> one_hot = convert_one_hot(seqs)
    >>> one_hot
    array([[[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]],

           [[0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.]]])
    >>> one_hot_shuffled = shuffle_onehot(one_hot)
    >>> one_hot_shuffled
    array([[[0., 0., 0., 1.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 1., 0.]],

           [[1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.]]])
    """

    if k == 1:
        L = one_hot.shape[1]  # one_hot has shape (N, L, A)
        rng = np.random.default_rng()
        one_hot_shuffled = []

        for x in one_hot:
            perm = rng.permutation(L)
            x_shuffled = x[perm, :]
            one_hot_shuffled.append(x_shuffled)

        one_hot_shuffled = np.array(one_hot_shuffled)

        return one_hot_shuffled

    elif k >= 2:
        # convert one_hot to pandas Series of letters, then string letters together
        # (for each Series)
        seqs = [seq.str.cat() for seq in convert_onehot_to_sequence(one_hot)]
        seqs_shuffled = []

        for i, seq in enumerate(seqs):
            seq = seq.upper()
            # dinucleotide shuffle
            seq_shuffled = shuffle(bytes(seq, "utf-8"), k).decode("utf-8")

            seqs_shuffled.append(seq_shuffled)

        one_hot_shuffled = convert_one_hot(seqs_shuffled)
        return one_hot_shuffled

    else:
        raise ValueError("k must be an integer greater than or equal to 1")


def shuffle_sequences(sequences, k=1):
    """Shuffle one-hot represented sequences while preserving k-let frequencies.

    Parameters
    ----------
    one_hot : np.ndarray
        One_hot encoded sequence with shape (N, L, A)
    k : int, optional
        k of k-let frequencies to preserve. For example, with k = 2, dinucleotide
        shuffle is performed. The default is k = 1 (i.e., single-nucleotide shuffle).

    Returns
    -------
    np.ndarray
        One-hot represented shuffled sequences, of the same shape as one_hot.

    Examples
    --------
    >>> seqs = ["AGCGTTCAA", "TACGAATCG"]
    >>> seqs_shuffled = shuffle_sequences(seqs, k=2) # dinucleotide shuffle
    >>> seqs_shuffled
    ['AAGTTCGCA', 'TCGATAACG']
    """
    sequences_shuffled = []

    for i, seq in enumerate(sequences):
        seq = seq.upper()
        seq_shuffled = shuffle(bytes(seq, "utf-8"), k).decode("utf-8")

        sequences_shuffled.append(seq_shuffled)

    return sequences_shuffled



