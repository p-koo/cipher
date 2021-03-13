from collections import OrderedDict
import random
import sys

import h5py
import numpy.random as npr
import numpy as np
from sklearn import preprocessing

################################################################################
# seq_hdf5.py
#
# Make an HDF5 file for Torch input out of a FASTA file and targets text file,
# dividing the data into training, validation, and test.
################################################################################

################################################################################
# main
################################################################################


def make_h5(
    fasta_file,
    targets_file,
    out_file,
    out_header_file,
    batch_size=None,
    extend_length=None,
    random_seed=1,
    test_pct=0,
    valid_pct=0,
):
    """Generate h5 format dataset from activity table file

    Parameters
    ----------
    fasta_file : str
        path to the fasta file with the genome
    targets_file : str
        path to the activity table txt file
    out_file : str
        path where .h5 will be saved
    out_header_file : str
        path where the summary file with headers will be saved
    batch_size : int
        batch_size
    extend_length : int
        Extend all sequences to this length
    random_seed : int
        numpy.random seed for shuffling the data
    test_pct : float
        Test set percentage
    valid_pct : float
        Validation set percentag
    """

    # seed rng before shuffle
    # TODO: numpy suggests creating an instance of a random number generator and
    # setting a seed during instantiation.
    # See https://numpy.org/doc/stable/reference/random/generator.html
    npr.seed(random_seed)

    #################################################################
    # load data
    #################################################################
    print("LOADING DATA")
    seqs, targets = load_data_1hot(
        fasta_file,
        targets_file,
        extend_len=extend_length,
        mean_norm=False,
        whiten=False,
        permute=False,
        sort=False,
    )

    # reshape sequences for torch
    seqs = seqs.reshape((seqs.shape[0], int(seqs.shape[1] / 4), 4))

    # read headers
    headers = []
    for line in open(fasta_file):
        if line[0] == ">":
            headers.append(line[1:].rstrip())
    headers = np.array(headers)

    # read labels
    target_labels = open(targets_file).readline().strip().split("\t")

    # permute
    order = npr.permutation(seqs.shape[0])
    seqs = seqs[order]
    targets = targets[order]
    headers = headers[order]

    assert test_pct + valid_pct <= 1.0

    #################################################################
    # divide data
    #################################################################
    print("DIVIDING DATA")

    test_count = int(0.5 + test_pct * seqs.shape[0])
    valid_count = int(0.5 + valid_pct * seqs.shape[0])
    print(test_count, valid_count)
    train_count = seqs.shape[0] - test_count - valid_count
    train_count = batch_round(train_count, batch_size)
    print("%d training sequences " % train_count, file=sys.stderr)

    test_count = batch_round(test_count, batch_size)
    print("%d test sequences " % test_count, file=sys.stderr)

    valid_count = batch_round(valid_count, batch_size)
    print("%d validation sequences " % valid_count, file=sys.stderr)

    i = 0
    train_seqs, train_targets = (
        seqs[i : i + train_count, :],
        targets[i : i + train_count, :],
    )
    i += train_count
    valid_seqs, valid_targets, _ = (
        seqs[i : i + valid_count, :],
        targets[i : i + valid_count, :],
        headers[i : i + valid_count],
    )
    i += valid_count
    test_seqs, test_targets, _ = (
        seqs[i : i + test_count, :],
        targets[i : i + test_count, :],
        headers[i : i + test_count],
    )

    #################################################################
    # construct hdf5 representation
    #################################################################
    print("MAKING hdf5")
    h5f = h5py.File(out_file, "w")
    # print(len(target_labels))
    target_labels = [n.encode("ascii", "ignore") for n in target_labels]
    # h5f.create_dataset('target_labels', data=target_labels)
    with open(out_header_file, "w") as f:
        for label in target_labels:
            # TODO: once we add testing, this can probably be simplified to
            # `f.write(label)`.
            f.write("%s\n" % label)

    if train_count > 0:
        h5f.create_dataset("x_train", data=train_seqs)
        h5f.create_dataset("y_train", data=train_targets)

    if valid_count > 0:
        h5f.create_dataset("x_valid", data=valid_seqs)
        h5f.create_dataset("y_valid", data=valid_targets)

    if test_count > 0:
        h5f.create_dataset("x_test", data=test_seqs)
        h5f.create_dataset("y_test", data=test_targets)
        # h5f.create_dataset('test_headers', data=test_headers)
    h5f.close()


def batch_round(count, batch_size):
    if batch_size is not None:
        count -= batch_size % count
    return count


def load_data_1hot(
    fasta_file,
    scores_file,
    extend_len=None,
    mean_norm=True,
    whiten=False,
    permute=True,
    sort=False,
):
    # load sequences
    seq_vecs = hash_sequences_1hot(fasta_file, extend_len)

    # load scores
    seq_scores = hash_scores(scores_file)

    # align and construct input matrix
    train_seqs, train_scores = align_seqs_scores_1hot(seq_vecs, seq_scores, sort)

    # whiten scores
    if whiten:
        train_scores = preprocessing.scale(train_scores)
    elif mean_norm:
        train_scores -= np.mean(train_scores, axis=0)

    # randomly permute
    if permute:
        order = npr.permutation(train_seqs.shape[0])
        train_seqs = train_seqs[order]
        train_scores = train_scores[order]

    return train_seqs, train_scores


def hash_sequences_1hot(fasta_file, extend_len=None):
    # determine longest sequence
    if extend_len is not None:
        seq_len = extend_len
    else:
        seq_len = 0
        seq = ""
        for line in open(fasta_file):
            if line[0] == ">":
                if seq:
                    seq_len = max(seq_len, len(seq))

                header = line[1:].rstrip()
                seq = ""
            else:
                seq += line.rstrip()

        if seq:
            seq_len = max(seq_len, len(seq))

    # load and code sequences
    seq_vecs = OrderedDict()
    seq = ""
    for line in open(fasta_file):
        if line[0] == ">":
            if seq:
                seq_vecs[header] = dna_one_hot(seq, seq_len)

            header = line[1:].rstrip()
            seq = ""
        else:
            seq += line.rstrip()

    if seq:
        seq_vecs[header] = dna_one_hot(seq, seq_len)

    return seq_vecs


def dna_one_hot(seq, seq_len=None, flatten=True, n_random=True):
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim : seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2

    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    if n_random:
        seq_code = np.zeros((seq_len, 4), dtype="bool")
    else:
        seq_code = np.zeros((seq_len, 4), dtype="float16")

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A":
                seq_code[i, 0] = 1
            elif nt == "C":
                seq_code[i, 1] = 1
            elif nt == "G":
                seq_code[i, 2] = 1
            elif nt == "T":
                seq_code[i, 3] = 1
            else:
                if n_random:
                    ni = random.randint(0, 3)
                    seq_code[i, ni] = 1
                else:
                    seq_code[i, :] = 0.25

    # flatten and make a column vector 1 x len(seq)
    if flatten:
        seq_vec = seq_code.flatten()[None, :]

    return seq_vec


def hash_scores(scores_file):
    seq_scores = {}

    for line in open(scores_file):
        a = line.split()

        try:
            seq_scores[a[0]] = np.array([float(a[i]) for i in range(1, len(a))])
        except Exception:
            print("Ignoring header line", file=sys.stderr)

    # consider converting the scores to integers
    int_scores = True
    for header in seq_scores:
        if not np.equal(np.mod(seq_scores[header], 1), 0).all():
            int_scores = False
            break

    if int_scores:
        for header in seq_scores:
            seq_scores[header] = seq_scores[header].astype("int8")

        """
        for header in seq_scores:
            if seq_scores[header] > 0:
                seq_scores[header] = np.array([0, 1], dtype=np.min_scalar_type(1))
            else:
                seq_scores[header] = np.array([1, 0], dtype=np.min_scalar_type(1))
        """

    return seq_scores


def align_seqs_scores_1hot(seq_vecs, seq_scores, sort=True):
    if sort:
        seq_headers = sorted(seq_vecs.keys())
    else:
        seq_headers = seq_vecs.keys()

    # construct lists of vectors
    train_scores = []
    train_seqs = []
    for header in seq_headers:
        train_seqs.append(seq_vecs[header])
        train_scores.append(seq_scores[header])

    # stack into matrices
    train_seqs = np.vstack(train_seqs)
    train_scores = np.vstack(train_scores)

    return train_seqs, train_scores
