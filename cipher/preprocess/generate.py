import os

import h5py
import numpy as np

from . import wrangle


def process_singletask(
    tf_path,
    dnase_path,
    genome_path,
    data_path,
    experiment,
    window=200,
    alphabet="ACGT",
    compression="gzip",
    max_len=300,
    gc_match=True,
    valid_frac=0.1,
    test_frac=0.2,
):
    """Preprocess data for a single-class task."""

    # remove extremely large peaks
    tf_filtered_path = os.path.join(data_path, experiment + "_pos_filtered.bed")
    wrangle.filter_max_length(tf_path, tf_filtered_path, max_len)

    # create new bed file with window size enforced
    pos_bed_path = os.path.join(data_path, experiment + "_pos_" + str(window) + ".bed")
    wrangle.enforce_constant_size(tf_filtered_path, pos_bed_path, window)

    # extract sequences from bed file and save to fasta file
    pos_fasta_path = os.path.join(data_path, experiment + "_pos.fa")
    wrangle.bedtools_getfasta(
        pos_bed_path, genome_path, output_path=pos_fasta_path, strand=True
    )

    # parse sequence from fasta file
    pos_seq, pos_names = wrangle.parse_fasta(pos_fasta_path)

    # filter sequences with absent nucleotides
    pos_seq, good_index = wrangle.filter_nonsense_sequences(pos_seq)
    pos_names = pos_names[good_index]

    # convert filtered sequences to one-hot representation
    pos_one_hot = wrangle.convert_one_hot(pos_seq, alphabet)

    # get non-overlap between pos peaks and neg peaks
    neg_bed_path = os.path.join(data_path, experiment + "_nonoverlap.bed")
    wrangle.bedtools_intersect(
        dnase_path, tf_path, neg_bed_path, write_a=True, nonoverlap=True
    )

    # create new bed file with window enforced
    neg_bed_path2 = os.path.join(data_path, experiment + "_neg_" + str(window) + ".bed")
    wrangle.enforce_constant_size(neg_bed_path, neg_bed_path2, window)

    # extract sequences from bed file and save to fasta file
    neg_fasta_path = os.path.join(data_path, experiment + "_neg.fa")
    wrangle.bedtools_getfasta(
        neg_bed_path2, genome_path, output_path=neg_fasta_path, strand=True
    )

    # parse sequence and chromosome from fasta file
    neg_seq, neg_names = wrangle.parse_fasta(neg_fasta_path)

    # filter sequences with absent nucleotides
    neg_seq, good_index = wrangle.filter_nonsense_sequences(neg_seq)
    neg_names = neg_names[good_index]

    # convert filtered sequences to one-hot representation
    neg_one_hot = wrangle.convert_one_hot(neg_seq, alphabet)

    if len(neg_one_hot) > len(pos_one_hot):
        # subselect background sequences according to gc-content
        if gc_match:
            # calling match_gc function to balance neg sequences with pos by GC content:
            f_pos = np.mean(pos_one_hot, axis=1)
            f_neg = np.mean(neg_one_hot, axis=1)

            # get GC content for pos and neg sequences
            gc_pos = np.sum(f_pos[:, 1:3], axis=1)
            gc_neg = np.sum(f_neg[:, 1:3], axis=1)

            index = wrangle.sample_b_matched_to_a(gc_pos, gc_neg)
        else:
            index = np.random.permutation(len(neg_one_hot))[: len(pos_one_hot)]
        neg_one_hot = neg_one_hot[index]
        neg_names = neg_names[index]

    # merge positive and negative labels
    one_hot = np.vstack([pos_one_hot, neg_one_hot])
    labels = np.vstack(
        [np.ones((len(pos_one_hot), 1)), np.zeros((len(neg_one_hot), 1))]
    )
    names = np.concatenate([pos_names, neg_names])
    names = names.astype("S")

    # shuffle indices for train, validation, and test sets
    train, valid, test, indices = wrangle.split_dataset(
        one_hot, labels, valid_frac=valid_frac, test_frac=test_frac
    )

    # save to hdf5 file
    file_path = os.path.join(data_path, experiment + "_" + str(window) + ".h5")
    with h5py.File(file_path, "w") as fout:
        fout.create_dataset("x_train", data=train[0], compression="gzip")
        fout.create_dataset("y_train", data=train[1], compression="gzip")
        fout.create_dataset("train_names", data=names[indices[0]], compression="gzip")
        fout.create_dataset("x_valid", data=valid[0], compression="gzip")
        fout.create_dataset("y_valid", data=valid[1], compression="gzip")
        fout.create_dataset("valid_names", data=names[indices[1]], compression="gzip")
        fout.create_dataset("x_test", data=test[0], compression="gzip")
        fout.create_dataset("y_test", data=test[1], compression="gzip")
        fout.create_dataset("test_names", data=names[indices[2]], compression="gzip")
    print("Saved to: " + file_path)
