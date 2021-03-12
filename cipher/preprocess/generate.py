import os, h5py
import numpy as np
import argparse
from . import wrangle

def process_singletask(tf_path, dnase_path, genome_path, data_path, experiment, 
                       window=200, alphabet='ACGT', compression='gzip', max_len=300, 
                       valid_frac=0.1, test_frac=0.2):

    # remove extremely large peaks
    tf_filtered_path = os.path.join(data_path, experiment+'_pos_filtered.bed')
    wrangle.filter_max_length(tf_path, tf_filtered_path, max_len)

    # create new bed file with window size enforced
    pos_bed_path = os.path.join(data_path, experiment+'_pos_'+str(window)+'.bed')
    wrangle.enforce_constant_size(tf_filtered_path, pos_bed_path, window)

    # extract sequences from bed file and save to fasta file
    pos_fasta_path = os.path.join(data_path, experiment+'_pos.fa')
    wrangle.bedtools_getfasta(pos_bed_path, genome_path, output_path=pos_fasta_path)

    # parse sequence from fasta file
    pos_seq, pos_names = wrangle.parse_fasta(pos_fasta_path)

    # filter sequences with absent nucleotides
    pos_seq, good_index = wrangle.filter_nonsense_sequences(pos_seq)
    pos_names = pos_names[good_index]

    # convert filtered sequences to one-hot representation
    pos_one_hot = wrangle.convert_one_hot(pos_seq, alphabet)

    # get non-overlap between pos peaks and neg peaks
    neg_bed_path = os.path.join(data_path, experiment + '_nonoverlap.bed')
    wrangle.bedtools_nonintersect(dnase_path, tf_path, neg_bed_path)

    # create new bed file with window enforced
    neg_bed_path2 = os.path.join(data_path, experiment + '_neg_'+str(window)+'.bed')
    wrangle.enforce_constant_size(neg_bed_path, neg_bed_path2, window)

    # extract sequences from bed file and save to fasta file
    neg_fasta_path = os.path.join(data_path, experiment + '_neg.fa')
    wrangle.bedtools_getfasta(neg_bed_path2, genome_path, output_path=neg_fasta_path)

    # parse sequence and chromosome from fasta file
    neg_seq, neg_names = wrangle.parse_fasta(neg_fasta_path)

    # filter sequences with absent nucleotides
    neg_seq, good_index = wrangle.filter_nonsense_sequences(neg_seq)
    neg_names = neg_names[index]

    # convert filtered sequences to one-hot representation
    neg_one_hot = wrangle.convert_one_hot(neg_seq, alphabet)

    # calling match_gc function to balance neg sequences with pos by GC content:
    neg_one_hot_gc, gc_index = wrangle.match_gc(pos_one_hot, neg_one_hot)
    neg_names = neg_names[gc_index]

    # merge postive and negative sequences
    one_hot = np.vstack([pos_one_hot, neg_one_hot_gc])
    labels = np.vstack([np.ones((len(pos_one_hot), 1)), np.zeros((len(neg_one_hot_gc), 1))])
    names = pos_names + neg_names

    # shuffle indices for train, validation, and test sets
    train, valid, test, indices = wrangle.split_dataset(one_hot, labels, valid_frac=valid_frac, test_frac=test_frac)
    
    # save to hdf5 file
    file_path = os.path.join(data_path, experiment+'_'+str(window)+'.h5')
    with h5py.File(file_path, 'w') as fout:
        x_train = fout.create_dataset('x_train', data=train[0], dtype='float32', compression="gzip")
        y_train = fout.create_dataset('y_train', data=train[1], dtype='int8',    compression="gzip")
        y_test = fout.create_dataset('train_names', data=names[indices[0]], dtype='str', compression="gzip")
        x_valid = fout.create_dataset('x_valid', data=valid[0], dtype='float32', compression="gzip")
        y_valid = fout.create_dataset('y_valid', data=valid[1], dtype='int8',    compression="gzip")
        y_test = fout.create_dataset('valid_names', data=names[indices[1]], dtype='str', compression="gzip")
        x_test  = fout.create_dataset('x_test',  data=test[0],  dtype='float32', compression="gzip")
        y_test  = fout.create_dataset('y_test',  data=test[1],  dtype='int8',    compression="gzip")
        y_test = fout.create_dataset('test_names', data=names[indices[2]], dtype='str', compression="gzip")
    print('Saved to: ' + file_path)
