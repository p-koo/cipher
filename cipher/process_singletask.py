'''
This script will take the following arguments in the command line, import helper functions from an external script, and conduct all preprocessing steps:
command line usage: python preprocess_wrapper.py --bed_pos = <> --neg_pos= <> ,...
Parameters
----------
bed_pos : <str>
bed_neg : <str>
window_size : <int>
data : <str>
experiment : <str>
ref : <str>
test_frac : <float>
validation_frac : <float>
Returns
_______
train, test, and validation .hdf5 files in 'data' folder
Example
-------
For .bed files and reference .fasta in folder called 'Data':
python preprocess_wrapper.py --bed_pos='Data/ENCFF252PLM.bed.gz' --bed_neg='Data/ENCFF209DJG.bed.gz' --window_size=200 --data_path='Data/' --experiment='CTCF' --ref='Data/hg19.fa'
'''

import os, h5py
import numpy as np
import argparse
from cipher.preprocess import wrangle

def main(args):

    # parse arguments
    tf_path = args.tf_path              # path to tf chip bed file
    dnase_path = args.dnase_path        # path to dnase bed file
    genome_path = args.genome_path      # path to reference genome
    data_path = args.data_path          # path for tmp files
    experiment = args.experiment        # name of experiment

    # optional arguments    
    window = args.window                # sequence length for dataset
    alphabet = args.alphabet            # alphabet 'ACGT'
    compression = args.compression      # compression of bed files (gzip or None)
    max_len = args.max_len              # maximum length of peaks (above this cutoff are filtered out)
    valid_frac = args.valid_frac        # validation set fraction
    test_frac = args.test_frac          # test set fraction

    # remove extremely large peaks
    tf_filtered_path = os.path.join(data_path, experiment+'_pos_filtered.bed')
    wrangle.filter_max_length(tf_path, tf_filtered_path, max_len)

    # create new bed file with window size enforced
    pos_bed_path = os.path.join(data_path, experiment+'_pos_'+str(window)+'.bed')
    wrangle.enforce_constant_size(tf_filtered_path, pos_bed_path, window)

    # extract sequences from bed file and save to fasta file
    pos_fasta_path = os.path.join(data_path, experiment+'_pos.fa')
    wrangle.bedtools_getfasta(pos_bed_path, genome_path, output_path=pos_fasta_path, strand=True)

    # parse sequence from fasta file
    pos_seq, pos_names = wrangle.parse_fasta(pos_fasta_path)

    # filter sequences with absent nucleotides
    pos_seq, good_index = wrangle.filter_nonsense_sequences(pos_seq)
    pos_names = pos_names[good_index]

    # convert filtered sequences to one-hot representation
    pos_one_hot = wrangle.convert_one_hot(pos_seq, alphabet)

    # get non-overlap between pos peaks and neg peaks
    neg_bed_path = os.path.join(data_path, experiment + '_nonoverlap.bed')
    wrangle.bedtools_intersect(dnase_path, tf_path, neg_bed_path, write_a=True, nonoverlap=False)

    # create new bed file with window enforced
    neg_bed_path2 = os.path.join(data_path, experiment + '_neg_'+str(window)+'.bed')
    wrangle.enforce_constant_size(neg_bed_path, neg_bed_path2, window)

    # extract sequences from bed file and save to fasta file
    neg_fasta_path = os.path.join(data_path, experiment + '_neg.fa')
    wrangle.bedtools_getfasta(neg_bed_path2, genome_path, output_path=neg_fasta_path, strand=True)

    # parse sequence and chromosome from fasta file
    neg_seq, neg_names = wrangle.parse_fasta(neg_fasta_path)

    # filter sequences with absent nucleotides
    neg_seq, good_index = wrangle.filter_nonsense_sequences(neg_seq)
    neg_names = neg_names[good_index]

    # convert filtered sequences to one-hot representation
    neg_one_hot = wrangle.convert_one_hot(neg_seq, alphabet)

    # calling match_gc function to balance neg sequences with pos by GC content:
    neg_one_hot_gc, gc_index = wrangle.match_gc(pos_one_hot, neg_one_hot)
    neg_names = neg_names[gc_index]

    # merge postive and negative sequences
    one_hot = np.vstack([pos_one_hot, neg_one_hot_gc])
    labels = np.vstack([np.ones((len(pos_one_hot), 1)), np.zeros((len(neg_one_hot_gc), 1))])
    names = np.concatenate([pos_names, neg_names])
    names = names.astype("S")
    
    # shuffle indices for train, validation, and test sets
    train, valid, test, indices = wrangle.split_dataset(one_hot, labels, valid_frac=valid_frac, test_frac=test_frac)

    # save to hdf5 file
    file_path = os.path.join(data_path, experiment+'_'+str(window)+'.h5')
    with h5py.File(file_path, "w") as fout:
        x_train = fout.create_dataset("x_train", data=train[0], compression="gzip")
        y_train = fout.create_dataset("y_train", data=train[1], compression="gzip")
        y_test = fout.create_dataset("train_names", data=names[indices[0]], compression="gzip")
        x_valid = fout.create_dataset("x_valid", data=valid[0], compression="gzip")
        y_valid = fout.create_dataset("y_valid", data=valid[1], compression="gzip")
        y_test = fout.create_dataset("valid_names", data=names[indices[1]], compression="gzip")
        x_test = fout.create_dataset("x_test", data=test[0], compression="gzip")
        y_test = fout.create_dataset("y_test", data=test[1], compression="gzip")
        y_test = fout.create_dataset("test_names", data=names[indices[2]], compression="gzip")
    print("Saved to: " + file_path)






if __name__ == "__main__":  
    ## ---------- Parse arguments ----------
    # Eg. 
    # $ process_singletask.py -tf tf_path -dnase dnase_path -g genome_path -d data_path -e experiment -w 200 -c gzip -m 300 

    parser = argparse.ArgumentParser(description='Pre-Processing TF ChIP-seq datasets for binary classification.')

    parser.add_argument('-tf', 'tf_path', dest='tf_path', type=str, required=True, help='Path to positive bed file.')
    parser.add_argument('-dnase', 'dnase_path', dest='dnase_path', type=str, required=True, help='Path to negative bed file.')
    parser.add_argument('-g', 'genome_path', dest='genome_path', type=str, required= True, help='Path to reference genome in fasta format.')
    parser.add_argument('-d', 'data_path', dest='data_path', type=str, required= True, help='Path to where files will be saved.')
    parser.add_argument('-e', 'experiment', dest='experiment', type=str, required= True, help='Experiment name -- used for saving files.')

    parser.add_argument('-w','--window', dest='window', type=int, default=200, help='Desired sequence length')
    parser.add_argument('-t','--test_frac', dest="test_frac", type=float, default=0.2, help="Fraction of data in test set")
    parser.add_argument('-v','--validation_frac', dest="validation_frac", type=float, default=0.1, help="Fraction of data in validation set")
    parser.add_argument('-a','--alphabet', dest="alphabet", type=str, default='ACGT', help="alphabet: for DNA = ACGT ")
    parser.add_argument('-c','--compression', dest="compression", type=str, default='gzip', help="compression of bed files ")
    parser.add_argument('-m','--max_len', dest="max_len", type=int, default=500, help="maximum length of a peak -- anything above will be removed.")

    # run main
    main(parser.parse_args())


