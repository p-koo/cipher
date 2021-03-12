"""Functions for data wrangling."""

import os
import numpy as np 
import pandas as pd
import subprocess


def filter_max_length(bed_path, output_path, max_len=1000):
    """
    Function to plot histogram of bed file peak sizes; automatically infers compression from extension and allows for user-input in removing outlier sequence

    Parameters
    -----------
    bed_path : <str>
        Path to bed file.
    output_path : <int>
        Path to filtered bed file.
    max_len: <int>
        Cutoff for maximum length of peak -- anything above will be filtered out.

    Returns 
    -----------
    .bed of the filtered peaks

    Example
    -----------

    """

    # check if bedfile is compressed
    if bed_path.split('.')[-1] == "gz" or bed_path.split('.')[-1] == "gzip": compression="gzip"

    # load bed file
    f = open(bed_path, 'rb')
    df = pd.read_table(f, header=None, compression=compression)
    start = df[1].to_numpy()
    end = df[2].to_numpy()

    # get peak sizes
    peak_sizes = end - start
    good_index = []
    for i, size in enumerate(peak_sizes):
        if size < max_len:
            good_index.append(i)

    # create dictionary for dataframe with filtered peaks
    data = {}
    for i in range(len(df.columns)):
        data[i] = df[i].to_numpy()[good_index]

    # create new dataframe
    df_new = pd.DataFrame(data);

    # save dataframe with fixed width window size to a bed file
    df_new.to_csv(output_path, sep='\t', header=None, index=False)



def enforce_constant_size(bed_path, output_path, window):
    """Generate a bed file where all peaks have same size centered on original peak
    
    Parameters
    ----------
    bed_path : <str>    
        The path to the unfiltered bed file downloaded from 
        ENCODE. 
    output_path : <str>
        Filtered bed file name with peak size clipped to specified 
        window size. 
    window : <int> 
        Size in bps which to clip the peak sequences. 

    Returns
    ----------
    None

    Example
    ----------
    >>> window = 200
    >>> bed_path = './ENCFF252PLM.bed.gz' 
    >>> output_path = './pos_'+str(window)+'.bed'
    >>> enforce_constant_size(pos_path, pos_bed_path, window, compression='gzip')
    """
    assert isinstance(window, int) and window > 0, 'Enter positive integer window size.'
    assert os.path.exists(bed_path), 'No such bed file.'

    # set up the compression argument 
    if bed_path.split('.')[-1] == 'gz':
        compression = 'gzip'
    else:
        compression = None
    # load bed file
    f = open(bed_path, 'rb')
    df = pd.read_table(f, header=None, compression=compression)
    chrom = df[0].to_numpy().astype(str)
    start = df[1].to_numpy()
    end = df[2].to_numpy()

    # calculate center point and create dataframe
    middle = np.round((start + end)/2).astype(int)
    half_window = np.round(window/2).astype(int)

    # calculate new start and end points
    start = middle - half_window
    end = middle + half_window

    # create dictionary for dataframe
    data = {}
    for i in range(len(df.columns)):
        data[i] = df[i].to_numpy()
    data[1] = start
    data[2] = end

    # create new dataframe
    df_new = pd.DataFrame(data);

    # save dataframe with fixed width window size to a bed file
    df_new.to_csv(output_path, sep='\t', header=None, index=False)




def parse_fasta(fasta_path):
    """Parse fasta file for sequences. 
    
    Parameters
    ----------
    fasta_path : <str>
        path to fasta file

    Returns
    -------
    sequences : <numpy.ndarray>
        The parsed sequences from the input fasta file as a numpy array of sequences. 

    Example
    -------
    >>> fasta_file_path = './CTCF_example.fa' 
    >>> pos_seq = parse_fasta(fasta_file_path)
    >>> pos_seq
    array(['TAAGACCCTGTCTCTAAAAAAATTTTAAAAATTAGCCA,
       'TTTGGTGGGGCAATGCTGTTGTTTATTTCTTCACCACAAACG,
       'TACTGAACACAACCAATCCTTCAAAAATCAATTCTCAAAATT,
       ...,
       'CTTAATAGAGTACAAAAGCAGCCTTGTACCTGTGCTTCTCTC,
       'CTTCTAGCTATTCAAGCATATGATGTATTTCCTCCGATAATT,
       'TGTGCTTGTAGTCCCAGCTACTTGGGAGGCTGAGCCCAGGAG,],
      dtype='<U38')
    """

    # parse sequence and chromosome from fasta file
    num_data = np.round(sum(1 for line in open(fasta_path))/2).astype(int)
    fin = open(fasta_path, "r")
    sequences = []
    for j in range(num_data):
        coord = fin.readline()
        line = fin.readline()[:-1].upper()
        sequences.append(line)
    sequences = np.array(sequences)
    return sequences


def convert_one_hot(sequence, alphabet='ACGT'):
    """Convert DNA/RNA sequences to a one-hot representation.

    Parameters
    ----------
    sequences : <iterable>
       A container with the sequences to transform to one-hot representation. 
    max_length : <int> 
       The maximum allowable length of the sequences. If the sequences argument 
       contains variable length sequences, all sequences will be set to length `max_length`.
       Longer sequences are trimmed and shorter sequences are zero-padded. 
       default: None
    dtype : <dtype>
       The datatype of the 

    Returns
    -------
    one_hot_seq : <numpy.ndarray>
    A numpy tensor of shape (len(sequences), max_length, A)
    Example
    -------
    >>> sequences = ['AGCAC', 'AGCGA']
    >>> convert_one_hot(sequences)
    [[[1. 0. 0. 0.]
    [0. 0. 1. 0.]
    [0. 1. 0. 0.]
    [1. 0. 0. 0.]
    [0. 1. 0. 0.]]
    [[1. 0. 0. 0.]
    [0. 0. 1. 0.]
    [0. 1. 0. 0.]
    [0. 0. 1. 0.]
    [1. 0. 0. 0.]]]
    """

    # create alphabet dictionary
    alphabet_dict = {a: i for i, a in enumerate(list(alphabet))}

    # convert sequences to one-hot
    one_hot = np.zeros((len(sequences),len(sequences[0]),len(alphabet)))
    for n, seq in enumerate(sequence):
        for l, s in enumerate(seq):
            one_hot[n,l,alphabet_dict[s]] = 1.
    return one_hot


def convert_onehot_to_sequence(one_hot, alphabet='ACGT'):
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
    assert alphabet in ['ACGT', 'ACGU'], 'Enter a valid alphabet'

    # convert alphabet to dictionary
    alphabet_dict = {i: a for i, a in enumerate(list(alphabet))}

    # get indices of one-hot
    seq_indices = np.argmax(one_hot, axis=2)  # (N, L)

    # convert index to sequence 
    sequences = []
    for seq_index in seq_indices:
        seq = pd.Series(seq_index).map(alphabet_dict)
        sequences.append(seq)
    return sequences





def filter_nonsense_sequences(sequences):
    """Filter sequences with N.

    Parameters
    ----------
    sequences : <numpy.ndarray> 
        A numpy vector of sequence strings. 
    
    Returns
    -------
    filter_sequences : <numpy.ndarray>
        The parsed sequences from the input fasta file as a numpy array of sequences. 
    good_index : <numpy.ndarray>
        A numpy array of indices corresponding to sequences without nonsense 'N' entries. 
    
    Example
    -------
    >>> print(sequences)
    ['GGCTGAAATGGCCACTGGAA' 'ACGCTCTCTCATCAAGTGGT' 'GCAGAANANCGAACACCAAC'
    'NNCNNCANCNACNGGGGAAC' 'GCCTAGTCCAGACATAATTC']
    >>> print(filter_nonsense_sequences(sequences))
    (array(['GGCTGAAATGGCCACTGGAA', 'ACGCTCTCTCATCAAGTGGT',
            'GCCTAGTCCAGACATAATTC'], dtype='<U20'), array([0, 1, 4]))
    """

    # filter sequences if contains at least one 'N' character
    good_index = []
    filter_sequences = []
    for i, seq in enumerate(sequences):
        if 'N' not in seq.upper():
            good_index.append(i)
            filter_sequences.append(seq)
    return np.array(filter_sequences), np.array(good_index)




def match_gc(pos_one_hot, neg_one_hot):
    """Given a set of one hot encoded positive and negative 
    sequences for TF binding, discard negative sequences that 
    do not the GC content in the set of positive sequences. 

    Parameters
    ----------
    pos_one_hot : <numpy.ndarray>
        One hot encoding of the positive sequences.
        
    neg_one_hot : <numpy.ndarray>
        One hot encoding of the negative sequences. 
    
    Returns
    -------
    neg_one_hot_filtered : <numpy.ndarray>
        Numpy matrix of one hot encoded negative sequences 
        that match gc content profile of positive sequences. 
    
    Example
    -------
    TODO. 
    """

    # nucleotide frequency matched background
    f_pos = np.mean(pos_one_hot, axis=2)
    f_neg = np.mean(neg_one_hot, axis=2)

    #get GC content for pos and neg sequences
    gc_pos = np.sum(f_pos[:,1:3], axis=1)
    gc_neg = np.sum(f_neg[:,1:3], axis=1)

    # sort by gc content
    gc_pos_sorted = np.sort(gc_pos)
    neg_index_sorted = np.argsort(gc_neg)
    gc_neg_sorted = np.sort(gc_neg)    

    # find nucleotide GC best matches between pos and neg sequences 
    match_index = []
    index = 0 
    for i, gc in enumerate(gc_pos_sorted):
        while index < len(gc_neg_sorted)-1: 
            if (abs(gc - gc_neg_sorted[index+1]) <=  abs(gc - gc_neg_sorted[index])):  
                index += 1
            else: 
                break         
        match_index.append(index)

    neg_one_hot_filtered = neg_one_hot[neg_index_sorted[match_index]]

    return neg_one_hot_filtered



def bedtools_getfasta(bed_path, genome_path, output_path, strand=True, exe_call="bedtools"):
    """Extract DNA sequences from a fasta file based on feature coordinates.
    Wrapper around `bedtools getfasta`. This function was made to
    work with bedtools version 2.27.1. It is not guaranteed to work
    with other versions. It is not even guaranteed to work with version 2.27.1, but
    it could and probably will.

    Parameters
    ----------
    genome_path : str, Path-like
        path to reference genome in fasta format.
    output_path : str, Path-like
        Output FASTA file.
    bed_path : str, Path-like
        BED/GFF/VCF file of ranges to extract from `input_fasta`.
    strand : bool
        Force strandedness. If the feature occupies the antisense
        strand, the squence will be reverse complemented.
    exe_call : Path-like
        The path to the `bedtools` executable. By default, uses `bedtools` in `$PATH`.
    
    Returns
    -------
    Instance of `subprocess.CompletedProcess`.
    """
    args = [str(exe_call), "getfasta"]
    if strand:
        args.append("-s")
    args.extend(["-fi", str(genome_path), "-bed", str(bed_path), "-fo", str(output_path)])
    try:
        return subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise subprocess.SubprocessError(e.stderr.decode()) from e


def bedtools_nonintersect(bed_path, bed_path2, output_bed_path, exe_call="bedtools"):
    """
    Gets non-overlap between dnase .bed file and tf .bed file.

    Parameters
    ----------
    bed_path : <str>
        base bed path
    bed_path2 : <str>
        bed path to compare against
    output_bed_path : <str>
        output path to non-intersected peaks
    
    Returns
    -------
    .bed file with no overlap with dnase .bed file in output_bed_path loc
    """ 
    # cmd = ['bedtools', 'intersect', '-v', '-wa', '-a', bed_path, '-b', bed_path2, '>', output_bed_path]
    os.system(exe_call+' intersect -v -wa -a '+bed_path+' -b '+bed_path2+' > '+output_bed_path)



def split_dataset(one_hot, labels, valid_frac=0.1, test_frac=0.2):
    """split dataset into training, cross-validation, and test set"""

    def split_index(num_data, valid_frac, test_frac):
        # split training, cross-validation, and test sets

        train_frac = 1 - valid_frac - test_frac
        cum_index = np.array(np.cumsum([0, train_frac, valid_frac, test_frac])*num_data).astype(int)
        shuffle = np.random.permutation(num_data)
        train_index = shuffle[cum_index[0]:cum_index[1]]
        valid_index = shuffle[cum_index[1]:cum_index[2]]
        test_index = shuffle[cum_index[2]:cum_index[3]]

        return train_index, valid_index, test_index


    # split training, cross-validation, and test sets
    num_data = len(one_hot)
    train_index, valid_index, test_index = split_index(num_data, valid_frac, test_frac)

    # split dataset
    train = (one_hot[train_index], labels[train_index,:])
    valid = (one_hot[valid_index], labels[valid_index,:])
    test = (one_hot[test_index], labels[test_index,:])
    indices = [train_index, valid_index, test_index]

    return train, valid, test, indices



# TODO: split according to chromosome

# TODO: function to generate fasta file

# TODO: function to load bed file to dataframe

# TODO: function to save dataframe to a bed file

# TODO: random split into k-fold cross validation




