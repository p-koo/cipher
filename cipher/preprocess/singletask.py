"""Functions for processing single-task ChIP-seq data generation."""


# TODO: Filter N

# TODO: Filter by size

# Function to make bed file constant window size
def enforce_constant_size(bed_path, output_path, window):
    """generate a bed file where all peaks have same size centered on original peak
    
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
    -------
    None


    Example
    -------
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

    # filter any negative start positions
    data = {}
    for i in range(len(df.columns)):
        data[i] = df[i].to_numpy()
    data[1] = start
    data[2] = end

    # create new dataframe
    df_new = pd.DataFrame(data);

    # save dataframe with fixed width window size to a bed file
    df_new.to_csv(output_path, sep='\t', header=None, index=False)

# Function to parse fasta file of sequences 
def parse_fasta(seq_path):
    """Parse fasta file for sequences. 

    Parameters
    ----------
    seq_path : <str>


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
    num_data = np.round(sum(1 for line in open(seq_path))/2).astype(int)
    fin = open(seq_path, "r")
    sequences = []
    for j in range(num_data):
        coord = fin.readline()
        line = fin.readline()[:-1].upper()
        sequences.append(line)
    sequences = np.array(sequences)
    return sequences

# Function filter out nonsense sequences. 
def filter_nonsense_sequences(sequences):
    """Parse fasta file for sequences

    Parameters
    ----------
    seq_path : <numpy.ndarray> 
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
       'GCAGAANANCGAACACCAAC', 'NNCNNCANCNACNGGGGAAC',
       'GCCTAGTCCAGACATAATTC'], dtype='<U20'), array([0, 1, 4]))
    """

    # parse sequence and chromosome from fasta file
    good_index = []
    filter_sequences = []

    for i, seq in enumerate(sequences):
        if 'N' not in seq.upper():
            good_index.append(i)
        filter_sequences.append(seq)

    filter_sequences = np.array(filter_sequences)
    good_index = np.array(good_index)
    return filter_sequences, good_index


def convert_one_hot(sequence, max_length=None, dtype=np.float32):
	"""convert DNA/RNA sequences to a one-hot representation

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

	one_hot_seq = []
	for seq in sequence:
		seq = seq.upper()
		seq_length = len(seq)
		one_hot = np.zeros((4,seq_length))
		index = [j for j in range(seq_length) if seq[j] == 'A']
		one_hot[0,index] = 1
		index = [j for j in range(seq_length) if seq[j] == 'C']
		one_hot[1,index] = 1
		index = [j for j in range(seq_length) if seq[j] == 'G']
		one_hot[2,index] = 1
		index = [j for j in range(seq_length) if (seq[j] == 'U') | (seq[j] == 'T')]
		one_hot[3,index] = 1

		# handle boundary conditions with zero-padding
		if max_length:
			offset1 = int((max_length - seq_length)/2)
			offset2 = max_length - seq_length - offset1

		if offset1:
			one_hot = np.hstack([np.zeros((4,offset1)), one_hot])
		if offset2:
			one_hot = np.hstack([one_hot, np.zeros((4,offset2))])
		one_hot_seq.append(one_hot)

	# convert to numpy array
	one_hot_seq = np.array(one_hot_seq)
	one_hot_seq = np.transpose(one_hot_seq, (0, 2, 1))

	return one_hot_seq

# GC match between positive and negative labels
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

# TODO: Save to hdf5 file





