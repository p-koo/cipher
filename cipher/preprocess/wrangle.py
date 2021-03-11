"""Functions for data wrangling."""
import os, sys, h5py
import numpy as np
import pandas as pd
import subprocess

# TODO: Function to convert sequence to one-hot
def convert_one_hot(sequence, max_length=None):
	"""convert DNA/RNA sequences to a one-hot representation"""

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

	return one_hot_seq

# TODO: Function to convert one-hot to sequence

# TODO: function to parse fasta file
def parse_fasta(seq_path):
    """Parse fasta file for sequences"""

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

# TODO: function to generate fasta file

# TODO: function to load bed file to dataframe

# TODO: function to save dataframe to a bed file

# TODO: Function to calculate GC-content

# TODO: bedtools getfasta
def bedtools_getfasta(
    *,
    input_fasta: PathType,
    output_fasta: PathType,
    bed_file: PathType,
    use_name=False,
    use_name_coords=False,
    split=False,
    tab_delim=False,
    force_strandedness=False,
    full_header=False,
    bedtools_exe: PathType = "bedtools",
) -> subprocess.CompletedProcess:
    """Extract DNA sequences from a fasta file based on feature coordinates.
    Wrapper around `bedtools getfasta`. This function was made to
    work with bedtools version 2.27.1. It is not guaranteed to work
    with other versions. It is not even guaranteed to work with version 2.27.1, but
    it could and probably will.
    Parameters
    ----------
    input_fasta : str, Path-like
        Input FASTA file.
    output_fasta : str, Path-like
        Output FASTA file.
    bed_file : str, Path-like
        BED/GFF/VCF file of ranges to extract from `input_fasta`.
    use_name : bool
        Use the name field for the FASTA header.
    use_name_coords : bool
        Use the name and coordinates for the FASTA header.
    split : bool
        Given BED12 format, extract and concatenate the sequences
        from the BED "blocks" (e.g., exons).
    tab_delim : bool
        Write output in TAB delimited format.
    force_strandedness : bool
        Force strandedness. If the feature occupies the antisense
        strand, the squence will be reverse complemented.
    full_header : bool
        Use full FASTA header.
    bedtools_exe : Path-like
        The path to the `bedtools` executable. By default, uses `bedtools` in `$PATH`.
    Returns
    -------
    Instance of `subprocess.CompletedProcess`.
    """
    args = [str(bedtools_exe), "getfasta"]
    if use_name:
        args.append("-name")
    if use_name_coords:
        args.append("-name+")
    if split:
        args.append("-split")
    if tab_delim:
        args.append("-tab")
    if force_strandedness:
        args.append("-s")
    if full_header:
        args.append("-fullHeader")
    args.extend(
        ["-fi", str(input_fasta), "-bed", str(bed_file), "-fo", str(output_fasta)]
    )
    try:
        return subprocess.run(
            args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        raise subprocess.SubprocessError(e.stderr.decode()) from e

# TODO: bedtools non overlap
def bedtools_intersect(dnase_bed_path,tf_bed_path,output_bed_path):
    """
    gets non-overlap between dnase .bed file and tf .bed file

    Parameters
    __________
    dnase_bed_path : <str>
    tf_bed_path : <str>
    output_bed_path : <str>
    
    Returns
    ________
    tf .bed file with no overlap with dnase .bed file in output_bed_path loc
    """	
    cmd = ['bedtools', 'intersect', '-v', '-wa', '-a', dnase_bed_path, '-b', tf_bed_path, '>', neg_bed_path]
    os.system('bedtools intersect -v -wa -a '+dnase_bed_path+' -b '+tf_bed_path+' > '+neg_bed_path)

# TODO: bedtools overlap

# TODO: random split into train test valid
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

# TODO: random split into k-fold cross validation

# TODO: split according to chromosome
