"""Functions to wrangle data."""

import gzip
import io
import os
import pathlib
import subprocess
import typing

import numpy as np
import pandas as pd
import scipy.stats


PathType = typing.Union[str, pathlib.Path]


def _is_gzipped(filepath: PathType) -> bool:
    """Return `True` if the file is gzip-compressed.

    This function does not depend on the suffix. Instead, the magic number of the file
    is compared to the GZIP magic number `1f 8b`. See
    https://en.wikipedia.org/wiki/Gzip#File_format for more information.
    """
    with open(filepath, "rb") as f:
        return f.read(2) == b"\x1f\x8b"



def filter_max_length(bed_path, output_path, max_len=1000):
    """Function to plot histogram of bed file peak sizes; automatically infers
    compression  from extension and allows for user-input in removing outlier sequence

    Parameters
    -----------
    bed_path : <str>
        Path to bed file.
    output_path : <int>
        Path to filtered bed file.
    max_len: <int>
        Cutoff for maximum length of peak -- anything above will be filtered out.

    Returns
    ----------
    None
    """

    # check if bedfile is compressed
    compression = "gzip" if _is_gzipped(bed_path) else None

    # load bed file
    df = pd.read_table(bed_path, header=None, sep="\t", compression=compression)
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
    df_new = pd.DataFrame(data)

    # save dataframe with fixed width window size to a bed file
    df_new.to_csv(output_path, sep="\t", header=None, index=False)



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
    assert isinstance(window, int) and window > 0, "Enter positive integer window size."
    assert os.path.exists(bed_path), "No such bed file."

    # set up the compression argument
    if bed_path.split(".")[-1] == "gz" or bed_path.split(".")[-1] == "gzip":
        compression = "gzip"
    else:
        compression = None

    # load bed file
    df = pd.read_table(bed_path, header=None, sep="\t", compression=compression)
    # chrom = df[0].to_numpy().astype(str)  # TODO: unused variable
    start = df[1].to_numpy()
    end = df[2].to_numpy()

    # calculate center point and create dataframe
    middle = np.round((start + end) / 2).astype(int)
    left_window = np.round(window / 2).astype(int)
    right_window = window - left_window

    # calculate new start and end points
    start = middle - left_window
    end = middle + right_window

    # create dictionary for dataframe
    data = {}
    for i in range(len(df.columns)):
        data[i] = df[i].to_numpy()
    data[1] = start
    data[2] = end

    # create new dataframe
    df_new = pd.DataFrame(data)

    # save dataframe with fixed width window size to a bed file
    df_new.to_csv(output_path, sep="\t", header=None, index=False)


def parse_fasta(filepath: PathType) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Parse FASTA file into arrays of descriptions and sequence data.

    Parameters
    ----------
    filepath : Path-like
        FASTA file to parse. Can be gzip-compressed.

    Returns
    -------
    tuple of two numpy arrays
        The first array contains the sequences, and the second array contains the name
        of each sequence. These arrays have the same length in the first dimension.
    """
    # FASTA format described here.
    # https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=BlastHelp
    descriptions: typing.List[str] = []
    sequences: typing.List[str] = []
    prev_line_was_sequence = False

    gzipped = _is_gzipped(filepath)
    openfile = gzip.open if gzipped else io.open
    with openfile(filepath, "rt") as f:  # type: ignore
        for line in f:
            line = line.strip()
            # handle blank lines
            if not line:
                continue
            is_description = line.startswith(">")
            if is_description:
                description = line[1:].strip()  # prune ">" char
                descriptions.append(description)
                prev_line_was_sequence = False
            else:  # is sequence data
                sequence = line.upper()
                if prev_line_was_sequence:
                    # This accounts for sequences that span multiple lines.
                    sequences[-1] += sequence
                else:
                    sequences.append(sequence)
                prev_line_was_sequence = True
    return np.array(sequences), np.array(descriptions)



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
        A numpy array of indices corresponding to sequences without nonsense 'N'
        entries.

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
        if "N" not in seq.upper():
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
    f_pos = np.mean(pos_one_hot, axis=1)
    f_neg = np.mean(neg_one_hot, axis=1)

    # get GC content for pos and neg sequences
    gc_pos = np.sum(f_pos[:, 1:3], axis=1)
    gc_neg = np.sum(f_neg[:, 1:3], axis=1)

    # sort by gc content
    gc_pos_sorted = np.sort(gc_pos)
    neg_index_sorted = np.argsort(gc_neg)
    gc_neg_sorted = np.sort(gc_neg)

    # find nucleotide GC best matches between pos and neg sequences
    match_index = []
    index = 0
    for i, gc in enumerate(gc_pos_sorted):
        while index < len(gc_neg_sorted) - 1:
            if abs(gc - gc_neg_sorted[index + 1]) <= abs(gc - gc_neg_sorted[index]):
                index += 1
            else:
                break
        match_index.append(index)

    index = neg_index_sorted[match_index]
    neg_one_hot_filtered = neg_one_hot[index]

    return neg_one_hot_filtered, index


def sample_b_matched_to_a(
    a: np.ndarray, b: np.ndarray, size: int = None, seed: int = None
) -> np.ndarray:
    """Return indices of `b` that are distributed similarly to `a`.

    Parameters
    ----------
    a : array
        One-dimensional array of samples.
    b : array
        One-dimensional array of samples from which to sample.
    size : int
        Number of samples to take from `b`. Default is `min(len(a), len(b))`.

    Returns
    -------
    Numpy array of indices with shape `(size,)`. If `size` is `None`, the shape is
    `(len(a),)`.

    Examples
    --------
    >>> a = np.array([1, 1, 2])
    >>> b = np.array([0, 0, 1, 2, 3, 4, 1])
    >>> mask = sample_b_matched_to_a(a, b, seed=42)
    >>> mask
    array([2, 6, 3])
    >>> b[mask]
    array([1, 1, 2])
    In the following example, two normal distributions are made. Despite the second
    distribution being bimodal, this function draws samples that are most similar to the
    first distribution.
    >>> rng = np.random.RandomState(seed=42)
    >>> x = rng.normal(size=1000)
    >>> y = np.concatenate((rng.normal(size=1000), rng.normal(loc=5, size=1000)))
    >>> _ = plt.hist(x, bins=25, range=(-3, 8))
    >>> plt.show()
    >>> _ = plt.hist(y, bins=25, range=(-3, 8))
    >>> plt.show()
    >>> mask = sample_b_matched_to_a(x, y, seed=42)
    >>> _ = plt.hist(y[mask], bins=25, range=(-3, 8))
    >>> plt.show()
    """
    a, b = np.asanyarray(a), np.asanyarray(b)
    if a.ndim != 1:
        raise ValueError("`a` must be one-dimensional")
    if b.ndim != 1:
        raise ValueError("`b` must be one-dimensional")
    if size is None:
        size = min(a.shape[0], b.shape[0])
    if size == b.shape[0]:
        # Optimize -- return range of indices if we would sample all of `b` anyway.
        return np.arange(b.shape[0])
    kde = scipy.stats.gaussian_kde(a)
    p = kde(b)
    p = p / p.sum()  # scale to sum to 1

    idxs = np.arange(b.shape[0])
    rng = np.random.RandomState(seed=seed)
    return rng.choice(idxs, size=size, replace=False, p=p)


def bedtools_getfasta(
    bed_path, genome_path, output_path, strand=True, bedtools_exe="bedtools"
):
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
    args = [str(bedtools_exe), "getfasta"]
    if strand:
        args.append("-s")
    args.extend(
        ["-fi", str(genome_path), "-bed", str(bed_path), "-fo", str(output_path)]
    )
    try:
        return subprocess.run(
            args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        raise subprocess.SubprocessError(e.stderr.decode()) from e


def bedtools_intersect(
    a: PathType,
    b: PathType,
    output_path: PathType,
    write_a=True,
    nonoverlap=False,
    bedtools_exe: PathType = "bedtools",
) -> subprocess.CompletedProcess:
    """Report overlaps between two feature files.
    This is an incomplete wrapper around `bedtools intersect` version 2.27.1.
    The set of arguments here does not include all of the command-line arguments.
    Parameters
    ----------
    a : Path-like
        First feature file <bed/gff/vcf/bam>.
    b : Path-like
        Second feature file <bed/gff/vcf/bam>.
    output_bedfile : Path-like
        Name of output file. Can be compressed (`.bed.gz`).
    write_a : bool
        Write the original entry in `a` for each overlap.
    write_b : bool
        Write the original entry in `b` for each overlap.
    invert_match : bool
        Only report those entries in `a` that have no overlaps with `b`.
    bedtools_exe : Path-like
        The path to the `bedtools` executable. By default, uses `bedtools` in `$PATH`.
    Returns
    -------
    Instance of `subprocess.CompletedProcess`.
    """
    args = [str(bedtools_exe), "intersect"]
    if write_a:
        args.append("-wa")
    if nonoverlap:
        args.append("-v")
    args.extend(["-a", str(a), "-b", str(b)])

    output_bedfile = pathlib.Path(output_path)
    gzipped_output = output_bedfile.suffix == ".gz"
    openfile = gzip.open if gzipped_output else io.open
    try:
        # We cannot write stdout directly to a gzip file.
        # See https://stackoverflow.com/a/2853396/5666087
        process = subprocess.run(
            args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if not process.stdout:
            raise subprocess.SubprocessError(
                f"empty stdout, aborting. stderr is {process.stderr.decode()}"
            )
        with openfile(output_path, mode="wb") as f:  # type: ignore
            f.write(process.stdout)
        return process
    except subprocess.CalledProcessError as e:
        raise subprocess.SubprocessError(e.stderr.decode()) from e


def split_dataset(one_hot, labels, valid_frac=0.1, test_frac=0.2):
    """split dataset into training, cross-validation, and test set"""

    def split_index(num_data, valid_frac, test_frac):
        # split training, cross-validation, and test sets

        train_frac = 1 - valid_frac - test_frac
        cum_index = np.array(
            np.cumsum([0, train_frac, valid_frac, test_frac]) * num_data
        ).astype(int)
        shuffle = np.random.permutation(num_data)
        train_index = shuffle[cum_index[0] : cum_index[1]]
        valid_index = shuffle[cum_index[1] : cum_index[2]]
        test_index = shuffle[cum_index[2] : cum_index[3]]

        return train_index, valid_index, test_index

    # split training, cross-validation, and test sets
    num_data = len(one_hot)
    train_index, valid_index, test_index = split_index(num_data, valid_frac, test_frac)

    # split dataset
    train = (one_hot[train_index], labels[train_index, :])
    valid = (one_hot[valid_index], labels[valid_index, :])
    test = (one_hot[test_index], labels[test_index, :])
    indices = [train_index, valid_index, test_index]

    return train, valid, test, indices


# TODO: split according to chromosome
def split_dataset_by_chr(one_hot, labels, names, chromosome_test, chromosome_valid):
    """takes list of chromosomes and which split to put them in
    takes names of peaks and extracts chromosomes

    Parameters
    ____________
    one_hot : < array of one_hot encoded sequences >
    labels : < array indicating class membership of corresponding sample >
    names : < array of sequence chromosomal coordinates
    chromosome_test : < list of chromosomes to be put in test >
    chromosome_valid : < list of chromosomes to be put in val >

    Returns
    _____________
    train,test,validation split and the corresponding indices

    Example
    _____________





    """
    # extracts chromosome from each peak
    chrs = [x.split(":")[0] for x in names]

    valid_index = []
    for i in chromosome_valid:
        valid_index.extend([x for x in range(len(chrs)) if chrs[x] == i])

    test_index = []
    for i in chromosome_test:
        test_index.extend([x for x in range(len(chrs)) if chrs[x] == i])

    num_data = len(one_hot)

    # produce shuffled list of all indices for downstream pruning:
    shuffle = np.random.permutation(num_data)
    shuffle = set(shuffle)

    # defining set of indices not in train:
    non_train_idx = valid_index
    non_train_idx.extend(test_index)
    non_train_idx = set(non_train_idx)

    # removing test/valid indices from shuffled data to produce train:
    # takes values in shuffle not in non_train_idx

    train_index = list(shuffle ^ non_train_idx)

    # split dataset
    train = (one_hot[train_index], labels[train_index, :])
    valid = (one_hot[valid_index], labels[valid_index, :])
    test = (one_hot[test_index], labels[test_index, :])
    indices = [train_index, valid_index, test_index]

    return train, valid, test, indices

