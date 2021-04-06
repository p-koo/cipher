from collections import OrderedDict
import random
import h5py
import numpy.random as npr
import numpy as np
from sklearn import preprocessing
import gzip
import os
import re
import subprocess
import sys

import numpy as np


def create_samplefile(
    data_dir,
    metadata_path,
    assembly="GRCh38",
    sample_output_path="sample_beds.txt",
    subset_output_path="selected_data.csv",
    criteria={},
    exp_accession_list=[],
):
    """Generate subset of metadata table and sample file for further processing

    Parameters
    ----------
    data_dir : str
        dataset directory with files
    metadata_path : str
        metadata containing experiments of interest
    assembly : str
        genome assembly to select
    sample_output_path : str
        path where the sample bed file will be saved
    subset_output_path : str
        path where the subset of metadata table used will be saved as a to_csv
    criteria : dict
        dictionary of column, value pairs to use in making a selection
    exp_accession_list : list
        list of experiments to select, if empty select all in the metadata table
    """
    assert metadata_path.endswith(".tsv") or metadata_path.endswith(".csv"), print(
        "Metadata should be a tsv or a csv file"
    )
    if metadata_path.endswith(".tsv"):
        metadata = pd.read_csv(metadata_path, sep="\t")
    elif metadata_path.endswith(".csv"):
        metadata = pd.read_csv(metadata_path, sep=",")

    if not exp_accession_list:
        print("Generating sample file from all of the metadata table")
        exp_accession_list = list(set(metadata["Experiment accession"]))

    summary = []
    selected_rows = []
    for exp_accession in exp_accession_list:
        # filter files and save selection
        selection = _process_exp(data_dir, exp_accession, metadata, assembly, criteria)
        if selection:
            selected_rows.append(selection[0])
            summary.append(selection[1])
    selected_data = pd.concat(selected_rows)
    selected_data["label"] = [entry[0] for entry in summary]
    selected_data["path"] = [entry[1] for entry in summary]

    with open(sample_output_path, "w") as filehandle:
        for line in summary:
            filehandle.write("{}\t{}\n".format(line[0], line[1]))
    selected_data.to_csv(subset_output_path)



def multitask_bed_generation(
    target_beds_file,
    feature_size=1000,
    merge_overlap=200,
    out_prefix="features",
    chrom_lengths_file=None,
    db_act_file=None,
    db_bed=None,
    ignore_auxiliary=False,
    no_db_activity=False,
    ignore_y=False,
):
    """Merge multiple bed files to select sample sequence regions with at least one
    peak.

    This function outputs a .bed file in the specified directory containing seven
    columns: chromosome, sequence start, sequence end, name, score, strand, and indexes
    of experiments that have a peak in this region.

    Parameters
    ----------
    target_beds_file: str
        Location of the sample file containing experiment label and their corresponding
        file locations. Should be a two column text file, first row contains label,
        second row contain directory for the .bed/.bed.gz file.
    feature_size: int, optional
        Length of the sequence region per sample in output. Default to 1000.
    merge_overlap: int, optional
        After adjusting peaks into feature_size, if two peak regions overlaps more than
        this amount, they will be re-centered and merged into a single sample. Defaults
        to 200.
    output_prefix: str, optional
        Location and naming of the output bed file. Default to 'features.bed'
    chrom_lenghts_file: str, optional
        Location of the chrom.sizes file. Default to None.
    db_act_file: str, optional
        Location of the existing database activity table. Defaults to None.
    db_bed: str, optional
        Location of the existing database .bed file. Defaults to None.
    ignore_auxiliary: bool, optional
        Ignore auxiliary chromosomes. Defaults to False.
    no_db_acticity: bool, optional
        Whether to pass along the activities of the database sequences. Defaults to
        False.
    ignor_y: bool, optional
        Ignore Y chromsosome features. Defaults to False.

    Returns
    -------
    None

    Examples
    --------
    >>> multitask_bed_generation(
        example_file,chrom_lengths_file='/data/hg38.chrom.size',
        feature_size=1000,merge_overlap=200,out_prefix='/data/multitask.bed')
    """

    if not target_beds_file:
        raise Exception(
            "Must provide file labeling the targets and providing BED file paths."
        )

    # determine whether we'll add to an existing DB
    db_targets = []
    db_add = False
    if db_bed is not None:
        db_add = True
        if not no_db_activity:
            if db_act_file is None:
                raise ValueError(
                    "Must provide both activity table or specify -n if you want to add"
                    " to an existing database"
                )
            else:
                # read db target names
                db_act_in = open(db_act_file)
                db_targets = db_act_in.readline().strip().split("\t")
                db_act_in.close()

    # read in targets and assign them indexes into the db
    target_beds = []
    target_dbi = []
    for line in open(target_beds_file):
        a = line.split()
        if len(a) != 2:
            print(a)
            print(
                "Each row of the target BEDS file must contain a label and BED file"
                " separated by whitespace",
                file=sys.stderr,
            )
            sys.exit(1)
        target_dbi.append(len(db_targets))
        db_targets.append(a[0])
        target_beds.append(a[1])

    # read in chromosome lengths
    chrom_lengths = {}
    if chrom_lengths_file is not None:
        chrom_lengths = {}
        for line in open(chrom_lengths_file):
            a = line.split()
            chrom_lengths[a[0]] = int(a[1])
    else:
        print(
            "Warning: chromosome lengths not provided, so regions near ends may be"
            " incorrect.",
            file=sys.stderr,
        )

    #################################################################
    # print peaks to chromosome-specific files
    #################################################################
    chrom_files = {}
    chrom_outs = {}

    peak_beds = target_beds
    if db_add:
        peak_beds.append(db_bed)

    for bi in range(len(peak_beds)):
        if peak_beds[bi][-3:] == ".gz":
            peak_bed_in = gzip.open(peak_beds[bi], "rt")
        else:
            peak_bed_in = open(peak_beds[bi])

        for line in peak_bed_in:
            if not line.startswith("#"):
                a = line.split("\t")
                a[-1] = a[-1].rstrip()

                # hash by chrom/strand
                chrom = a[0]
                strand = "+"
                if len(a) > 5 and a[5] in "+-":
                    strand = a[5]
                chrom_key = (chrom, strand)

                # adjust coordinates to midpoint
                start = int(a[1])
                end = int(a[2])
                mid = int(_find_midpoint(start, end))
                a[1] = str(mid)
                a[2] = str(mid + 1)

                # open chromosome file
                if chrom_key not in chrom_outs:
                    chrom_files[chrom_key] = "%s_%s_%s.bed" % (
                        out_prefix,
                        chrom,
                        strand,
                    )
                    chrom_outs[chrom_key] = open(chrom_files[chrom_key], "w")

                # if it's the db bed
                if db_add and bi == len(peak_beds) - 1:
                    if no_db_activity:
                        # set activity to null
                        a[6] = "."
                        print("\t".join(a[:7]), file=chrom_outs[chrom_key])
                        # print >> chrom_outs[chrom_key], '\t'.join(a[:7])
                    else:
                        print(line, chrom_outs[chrom_key])
                        # print >> chrom_outs[chrom_key], line,

                # if it's a new bed
                else:
                    # specify the target index
                    while len(a) < 7:
                        a.append("")
                    a[5] = strand
                    a[6] = str(target_dbi[bi])
                    print("\t".join(a[:7]), file=chrom_outs[chrom_key])
                    # print >> chrom_outs[chrom_key], '\t'.join(a[:7])

        peak_bed_in.close()

    # close chromosome-specific files
    for chrom_key in chrom_outs:
        chrom_outs[chrom_key].close()

    # ignore Y
    if ignore_y:
        for orient in "+-":
            chrom_key = ("chrY", orient)
            if chrom_key in chrom_files:
                print("Ignoring chrY %s" % orient, file=sys.stderr)
                # print >> sys.stderr, 'Ignoring chrY %s' % orient
                os.remove(chrom_files[chrom_key])
                del chrom_files[chrom_key]

    # ignore auxiliary
    if ignore_auxiliary:
        # TODO: \d appears to be an invalid escape sequence. And re.compile will escape
        # \d anyway to \\d.
        primary_re = re.compile("chr\\d+$")
        for chrom_key in chrom_files.keys():
            chrom, strand = chrom_key
            primary_m = primary_re.match(chrom)
            if not primary_m and chrom != "chrX":
                print("Ignoring %s %s" % (chrom, strand), file=sys.stderr)
                # print >> sys.stderr, 'Ignoring %s %s' % (chrom,strand)
                os.remove(chrom_files[chrom_key])
                del chrom_files[chrom_key]

    #################################################################
    # sort chromosome-specific files
    #################################################################
    for chrom_key in chrom_files:
        chrom, strand = chrom_key
        chrom_sbed = "%s_%s_%s_sort.bed" % (out_prefix, chrom, strand)
        sort_cmd = "sortBed -i %s > %s" % (chrom_files[chrom_key], chrom_sbed)
        subprocess.call(sort_cmd, shell=True)
        os.remove(chrom_files[chrom_key])
        chrom_files[chrom_key] = chrom_sbed

    #################################################################
    # parse chromosome-specific files
    #################################################################
    final_bed_out = open("%s.bed" % out_prefix, "w")

    for chrom_key in chrom_files:
        chrom, strand = chrom_key

        open_peaks = []
        for line in open(chrom_files[chrom_key], "rt"):
            a = line.split("\t")
            a[-1] = a[-1].rstrip()

            # construct Peak
            peak_start = int(a[1])
            peak_end = int(a[2])
            peak_act = _activity_set(a[6])
            peak = Peak(peak_start, peak_end, peak_act)
            peak.extend(feature_size, chrom_lengths.get(chrom, None))

            if len(open_peaks) == 0:
                # initialize open peak
                open_end = peak.end
                open_peaks = [peak]

            else:
                # operate on exiting open peak

                # if beyond existing open peak
                if open_end - merge_overlap <= peak.start:
                    # close open peak
                    mpeaks = _merge_peaks(
                        open_peaks,
                        feature_size,
                        merge_overlap,
                        chrom_lengths.get(chrom, None),
                    )

                    # print to file
                    for mpeak in mpeaks:
                        print(mpeak.bed_str(chrom, strand), file=final_bed_out)
                        # print >> final_bed_out, mpeak.bed_str(chrom, strand)

                    # initialize open peak
                    open_end = peak.end
                    open_peaks = [peak]

                else:
                    # extend open peak
                    open_peaks.append(peak)
                    open_end = max(open_end, peak.end)

        if len(open_peaks) > 0:
            # close open peak
            mpeaks = _merge_peaks(
                open_peaks, feature_size, merge_overlap, chrom_lengths.get(chrom, None)
            )

            # print to file
            for mpeak in mpeaks:
                print(mpeak.bed_str(chrom, strand), file=final_bed_out)
                # print >> final_bed_out, mpeak.bed_str(chrom, strand)

    final_bed_out.close()

    # clean
    for chrom_key in chrom_files:
        os.remove(chrom_files[chrom_key])

    #################################################################
    # construct/update activity table
    #################################################################
    final_act_out = open("%s_act.txt" % out_prefix, "w")

    # print header
    cols = [""] + db_targets
    print("\t".join(cols), file=final_act_out)
    # print >> final_act_out, '\t'.join(cols)

    # print sequences
    for line in open("%s.bed" % out_prefix):
        a = line.rstrip().split("\t")
        # index peak
        peak_id = "%s:%s-%s(%s)" % (a[0], a[1], a[2], a[5])

        # construct full activity vector
        peak_act = [0] * len(db_targets)
        for ai in a[6].split(","):
            if ai != ".":
                peak_act[int(ai)] = 1

        # print line
        cols = [peak_id] + peak_act
        print("\t".join([str(c) for c in cols]), file=final_act_out)
        # print >> final_act_out, '\t'.join([str(c) for c in cols])

    final_act_out.close()


class Peak:
    """Peak representation
    Attributes:
        start (int) : peak start
        end   (int) : peak end
        act   (set[int]) : set of target indexes where this peak is active.
    """

    def __init__(self, start, end, act):
        self.start = start
        self.end = end
        self.act = act

    def extend(self, ext_len, chrom_len):
        """Extend the peak to the given length
        Args:
            ext_len (int) : length to extend the peak to
            chrom_len (int) : chromosome length to cap the peak at
        """
        mid = _find_midpoint(self.start, self.end)
        self.start = max(0, mid - ext_len / 2)
        self.end = self.start + ext_len
        if chrom_len and self.end > chrom_len:
            self.end = chrom_len
            self.start = self.end - ext_len

    def bed_str(self, chrom, strand):
        """Return a BED-style line
        Args:
            chrom (str)
            strand (str)
        """
        if len(self.act) == 0:
            act_str = "."
        else:
            act_str = ",".join([str(ai) for ai in sorted(list(self.act))])
        cols = (
            chrom,
            str(int(self.start)),
            str(int(self.end)),
            ".",
            "1",
            strand,
            act_str,
        )
        return "\t".join(cols)

    def merge(self, peak2, ext_len, chrom_len):
        """Merge the given peak2 into this peak
        Args:
            peak2 (Peak)
            ext_len (int) : length to extend the merged peak to
            chrom_len (int) : chromosome length to cap the peak at
        """
        # find peak midpoints
        peak_mids = [_find_midpoint(self.start, self.end)]
        peak_mids.append(_find_midpoint(peak2.start, peak2.end))

        # weight peaks
        peak_weights = [1 + len(self.act)]
        peak_weights.append(1 + len(peak2.act))

        # compute a weighted average
        merge_mid = int(0.5 + np.average(peak_mids, weights=peak_weights))

        # extend to the full size
        merge_start = max(0, merge_mid - ext_len / 2)
        merge_end = merge_start + ext_len
        if chrom_len and merge_end > chrom_len:
            merge_end = chrom_len
            merge_start = merge_end - ext_len

        # merge activities
        merge_act = self.act | peak2.act

        # set merge to this peak
        self.start = merge_start
        self.end = merge_end
        self.act = merge_act


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
    seqs, targets = _load_data_1hot(
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
    train_count = _batch_round(train_count, batch_size)
    print("%d training sequences " % train_count, file=sys.stderr)

    test_count = _batch_round(test_count, batch_size)
    print("%d test sequences " % test_count, file=sys.stderr)

    valid_count = _batch_round(valid_count, batch_size)
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




#------------------------------------------------------------
# Useful functions
#------------------------------------------------------------


def _process_exp(data_dir, exp_accession, metadata, assembly, criteria):
    """Process one experiment from the metadata table.

    Parameters
    ----------
    data_dir : str
        path to the directory with bedfiles
    exp_accession : str
        ENCODE Experiment Accession ID from the metadata table
    metadata : pandas.DataFrame
        meatdata table rendered as a pandas dataframe
    assembly : str
        genome assembly to filter
    criteria : dict
        dictionary of column, value pairs to use in making a selection

    Returns
    -------
    list or bool
        [row selected, summary line for the sample file] if experiment found.
        False if no entries found.
    """
    print("Processing ", exp_accession)
    # get the rows of the metadata table corresponding to the experiment and of
    # the data type of interest
    exp_df = metadata[
        (metadata["Experiment accession"] == exp_accession)
        & (metadata["File assembly"] == assembly)
    ]
    # check if any found
    if exp_df.size == 0:
        # continue with a warning
        print(
            "Warning: no records found in metadata, skipping experiment with accession "
            + exp_accession
        )
        return
    else:

        bed = exp_df[(exp_df["File type"] == "bed")]
        if criteria:
            for col, value in criteria.items():
                bed = bed[(exp_df[col] == value)]
        # pick the first bed file
        bed = bed.iloc[[0]]
        filepath = _get_filepath(data_dir, bed["File accession"].values[0])
        if filepath:
            summary_line = [_make_label(bed), filepath]
            return [bed, summary_line]
        else:
            print(
                "Warning: file not found in the data directory, skipping file "
                + bed["File accession"].values[0]
            )
            return False


def _make_label(df_row):
    """Generate a unique label for each row selected.

    Parameters
    ----------
    df_row : pd.Series
        Row of interest.

    Returns
    -------
    str
        Label made by concatenation of relevant columns.
    """
    label_list = [
        str(c.values[0])
        for c in [
            df_row["Assay"],
            df_row["Experiment target"],
            df_row["Biosample term name"],
            df_row["Lab"],
            df_row["Experiment accession"],
        ]
    ]
    return "_".join(label_list).replace(" ", "-")


def _get_filepath(data_dir, filename):
    """Generate path where the file is found

    Parameters
    ----------
    data_dir : str
        dataset directory with files
    filename : str
        file identifier, i.e. the file accession

    Returns
    -------
    str
        file path if present
        empty string if absent
    """
    # TODO: replace thsee operations with pathlib.
    filepath = os.path.abspath(os.path.join(data_dir, filename + ".bed"))
    if os.path.isfile(filepath):
        return filepath

    elif os.path.isfile(filepath + ".gz"):
        return filepath + ".gz"
    else:
        return ""



def _activity_set(act_cs):
    """Return a set of ints from a comma-separated list of int strings.
    Attributes:
        act_cs (str) : comma-separated list of int strings
    Returns:
        set (int) : int's in the original string
    """
    ai_strs = [ai for ai in act_cs.split(",")]

    if ai_strs[-1] == "":
        ai_strs = ai_strs[:-1]

    if ai_strs[0] == ".":
        aset = set()
    else:
        aset = set([int(ai) for ai in ai_strs])

    return aset


def _find_midpoint(start, end):
    """ Find the midpoint coordinate between start and end """
    mid = (start + end) / 2
    return int(mid)


def _merge_peaks(peaks, peak_size, merge_overlap, chrom_len):
    """Merge and the list of Peaks.
    Repeatedly find the closest adjacent peaks and consider
    merging them together, until there are no more peaks
    we want to merge.
    Attributes:
        peaks (list[Peak]) : list of Peaks
        peak_size (int) : desired peak extension size
        chrom_len (int) : chromsome length
    Returns:
        Peak representing the merger
    """
    max_overlap = merge_overlap
    while len(peaks) > 1 and max_overlap >= merge_overlap:
        # find largest overlap
        max_i = 0
        max_overlap = peaks[0].end - peaks[1].start
        for i in range(1, len(peaks) - 1):
            peaks_overlap = peaks[i].end - peaks[i + 1].start
            if peaks_overlap > max_overlap:
                max_i = i
                max_overlap = peaks_overlap

        if max_overlap >= merge_overlap:
            # merge peaks
            peaks[max_i].merge(peaks[max_i + 1], peak_size, chrom_len)

            # remove merged peak
            peaks = peaks[: max_i + 1] + peaks[max_i + 2 :]

    return peaks


def merge_peaks_dist(peaks, peak_size, chrom_len):
    """Merge and grow the Peaks in the given list.
    Obsolete
    Attributes:
        peaks (list[Peak]) : list of Peaks
        peak_size (int) : desired peak extension size
        chrom_len (int) : chromsome length
    Returns:
        Peak representing the merger
    """
    # determine peak midpoints
    peak_mids = []
    peak_weights = []
    for p in peaks:
        mid = (p.start + p.end - 1) / 2.0
        peak_mids.append(mid)
        peak_weights.append(1 + len(p.act))

    # take the mean
    merge_mid = int(0.5 + np.average(peak_mids, weights=peak_weights))

    # extend to the full size
    merge_start = max(0, merge_mid - peak_size / 2)
    merge_end = merge_start + peak_size
    if chrom_len and merge_end > chrom_len:
        merge_end = chrom_len
        merge_start = merge_end - peak_size

    # merge activities
    merge_act = set()
    for p in peaks:
        merge_act |= p.act

    return Peak(merge_start, merge_end, merge_act)




def _batch_round(count, batch_size):
    if batch_size is not None:
        count -= batch_size % count
    return count


def _load_data_1hot(
    fasta_file,
    scores_file,
    extend_len=None,
    mean_norm=True,
    whiten=False,
    permute=True,
    sort=False,
):
    # load sequences
    seq_vecs = _hash_sequences_1hot(fasta_file, extend_len)

    # load scores
    seq_scores = _hash_scores(scores_file)

    # align and construct input matrix
    train_seqs, train_scores = _align_seqs_scores_1hot(seq_vecs, seq_scores, sort)

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


def _hash_sequences_1hot(fasta_file, extend_len=None):
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
                seq_vecs[header] = _dna_one_hot(seq, seq_len)

            header = line[1:].rstrip()
            seq = ""
        else:
            seq += line.rstrip()

    if seq:
        seq_vecs[header] = _dna_one_hot(seq, seq_len)

    return seq_vecs


def _dna_one_hot(seq, seq_len=None, flatten=True, n_random=True):
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


def _hash_scores(scores_file):
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


def _align_seqs_scores_1hot(seq_vecs, seq_scores, sort=True):
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
