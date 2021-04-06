"""Functions for quality control of data."""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_bed_histogram(bed_path, bins=20, ax=None, fontsize=12, **kwargs):
    """Function to plot histogram of bed file peak sizes; automatically infers
    compression from extension and allows for user-input in removing outlier sequence

    Parameters
    -----------
    bed_path : str
        path to bed file
    bins : int
        Number of bins for histogram
    ax : matplotlib Axis
        Figure handle to plot. If None, then generates a new figure
    fontsize : int
        font size for labels and ticks
    kwargs:
        optional dictionary arguments for matplotlib.pyplot.hist

    Returns
    -------
    None

    Examples
    --------
    >>> pos_path = ENCFF252PLM.bed.gz
    >>> plot_bed_histogram(pos_path, bins=20, ax=None, fontsize=12)

    >>> plt.figure()
    >>> ax = plt.subplot(1,1,1)
    >>> plot_bed_histogram(pos_path, bins=20, ax=ax, fontsize=12)
    """

    # check if bedfile is compressed
    if bed_path.split(".")[-1] == "gz" or bed_path.split(".")[-1] == "gzip":
        compression = "gzip"

    # TODO: when tests are added for this, pass the filename to read_table instead of
    # the file handle.

    # load bed file
    f = open(bed_path, "rb")
    df = pd.read_table(f, header=None, compression=compression)
    start = df[1].to_numpy()
    end = df[2].to_numpy()

    # get peak sizes
    peak_sizes = end - start

    # plot histogram
    bins = Counter(peak_sizes)
    if ax is not None:
        ax.hist(peak_sizes, bins, **kwargs)
    else:
        plt.hist(peak_sizes, bins, **kwargs)
    plt.xlabel("Peak size (nt)", fontsize=fontsize)
    plt.ylabel("Counts", fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)


def gc_content_histgram(one_hot, bins=15, ax=None, fontsize=12, **kwargs):
    """
    Function to plot histogram of of GC content across one-hot sequences

    Parameters
    -----------
    one_hot : np.ndarray
        path to bed file
    bins : int
        Number of bins for histogram
    ax : matplotlib Axis
        Figure handle to plot. If None, then generates a new figure
    fontsize : int
        font size for labels and ticks
    kwargs:
        optional dictionary arguments for matplotlib.pyplot.hist

    Returns
    -------
    None

    Examples
    --------
    >>> gc_content_histgram(one_hot, bins=15, ax=None, fontsize=12)
    """

    # nucleotide frequency matched background
    freq = np.mean(one_hot, axis=1)

    # summing g+c count for each sequence:
    gc = np.sum(freq[:, 1:3], axis=1)

    # plot histogram of gc content
    if ax is not None:
        ax.hist(gc, bins, **kwargs)
    else:
        plt.hist(gc, bins, **kwargs)
    plt.xlabel("GC-content", fontsize=fontsize)
    plt.ylabel("Counts", fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
