"""Useful functions."""

import os


# TODO: Function to load tsv to dataframe

# TODO: function to save dataframe to tsv


def make_directory(dirpath, verbose=1):
    """make a directory.

    Parameters
    ----------
    dirpath : string
        String of path to directory.

    Returns
    -------


    Examples
    --------
    >>> dirpath = './results'
    >>> make_directory(dirpath)

    """
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
        print("making directory: " + dirpath)







