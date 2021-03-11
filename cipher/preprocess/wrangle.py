import os
import numpy as np
import pandas as pd

"""Functions for data wrangling."""

# TODO: Function to convert sequence to one-hot

# TODO: Function to convert one-hot to sequence

# TODO: function to parse fasta file

# TODO: function to generate fasta file

# TODO: function to load bed file to dataframe
def filter_encode_metatable(file_path, save_filtered_table=True):
    """Filter ENCODE metatable for relevant rows.

    Parameters
    ----------
    file_path : <str>
        Path to ENCODE metatable file in TSV format.
    save_filtered_table : <bool>, optional
        Optional flag denoting whether filtered table should be saved in same directory as original metatable.

    Returns
    -------
    metatable_filtered : <pandas.DataFrame>
        Filtered ENCODE metatable.

    Example
    -------
    >>> metatable_path = "./k562.tsv"
    >>> metatable_filtered = filter_encode_metatable(metatable_path)
    """

    metatable = pd.read_csv(file_path, sep='\t')
    metatable_filtered = pd.DataFrame(columns=metatable.columns) # make empty DataFrame to hold all desired datasets

    for accession in metatable["Experiment accession"].unique():
        criterion = (metatable["Experiment accession"] == accession) & \
        (metatable["Output type"] == "IDR thresholded peaks") & \
        (metatable["File assembly"] == "GRCh38") & \
        (metatable["Biological replicate(s)"] == "1, 2")

        metatable_filtered = pd.concat( [metatable_filtered, metatable[criterion]] ) # add filtered metatable (i.e., metatable[criterion]) to datasets

    if save_filtered_table:
        save_path, _ = os.path.split(file_path)
        file_name = os.path.splitext(file_path)[0]
        save_file = os.path.join(save_path, file_name + "_filtered.tsv")
        metatable_filtered.to_csv(save_file, sep='\t', index=False)

    return metatable_filtered

def extract_table_information(filtered_table):
    """Extract filtered ENCODE metatable for columns.

    Parameters
    ----------
    filtered_table : <pandas.DataFrame>
        Filtered ENCODE metatable.

    Returns
    -------
    tf_list : <list>
        List of transcription factors in the ENCODE metatable.
    cell_type_list : <list>
        List of cell types in the ENCODE metatable.
    file_accession_list : <list>
        List of file acessions in the ENCODE metatable.
    url_list : <list>
        List of URLs in the ENCODE metatable.
    audit_warning_list : <list>
        List of audit warnings in the ENCODE metatable.

    Example
    -------
    >>> tf, cell_type, file_accession, url, audit = extract_table_information(filtered_table)
    """

    filtered_table = filtered_table[['File accession', 'Biosample term name', 'Experiment target', 'Lab', 'File download URL', 'Audit WARNING']].copy()
    filtered_table['Experiment target'] = filtered_table['Experiment target'].str.split('-', expand=True)[0]

    index_list = filtered_table.index.tolist()
    tf_list = filtered_table['Experiment target'].tolist()
    cell_type_list = filtered_table['Biosample term name'].tolist()
    file_accession_list = filtered_table['File accession'].tolist()
    url_list = filtered_table['File download URL'].tolist()
    audit_warning_list = filtered_table['Audit WARNING'].tolist()

    return tf_list, cell_type_list, file_accession_list, url_list, audit_warning_list

# TODO: function to save dataframe to a bed file

# TODO: Function to calculate GC-content

# TODO: bedtools getfasta

# TODO: bedtools non overlap

# TODO: bedtools overlap

# TODO: random split into train test valid

# TODO: random split into k-fold cross validation

# TODO: split according to chromosome
