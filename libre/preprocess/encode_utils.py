import os
import subprocess

import pandas as pd

#---------------------------------------------------------
# Single task datasets
#---------------------------------------------------------

def download_cell_line_data(metadata_path, tfchipdir):
    """This function parses a raw meta data file downloaded from
    the ENCODE website, downloads a curated list of ChIPseq bed files
    into a directory organized by cell line and TF.
    Additional meta data is also saved.

    Parameters
    ----------
    metadata_path : str
        The path to the input meta data file in tsv format.
    tfchipdir : str
        The path to the directory in which the bed files are to downloaded and saved.

    Returns
    -------
    None

    Example
    -------
    >>> metadata_path = './A549.tsv'
    >>> tfchipdir = "./tf_chip/"
    >>> download_cell_line_data(metadata_path, tfchipdir)
    """

    # load the meta data file and filter its contents
    metatable_filtered = _filter_encode_metatable(metadata_path)
    res = _extract_metatable_information(metatable_filtered)
    df = pd.DataFrame.from_dict(res)

    # loop through all the rows of the cell type metadata table.
    for idx in df.index:
        row = df.iloc[idx]

        # get the output directory and output path for the bed file
        tf = row["tf_list"]
        cell_type = row["cell_type_list"]
        url = row["url_list"]
        file_accession = row["file_accession_list"]
        outdir = os.path.join(tfchipdir, cell_type, tf)
        outpath = os.path.join(outdir, file_accession + ".bed.gz")

        # get the meta data path
        meta_df_name = os.path.join(outdir, "metadata.tsv")

        # load the meta data if it already exists ; create new one if not.
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            meta_df = pd.DataFrame(data=[], columns=list(df.columns[2:]))
        else:
            meta_df = pd.read_csv(meta_df_name, sep="\t")

        # download the bed file
        _download_url(url, outpath=outpath)

        # update the metadata table
        remaining_metadata = row.iloc[2:]
        meta_df = meta_df.append(remaining_metadata)

        # save the meta data table
        meta_df.to_csv(meta_df_name, sep="\t")



def _filter_encode_metatable(file_path, save_filtered_table=True):
    """Filter ENCODE metatable for relevant rows.

    Parameters
    ----------
    file_path : <str>
        Path to ENCODE metatable file in TSV format.
    save_filtered_table : <bool>, optional
        Optional flag denoting whether filtered table should be saved in same directory
        as original metatable.

    Returns
    -------
    metatable_filtered : <pandas.DataFrame>
        Filtered ENCODE metatable.

    Example
    -------
    >>> metatable_path = "./k562.tsv"
    >>> metatable_filtered = filter_encode_metatable(metatable_path)
    """

    metatable = pd.read_csv(file_path, sep="\t")
    metatable_filtered = pd.DataFrame(
        columns=metatable.columns
    )  # make empty DataFrame to hold all desired datasets

    for accession in metatable["Experiment accession"].unique():
        criterion = (
            (metatable["Experiment accession"] == accession)
            & (metatable["Output type"] == "IDR thresholded peaks")
            & (metatable["File assembly"] == "GRCh38")
            & (metatable["Biological replicate(s)"] == "1, 2")
        )

        metatable_filtered = pd.concat(
            [metatable_filtered, metatable[criterion]]
        )  # add filtered metatable (i.e., metatable[criterion]) to datasets

    if save_filtered_table:
        save_path, _ = os.path.split(file_path)
        file_name = os.path.splitext(file_path)[0]
        save_file = os.path.join(save_path, file_name + "_filtered.tsv")
        metatable_filtered.to_csv(save_file, sep="\t", index=False)

    return metatable_filtered


def _extract_metatable_information(metatable_filtered):
    """Extract filtered ENCODE metatable for columns.
    Parameters
    ----------
    metatable_filtered : <pandas.DataFrame>
        Filtered ENCODE metatable.
    Returns
    -------
    res_dict : <dict>
        A dictionary containing the following key::value pairs:
            tf_list : <list>
                List of transcription factors in the ENCODE metatable.
            cell_type_list : <list>
                List of cell types in the ENCODE metatable.
            file_accession_list : <list>
                List of file acessions in the ENCODE metatable.
            expt_accession_list : <list>
                List of experiment accessions in the ENCODE metatable.
            url_list : <list>
                List of URLs in the ENCODE metatable.
            audit_warning_list : <list>
                List of audit warnings in the ENCODE metatable.
    Example
    -------
    >>> metatable_filtered = filter_encode_metatable(
        file_path, save_filtered_table=True)
    >>> tf, cell_type, file_accession, url, audit = extract_table_information(
        metatable_filtered)
    """

    metatable_filtered = metatable_filtered[
        [
            "File accession",
            "Experiment accession",
            "Biosample term name",
            "Experiment target",
            "Lab",
            "File download URL",
            "Audit WARNING",
        ]
    ].copy()
    metatable_filtered["Experiment target"] = metatable_filtered[
        "Experiment target"
    ].str.split("-", expand=True)[0]

    tf_list = metatable_filtered["Experiment target"].tolist()
    cell_type_list = metatable_filtered["Biosample term name"].tolist()
    file_accession_list = metatable_filtered["File accession"].tolist()
    expt_accession_list = metatable_filtered["Experiment accession"].tolist()
    url_list = metatable_filtered["File download URL"].tolist()
    audit_warning_list = metatable_filtered["Audit WARNING"].tolist()

    res_dict = {
        "tf_list": tf_list,
        "cell_type_list": cell_type_list,
        "file_accession_list": file_accession_list,
        "expt_accession_list": expt_accession_list,
        "url_list": url_list,
        "audit_warning_list": audit_warning_list,
    }

    return res_dict


def _download_url(url, outpath=None):
    """Download a file from a given url and save it with a specified output file
    if necessary.

    Parameters
    ----------
    url : <str>
        The url of the file to download.
    outpath : <str>
        The full output file path. If None specified, the file is saved
        in the current working directory with its original name.

    Returns
    -------
    None

    Example
    -------
    >>> url = "https://www.encodeproject.org/files/ENCFF695MMQ/@@download/ENCFF695MMQ.bed.gz"  # noqa: E501
    >>> outpath = "./downloads/out.bed.gz"
    >>> _download_url(url, outpath)
    """

    # TODO: this requires wget to be available. This might not be available on some
    # systems, like windows.
    if outpath is None:
        cmd = ["wget", url]
    else:
        cmd = ["wget", url, "-O", outpath]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

