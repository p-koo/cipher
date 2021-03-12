import os
import numpy as np
import pandas as pd
import subprocess 
from wrangle import filter_encode_metatable, extract_metatable_information

def _download_url(url, outpath=None):
    """
    Download a file from a given url and save it with a specified output file 
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
    >>> url = "https://www.encodeproject.org/files/ENCFF695MMQ/@@download/ENCFF695MMQ.bed.gz"
    >>> outpath = "./downloads/out.bed.gz"
    >>> _download_url(url, outpath) 

    """
    if outpath is None:
        cmd = ['wget', url]
    else:
        cmd = ['wget', url, '-O', outpath]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def download_cell_line_data(metadata_path, tfchipdir):
    """This function parses a raw meta data file downloaded from 
    the ENCODE website, downloads a curated list of ChIPseq bed files 
    into a directory organized by cell line and TF. 
    Additional meta data is also saved. 
    
    Parameters
    ----------
    metadata_path : <str>
        The path to the input meta data file in tsv format. 

    tfchipdir : <str>
        The path to the directory in which the bed files are to downloaded and saved onto. 


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
    metatable_filtered = filter_encode_metatable(metatable_path)
    res = extract_metatable_information(metatable_filtered)
    df = pd.DataFrame.from_dict(res)

    # loop through all the rows of the cell type metadata table.
    for idx in df.index:
        row = df.iloc[idx]

        # get the output directory and output path for the bed file
        tf = row['tf_list']
        cell_type = row['cell_type_list']
        url = row['url_list']
        file_accession = row['file_accession_list']
        outdir = os.path.join(tfchipdir, cell_type, tf)
        outpath = os.path.join(outdir, file_accession+".bed.gz")

        # get the meta data path 
        meta_df_name = os.path.join(outdir, 'metadata.tsv')

        # load the meta data if it already exists ; create new one if not. 
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            meta_df = pd.DataFrame(data=[], columns=list(df.columns[2:]))
        else:
            meta_df = pd.read_csv(meta_df_name, sep='\t')
        
        # download the bed file 
        _download_url(url, outpath=outpath)

        # update the metadata table
        remaining_metadata = row.iloc[2:] 
        meta_df = meta_df.append(remaining_metadata)

        # save the meta data table 
        meta_df.to_csv(meta_df_name, sep='\t')
