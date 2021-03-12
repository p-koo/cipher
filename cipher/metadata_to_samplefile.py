#!/usr/bin/env python
import numpy as np
import pandas as pd
import os




def make_label(df_row):

    '''Generate a unique label for each row selected

    Parameters
    ----------
    df_row : pandas.Series
        one row of interest

    Returns
    -------
    str
    label made by concatenation of relevant columns.
    '''
    label_list = [str(c.values[0]) for c in [df_row['Assay'],
                         df_row['Experiment target'],
                         df_row['Biosample term name'],
                         df_row['Lab'],
                         df_row['Experiment accession']]]
    return('_'.join(label_list).replace(" ", "-"))


def process_exp(data_dir, exp_accession, metadata, assembly, criteria):
    '''Process one experiment from the metadata table.

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
        [row selected, summary line for the sample file] if experiment found
        False if no entries found


    '''
    print('Processing ', exp_accession)
    #get the rows of the metadata table corresponding to the experiment and of
    #the data type of interest
    exp_df = metadata[(metadata['Experiment accession']==exp_accession) &
                      (metadata['File assembly']==assembly)]
    # check if any found
    if exp_df.size == 0:
        #continue with a warning
        print('Warning: no records found in metadata, skipping experiment with accession '+exp_accession)
        return
    else:

        bed = exp_df[(exp_df['File type'] == 'bed')]
        if criteria:
            for col, value in criteria.items():
                bed = bed[(exp_df[col]==value)]
        # pick the first bed file
        bed = bed.iloc[[0]]
        filepath = get_filepath(data_dir, bed['File accession'].values[0])
        if filepath:
            summary_line = [make_label(bed), filepath]
            return [bed, summary_line]
        else:
            print('Warning: file not found in the data directory, skipping file '+bed['File accession'].values[0])
            return False

def get_filepath(data_dir, filename):
    ''' Generate path where the file is found
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
    '''
    filepath = os.path.abspath(os.path.join(data_dir, filename+'.bed'))
    if os.path.isfile(filepath):
        return filepath

    elif os.path.isfile(filepath+'.gz'):
        return filepath+'.gz'
    else:
        return ''


# assembly = 'GRCh38'
# data_dir = '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/hackathon'
# metadata_path = 'metadata_1.tsv'
# output_path = 'test_HepG2'
# exp_accession_list = ['ENCSR580HOI', 'ENCSR956OSX', 'nonexistent_bed'] #optional

def create_samplefile(data_dir, metadata_path, assembly = 'GRCh38',
                      sample_output_path='sample_beds.txt',
                      subset_output_path='selected_data.csv',
                      criteria={},
                      exp_accession_list=[]):
    '''Generate subset of metadata table and sample file for further processing
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

    '''
    assert metadata_path.endswith('.tsv') or metadata_path.endswith('.csv'), print('Metadata should be a tsv or a csv file')
    if metadata_path.endswith('.tsv'):
        metadata = pd.read_csv(metadata_path, sep='\t')
    elif metadata_path.endswith('.csv'):
        metadata = pd.read_csv(metadata_path, sep=',')


    if not exp_accession_list:
        print('Generating sample file from all of the metadata table')
        exp_accession_list = list(set(metadata['Experiment accession']))

    summary = []
    selected_rows = []
    for exp_accession in exp_accession_list:
        # filter files and save selection
        selection = process_exp(data_dir, exp_accession, metadata, assembly, criteria)
        if selection:
            selected_rows.append(selection[0])
            summary.append(selection[1])
    selected_data = pd.concat(selected_rows)
    selected_data['label'] = [entry[0] for entry in summary]
    selected_data['path'] = [entry[1] for entry in summary]

    with open(sample_output_path, 'w') as filehandle:
        for l in summary:

            filehandle.write('{}\t{}\n'.format(l[0], l[1]))
    selected_data.to_csv(subset_output_path)
