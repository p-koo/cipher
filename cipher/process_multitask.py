"""
This script will take the following arguments in the command line, import helper functions from an external script, and conduct all preprocessing steps

Parameters:
--------------


metadata: str
    location for metadata table containing experiment information
data_dir: str
    directory containing corresponding sequencing data files
output:str
    Optional.Directory for output file. Default to current directory
assembly: str
    Optional. Default to be 'GRCh38'
chrom_size:str
    Optional. Location for .chome.size file
subset: str
    Optional. Path where the subset of metadata table used will be saved as a to_csv.
    Default to current directory
criteria: dict
    Optional.Dictionary of column, value pairs to use in making a selection
    Defulat to {'Output type':'IDR thresholded peaks'}
exp_accession_list:list
    Optional. List of experiments to select, if empty select all in the metadata table

"""


from optparse import OptionParser
import subprocess
from metadata_to_samplefile import create_samplefile
from bed_generation import multitask_bed_generation
from seq_hdf5 import make_h5

def main():
    parser = OptionParser()
    parser.add_option('--metadata', dest='metadata_path', help='location for metadata table containing experiment information')
    parser.add_option('--data_dir', dest='data_dir', help='directory containing corresponding sequencing data files')
    parser.add_option('--feature_size',dest = 'feature_size',default = 1000,help='length of selected sequence regions')
    parser.add_option('--fasta',dest = 'fasta',help='length of selected sequence regions')
    parser.add_option('--output', dest='h5_output', default ='output.h5', help='Directory for output h5 file. Default to current directory')
    parser.add_option('--sample_output', dest='exp_output', default ='sample_beds.txt', help='Directory for output sample file. Default to current directory')
    parser.add_option('--bed_output', dest='bed_output', default ='merged_features', help='Directory for output merged peak bed file. Default to current directory')
    parser.add_option('--header_output', dest='header_output', default ='output_header', help='Directory for output h5 file. Default to current directory')
    parser.add_option('--fasta_output', dest='fa_output', default ='selected_region.fa', help='Directory for output sub-fasta file. Default to current directory')
    parser.add_option('--assembly', dest='g_assembly',default ='GRCh38', help='genome assembly used for reference. Optional.')
    parser.add_option('--chrom_size', dest='chrom_size', help='Location of chromosome size file')
    parser.add_option('--subset', dest='subset_output',default = 'selected_data.csv',help='path where the subset of metadata table used will be saved as a to_csv')
    parser.add_option('--criteria', dest='criteria', default = {} ,help='dictionary of column, value pairs to use in making a selection')
    parser.add_option('--exp_accession_list', dest='exp_accession',default=None, help='List of experiments to select, if empty select all in the metadata table')
    parser.add_option('--merge_overlap',dest = 'overlap',default=200, help='if two peak regions overlaps more than this amount, they will be re-centered and merged into a single sample')
    (options,args) = parser.parse_args()

    #call package functiond
    create_samplefile(options.data_dir, options.metadata_path, assembly = options.g_assembly,
                      sample_output_path=options.subset_output,
                      subset_output_path=options.exp_output,
                      criteria=options.criteria,
                      exp_accession_list=options.exp_accession)

    multitask_bed_generation(exp_output,chrom_lengths_file=options.chrom_size,
                            feature_size=options.feature_size,merge_overlap=options.overlap,
                             out_prefix=options.bed_output)

    subprocess.call('bedtools getfasta -fi {} -s -bed {} -fo {}'.format(options.fasta,options.bed_output,options.fasta_output), shell=True)
    make_h5(options.fasta,bed_output+'_act.txt',options.h5_output,options.header_output)


if __name__ == "__main__":
    main()
