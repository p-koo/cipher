"""
This script will take the following arguments in the command line, import helper
functions from an external script, and conduct all preprocessing steps

Parameters:
-----------
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

# TODO: the documentation is likely part of the parser help message. We can probably
# remove it.

from optparse import OptionParser
import subprocess

from libre.preprocess import multitask


def main():
    parser = OptionParser()
    parser.add_option(
        "--feature_size",
        dest="feature_size",
        default=1000,
	type='int',
        help="length of selected sequence regions",
    )
    parser.add_option(
        "--fasta", dest="fasta", help="length of selected sequence regions"
    )
    parser.add_option(
        "--output",
        dest="h5_output",
        default="output.h5",
        help="Directory for output h5 file. Default to current directory",
    )
    parser.add_option(
        "--sample_output",
        dest="exp_output",
        default="sample_beds.txt",
        help="Directory for output sample file. Default to current directory",
    )
    parser.add_option(
        "--bed_output",
        dest="bed_output",
        default="merged_features",
        help="Directory for output merged peak bed file. Default to current directory",
    )
    parser.add_option(
        "--header_output",
        dest="header_output",
        default="output_header",
        help="Directory for output h5 file. Default to current directory",
    )
    parser.add_option(
        "--fasta_output",
        dest="fa_output",
        default="selected_region.fa",
        help="Directory for output sub-fasta file. Default to current directory",
    )
    parser.add_option(
        "--assembly",
        dest="g_assembly",
        default="GRCh38",
        help="genome assembly used for reference. Optional.",
    )
    parser.add_option(
        "--chrom_size", dest="chrom_size", help="Location of chromosome size file"
    )
    parser.add_option(
        "--subset",
        dest="subset_output",
        default="selected_data.csv",
        help="path where the subset of metadata table used will be saved as a to_csv",
    )
    parser.add_option(
        "--criteria",
        dest="criteria",
        default={},
        help="dictionary of column, value pairs to use in making a selection",
    )
    parser.add_option(
        "--exp_accession_list",
        dest="exp_accession",
        default=None,
        help="List of experiments to select, if empty select all in the metadata table",
    )
    parser.add_option(
	    "--seed", 
	    dest="seed", 
	    default=42, 
	    type="int", 
	    help="random split for data shuffle"
    )
    parser.add_option(
	    "--valid_frac", 
	    dest="valid_pct", 
	    default=0.1, 
	    type="float", 
	    help="fraction of data allocated to the validation set"
    )
    parser.add_option(
	    '--test_frac', 
	    dest = 'test_pct', 
	    default=0.1, 
	    type='float', 
	    help='fraction of data allocated to the test set'
    )
	
    parser.add_option(
        "--merge_overlap",
        dest="overlap",
        default=200,
	type='int',
        help=
            "if two peak regions overlaps more than this amount, they will be"
            " re-centered and merged into a single sample"
        
    )
    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error("Must provide data directory and metadata table path.")
    else:
        data_dir = args[0]
        metadata_path = args[1]

    # call package functiond
    multitask.create_samplefile(
        data_dir,
        metadata_path,
        assembly=options.g_assembly,
        sample_output_path=options.exp_output,
        subset_output_path=options.subset_output,
        criteria=options.criteria,
        exp_accession_list=options.exp_accession,
    )

    multitask.multitask_bed_generation(
        options.exp_output,
        chrom_lengths_file=options.chrom_size,
        feature_size=options.feature_size,
        merge_overlap=options.overlap,
        out_prefix=options.bed_output,
    )

    # TODO: shell=True is probably not necessary here. Remove once tests are in place.
    subprocess.call(
        "bedtools getfasta -fi {} -s -bed {} -fo {}".format(
            options.fasta, options.bed_output + ".bed", options.fa_output
        ),
        shell=True,
    )

    multitask.make_h5(
        options.fa_output,
        options.bed_output + "_act.txt",
        options.h5_output,
        options.header_output,
	random_seed=options.seed,
	test_pct=options.test_pct,
	valid_pct=options.valid_pct
    )


if __name__ == "__main__":
    main()
