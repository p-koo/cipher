import argparse
'''
This script will take the following arguments in the command line, import helper functions from an external script, and conduct all preprocessing steps:
command line usage: python preprocess_wrapper.py --bed_pos = <> --neg_pos= <> ,...

Parameters
----------
bed_pos : <str>
bed_neg : <str>
window_size : <int>
data : <str>
experiment : <str>
ref : <str>
test_frac : <float>
validation_frac : <float>

Returns
_______
train, test, and validation .hdf5 files in 'data' folder


Example
-------
For .bed files and reference .fasta in folder called 'Data':

python preprocess_wrapper.py --bed_pos='Data/ENCFF252PLM.bed.gz' --bed_neg='Data/ENCFF209DJG.bed.gz' --window_size=200 --data_path='Data/' --experiment='CTCF' --ref='Data/hg19.fa'

'''
parser = argparse.ArgumentParser(description='Pre-Processing Functions.')
parser.add_argument('--bed_pos', dest='bed_pos', required= True, help='Path to positive-sample bed file.')
parser.add_argument('--bed_neg', dest='bed_neg', required= True, help='Path to negative-sample bed file.')
parser.add_argument('--window_size', dest='window',type=int, required= True, help='Desired sequence length')
parser.add_argument('--data_path', dest='data', required= True, help='Path to input data')
parser.add_argument('--experiment',dest='experiment',required=True,help="Transcription factor of interest")
parser.add_argument('--ref',dest='genome_filename',required=True,help="Reference Genome (hg19/38.fa)")
parser.add_argument('--test_frac',dest="test_frac",type=float,default=0.2,help="Fraction of data in test cohort")
parser.add_argument('--validation_frac',dest="validation_frac",type=float,default=0.1,help="Fraction of data in validation cohort")


args = parser.parse_args()
pos_filename = args.bed_pos
neg_filename = args.bed_neg
window = args.window
data_path = args.data
experiment = args.experiment
genome_file = args.genome_filename
valid_frac = args.validation_frac
test_frac = args.test_frac
train_frac = 1 -(test_frac + valid_frac)

#### import functions from respective files (here called script):
# import enforce_constant_size from script
# import parse_fasta from script
## or if all helper functions in one script in working dir:

from wrangle import (parse_fasta,filter_nonsense_sequences,convert_one_hot,bedtools_getfasta,bedtools_intersect,split_dataset) 

# for histograms:
from inspect import (plot_bed_histogram,gc_content_histgram_from_one_hot)

from singletask import (match_gc,enforce_constant_size)



# set paths
genome_path = os.path.join(data_path, genome_filename)
tf_bed_path  = os.path.join(data_path, pos_filename)
dnase_bed_path = os.path.join(data_path, neg_filename)

# create new bed file with window enforced
pos_bed_path = os.path.join(data_path, experiment + '_pos_'+str(window)+'.bed')
enforce_constant_size(tf_bed_path, pos_bed_path, window, compression='gzip')

# extract sequences from bed file and save to fasta file
pos_fasta_path = os.path.join(data_path, experiment + '_pos.fa')
#cmd = ['bedtools', 'getfasta','-s','-fi', genome_path, '-bed', pos_bed_path, '-fo', pos_fasta_path]
#process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#stdout, stderr = process.communicate()

## as function:
bedtools_getfasta(input_fasta = genome_path,output_fasta=pos_fasta_path,bed_file=pos_bed_path)


# parse sequence and chromosome from fasta file
pos_seq = parse_fasta(pos_fasta_path)

# filter sequences with absent nucleotides
pos_seq, _ = filter_nonsense_sequences(pos_seq)

# convert filtered sequences to one-hot representation
pos_one_hot = convert_one_hot(pos_seq, max_length=window)

# get non-overlap between pos peaks and neg peaks
neg_bed_path = os.path.join(data_path, experiment + '_nonoverlap.bed')
bedtools_intersect(dnase_bed_path,tf_bed_path,neg_bed_path)

#cmd = ['bedtools', 'intersect', '-v', '-wa', '-a', dnase_bed_path, '-b', tf_bed_path, '>', neg_bed_path]
#os.system('bedtools intersect -v -wa -a '+dnase_bed_path+' -b '+tf_bed_path+' > '+neg_bed_path)



# create new bed file with window enforced
neg_bed_path2 = os.path.join(data_path, experiment + '_neg_'+str(window)+'.bed')
enforce_constant_size(neg_bed_path, neg_bed_path2, window, compression=None)

# extract sequences from bed file and save to fasta file
neg_fasta_path = os.path.join(data_path, experiment + '_neg.fa')
#cmd = ['bedtools', 'getfasta','-s','-fi', genome_path, '-bed', neg_bed_path2, '-fo', neg_fasta_path]
#process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#stdout, stderr = process.communicate()

# as function:
bedtools_getfasta(input_fasta = genome_path,output_fasta=neg_fasta_path,bed_file=neg_bed_path2)



# parse sequence and chromosome from fasta file
neg_seq = parse_fasta(neg_fasta_path)

# filter sequences with absent nucleotides
neg_seq, _ = filter_nonsense_sequences(neg_seq)

# convert filtered sequences to one-hot representation
neg_one_hot = convert_one_hot(neg_seq, max_length=window)


# generate histogram of peak sizes and GC content by calling inspect.py
plot_bed_histogram(pos_path, None)
plot_bed_histogram(neg_path, None)
gc_content_histgram_from_one_hot(pos_one_hot,neg_one_hot)

# calling match_gc function to balance neg sequences with pos by GC content:
neg_one_hot_gc = match_gc(pos_one_hot, neg_one_hot)

# merge postive and negative sequences
one_hot = np.vstack([pos_one_hot, neg_one_hot_gc])
labels = np.vstack([np.ones((len(pos_one_hot), 1)), np.zeros((len(neg_one_hot_gc), 1))])

# shuffle indices for train, validation, and test sets


num_data = len(one_hot)
cum_index = np.array(np.cumsum([0, train_frac, valid_frac, test_frac])*num_data).astype(int)
shuffle = np.random.permutation(num_data)
train_index = shuffle[cum_index[0]:cum_index[1]]
valid_index = shuffle[cum_index[1]:cum_index[2]]
test_index = shuffle[cum_index[2]:cum_index[3]]


# train: train = (one_hot[train_index], labels[train_index,:])
train, valid, test, indices = split_dataset(one_hot, labels, valid_frac= valid_frac, test_frac=test_frac)


filename = experiment+'_'+str(window)+'.h5'
file_path = os.path.join(data_path, filename)

with h5py.File(file_path, 'w') as fout:
    X_train = fout.create_dataset('x_train', data=list(train)[0], dtype='float32', compression="gzip")
    Y_train = fout.create_dataset('y_train', data=list(train)[1], dtype='int8', compression="gzip")
    X_valid = fout.create_dataset('x_valid', data=list(valid)[0], dtype='float32', compression="gzip")
    Y_valid = fout.create_dataset('y_valid', data=list(valid)[1], dtype='int8', compression="gzip")
    X_test = fout.create_dataset('x_test', data=list(test)[0], dtype='float32', compression="gzip")
    Y_test = fout.create_dataset('y_test', data=list(test)[1], dtype='int8', compression="gzip")
print('Saved to: ' + file_path)




#with h5py.File(file_path, 'w') as fout:
#    X_train = fout.create_dataset('x_train', data=one_hot[train_index], dtype='float32', compression="gzip")
#    Y_train = fout.create_dataset('y_train', data=labels[train_index,:], dtype='int8', compression="gzip")
#    X_valid = fout.create_dataset('x_valid', data=one_hot[valid_index], dtype='float32', compression="gzip")
#    Y_valid = fout.create_dataset('y_valid', data=labels[valid_index,:], dtype='int8', compression="gzip")
#    X_test = fout.create_dataset('x_test', data=one_hot[test_index], dtype='float32', compression="gzip")
#    Y_test = fout.create_dataset('y_test', data=labels[test_index,:], dtype='int8', compression="gzip")
#print('Saved to: ' + file_path)

