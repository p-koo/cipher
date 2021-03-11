"""Functions for quality control of data."""

# TODO: Function to plot histogram of bed sizes
def plot_bed_histogram(bed_path: str,cutoff : None):
  '''
  Function to plot histogram of bed file peak sizes; automatically infers compression from extension
  
  Parameters:

  bed_path --> path to .bed file
  
  cutoff --> user-defined value to remove outlier samples

  _________

  Output:
  
  histogram of peaks
  ___________
  
  Usage:
  pos_path=ENCFF252PLM.bed.gz
  plot_bed_histogram(pos_path, None)
  '''
  import matplotlib.pyplot as plt
  from collections import Counter
  # load bed file
  f = open(bed_path, 'rb')
  if bed_path.split('.')[-1] == "gz" or bed_path.split('.')[-1] == "gzip": compression="gzip"
  df = pd.read_table(f, header=None, compression=compression)
  chrom = df[0].to_numpy().astype(str)
  start = df[1].to_numpy()
  end = df[2].to_numpy()
  peak_sizes=end-start
  print("Number of peaks: "+str(len(peak_sizes))+'\n'+"Mean peak size: "+str(peak_sizes.mean()) + '\n'+"Std in peak size: "+str(peak_sizes.std()))
  bins = Counter(peak_sizes)
  print(bins)
  if cutoff is not None:
    print(list(bins.values()).index(cutoff))
    cutoff_idx=list(bins.values()).index(cutoff)
    bins.pop(list(bins.keys())[cutoff_idx])
  plt.bar(bins.keys(), bins.values())

# TODO: Function to plot histogram of GC content
def gc_content_histgram_from_one_hot(pos_one_hot, neg_one_hot):
  '''
  Function to plot histogram of of GC content across sequences for pos/neg sequences
  
  Parameters:

  pos/neg_one_hot --> one_hot encodings produced by convert_one_hot
  

  _________

  Output:
  
  histograms of:
  1. GC content for pos sequences (x = % GC; y = count of said instances)
  2. GC content for neg sequences
  3. Overlap
  ___________
  
  Usage:
  # get non-overlap between pos peaks and neg peaks
  print(neg_path)
  neg_bed_path = os.path.join(data_path, experiment + '_nonoverlap.bed')
  cmd = ['bedtools', 'intersect', '-v', '-wa', '-a', neg_path, '-b', pos_path, '>', neg_bed_path]
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()

  # create new bed file with window enforced
  neg_bed_path2 = os.path.join(data_path, experiment + '_neg_'+str(window)+'.bed')
  enforce_constant_size(neg_path, neg_bed_path2, window, compression="gzip")

  # extract sequences from bed file and save to fasta file
  neg_fasta_path = os.path.join(data_path, experiment + '_neg.fa')
  cmd = ['bedtools', 'getfasta','-s','-fi', genome_path, '-bed', neg_bed_path2, '-fo', neg_fasta_path]
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()

  # parse sequence and chromosome from fasta file
  neg_seq = parse_fasta(neg_fasta_path)

  # filter sequences with absent nucleotides
  neg_seq, _ = filter_nonsense_sequences(neg_seq)

  # convert filtered sequences to one-hot representation
  neg_one_hot = convert_one_hot(neg_seq, max_length=window)

  gc_content_histgram_from_one_hot(pos_one_hot,neg_one_hot)
  '''
  # nucleotide frequency matched background
  seq_pos = np.squeeze(np.argmax(pos_one_hot, axis=1))
  seq_neg = np.squeeze(np.argmax(neg_one_hot, axis=1))

  # get nucleotide frequency for pos sequences
  f_pos = []
  for s in seq_pos:
      f_pos.append([np.sum(s==0), np.sum(s==1), np.sum(s==2), np.sum(s==3)]) 
  f_pos = np.array(f_pos)/window
  # summing g+c count for each sequence and rounding value to nearest 2 decimal places: 
  gc_pos= [np.round(x[1:3].sum(),2) for x in f_pos]

  from collections import Counter
  bins = Counter(gc_pos)

  import matplotlib.pyplot as plt
  plt.bar(bins.keys(), bins.values(),align="edge",width=-0.1,label="GC Pos")
  plt.legend(loc="best")
  plt.show()

  # get nucleotide frequency for neg sequences
  f_neg = []
  for s in seq_neg:
      f_neg.append([np.sum(s==0), np.sum(s==1), np.sum(s==2), np.sum(s==3)]) 
  f_neg = np.array(f_neg)/window
  gc_neg= [np.round(x[1:3].sum(),2) for x in f_neg]
  bins_n = Counter(gc_neg)
  plt.bar(bins_n.keys(), bins_n.values(),align="edge",width=-0.1,label="GC Neg")
  plt.legend(loc="best")
  plt.show()

  import seaborn as sns
  sns.histplot(data=gc_pos , color="skyblue", label="GC Pos", kde=True)
  sns.histplot(data=gc_neg, color="red", label="GC Neg", kde=True)
  plt.legend() 
  plt.show()

# TODO: Table to look at N content, GC content, overlap w.r.t. genes, transposons, repeats 




