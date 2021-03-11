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

# TODO: Table to look at N content, GC content, overlap w.r.t. genes, transposons, repeats 




