"""Functions for processing single-task ChIP-seq data generation."""


# TODO: Filter N

# TODO: Filter by size

# TODO: Function to make bed file constant window size
def enforce_constant_size(bed_path, output_path, window, compression=None):
    """generate a bed file where all peaks have same size centered on original peak"""

    # load bed file
    f = open(bed_path, 'rb')
    df = pd.read_table(f, header=None, compression=compression)
    chrom = df[0].to_numpy().astype(str)
    start = df[1].to_numpy()
    end = df[2].to_numpy()

    # calculate center point and create dataframe
    middle = np.round((start + end)/2).astype(int)
    half_window = np.round(window/2).astype(int)

    # calculate new start and end points
    start = middle - half_window
    end = middle + half_window

    # filter any negative start positions
    data = {}
    for i in range(len(df.columns)):
        data[i] = df[i].to_numpy()
    data[1] = start
    data[2] = end

    # create new dataframe
    df_new = pd.DataFrame(data);

    # save dataframe with fixed width window size to a bed file
    df_new.to_csv(output_path, sep='\t', header=None, index=False)


# TODO: GC match between positive and negative labels
def match_gc(pos_one_hot, neg_one_hot):   #Antonio's revision. 

    # nucleotide frequency matched background
    f_pos = np.mean(pos_one_hot, axis=2)
    f_neg = np.mean(neg_one_hot, axis=2)

    #get GC content for pos and neg sequences
    gc_pos = np.sum(f_pos[:,1:3], axis=1)
    gc_neg = np.sum(f_neg[:,1:3], axis=1)

    # sort by gc content
    gc_pos_sorted = np.sort(gc_pos)
    neg_index_sorted = np.argsort(gc_neg)
    gc_neg_sorted = np.sort(gc_neg)    

    # find nucleotide GC best matches between pos and neg sequences 
    match_index = []
    index = 0 
    for i, gc in enumerate(gc_pos_sorted):
        while index < len(gc_neg_sorted)-1: 
            if (abs(gc - gc_neg_sorted[index+1]) <=  abs(gc - gc_neg_sorted[index])):  
                index += 1
            else: 
                break         
        match_index.append(index)
  
    return neg_one_hot[neg_index_sorted[match_index]]

# TODO: Save to hdf5 file
