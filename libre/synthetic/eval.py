import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

#------------------------------------------------------------------------
# Attribution analysis  
#------------------------------------------------------------------------

def interpretability_performance(scores, x_model, threshold=0.01):
    """Compare attribution scores to ground truth (e.g. x_model).
    scores --> (N,L)
    x_model --> (N,L,A)
    """

    pr_score = []
    roc_score = []
    for j, score in enumerate(scores):

        # calculate information of ground truth
        gt_info = np.log2(4) + np.sum(x_model[j] * np.log2(x_model[j] + 1e-10), axis=1)

        # set label if information is greater than 0
        label = np.zeros(gt_info.shape)
        label[gt_info > threshold] = 1

        # (don't evaluate over low info content motif positions)
        index = np.where((gt_info > threshold) | (gt_info == np.min(gt_info)))[0]

        # precision recall metric
        precision, recall, thresholds = precision_recall_curve(
            label[index], score[index]
        )
        pr_score.append(auc(recall, precision))

        # roc curve
        fpr, tpr, thresholds = roc_curve(label[index], score[index])
        roc_score.append(auc(fpr, tpr))

    roc_score = np.array(roc_score)
    pr_score = np.array(pr_score)

    return roc_score, pr_score


def signal_noise_stats(scores, x_model, top_k=10, threshold=0.01):
    """averate saliency score at signals and average noise level. Signal and 
     noise are determined by information content of sequence model (x_model)"""

    signal = []
    noise_mean = []
    noise_max = []
    noise_topk = []
    for j, score in enumerate(scores):

        # calculate information of ground truth
        gt_info = np.log2(4) + np.sum(x_model[j]*np.log2(x_model[j]+1e-10), axis=1)

        # (don't evaluate over low info content motif positions)  
        index = np.where(gt_info > threshold)[0]

        # evaluate noise levels
        index2 = np.where((score > 0) & (gt_info == np.min(gt_info)))[0]

        if len(index2) < top_k:
          signal.append(0)
          noise_max.append(0)
          noise_mean.append(0)
          noise_topk.append(0)
        else:
          signal.append(np.mean(score[index]))
          noise_max.append(np.max(score[index2]))
          noise_mean.append(np.mean(score[index2]))
          sort_score = np.sort(score[index2])[::-1]
          noise_topk.append(np.mean(sort_score[:top_k]))
          
    return (
        np.array(signal),
        np.array(noise_max),
        np.array(noise_mean),
        np.array(noise_topk),
    )


def calculate_snr(signal, noise):
  snr = signal/noise
  snr[np.isnan(snr)] = 0
  return snr
  

#------------------------------------------------------------------------
# Filter analysis  
#------------------------------------------------------------------------


def synthetic_multiclass_motifs():

    arid3 = ["MA0151.1", "MA0601.1", "PB0001.1"]
    cebpb = ["MA0466.1", "MA0466.2"]
    fosl1 = ["MA0477.1"]
    gabpa = ["MA0062.1", "MA0062.2"]
    mafk = ["MA0496.1", "MA0496.2"]
    max1 = ["MA0058.1", "MA0058.2", "MA0058.3"]
    mef2a = ["MA0052.1", "MA0052.2", "MA0052.3"]
    nfyb = ["MA0502.1", "MA0060.1", "MA0060.2"]
    sp1 = ["MA0079.1", "MA0079.2", "MA0079.3"]
    srf = ["MA0083.1", "MA0083.2", "MA0083.3"]
    stat1 = ["MA0137.1", "MA0137.2", "MA0137.3", "MA0660.1", "MA0773.1"]
    yy1 = ["MA0095.1", "MA0095.2"]
    motif_ids = [arid3, cebpb, fosl1, gabpa, mafk, max1, mef2a, nfyb, sp1, srf, stat1, yy1]

    motif_names = [
        "Arid3",
        "CEBPB",
        "FOSL1",
        "GABPA",
        "MAFK",
        "MAX",
        "MEF2A",
        "NFYB",
        "SP1",
        "SRF",
        "STAT1",
        "YY1",
    ]
    return motif_names, motif_ids



def match_hits_to_ground_truth(file_path, motif_names, motif_ids, num_filters=32):
    """works with Tomtom version 5.1.0
    inputs:
        - file_path: .tsv file output from tomtom analysis
        - motif_ids: list of list of JASPAR ids
        - motif_names: name of motifs in the list
        - num_filters: number of filters in conv layer (needed to normalize -- tomtom
            doesn't always give results for every filter)

    outputs:
        - match_fraction: fraction of hits to ground truth motifs
        - match_any: fraction of hits to any motif in JASPAR (except Gremb1)
        - filter_match: the motif of the best hit (to a ground truth motif)
        - filter_qvalue: the q-value of the best hit to a ground truth motif
            (1.0 means no hit)
        - motif_qvalue: for each ground truth motif, gives the best qvalue hit
        - motif_counts for each ground truth motif, gives number of filter hits
    """

    # add a zero for indexing no hits
    motif_ids = motif_ids.copy()
    motif_names = motif_names.copy()
    motif_ids.insert(0, [""])
    motif_names.insert(0, "")

    # get dataframe for tomtom results
    df = pd.read_csv(file_path, delimiter="\t")

    # loop through filters
    filter_qvalue = np.ones(num_filters)
    best_match = np.zeros(num_filters).astype(int)
    correction = 0
    for name in np.unique(df["Query_ID"][:-3].to_numpy()):
        filter_index = int(name.split("r")[1])

        # get tomtom hits for filter
        subdf = df.loc[df["Query_ID"] == name]
        targets = subdf["Target_ID"].to_numpy()

        # loop through ground truth motifs
        for k, motif_id in enumerate(motif_ids):

            # loop through variations of ground truth motif
            for id in motif_id:

                # check if there is a match
                index = np.where(targets == id)[0]
                if len(index) > 0:
                    qvalue = subdf["q-value"].to_numpy()[index]

                    # check to see if better motif hit, if so, update
                    if filter_qvalue[filter_index] > qvalue:
                        filter_qvalue[filter_index] = qvalue
                        best_match[filter_index] = k

        # dont' count hits to Gmeb1 (because too many)
        index = np.where(targets == "MA0615.1")[0]
        if len(index) > 0:
            if len(targets) == 1:
                correction += 1

    # get names of best match motifs
    filter_match = [motif_names[i] for i in best_match]

    # get hits to any motif
    # 3 is correction because of last 3 lines of comments in the tsv file (may change
    # across tomtom versions)
    num_matches = len(np.unique(df["Query_ID"])) - 3.0
    # counts hits to any motif (not including Grembl)
    match_any = (num_matches - correction) / num_filters

    # match fraction to ground truth motifs
    match_index = np.where(filter_qvalue != 1.0)[0]
    if any(match_index):
        match_fraction = len(match_index) / float(num_filters)
    else:
        match_fraction = 0.0

    # get the number of hits and minimum q-value for each motif
    num_motifs = len(motif_ids) - 1
    motif_qvalue = np.zeros(num_motifs)
    motif_counts = np.zeros(num_motifs)
    for i in range(num_motifs):
        index = np.where(best_match == i + 1)[0]
        if len(index) > 0:
            motif_qvalue[i] = np.min(filter_qvalue[index])
            motif_counts[i] = len(index)

    # TODO: consider changing this to a namedtuple to make the output explicit.
    return (
        match_fraction,
        match_any,
        filter_match,
        filter_qvalue,
        motif_qvalue,
        motif_counts,
    )
