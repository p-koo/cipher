import pandas as pd
import numpy as np
from tensorflow import keras
import subprocess

# MOANA (MOtif ANAlysis)

# ---------------------------------------------------------------------------------------
# Get position probability matrix of conv filters


def filter_activations(x_test, model, layer=3, window=20, threshold=0.5, min_align=2):
    """get alignment of filter activations for visualization"""

    # get feature maps of 1st convolutional layer after activation
    intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output)
    fmap = intermediate.predict(x_test)
    N, L, A = x_test.shape

    # Set the left and right window sizes
    window_left = int(window / 2)
    window_right = window - window_left

    W = []
    for filter_index in range(fmap.shape[-1]):

        # Find regions above threshold
        coords = np.where(
            fmap[:, :, filter_index] > np.max(fmap[:, :, filter_index]) * threshold
        )
        x, y = coords

        # Sort score
        index = np.argsort(fmap[x, y, filter_index])[::-1]
        data_index = x[index].astype(int)
        pos_index = y[index].astype(int)

        # Make a sequence alignment centered about each activation (above threshold)
        seq_align = []
        for i in range(len(pos_index)):

            # Determine position of window about each filter activation
            start_window = pos_index[i] - window_left
            end_window = pos_index[i] + window_right

            # Check to make sure positions are valid
            if (start_window > 0) & (end_window < L):
                seq = x_test[data_index[i], start_window:end_window, :]
                seq_align.append(seq)

        # Calculate position probability matrix
        if len(seq_align) >= min_align:
            W.append(np.mean(seq_align, axis=0))
        else:
            W.append(np.ones((window, A)) / 4)
    return np.array(W)


# ---------------------------------------------------------------------------------------
# utilities to process filters


def clip_filters(W, threshold=0.5, pad=3):
    """clip uninformative parts of conv filters"""
    W_clipped = []
    for w in W:
        L, A = w.shape
        entropy = np.log2(4) + np.sum(w * np.log2(w + 1e-7), axis=1)
        index = np.where(entropy > threshold)[0]
        if index.any():
            start = np.maximum(np.min(index) - pad, 0)
            end = np.minimum(np.max(index) + pad + 1, L)
            W_clipped.append(w[start:end, :])
        else:
            W_clipped.append(w)

    return W_clipped


def meme_generate(W, output_file="meme.txt", prefix="filter"):
    """generate a meme file for a set of filters, W ∈ (N,L,A)"""

    # background frequency
    nt_freqs = [1.0 / 4 for i in range(4)]

    # open file for writing
    f = open(output_file, "w")

    # print intro material
    f.write("MEME version 4\n")
    f.write("\n")
    f.write("ALPHABET= ACGT\n")
    f.write("\n")
    f.write("Background letter frequencies:\n")
    f.write("A %.4f C %.4f G %.4f T %.4f \n" % tuple(nt_freqs))
    f.write("\n")

    for j, pwm in enumerate(W):
        L, A = pwm.shape
        f.write("MOTIF %s%d \n" % (prefix, j))
        f.write("letter-probability matrix: alength= 4 w= %d nsites= %d \n" % (L, L))
        for i in range(L):
            f.write("%.4f %.4f %.4f %.4f \n" % tuple(pwm[i, :]))
        f.write("\n")

    f.close()


def count_meme_entries(motif_path):
    """Count number of meme entries"""
    with open(motif_path, "r") as f:
        counter = 0
        for line in f:
            if line[:6] == "letter":
                counter += 1
    return counter


# ---------------------------------------------------------------------------------------
# motif comparison


def tomtom(
    motif_path,
    jaspar_path,
    output_path,
    evalue=False,
    thresh=0.5,
    dist="pearson",
    png=None,
    tomtom_path="tomtom",
):
    """ perform tomtom analysis """
    # "dist options: allr | ​ ed | ​ kullback | ​ pearson | ​ sandelin"
    cmd = [tomtom_path, "-thresh", str(thresh), "-dist", dist]
    if evalue:
        cmd.append("-evalue")
    if png:
        cmd.append("-png")
    cmd.extend(["-oc", output_path, motif_path, jaspar_path])

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr


