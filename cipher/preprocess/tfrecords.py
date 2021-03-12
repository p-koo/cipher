"""This module includes functions to convert data between HDF5 and TFRecord format.

TFRecord files are composed of Protocol Buffer messages, and the message here have
the following keys and types:

- feature/value - (bytes) feature array. Dtype and shape must be known to reconstruct
    this to a tensor.
- feature/dtype - (bytes) name of the datatype of the features.
- feature/ndim - (int) the number of dimensions
- feature/shape/0 - (int) length of dimension 0.
- feature/shape/1 - (int) length of dimension 1.
- feature/shape/N - (int) length of dimension N.
- label/value - (bytes) feature array. Dtype and shape must be known to reconstruct
    this to a tensor.
- label/dtype - (bytes) name of the datatype of the labels.
- label/ndim - (int) the number of dimensions in the labels array.
- label/shape/0 - (int) length of dimension 0.
- label/shape/1 - (int) length of dimension 1.
- label/shape/N - (int) length of dimension N.

Given an HDF5 file with arrays of features and labels, one can convert to multiple
TFRecord files as below.

.. code-block:: python

    hdf5_to_tfrecords(
        hdf5_path="path/to/data.h5",
        feature_dataset="/train/features",
        label_dataset="/train/labels",
        tfrecord_path="tfrecords/data-train_shard-{shard:02d}.tfrec",
        feature_dtype="uint8",
        label_dtype="uint8",
        chunksize=100,  # number of samples per TFRecord file
    )

Given existing TFRecord files, one can construct a data pipeline using :code:`tf.data`.

.. code-block:: python

    glob_pattern = "tfrecords/data-train_shard-*.tfrec"
    dset = tf.data.Dataset.list_files(glob_pattern)
    dset = dset.interleave(
        lambda f: tf.data.TFRecordDataset(f, compression_type="GZIP")
    )
    parse_example = get_parse_tfrecord_example_fn(
        feature_dtype="uint8",
        label_dtype="uint8",
        feature_shape=(1000, 4),
        label_shape=(200,),
    )
    dset = dset.map(parse_example)
"""

import pathlib
import string
import typing

import h5py
import numpy as np
import tensorflow as tf

# Type that represents a path on the filesystem.
PathLike = typing.TypeVar("PathLike", str, pathlib.Path)


def _serialize_example(x, y, x_dtype=np.uint8, y_dtype=np.uint8) -> bytes:
    """Creates a serialized `tf.train.Example` message ready to be written to a file.

    Parameters
    ----------
    x : numpy array
        Array of a single sequence. This must be two-dimensional with shape
        `(sequence_length, alphabet_length)`. For example, the shape could be (1000, 4)
        for a 1000-nt-long sequence with 4 possible nucleotides.
    y : numpy array
        Array for the targets of a single sequence. This must be one-dimensional. The
        length of the array is the number of possible targets.
    x_dtype : str, dtype, optional
        Data type of the features. Default is uint8.
    y_dtype : str, dtype, optional
        Data type of the labels. Default is uint8.

    Returns
    -------
    bytes
        Protobuf message ready to be written to a TFRecord file.
    """

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = (
                value.numpy()
            )  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    x = np.asanyarray(x).astype(x_dtype)
    y = np.asanyarray(y).astype(y_dtype)

    # This is a dictionary used to construct a protobuf message.
    tf_feature = {
        "feature/value": _bytes_feature(x.tobytes()),
        "feature/dtype": _bytes_feature(x.dtype.name.encode()),
        "feature/ndim": _int64_feature(x.ndim),
    }
    # Add shape info. This part is dynamic because the array could have any
    # number of dimensions.
    tf_feature.update(
        {f"feature/shape/{i}": _int64_feature(s) for i, s in enumerate(x.shape)}
    )

    # Update with information about labels. We add label information after all
    # feature information has been added so that feature information all
    # stays together, and all label information stays together.
    # Otherwise, feature and label info would be interleaved.
    tf_feature.update(
        {
            "label/value": _bytes_feature(y.tobytes()),
            "label/dtype": _bytes_feature(y.dtype.name.encode()),
            "label/ndim": _int64_feature(y.ndim),
        }
    )
    tf_feature.update(
        {f"label/shape/{i}": _int64_feature(s) for i, s in enumerate(y.shape)}
    )

    example_proto = tf.train.Example(features=tf.train.Features(feature=tf_feature))

    return example_proto.SerializeToString()


def _get_str_format_keys(s: str) -> list:
    """Return the formatting keys present in a string `s`.

    Examples
    --------
    >>> _get_str_format_keys("foo {bar}")
    ['bar']
    >>> _get_str_format_keys("foo")
    []
    """
    # https://stackoverflow.com/a/46161774/5666087

    return [tup[1] for tup in string.Formatter().parse(s) if tup[1] is not None]


def hdf5_to_tfrecords(
    hdf5_path: PathLike,
    feature_dataset: str,
    label_dataset: str,
    tfrecord_path: str,
    feature_dtype=np.uint8,
    label_dtype=np.uint8,
    chunksize=10000,
    gzip=True,
    first_shard=0,
    verbose=1,
) -> None:
    """Convert data in HDF5 to TFRecord format.

    Data in features and labels datasets must have same length in the first dimension.

    Parameters
    ----------
    hdf5_path : Pathlike
        Path to HDF5 file with data.
    feature_dataset : str
        Path to dataset of features in the HDF5 file.
    label_dataset : str
        Path to dataset of labels in the HDF5 file.
    tfrecord_path : Pathlike
        Path to save TFRecord files. This **must** contain the formatting key `shard`,
        which accepts an integer. The data is split among many TFRecord files, because
        TFRecord supports parallel reading. Each TFRecord file is one shard of the
        dataset.
    feature_dtype : str, dtype
        Data type of the features.
    label_dtype : str, dtype
        Data type of the labels.
    chunksize : int
        Size of one chunk. This many samples are saved to a single TFRecord file.
    gzip : bool
        Whether to gzip-compress. It is highly recommended to set this to `True`.
    first_shard : int
        The beginning of the numbering scheme. Usually, 0 is the best choice, but if
        one must use this function multiple times on the same dataset, then one can
        call this function with a higher `first_shard` to continue the numbering from
        the previous function call.
    verbose : int
        Verbosity mode: 0 (silent), 1 (verbose), 2 (semi-verbose).


    Returns
    -------
    None
    """

    tfrecord_path = str(tfrecord_path)
    if "shard" not in _get_str_format_keys(tfrecord_path):
        raise ValueError("tfrecord_path must contain string formatting key `shard`")

    # Make sure lengths of features and labels are the same.
    with h5py.File(hdf5_path, mode="r") as f:
        feature_shape = f[feature_dataset].shape
        label_shape = f[label_dataset].shape

    if feature_shape[0] != label_shape[0]:
        raise ValueError(
            "Features and labels must have the same length in the first dimension but"
            f" found {feature_shape[0]} and {label_shape[0]}"
        )

    # Slices that allow us to index chunks of the arrays.
    # something like `[ [0:10], [10:20], [20:30] ]`.
    slices = [slice(i, i + chunksize) for i in range(0, feature_shape[0], chunksize)]

    if gzip:
        options = tf.io.TFRecordOptions(compression_type="GZIP", compression_level=6)
    else:
        options = None

    progress = tf.keras.utils.Progbar(len(slices), verbose=verbose)
    progress.update(0)

    for i, sl in enumerate(slices):
        # We read a chunk at a time to be memory efficient.
        with h5py.File(hdf5_path, "r") as f:
            this_x_chunk = f[feature_dataset][sl]
            this_y_chunk = f[label_dataset][sl]
        # Sanity check
        if this_x_chunk.shape[0] != this_y_chunk.shape[0]:
            raise ValueError("Number of samples in this chunk is not equal.")
        this_file = tfrecord_path.format(shard=i)
        with tf.io.TFRecordWriter(path=this_file, options=options) as writer:
            for j in range(this_x_chunk.shape[0]):
                example = _serialize_example(
                    x=this_x_chunk[j],
                    y=this_y_chunk[j],
                    x_dtype=feature_dtype,
                    y_dtype=label_dtype,
                )
                writer.write(example)
        progress.add(1)


def get_parse_tfrecord_example_fn(
    feature_dtype, label_dtype, feature_shape=None, label_shape=None
):
    """Return a function that can be used to parse a single TFRecord example.

    Parameters
    ----------
    x_dtype : str, dtype
        Data type of the features.
    y_dtype : str, dtype
        Data type of the labels.
    x_shape : int or tuple of ints, optional
        Shape of the features. If not specified, the feature tensor will be flat.
    y_shape : int or tuple of ints, optional
        Shape of the labels. If not specified, the label tensor will be flat.

    Returns
    -------
    Callable
        Function that takes one argument, a byte-encoded example.
    """

    def parse_tfrecord_example(serialized: bytes) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        """Return tuple (features, labels) from one serialized TFRecord example.

        Parameters
        ----------
        serialized : bytes
            The byte-encoded example.

        Returns
        -------
        tuple
            Tuple of (features, labels).
        """
        features = {
            "feature/value": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            "label/value": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        }
        example = tf.io.parse_single_example(serialized, features)
        x = tf.io.decode_raw(example["feature/value"], feature_dtype)
        y = tf.io.decode_raw(example["label/value"], label_dtype)
        # The shapes are encoded in the TFRecord file, but we cannot use
        # them dynamically (aka reshape according to the shape in this example).
        if feature_shape is not None:
            x = tf.reshape(x, shape=feature_shape)
        if label_shape is not None:
            y = tf.reshape(y, shape=label_shape)
        return x, y

    return parse_tfrecord_example
