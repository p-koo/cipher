"""Tests for `cipher.preprocess.tfrecords`."""

import pathlib

import h5py
import numpy as np
import tensorflow as tf

from cipher.preprocess.tfrecords import _get_str_format_keys
from cipher.preprocess.tfrecords import _serialize_example
from cipher.preprocess.tfrecords import hdf5_to_tfrecords
from cipher.preprocess.tfrecords import get_parse_tfrecord_example_fn


def test__get_str_format_keys():
    assert _get_str_format_keys("foo bar baz") == []
    assert _get_str_format_keys("foo {bar} baz") == ["bar"]
    assert _get_str_format_keys("foo {bar} {baz}") == ["bar", "baz"]
    assert _get_str_format_keys("foo {bar:03d}") == ["bar"]


def test__serialize_example():
    x = np.random.random_sample([2, 4])
    y = np.random.random_sample([6])
    serialized = _serialize_example(x, y, x_dtype="uint8", y_dtype="uint8")

    example = tf.train.Example()
    example.ParseFromString(serialized)
    example_dict: dict = example.features.feature

    assert example_dict["feature/value"].bytes_list.value == [
        x.astype("uint8").tobytes()
    ]
    assert example_dict["feature/dtype"].bytes_list.value == [b"uint8"]
    assert example_dict["feature/ndim"].int64_list.value == [2]
    assert example_dict["feature/shape/0"].int64_list.value == [2]
    assert example_dict["feature/shape/1"].int64_list.value == [4]

    assert example_dict["label/dtype"].bytes_list.value == [b"uint8"]
    assert example_dict["label/value"].bytes_list.value == [y.astype("uint8").tobytes()]
    assert example_dict["label/ndim"].int64_list.value == [1]
    assert example_dict["label/shape/0"].int64_list.value == [6]


def test_integration_hdf5_to_tfrecord(tmp_path: pathlib.Path):
    # Create HDF5 file, then convert to TFRecords, then read into numpy arrays.
    # Make sure that the arrays are the same on the way in and out.

    # 10 sequences, where each sequence has length 20 and 4 nts.
    x = np.random.randint(0, 2, size=[10, 20, 4]).astype(np.int32)
    # 10 sequences and 6 classes.
    y = np.random.randint(0, 2, size=[10, 6]).astype(np.float32)

    hdf5_path = tmp_path / "data.h5"
    with h5py.File(str(hdf5_path), mode="w") as f:
        f.create_dataset("/features", data=x, compression="gzip")
        f.create_dataset("/labels", data=y, compression="gzip")

    tfrec_dir = tmp_path / "tfrecords"
    tfrec_dir.mkdir(exist_ok=True)
    hdf5_to_tfrecords(
        hdf5_path=hdf5_path,
        feature_dataset="/features",
        label_dataset="/labels",
        tfrecord_path=str(tfrec_dir / "data_shard-{shard:02d}.tfrec"),
        feature_dtype="int32",  # use different dtypes to ensure correct behavior
        label_dtype="float32",
        chunksize=1,
        gzip=True,
        verbose=0,
    )

    tfrecs = list(tfrec_dir.glob("*.tfrec"))
    # chunksize=1 implies one tfrec per sample.
    assert len(tfrecs) == x.shape[0]

    parse_fn = get_parse_tfrecord_example_fn(
        feature_dtype="int32",
        label_dtype="float32",
        feature_shape=(20, 4),
        label_shape=(6,),
    )
    # Important to sort here...
    tfrecs_str = sorted([str(path) for path in tfrecs])
    dset = tf.data.TFRecordDataset(tfrecs_str, compression_type="GZIP")
    dset = dset.map(parse_fn)
    xs = []
    ys = []
    for feature, label in dset:
        xs.append(feature)
        ys.append(label)
    xs = np.stack(xs)
    ys = np.stack(ys)

    np.testing.assert_array_equal(x, xs)
    np.testing.assert_array_equal(y, ys)

    # Test again without specifying shape.
    parse_fn = get_parse_tfrecord_example_fn(
        feature_dtype="int32",
        label_dtype="float32",
    )
    dset = tf.data.TFRecordDataset(tfrecs_str, compression_type="GZIP")
    dset = dset.map(parse_fn)
    xs = []
    ys = []
    for feature, label in dset:
        xs.append(feature)
        ys.append(label)
    xs = np.stack(xs)
    ys = np.stack(ys)

    assert xs.shape == (10, 20 * 4)  # type: ignore
    assert ys.shape == (10, 6)  # type: ignore
