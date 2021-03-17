"""Tests for `cipher.preprocess.wrangle`."""

import gzip
import io
import pathlib
import typing

import numpy as np
import pytest

from cipher.preprocess.singletask import _is_gzipped
from cipher.preprocess.singletask import filter_max_length
from cipher.preprocess.singletask import enforce_constant_size
from cipher.preprocess.singletask import parse_fasta
from cipher.preprocess.singletask import convert_one_hot
from cipher.preprocess.singletask import convert_onehot_to_sequence
from cipher.preprocess.singletask import filter_nonsense_sequences


def test__is_gzipped(tmp_path: pathlib.Path):
    p = tmp_path / "test.txt.gz"
    with gzip.open(p, "wt") as f:
        f.write("foobar")
    assert _is_gzipped(p)

    # Overwrite file.
    p.write_text("foobar")
    assert not _is_gzipped(p)

    with pytest.raises(FileNotFoundError):
        _is_gzipped("non-existing-path.txt")


@pytest.mark.xfail
def test_filter_encode_metatable(tmp_path: pathlib.Path):
    raise NotImplementedError()


@pytest.mark.xfail
def test_extract_metatable_information(tmp_path: pathlib.Path):
    raise NotImplementedError()


@pytest.mark.parametrize(
    "length,val,ref",
    [
        (
            1000,
            "chr7\t0\t2000\nchr7\t4000\t4500\nchr7\t20000\t20001",
            "chr7\t4000\t4500\nchr7\t20000\t20001\n",
        )
    ],
)
@pytest.mark.parametrize("opener", [io.open, gzip.open])
def test_filter_max_length(
    length: int, val: str, ref: str, opener, tmp_path: pathlib.Path
):
    suffix = ".bed.gz" if opener is gzip.open else ".bed"
    orig = tmp_path / f"foobar{suffix}"
    with opener(orig, mode="wt") as f:
        f.write(val)
    out = orig.with_suffix(f".new{suffix}")
    filter_max_length(bed_path=str(orig), output_path=str(out), max_len=length)
    with opener(out, mode="rt") as f:
        assert f.read() == ref


@pytest.mark.parametrize(
    "length,val,ref",
    [
        (
            1000,
            "chr7\t0\t2000\nchr7\t4000\t4500\nchr7\t20000\t20001",
            "chr7\t500\t1500\nchr7\t3750\t4750\nchr7\t19500\t20500\n",
        ),
        (
            100,
            "chr7\t0\t2000\nchr7\t4000\t4500\nchr7\t20000\t20001",
            "chr7\t950\t1050\nchr7\t4200\t4300\nchr7\t19950\t20050\n",
        ),
    ],
)
@pytest.mark.parametrize("opener", [io.open, gzip.open])
def test_enforce_constant_size(
    length: int, val: str, ref: str, opener, tmp_path: pathlib.Path
):
    suffix = ".bed.gz" if opener is gzip.open else ".bed"
    orig = tmp_path / f"foobar{suffix}"
    with opener(orig, mode="wt") as f:
        f.write(val)
    out = orig.with_suffix(f".new{suffix}")
    enforce_constant_size(bed_path=str(orig), output_path=str(out), window=length)
    with opener(out, mode="rt") as f:
        assert f.read() == ref


@pytest.mark.parametrize(
    "val,ref_seqs,ref_descs",
    [
        (
            """>SEQUENCE_1
MTEITAAMVKELREKAAKKADRLAAEG
LVSVKVSDDPEHK


>SEQUENCE_2
SATVSEINSEEYLKSQI
ATIGENLVVRRFATLKAGQICMH""",
            [
                "MTEITAAMVKELREKAAKKADRLAAEGLVSVKVSDDPEHK",
                "SATVSEINSEEYLKSQIATIGENLVVRRFATLKAGQICMH",
            ],
            ["SEQUENCE_1", "SEQUENCE_2"],
        )
    ],
)
@pytest.mark.parametrize("opener", [io.open, gzip.open])
def test_parse_fasta(
    val: str,
    ref_seqs: typing.List[str],
    ref_descs: typing.List[str],
    opener,
    tmp_path: pathlib.Path,
):
    orig = tmp_path / "data.fa"
    with opener(orig, mode="wt") as f:
        f.write(val)
    seqs, descs = parse_fasta(orig)
    np.testing.assert_array_equal(seqs, ref_seqs)
    np.testing.assert_array_equal(descs, ref_descs)


@pytest.mark.parametrize(
    "val,ref,alphabet",
    [
        (
            ["ABCD", "DCBA", "AABD"],
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            "ABCD",
        ),
        # Fail when sequences are not the same length.
        (["DDBC", "GG"], ValueError, "BCDG"),
    ],
)
def test_one_hot(val: typing.List[str], ref, alphabet: str):
    if ref is ValueError:
        with pytest.raises(ref):
            convert_one_hot(val, alphabet=alphabet)
    else:
        out = convert_one_hot(val, alphabet=alphabet)
        np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize(
    "val,ref,alphabet",
    [
        (
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            [["A", "C", "G", "T"], ["T", "G", "C", "A"], ["A", "A", "C", "T"]],
            "ACGT",
        )
    ],
)
def test_convert_onehot_to_sequence(
    val: typing.List, ref: typing.List[typing.List[str]], alphabet: str
):
    out = convert_onehot_to_sequence(val, alphabet=alphabet)
    np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize(
    "val,ref,inds",
    [(["ABCD", "ADDCCBN", "DACDGAEN", "ABBA"], ["ABCD", "ABBA"], [0, 3])],
)
def test_filter_nonsense_sequences(
    val: typing.List[str], ref: typing.List[str], inds: typing.List[int]
):
    out_seqs, out_inds = filter_nonsense_sequences(val)
    np.testing.assert_array_equal(out_seqs, ref)
    np.testing.assert_array_equal(out_inds, inds)


@pytest.mark.xfail
def test_match_gc():
    raise NotImplementedError()


@pytest.mark.xfail
def test_bedtools_getfasta():
    raise NotImplementedError()


@pytest.mark.xfail
def test_bedtools_intersect():
    raise NotImplementedError()


@pytest.mark.xfail
def test_split_dataset():
    raise NotImplementedError()
