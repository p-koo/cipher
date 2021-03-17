import typing

import numpy as np
import pytest

from cipher.utils import convert_one_hot
from cipher.utils import convert_onehot_to_sequence




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
