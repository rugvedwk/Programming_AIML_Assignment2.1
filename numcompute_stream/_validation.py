"""Validation helpers shared by the streaming components."""

import numpy as np

from numcompute.utils import check_array


def _integer(value, name, minimum):
    """Validate an integer setting while rejecting boolean flags."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")
    return int(value)


def _numeric_matrix(X):
    X = check_array(X, ensure_2d=True, dtype=np.float64, allow_inf=False)
    if not X.shape[1]:
        raise ValueError("X must have at least one feature")
    return X


def _wide_integer(source, target):
    """Whether converting the source integer dtype can lose bits in a float."""
    return (
        source.kind in "iu"
        and target.kind == "f"
        and np.iinfo(source).bits - (source.kind == "i") > np.finfo(target).nmant + 1
    )


def _discrete_kind(values):
    """Identify a nonempty, validated vector of comparable discrete values."""
    if values.dtype.kind != "O":
        return values.dtype.kind if values.dtype.kind in "US" else "numeric"
    kinds = set()
    for value in values:
        if isinstance(value, str):
            kinds.add("U")
        elif isinstance(value, bytes):
            kinds.add("S")
        else:
            kinds.add("numeric")
    if len(kinds) != 1:
        raise TypeError("values must be mutually comparable and of one kind")
    return kinds.pop()
