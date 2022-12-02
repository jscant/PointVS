"""Tests for data pipeline."""
import numpy as np
import pandas as pd
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from point_vs.preprocessing.preprocessing import angle_3d, generate_edges, \
    extract_coords


# Tests should be repeatable
np.random.seed(2)


struct = pd.DataFrame({
    'x': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'y': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    'z': [0, 0, 0, 0, 2, 2, 2, 2, 6, 6, 6, 6],
    'atomic_number': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    'types': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    'bp': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
})


def test_angle_3d():
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    assert angle_3d(v1, v2) == pytest.approx(np.pi / 2)


def test_generate_edges():
    global struct
    out_struct, edge_indices, edge_attrs = generate_edges(
        struct.copy(), inter_radius=2.1, intra_radius=1.1, prune=False)
    assert_array_equal(
        edge_indices[0],
        np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
             7, 7, 8, 8, 9, 9, 10, 10, 11, 11]))
    assert_array_equal(
        edge_indices[1],
        np.array(
            [4, 5, 6, 7, 0, 1, 2, 3, 1, 2, 0, 3, 0, 3, 1, 2, 5, 6, 4, 7, 4, 7,
             5, 6, 9, 10, 8, 11, 8, 11, 9, 10])
    )
    assert_array_equal(
        edge_attrs, np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))


def test_generate_edges_prune():
    global struct
    out_struct, edge_indices, edge_attrs = generate_edges(
        struct.copy(), inter_radius=2.1, intra_radius=1.1, prune=True)
    assert_array_equal(
        edge_indices[0],
        np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
             7, 7]))
    assert_array_equal(
        edge_indices[1],
        np.array(
            [4, 5, 6, 7, 0, 1, 2, 3, 1, 2, 0, 3, 0, 3, 1, 2, 5, 6, 4, 7, 4, 7,
             5, 6])
    )
    assert_array_equal(
        edge_attrs, np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2,
             2, 2]))


def test_extract_coords():
    global struct
    assert_array_equal([[0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [1, 1, 0]], extract_coords(struct, bp=0))
    assert_array_equal([[0, 0, 2],
                        [1, 0, 2],
                        [0, 1, 2],
                        [1, 1, 2],
                        [0, 0, 6],
                        [1, 0, 6],
                        [0, 1, 6],
                        [1, 1, 6]], extract_coords(struct, bp=1))
