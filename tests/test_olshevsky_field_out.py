"""CPU-only regression tests for filling preallocated benchmark buffers."""

import numpy as np
import pytest

from benchmarks.olshevsky2020_field import create_grid, olshevsky2020_field


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_filling_existing_fortran_views(dtype):
    c = create_grid(12)
    reference = olshevsky2020_field(c, c, c, dtype=dtype)
    flat = [np.empty(c.size**3, dtype=dtype) for _ in range(3)]
    views = tuple(a.reshape((c.size,) * 3, order="F") for a in flat)

    result = olshevsky2020_field(c, c, c, out=views)
    for original, filled, underlying in zip(reference, result, flat, strict=True):
        assert np.shares_memory(filled, underlying)
        assert filled.flags.f_contiguous
        np.testing.assert_array_equal(original, filled)


@pytest.mark.parametrize(
    ("buffers", "error"),
    [
        ((np.empty((1, 1, 1)),) * 3, ValueError),
        ((np.zeros((12, 12, 12), dtype=np.int32),) * 3, TypeError),
        ((np.zeros((12, 12, 12), dtype=np.float32),) * 2, ValueError),
    ],
)
def test_rejects_invalid_out_buffers(buffers, error):
    c = create_grid(12)
    with pytest.raises(error):
        olshevsky2020_field(c, c, c, out=buffers)
