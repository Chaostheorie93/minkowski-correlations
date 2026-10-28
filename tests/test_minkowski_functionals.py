import numpy as np
import pytest

from minkowski_correlations import minkowski_functionals


POINTS = np.array([
    [0.2, 0.2],
    [0.5, 0.5],
    [0.8, 0.8],
])

WINDOW = (0.0, 1.0, 0.0, 1.0)


def test_no_boundary():
    mf = minkowski_functionals(
        POINTS,
        inner_radius=0.02,
        outer_radius=0.05,
        jobs=1,
    )

    assert len(mf) == 3
    assert np.all(np.isfinite(mf))


def test_periodic_boundary():
    mf = minkowski_functionals(
        POINTS,
        inner_radius=0.02,
        outer_radius=0.05,
        window=WINDOW,
        boundary_condition="periodic",
        jobs=1,
    )

    assert len(mf) == 3
    assert np.all(np.isfinite(mf))


def test_minus_sampling():
    mf = minkowski_functionals(
        POINTS,
        inner_radius=0.02,
        outer_radius=0.05,
        window=WINDOW,
        boundary_condition="minus",
        jobs=1,
    )

    assert len(mf) == 3
    assert np.all(np.isfinite(mf))


def test_invalid_radii():
    with pytest.raises(ValueError):
        minkowski_functionals(
            POINTS,
            inner_radius=0.1,
            outer_radius=0.05,
            jobs=1,
        )


@pytest.mark.parametrize(
    "boundary_condition",
    ["periodic", "minus"],
)
def test_missing_window(boundary_condition):
    with pytest.raises(ValueError):
        minkowski_functionals(
            POINTS,
            inner_radius=0.02,
            outer_radius=0.05,
            boundary_condition=boundary_condition,
            jobs=1,
        )