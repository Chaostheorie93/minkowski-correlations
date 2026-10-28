# minkowski-correlations

[![DOI](https://zenodo.org/badge/1436512511.svg)](https://doi.org/10.5281/zenodo.23718749)

**Minkowski functionals of two-dimensional point patterns**

`minkowski-correlations` is a Python package for computing the Minkowski functionals of the union of disks or annuli centered at points in a two-dimensional point pattern. The functionals characterize the area, boundary length, and topology of the resulting geometric set.

For a fixed inner radius and outer radius, the package returns three Minkowski functionals:

* **M0:** Area of the union.
* **M1:** Half the perimeter of the union.
* **M2:** Euler characteristic of the union.

Evaluating these quantities across a range of radii, and averaging over realizations when appropriate, provides radius-dependent Minkowski-functional correlation measures.

## Installation

The package requires Python 3.10 or newer. To install it from a local clone of this repository, run the following command in the repository's root directory:

```bash
python -m pip install -e .
```

The required dependencies (`numpy`, `scipy`, and `joblib`) are installed automatically if needed.

## Quick start

```python
import numpy as np
from minkowski_correlations import minkowski_functionals

rng = np.random.default_rng(42)
points = rng.uniform(0.0, 1.0, size=(30, 2))

mf = minkowski_functionals(
    data_points=points,
    inner_radius=0.03,
    outer_radius=0.08,
    boundary_condition="none",
    jobs=1,
)

print("M0 (area):", mf[0])
print("M1 (half perimeter):", mf[1])
print("M2 (Euler characteristic):", mf[2])
```

Setting `inner_radius=0` computes the functionals for a union of disks rather than annuli.

## Boundary conditions

The `boundary_condition` argument supports three modes:

|Mode|Description|Observation window|
|-|-|-|
|`"none"`|Compute the functionals of the full union, without restricting it to a window.|Not required|
|`"periodic"`|Apply periodic boundary conditions on a rectangular observation window.|Required|
|`"minus"`|Use a reduced observation window, shrinking each side by `outer_radius`; contributions from points outside the reduced window are retained.|Required|

For periodic and minus boundary conditions, provide the observation window as `(x_min, x_max, y_min, y_max)`:

```python
mf = minkowski_functionals(
    data_points=points,
    inner_radius=0.03,
    outer_radius=0.08,
    window=(0.0, 1.0, 0.0, 1.0),
    boundary_condition="periodic",
    jobs=1,
)
```

For `"minus"`, the optional argument `include_boundary=True` includes contributions from the covered boundary of the reduced observation window to the perimeter and Euler characteristic. By default, these contributions are excluded.

## API reference

For detailed documentation of the function parameters, return values, and implementation notes, see the [API Reference (PDF)](docs/minkowski-correlations-documentation.pdf).

```python
minkowski_functionals(
    data_points,
    inner_radius,
    outer_radius,
    window=None,
    boundary_condition="none",
    include_boundary=False,
    jobs=-1,
    return_geometry=False,
)
```

|Parameter|Description|
|-|-|
|`data_points`|NumPy array of shape `(N, 2)` containing the point coordinates. Duplicate points are removed automatically.|
|`inner_radius`|Nonnegative inner radius; must be smaller than `outer_radius`.|
|`outer_radius`|Positive outer radius.|
|`window`|Rectangle `(x_min, x_max, y_min, y_max)`. Required for `"periodic"` and `"minus"`; ignored for `"none"`.|
|`boundary_condition`|One of `"none"` (default), `"periodic"`, or `"minus"`.|
|`include_boundary`|Whether to include covered window-boundary contributions for minus sampling (default: `False`).|
|`jobs`|Number of parallel workers used by `joblib` (default: `-1`, requesting all available workers).|
|`return_geometry`|If `True`, also return intermediate geometric results (default: `False`).|

By default, the function returns a list `[M0, M1, M2]`. With `return_geometry=True`, it additionally returns the inner and outer circle-boundary data, observation-window boundary segments, and point coordinates used in the calculation.

## Scientific background

Minkowski functionals provide geometric and topological descriptors of spatial patterns. Applied to unions of disks or annuli centered at the observed points, they capture information beyond pair correlations and can be studied as functions of the disk or annulus radii.

## Tests

Install `pytest` if necessary, then run the tests from the repository root:

```bash
python -m pytest -q
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

