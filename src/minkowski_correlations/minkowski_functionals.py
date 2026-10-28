import math
import numpy as np
from joblib import Parallel, delayed

from . import ring_surfaces as RS
from . import rectangle_surfaces as REC





def minkowski_functionals(
    data_points: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    window: tuple[float, float, float, float] | None = None,
    boundary_condition: str = "none",
    include_boundary: bool = False,
    jobs: int = -1,
    return_geometry: bool = False,
):
    """Compute the Minkowski functionals of a union of annuli.

    Each input point defines an annulus with a common inner and outer
    radius. The Minkowski functionals are evaluated analytically from
    the exposed boundary arcs, including contributions from the
    observation window where applicable.

    Parameters
    ----------
    data_points : numpy.ndarray
        Point coordinates as an array of shape (N, 2).
        Duplicate points are automatically removed while
        preserving the order of their first occurrence.
    inner_radius : float
        Inner radius of the annuli. Must be nonnegative and smaller
        than outer_radius. A value of zero produces disks.
    outer_radius : float
        Outer radius of the annuli. Must be positive.
    window : tuple of float or None, optional
        Rectangular observation window given by
        (x_min, x_max, y_min, y_max).
        Required for "periodic" and "minus" boundary conditions.
        Ignored when boundary_condition is "none".
        Default is None.
    
    boundary_condition : {"none", "periodic", "minus"}, optional
        Method used to handle edge effects caused by the finite
        observation window:

        - "none": Ignore the observation window and compute the
          Minkowski functionals of the full union of annuli.
        - "periodic": Apply periodic boundary conditions, treating
          the rectangular observation window as a periodic unit cell.
          Annuli crossing one boundary re-enter from the opposite side.
        - "minus": Apply minus sampling by shrinking each side of
          the rectangular observation window by outer_radius.
          Points outside the reduced window are retained to account
          for their contributions inside it.

        The default is "none". Periodic boundary conditions and
        minus sampling require an axis-aligned rectangular window.

    include_boundary : bool, optional
        If True and boundary_condition is "minus", include the
        contributions of the covered observation-window boundary
        to the perimeter and Euler characteristic.
        By default, these contributions are excluded.

    jobs : int, optional
        Number of parallel workers used by joblib. A value of -1
        requests all available workers. Default is -1.
    return_geometry : bool, optional
        If True, return the geometric intermediate results in
        addition to the Minkowski functionals. Default is False.

    Returns
    -------
    MF : list of float
        Minkowski functionals in the order
        (area, half the perimeter, Euler characteristic),
        corresponding to (M0, M1, M2).
    innerCircles_new : list, optional
        Inner circle boundaries after intersection with the
        observation window. Returned if return_geometry is True.
    outerCircles_new : list, optional
        Outer circle boundaries after intersection with the
        observation window. Returned if return_geometry is True.
    Rectangle : RectangleSurface, optional
        Accumulated observation-window boundary segments.
        Returned if return_geometry is True.
    points : list, optional
        Point coordinates used in the calculation, including
        periodic copies where applicable.
        Returned if return_geometry is True.

    Raises
    ------
    ValueError
        If the point coordinates, radii, observation window, or
        boundary condition are invalid.
    TypeError
        If the input points cannot be converted to numeric values.

    Notes
    -----
    The implementation evaluates boundary integrals over the
    exposed arcs of the union of annuli. The Euler characteristic
    is obtained from curvature and vertex contributions.
    """

    # Validate input points.
    data_points = np.asarray(data_points, dtype=float)

    if data_points.ndim != 2 or data_points.shape[1] != 2:
        raise ValueError(
            "data_points must have shape (N, 2)."
        )

    if not np.all(np.isfinite(data_points)):
        raise ValueError(
            "data_points must contain only finite coordinates."
        )

    _, unique_indices = np.unique(
        data_points, axis=0, return_index=True
    )
    data_points = data_points[np.sort(unique_indices)]

    # Validate radii.
    if not (
        np.isfinite(inner_radius)
        and np.isfinite(outer_radius)
        and 0 <= inner_radius < outer_radius
    ):
        raise ValueError(
            "Radii must satisfy 0 <= inner_radius < outer_radius "
            "and must be finite."
        )

    # Validate boundary condition.
    if boundary_condition not in ("none", "periodic", "minus"):
        raise ValueError(
            "boundary_condition must be 'none', 'periodic', "
            "or 'minus'."
        )

    # Validate observation window when required.
    if boundary_condition in ("periodic", "minus"):
        if window is None or len(window) != 4:
            raise ValueError(
                "window must contain four values: "
                "(x_min, x_max, y_min, y_max)."
            )

        if not np.all(np.isfinite(window)):
            raise ValueError(
                "window must contain only finite values."
            )

        if window[0] >= window[1] or window[2] >= window[3]:
            raise ValueError(
                "Window bounds must satisfy "
                "x_min < x_max and y_min < y_max."
            )

        if (
            window[1] - window[0] <= 2 * outer_radius
            or window[3] - window[2] <= 2 * outer_radius
        ):
            raise ValueError(
                "Window is too small for the specified outer radius."
            )

    # Prepare point coordinates and boundary flags.
    points = data_points.tolist()
    n = len(points)
    completely_IN = [True for _ in range(n)]

    # Apply periodic boundary conditions.
    if boundary_condition == "periodic":
        window_final = window

        for i in range(n):
            x_min = points[i][0] - window[0] < outer_radius
            x_max = window[1] - points[i][0] < outer_radius
            y_min = points[i][1] - window[2] < outer_radius
            y_max = window[3] - points[i][1] < outer_radius

            if x_min:
                q = [
                    points[i][0] + window[1] - window[0],
                    points[i][1],
                ]
                points.append(q)
                completely_IN[i] = False
                completely_IN.append(False)

                if y_min:
                    q1 = [
                        points[i][0] + window[1] - window[0],
                        points[i][1] + window[3] - window[2],
                    ]
                    points.append(q1)
                    completely_IN.append(False)

                if y_max:
                    q2 = [
                        points[i][0] + window[1] - window[0],
                        points[i][1] - window[3] + window[2],
                    ]
                    points.append(q2)
                    completely_IN.append(False)

            if x_max:
                q = [
                    points[i][0] - window[1] + window[0],
                    points[i][1],
                ]
                points.append(q)
                completely_IN[i] = False
                completely_IN.append(False)

                if y_min:
                    q1 = [
                        points[i][0] - window[1] + window[0],
                        points[i][1] + window[3] - window[2],
                    ]
                    points.append(q1)
                    completely_IN.append(False)

                if y_max:
                    q2 = [
                        points[i][0] - window[1] + window[0],
                        points[i][1] - window[3] + window[2],
                    ]
                    points.append(q2)
                    completely_IN.append(False)

            if y_min:
                q = [
                    points[i][0],
                    points[i][1] + window[3] - window[2],
                ]
                points.append(q)
                completely_IN[i] = False
                completely_IN.append(False)

            if y_max:
                q = [
                    points[i][0],
                    points[i][1] - window[3] + window[2],
                ]
                points.append(q)
                completely_IN[i] = False
                completely_IN.append(False)

    # Apply minus-sampling boundary conditions.
    elif boundary_condition == "minus":
        window_final = [
            window[0] + outer_radius,
            window[1] - outer_radius,
            window[2] + outer_radius,
            window[3] - outer_radius,
        ]

        for i in range(n):
            if not (
                window_final[0] + outer_radius
                <= points[i][0]
                <= window_final[1] - outer_radius
                and window_final[2] + outer_radius
                <= points[i][1]
                <= window_final[3] - outer_radius
            ):
                completely_IN[i] = False

    else:
        window_final = None

    # Determine the exposed inner and outer circle boundaries.
    innerCircles, outerCircles = RS.calculate_ring_surface(
        points, inner_radius, outer_radius, jobs=jobs
    )

    # Evaluate local contributions in parallel.
    integral_results = Parallel(n_jobs=jobs)(
        delayed(_compute_integrals)(
            points[i],
            i,
            inner_radius,
            outer_radius,
            innerCircles[i],
            outerCircles[i],
            completely_IN[i],
            window_final,
        )
        for i in range(len(completely_IN))
    )

    # Accumulate Minkowski functionals and geometric results.
    MF = [0 for _ in range(3)]
    innerCircles_new = [None for _ in range(len(completely_IN))]
    outerCircles_new = [None for _ in range(len(completely_IN))]
    Rectangle = REC.RectangleSurface()

    for M, Rect, innerCircle, outerCircle, i in integral_results:
        for j in range(3):
            MF[j] += M[j]

        innerCircles_new[i] = innerCircle
        outerCircles_new[i] = outerCircle
        Rectangle.union(Rect)

    # Add boundary contributions to the area integral.
    if boundary_condition in ("periodic", "minus"):
        value = [
            -window_final[0],
            window_final[1],
            -window_final[2],
            window_final[3],
        ]

        for j in range(4):
            segments = getattr(Rectangle, Rectangle.SIDES[j])
            for seg in segments:
                MF[2] += value[j] * (seg.end - seg.start)

    # Optionally include boundary contributions for minus sampling.
    if boundary_condition == "minus" and include_boundary:
        for j in range(4):
            segments = getattr(Rectangle, Rectangle.SIDES[j])

            for seg in segments:
                MF[1] += seg.end - seg.start

                if seg.rot_start is not None:
                    MF[0] += seg.rot_start

                if seg.rot_end is not None:
                    MF[0] += seg.rot_end

    # Normalize and reorder the Minkowski functionals as (M0, M1, M2).
    MF[0] *= 1 / (2 * math.pi)
    MF[1] *= 0.5
    MF[2] *= 0.5  
    MF = [MF[2], MF[1], MF[0]]

    if return_geometry:
        return (
            MF,
            innerCircles_new,
            outerCircles_new,
            Rectangle,
            points,
        )

    return MF




def _compute_integrals(
    point: tuple[float, float] | list[float],
    i: int,
    inner_radius: float,
    outer_radius: float,
    innerCircle,
    outerCircle,
    completely_in: bool,
    window: tuple[float, float, float, float] | list[float] | None,
):
    """Compute the local contributions to the Minkowski functionals.

    Evaluate the boundary integrals over the exposed inner and outer
    circular arcs associated with a single point. If necessary,
    intersect these arcs with the rectangular observation window.

    Parameters
    ----------
    point : tuple or list of float
        Coordinates (x, y) of the center of the annulus.
    i : int
        Index of the point in the point configuration.
    inner_radius : float
        Inner radius of the annulus.
    outer_radius : float
        Outer radius of the annulus.
    innerCircle : CircleSurface
        Exposed segments of the inner circular boundary.
    outerCircle : CircleSurface
        Exposed segments of the outer circular boundary.
    completely_in : bool
        Whether the annulus can be evaluated without clipping
        its boundary to the observation window.
    window : tuple or list of float or None
        Rectangular observation window given by
        (x_min, x_max, y_min, y_max).
        Required if completely_in is False.

    Returns
    -------
    M : list of float
        Unnormalized local contributions in the order
        (Euler characteristic, perimeter, area).
        These contributions are normalized and reordered by
        the main function.
    Rect : RectangleSurface or None
        Parts of the observation-window boundary covered by
        the annulus, or None if no clipping is performed.
    innerCircle : CircleSurface
        Inner circular boundary after intersection with
        the observation window, if applicable.
    outerCircle : CircleSurface
        Outer circular boundary after intersection with
        the observation window, if applicable.
    i : int
        Index of the corresponding point.
    """

    M = [0 for _ in range(3)]

    # Restrict the exposed circular arcs to the observation window.
    if not completely_in:
        Rect, innerCircle_inside, outerCircle_inside = (
            REC.ring_rectangle_intersection(
                point, inner_radius, outer_radius, window
            )
        )
        innerCircle.intersect(innerCircle_inside)
        outerCircle.intersect(outerCircle_inside)

    else:
        Rect = None

    # Integrate over the exposed inner boundary arcs.
    # The negative orientation accounts for the inner boundary.
    if inner_radius != 0:
        for arc in innerCircle.arcs:
            vz = -1
            radius = inner_radius

            M[0] += vz * (arc.end - arc.start)
            M[1] += (arc.end - arc.start) * radius
            M[2] += vz * radius * (
                point[0] * (math.sin(arc.end) - math.sin(arc.start))
                + point[1] * (math.cos(arc.start) - math.cos(arc.end))
                + radius * (arc.end - arc.start)
            )

            # Include angular contributions at arc endpoints.
            if arc.rot_start is not None:
                M[0] -= arc.rot_start / 2
            if arc.rot_end is not None:
                M[0] -= arc.rot_end / 2

    # Integrate over the exposed outer boundary arcs.
    for arc in outerCircle.arcs:
        vz = 1
        radius = outer_radius

        M[0] += vz * (arc.end - arc.start)
        M[1] += (arc.end - arc.start) * radius
        M[2] += vz * radius * (
            point[0] * (math.sin(arc.end) - math.sin(arc.start))
            + point[1] * (math.cos(arc.start) - math.cos(arc.end))
            + radius * (arc.end - arc.start)
        )

        # Include angular contributions at arc endpoints.
        if arc.rot_start is not None:
            M[0] -= arc.rot_start / 2
        if arc.rot_end is not None:
            M[0] -= arc.rot_end / 2

    return M, Rect, innerCircle, outerCircle, i
