"""Represent circular boundary arcs and compute exposed annulus boundaries."""

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from . import geometric_tools as GT


@dataclass
class Arc:
    """A circular arc with optional angular contributions at its endpoints.

    Parameters
    ----------
    start, end : float
        Start and end angles of the arc in radians.
    rot_start, rot_end : float or None, optional
        Angular contributions associated with the arc endpoints.
    """

    start: float
    end: float
    rot_start: float | None = None
    rot_end: float | None = None


class CircleSurface:
    """Represent a circular boundary as a collection of nonoverlapping arcs.

    Parameters
    ----------
    arcs : bool or iterable of Arc, optional
        If True, initialize the complete circle. If False, initialize an
        empty boundary. Otherwise, add the supplied arcs in sequence.
    """

    def __init__(self, arcs: bool | list[Arc] = False) -> None:
        if arcs is True:
            self.arcs = [Arc(0.0, 2 * np.pi)]
        elif arcs is False:
            self.arcs = []
        else:
            self.arcs = []
            if arcs:
                for s in arcs:
                    self.add_arc(s.start, s.end, s.rot_start, s.rot_end)

    def __str__(self) -> str:
        """Format arc endpoints and angular contributions in units of 2π."""

        def _fmt_rot(v: float | None) -> str:
            return "None" if v is None else f"{(v / (2 * np.pi)):.4f}"

        parts = []
        for s in self.arcs:
            parts.append(
                f"[{s.start / (2 * np.pi):.4f}, {s.end / (2 * np.pi):.4f}] "
                f"({_fmt_rot(s.rot_start)}, {_fmt_rot(s.rot_end)})"
            )
        return "RingSurface: " + "; ".join(parts)

    def add_arc(
        self,
        t1: float,
        t2: float,
        rot1: float | None = None,
        rot2: float | None = None,
    ) -> None:
        """Add an arc, handling angle wrapping and merging overlapping arcs.

        Parameters
        ----------
        t1, t2 : float
            Start and end angles in radians.
        rot1, rot2 : float or None, optional
            Angular contributions at the respective arc endpoints.

        Notes
        -----
        Existing endpoint contributions are retained when arcs are merged.
        This method modifies the instance in place.
        """
        t1 = GT.norm_angle(t1)
        if not GT.ang_eq(t2, 2 * np.pi):
            t2 = GT.norm_angle(t2)

        if rot1 is not None:
            rot1 = GT.norm_angle(rot1)
        if rot2 is not None:
            rot2 = GT.norm_angle(rot2)

        if t2 <= t1:
            if t1 == t2:
                return
            # Split an arc crossing the zero-angle direction.
            self.add_arc(t1, 2 * np.pi, rot1, None)
            self.add_arc(0.0, t2, None, rot2)
            return

        inserted = False
        start = t1
        end = t2
        rot_start = rot1
        rot_end = rot2
        for i in range(len(self.arcs) - 1, -1, -1):
            if start > self.arcs[i].end:
                self.arcs.insert(i + 1, Arc(start, end, rot_start, rot_end))
                inserted = True
                break

            elif start < self.arcs[i].start:
                if end >= self.arcs[i].start:
                    if end < self.arcs[i].end:
                        end = self.arcs[i].end
                        rot_end = self.arcs[i].rot_end
                    self.arcs.pop(i)

            elif end > self.arcs[i].end:
                start = self.arcs[i].start
                rot_start = self.arcs[i].rot_start
                self.arcs.pop(i)

            else:
                inserted = True
                break

        if not inserted:
            self.arcs.insert(0, Arc(start, end, rot_start, rot_end))

    def intersect(self, other: "CircleSurface | None") -> None:
        """Intersect this arc collection with another, modifying it in place.

        Parameters
        ----------
        other : CircleSurface or None
            Boundary to intersect with. If None, leave this instance
            unchanged (used for disjoint pairs of annuli).
        """
        if other is None:
            return

        self_angles = []
        for arc in self.arcs:
            self_angles.append((arc.start, arc.rot_start))
            self_angles.append((arc.end, arc.rot_end))
        other_angles = []
        for arc in other.arcs:
            other_angles.append((arc.start, arc.rot_start))
            other_angles.append((arc.end, arc.rot_end))

        merge = self_angles + other_angles
        truth = [True] * len(self_angles) + [False] * len(other_angles)
        sorted_indices = sorted(range(len(merge)), key=lambda x: merge[x][0])
        merge = [merge[i] for i in sorted_indices]
        truth = [truth[i] for i in sorted_indices]
        IN = [False, False]
        new_arcs = []

        for i in range(len(merge)):
            if IN[0] and IN[1]:
                new_arcs.append(merge[i])
            if truth[i]:
                IN[0] = not IN[0]
            else:
                IN[1] = not IN[1]
            if IN[0] and IN[1]:
                new_arcs.append(merge[i])

        self.arcs = []
        for i in range(0, len(new_arcs), 2):
            start, rot_start = new_arcs[i]
            end, rot_end = new_arcs[i + 1]
            self.arcs.append(Arc(start, end, rot_start, rot_end))


def calculate_surface_two_rings(
    center1: list[float] | tuple[float, float],
    center2: list[float] | tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    indices: tuple[int, int],
) -> tuple[list[CircleSurface | None], list[CircleSurface | None], tuple[int, int]]:
    """Calculate exposed inner and outer boundary arcs for two annuli.

    Parameters
    ----------
    center1, center2 : list or tuple of float
        Center coordinates of the two annuli.
    inner_radius, outer_radius : float
        Common inner and outer radii of the annuli.
    indices : tuple of int
        Indices identifying the two centers in the point configuration.

    Returns
    -------
    innerCircles : list of CircleSurface or None
        Inner boundary arcs associated with the two annuli. Both entries
        are None if the outer disks are disjoint.
    outerCircles : list of CircleSurface or None
        Outer boundary arcs associated with the two annuli. Both entries
        are None if the outer disks are disjoint.
    indices : tuple of int
        Unchanged indices of the two centers.
    """
    # Calculate the distance between the two centers.
    distance = np.linalg.norm(np.array(center1) - np.array(center2))

    # Rings are disjoint.
    if distance >= 2 * outer_radius:
        innerCircles = [None, None]
        outerCircles = [None, None]
        return innerCircles, outerCircles, indices

    else:
        innerCircles = [CircleSurface(False), CircleSurface(False)]
        outerCircles = [CircleSurface(False), CircleSurface(False)]

        # Surface 1: outer circular boundaries.
        intersection1, intersection2, theta = GT.calculate_intersection_points_withAngle(
            center1, outer_radius, center2, outer_radius, distance
        )

        angle1_center1 = np.arctan2(intersection1[1] - center1[1], intersection1[0] - center1[0])
        angle2_center1 = np.arctan2(intersection2[1] - center1[1], intersection2[0] - center1[0])
        angle1_center2 = np.arctan2(intersection1[1] - center2[1], intersection1[0] - center2[0])
        angle2_center2 = np.arctan2(intersection2[1] - center2[1], intersection2[0] - center2[0])

        angle1_center1, angle2_center1, switch = GT.order_angles_for_longer_arc(angle1_center1, angle2_center1)
        angle1_center2, angle2_center2, switch = GT.order_angles_for_longer_arc(angle1_center2, angle2_center2)

        outerCircles[0].add_arc(angle1_center1, angle2_center1, theta, theta)
        outerCircles[1].add_arc(angle1_center2, angle2_center2, theta, theta)

        # Surface 2: both inner boundaries remain completely visible.
        if distance >= inner_radius + outer_radius:
            innerCircles[0].add_arc(0.0, 2 * np.pi)
            innerCircles[1].add_arc(0.0, 2 * np.pi)
            return innerCircles, outerCircles, indices

        else:
            # Surface 3: intersections between inner and outer circles.
            if distance >= outer_radius - inner_radius:
                intersection1, intersection2, theta = GT.calculate_intersection_points_withAngle(
                    center1, inner_radius, center2, outer_radius, distance
                )
                theta = np.pi - theta

                angle1_center2 = np.arctan2(intersection1[1] - center2[1], intersection1[0] - center2[0])
                angle2_center2 = np.arctan2(intersection2[1] - center2[1], intersection2[0] - center2[0])
                angle1_center2, angle2_center2, switch = GT.order_angles_for_longer_arc(angle1_center2, angle2_center2)

                outerCircles[1].add_arc(angle2_center2, angle1_center2, theta, theta)

                angle1_center2_ref = (angle1_center2 + np.pi) % (2 * np.pi)
                angle2_center2_ref = (angle2_center2 + np.pi) % (2 * np.pi)

                outerCircles[0].add_arc(angle2_center2_ref, angle1_center2_ref, theta, theta)

                angle1_center1 = np.arctan2(intersection1[1] - center1[1], intersection1[0] - center1[0]) % (2 * np.pi)
                angle2_center1 = np.arctan2(intersection2[1] - center1[1], intersection2[0] - center1[0]) % (2 * np.pi)
                angle1_center1_ref = (angle1_center1 + np.pi) % (2 * np.pi)
                angle2_center1_ref = (angle2_center1 + np.pi) % (2 * np.pi)

                if switch == False:
                    innerCircles[0].add_arc(angle2_center1, angle1_center1, theta, theta)
                    innerCircles[1].add_arc(angle2_center1_ref, angle1_center1_ref, theta, theta)
                else:
                    innerCircles[0].add_arc(angle1_center1, angle2_center1, theta, theta)
                    innerCircles[1].add_arc(angle1_center1_ref, angle2_center1_ref, theta, theta)

            # Surface 4: intersection between the two inner circles.
            if distance < 2 * inner_radius:
                intersection1, intersection2, theta = GT.calculate_intersection_points_withAngle(
                    center1, inner_radius, center2, inner_radius, distance
                )

                angle1_center1 = np.arctan2(intersection1[1] - center1[1], intersection1[0] - center1[0])
                angle2_center1 = np.arctan2(intersection2[1] - center1[1], intersection2[0] - center1[0])
                angle1_center1, angle2_center1, switch = GT.order_angles_for_longer_arc(angle1_center1, angle2_center1)

                innerCircles[0].add_arc(angle2_center1, angle1_center1, theta, theta)

                angle1_center2 = np.arctan2(intersection1[1] - center2[1], intersection1[0] - center2[0])
                angle2_center2 = np.arctan2(intersection2[1] - center2[1], intersection2[0] - center2[0])
                angle1_center2, angle2_center2, switch = GT.order_angles_for_longer_arc(angle1_center2, angle2_center2)

                innerCircles[1].add_arc(angle2_center2, angle1_center2, theta, theta)

            return innerCircles, outerCircles, indices


def calculate_ring_surface(
    points: list[list[float]],
    inner_radius: float,
    outer_radius: float,
    jobs: int = -1,
) -> tuple[list[CircleSurface], list[CircleSurface]]:
    """Compute exposed circular boundaries for all annuli in a point set.

    Parameters
    ----------
    points : list of list of float
        Coordinates of the annulus centers, each with two components.
    inner_radius, outer_radius : float
        Common inner and outer radii of the annuli.
    jobs : int, optional
        Number of parallel workers used by joblib. The default, -1,
        requests all available workers.

    Returns
    -------
    innerCircles : list of CircleSurface
        Exposed inner circular boundary arcs for each input point.
    outerCircles : list of CircleSurface
        Exposed outer circular boundary arcs for each input point.

    Notes
    -----
    Pairwise boundary constraints are evaluated in parallel and then
    intersected in the original pair order.
    """
    pair_indices = [
        (i, j) for i in range(len(points)) for j in range(i + 1, len(points))
    ]
    SurfacePairs = Parallel(n_jobs=jobs)(
        delayed(calculate_surface_two_rings)(
            points[i], points[j], inner_radius, outer_radius, (i, j)
        )
        for i, j in pair_indices
    )

    innerCircles = [CircleSurface(True) for i in range(len(points))]
    outerCircles = [CircleSurface(True) for i in range(len(points))]

    for inner, outer, indices in SurfacePairs:
        innerCircles[indices[0]].intersect(inner[0])
        innerCircles[indices[1]].intersect(inner[1])
        outerCircles[indices[0]].intersect(outer[0])
        outerCircles[indices[1]].intersect(outer[1])

    return innerCircles, outerCircles
