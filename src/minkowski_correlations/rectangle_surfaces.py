"""Represent rectangular boundary segments and clip annuli to a window."""

import math
from dataclasses import dataclass

import numpy as np

from . import geometric_tools as GT
from . import ring_surfaces as RS


@dataclass
class Segment:
    """Represent a segment on one side of an axis-aligned rectangle.

    Parameters
    ----------
    start, end : float
        Coordinates along the corresponding rectangle side. For the
        left and right sides these are y-coordinates; for the down and
        up sides these are x-coordinates.
    rot_start, rot_end : float or None, optional
        Angular contributions at the segment endpoints, in radians.
    """

    start: float
    end: float
    rot_start: float | None = None
    rot_end: float | None = None


class RectangleSurface:
    """Store boundary segments on the four sides of a rectangle.

    Segments are stored separately for the ``left``, ``right``, ``down``,
    and ``up`` sides. Iteration yields pairs of side names and segment
    lists in this fixed order. The ``add_seg`` method merges overlapping
    segments on the same side.

    Attributes
    ----------
    SIDES : tuple of str
        Side names in their fixed iteration order.
    left, right, down, up : list of Segment
        Segments associated with each rectangle side.
    """

    SIDES = ("left", "right", "down", "up")

    def __init__(self) -> None:
        """Initialize an empty collection of boundary segments."""
        self.left: list[Segment] = []
        self.right: list[Segment] = []
        self.down: list[Segment] = []
        self.up: list[Segment] = []

    def __iter__(self):
        """Yield ``(side_name, segments)`` for each rectangle side."""
        for name in self.SIDES:
            yield name, getattr(self, name)

    def __str__(self) -> str:
        """Format segments and endpoint rotations for inspection.

        Segment coordinates are displayed directly, whereas endpoint
        rotations are expressed in units of 2π.
        """

        def _fmt_rot(v: float | None) -> str:
            return "None" if v is None else f"{(v / (2 * np.pi)):.4f}"

        lines = ["RectangleSurface:"]
        for side in ["left", "right", "down", "up"]:
            segs = getattr(self, side)
            if not segs:
                continue
            parts = []
            for s in segs:
                parts.append(
                    f"[{s.start:.4f}, {s.end:.4f}] "
                    f"({_fmt_rot(s.rot_start)}, {_fmt_rot(s.rot_end)})"
                )
            lines.append(f" {side}: " + "; ".join(parts))
        return "\n".join(lines)

    def add_seg(
        self,
        where: str,
        a: float,
        b: float,
        rot1: float | None = None,
        rot2: float | None = None,
    ) -> None:
        """Add a segment to a rectangle side, merging overlapping segments.

        Parameters
        ----------
        where : {"left", "right", "down", "up"}
            Side to which the segment belongs.
        a, b : float
            Start and end coordinates of the segment along that side.
        rot1, rot2 : float or None, optional
            Angular contributions at the start and end of the segment.

        Notes
        -----
        The segment collection is modified in place. Endpoint angular
        contributions are retained according to the existing merging
        rules.
        """
        inserted = False
        start = a
        end = b
        rot_start = rot1
        rot_end = rot2

        for i in range(len(getattr(self, where)) - 1, -1, -1):
            if start > getattr(self, where)[i].end:
                getattr(self, where).insert(
                    i + 1, Segment(start, end, rot_start, rot_end)
                )
                inserted = True
                break

            elif start < getattr(self, where)[i].start:
                if end >= getattr(self, where)[i].start:
                    if end < getattr(self, where)[i].end:
                        end = getattr(self, where)[i].end
                        rot_end = getattr(self, where)[i].rot_end
                    getattr(self, where).pop(i)

            elif end > getattr(self, where)[i].end:
                start = getattr(self, where)[i].start
                rot_start = getattr(self, where)[i].rot_start
                getattr(self, where).pop(i)

            else:
                inserted = True
                break

        if not inserted:
            getattr(self, where).insert(
                0, Segment(start, end, rot_start, rot_end)
            )

    def union(self, other: "RectangleSurface | None") -> None:
        """Merge the segments of another rectangle boundary in place.

        Parameters
        ----------
        other : RectangleSurface or None
            Boundary segments to add. If None, leave this instance
            unchanged.
        """
        if other is None:
            return

        for side, segments in other:
            for s in segments:
                self.add_seg(side, s.start, s.end, s.rot_start, s.rot_end)


def ring_rectangle_intersection(
    x: tuple[float, float] | list[float],
    inner_radius: float,
    outer_radius: float,
    window: tuple[float, float, float, float] | list[float],
) -> tuple[RectangleSurface, RS.CircleSurface, RS.CircleSurface]:
    """Determine the intersection of an annulus with a rectangular window.

    Calculate the covered portions of the window boundary and the inner
    and outer circular arcs lying inside the window. The resulting
    circular boundaries can be intersected with the exposed arcs of an
    annulus in the main calculation.

    Parameters
    ----------
    x : tuple or list of float
        Coordinates (x, y) of the annulus center.
    inner_radius : float
        Inner radius of the annulus.
    outer_radius : float
        Outer radius of the annulus.
    window : tuple or list of float
        Axis-aligned observation window specified as
        (x_min, x_max, y_min, y_max).

    Returns
    -------
    Rect : RectangleSurface
        Portions of the four window sides covered by the annulus,
        including endpoint angular contributions.
    innerCircle : CircleSurface
        Inner circular arcs inside the observation window.
    outerCircle : CircleSurface
        Outer circular arcs inside the observation window.

    Notes
    -----
    This function preserves the original intersection and endpoint
    handling, including comparisons at the window boundaries.
    """
    # Each edge is described by its fixed coordinate and interval bounds.
    # The order is left, right, down, up.
    edges = [
        [window[0], window[2], window[3]],
        [window[1], window[2], window[3]],
        [window[2], window[0], window[1]],
        [window[3], window[0], window[1]],
    ]

    Rect = RectangleSurface()
    IN = window[0] <= x[0] <= window[1] and window[2] <= x[1] <= window[3]

    inner_angles = []
    outer_angles = []

    # Collect circle intersections and covered segments on each side.
    for j in range(4):
        where = Rect.SIDES[j]
        if j <= 1:
            vertical = True
            coord_IN = window[0] <= x[0] <= window[1]
        else:
            vertical = False
            coord_IN = window[2] <= x[1] <= window[3]

        rectangle_point_inner = GT.circle_axis_intersection(
            x, inner_radius, edges[j][0], vertical
        )
        rectangle_point_outer = GT.circle_axis_intersection(
            x, outer_radius, edges[j][0], vertical
        )
        intersection_points = rectangle_point_inner + rectangle_point_outer

        if intersection_points:
            which_circle = [0 for i in range(len(rectangle_point_inner))] + [
                1 for i in range(len(rectangle_point_outer))
            ]
            intersection_points = [
                (x, theta, angle, cid)
                for (x, theta, angle), cid in zip(
                    intersection_points, which_circle
                )
            ]
            intersection_points.sort(key=lambda t: t[0])

            smaller = 0
            relevant_points = []

            for i in range(len(intersection_points)):
                if intersection_points[i][0] <= edges[j][1]:
                    smaller += 1
                elif intersection_points[i][0] < edges[j][2]:
                    relevant_points.append(intersection_points[i])
                    if intersection_points[i][3] == 0:
                        inner_angles.append(intersection_points[i][2])
                    else:
                        outer_angles.append(intersection_points[i][2])

            if smaller % 2 == 1:
                relevant_points.insert(0, (edges[j][1], math.pi / 4, None, None))
            if len(relevant_points) % 2 == 1:
                relevant_points.append((edges[j][2], math.pi / 4, None, None))

            segments = []
            for i in range(0, len(relevant_points), 2):
                start, rot_start, angle1, cid1 = relevant_points[i]
                end, rot_end, angle2, cid2 = relevant_points[i + 1]
                if not coord_IN:
                    # Select the complementary endpoint angle when needed.
                    if cid1 is not None:
                        rot_start = math.pi - rot_start
                    if cid2 is not None:
                        rot_end = math.pi - rot_end
                if cid1 == 0:
                    rot_start = math.pi - rot_start
                if cid2 == 0:
                    rot_end = math.pi - rot_end
                segments.append(Segment(start, end, rot_start, rot_end))

            setattr(Rect, where, segments)

    inner_angles.sort()
    outer_angles.sort()

    inner_start = (
        window[0] <= x[0] + inner_radius <= window[1]
        and window[2] <= x[1] <= window[3]
    )
    outer_start = (
        window[0] <= x[0] + outer_radius <= window[1]
        and window[2] <= x[1] <= window[3]
    )

    # Complete the angular intervals across the 0/2π boundary.
    if inner_angles and inner_angles[0] == 0:
        t = 0.5 * inner_angles[1]
        start_point = (
            window[0] <= x[0] + inner_radius * math.cos(t) <= window[1]
            and window[2] <= x[1] + inner_radius * math.sin(t) <= window[3]
        )
        if not start_point:
            inner_angles.pop(0)
            inner_angles.append(2 * math.pi)
    elif inner_angles:
        if inner_start:
            inner_angles.insert(0, 0.0)
            inner_angles.append(2 * math.pi)

    if outer_angles and outer_angles[0] == 0:
        t = 0.5 * outer_angles[1]
        start_point = (
            window[0] <= x[0] + outer_radius * math.cos(t) <= window[1]
            and window[2] <= x[1] + outer_radius * math.sin(t) <= window[3]
        )
        if not start_point:
            outer_angles.pop(0)
            outer_angles.append(2 * math.pi)
    elif outer_angles:
        if outer_start:
            outer_angles.insert(0, 0.0)
            outer_angles.append(2 * math.pi)

    # Construct the circular boundary pieces inside the window.
    if inner_angles:
        arcs_inner = []
        for i in range(0, len(inner_angles), 2):
            start = inner_angles[i]
            end = inner_angles[i + 1]
            arcs_inner.append(RS.Arc(start, end))
        innerCircle = RS.CircleSurface(False)
        innerCircle.arcs = arcs_inner
    else:
        if IN:
            innerCircle = RS.CircleSurface(True)
        else:
            innerCircle = RS.CircleSurface(False)

    if outer_angles:
        arcs_outer = []
        for i in range(0, len(outer_angles), 2):
            start = outer_angles[i]
            end = outer_angles[i + 1]
            arcs_outer.append(RS.Arc(start, end))
        outerCircle = RS.CircleSurface(False)
        outerCircle.arcs = arcs_outer
    else:
        if IN:
            outerCircle = RS.CircleSurface(True)
        else:
            outerCircle = RS.CircleSurface(False)

    return Rect, innerCircle, outerCircle
