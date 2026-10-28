"""Geometric utilities for circular arcs and circle intersections."""

import math

import numpy as np


def norm_angle(t: float) -> float:
    """Normalize an angle to the interval [0, 2π).

    Parameters
    ----------
    t : float
        Angle in radians.

    Returns
    -------
    float
        Angle modulo 2π.
    """
    return t % (2 * np.pi)


def ang_eq(a: float, b: float) -> bool:
    """Test whether two normalized angles differ by less than 1e-9.

    Parameters
    ----------
    a, b : float
        Angles in radians.

    Returns
    -------
    bool
        True if the absolute difference of the normalized angles
        is strictly less than 1e-9.

    Notes
    -----
    This compares normalized values directly; angles immediately on
    opposite sides of the 0/2π boundary are not treated as close.
    """
    return abs(norm_angle(a) - norm_angle(b)) < 1e-9


def order_angles_for_longer_arc(
    angle1: float, angle2: float
) -> tuple[float, float, bool]:
    """Order two angles to parameterize the longer connecting arc.

    Parameters
    ----------
    angle1, angle2 : float
        Input angles in radians, normalized internally to [0, 2π).

    Returns
    -------
    angle1 : float
        First angle of the ordered pair.
    angle2 : float
        Second angle of the ordered pair.
    switch : bool
        Whether the final ordering swaps the two input angles.

    Notes
    -----
    The arc is interpreted in the positive angular direction from the
    first returned angle to the second, allowing wraparound at 2π.
    """
    # Normalize angles to the range [0, 2π).
    angle1 = angle1 % (2 * np.pi)
    angle2 = angle2 % (2 * np.pi)

    # Start with the angles in ascending order.
    if angle1 > angle2:
        angle1, angle2 = angle2, angle1
        switch = True
    else:
        switch = False

    # Compare the two possible arc lengths.
    arc_length1 = (angle2 - angle1) % (2 * np.pi)
    arc_length2 = (2 * np.pi) - arc_length1

    # Reverse the order if the complementary arc is longer.
    if arc_length1 < arc_length2:
        angle1, angle2 = angle2, angle1
        switch = not switch

    return angle1, angle2, switch


def calculate_intersection_points_withAngle(
    center1: tuple[float, float] | list[float] | np.ndarray,
    radius1: float,
    center2: tuple[float, float] | list[float] | np.ndarray,
    radius2: float,
    distance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate the intersection points and intersection angle of two circles.

    Parameters
    ----------
    center1, center2 : array-like of float, shape (2,)
        Coordinates of the circle centers.
    radius1, radius2 : float
        Radii of the circles.
    distance : float
        Distance between the circle centers, supplied by the caller.

    Returns
    -------
    intersection1 : numpy.ndarray
        Coordinates of the first intersection point.
    intersection2 : numpy.ndarray
        Coordinates of the second intersection point.
    theta : float
        Angle in radians between the vectors from the first
        intersection point to the two circle centers.

    Notes
    -----
    The caller must supply a configuration with two distinct circle
    centers and valid circle intersections. This function does not
    independently check those geometric preconditions.
    """
    # Vector between the centers.
    vec_center1_to_center2 = np.array(center2) - np.array(center1)

    # Distance from center1 to the chord midpoint.
    a = (radius1**2 - radius2**2 + distance**2) / (2 * distance)

    # Perpendicular distance from the chord midpoint to either intersection.
    h = np.sqrt(radius1**2 - a**2)

    # Midpoint of the chord connecting the intersection points.
    midpoint = np.array(center1) + a * vec_center1_to_center2 / distance

    # Calculate both intersection points.
    intersection1 = midpoint + h * np.array(
        [-vec_center1_to_center2[1], vec_center1_to_center2[0]]
    ) / distance
    intersection2 = midpoint - h * np.array(
        [-vec_center1_to_center2[1], vec_center1_to_center2[0]]
    ) / distance

    # Angle between the center-directed vectors at the first intersection.
    v1 = np.array(center1) - intersection1
    v2 = np.array(center2) - intersection1

    skalarprodukt = v1[0] * v2[0] + v1[1] * v2[1]
    norm_v1 = math.hypot(*v1)
    norm_v2 = math.hypot(*v2)

    cos_theta = skalarprodukt / (norm_v1 * norm_v2)
    theta = math.acos(cos_theta)

    return intersection1, intersection2, theta


def circle_axis_intersection(
    z: tuple[float, float] | list[float] | np.ndarray,
    radius: float,
    value: float,
    vertical: bool = True,
) -> list[tuple[float, float, float]]:
    """Find intersections of a circle with an axis-aligned line.

    Parameters
    ----------
    z : array-like of float, shape (2,)
        Coordinates of the circle center.
    radius : float
        Circle radius.
    value : float
        Fixed x-coordinate of a vertical line or y-coordinate of a
        horizontal line.
    vertical : bool, optional
        If True, intersect with the vertical line x = value.
        Otherwise, intersect with the horizontal line y = value.
        Default is True.

    Returns
    -------
    intersections : list of tuple of float
        Each tuple contains (w, theta, angle), where w is the free
        coordinate of an intersection point, theta is the smaller
        angle between the tangent and the axis-aligned line, and
        angle is the polar angle of the intersection point relative
        to the circle center.

        Returns an empty list if the line does not cross the circle
        at two distinct points (including tangency).
    """
    if vertical:
        dx = value - z[0]
        if abs(dx) >= radius:
            return []
        else:
            dy = math.sqrt(radius**2 - dx**2)
            angle1 = math.atan2(dy, dx) % (2 * math.pi)
            angle2 = math.atan2(-dy, dx) % (2 * math.pi)

            theta = math.acos(-dx / radius)
            theta = min(theta, math.pi - theta)

            return [(z[1] + dy, theta, angle1), (z[1] - dy, theta, angle2)]

    else:
        dy = value - z[1]
        if abs(dy) >= radius:
            return []
        else:
            dx = math.sqrt(radius**2 - dy**2)
            angle1 = math.atan2(dy, dx) % (2 * math.pi)
            angle2 = math.atan2(dy, -dx) % (2 * math.pi)

            theta = math.acos(dy / radius)
            theta = min(theta, math.pi - theta)

            return [(z[0] + dx, theta, angle1), (z[0] - dx, theta, angle2)]
