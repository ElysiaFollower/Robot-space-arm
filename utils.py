import math
import numpy as np


def angle_interpolate_2(start, end, t_start, t_end, t):
    """
    Interpolate angle between start and end angles at time t using quintic polynomial
    Args:
        start: start angle
        end: end angle
        t_start: start time
        t_end: end time
        t: current time

    Returns:
        interpolated angle
    """
    # Handle boundary conditions
    if t <= t_start:
        return start
    if t >= t_end:
        return end

    # Calculate time duration
    T = t_end - t_start
    if T <= 0:
        return start

    # Normalize time to [0, T]
    tau = t - t_start

    # Find shortest path for angle interpolation
    angle_diff = end - start
    if angle_diff > math.pi:
        end = end - 2 * math.pi
    elif angle_diff < -math.pi:
        end = end + 2 * math.pi

    # Calculate quintic polynomial coefficients
    # Using zero velocity and acceleration at boundaries for smooth motion
    a0 = start
    a1 = 0  # v_start = 0
    a2 = 0  # a_start = 0

    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T

    a3 = (20 * (end - start) - 8 * 0 * T - 12 * 0 * T) / (2 * T3)  # v_end = 0, a_end = 0
    a4 = (30 * (start - end) + 14 * 0 * T + 16 * 0 * T) / (2 * T4)
    a5 = (12 * (end - start) - 6 * 0 * T - 6 * 0 * T) / (2 * T5)

    # Evaluate quintic polynomial
    angle = (
        a0
        + a1 * tau
        + a2 * tau * tau
        + a3 * tau * tau * tau
        + a4 * tau * tau * tau * tau
        + a5 * tau * tau * tau * tau * tau
    )

    # Normalize angle to [-pi, pi]
    angle = math.remainder(angle, 2 * math.pi)

    return angle


def angle_interpolate_3(start, mid, end, t_start, t_mid, t_end, t):
    """
    Interpolate angle through start, mid, and end points at time t using quintic polynomials
    Args:
        start: start angle
        mid: middle angle
        end: end angle
        t_start: start time
        t_mid: middle time
        t_end: end time
        t: current time

    Returns:
        interpolated angle
    """
    # Handle boundary conditions
    if t <= t_start:
        return start
    if t >= t_end:
        return end

    # Calculate time durations
    T1 = t_mid - t_start
    T2 = t_end - t_mid

    if T1 <= 0 or T2 <= 0:
        return start

    # Process angles for continuity
    # Find shortest path for angle interpolation
    mid_processed = mid
    end_processed = end

    # Process mid angle
    diff_mid = mid - start
    if diff_mid > math.pi:
        mid_processed = mid - 2 * math.pi
    elif diff_mid < -math.pi:
        mid_processed = mid + 2 * math.pi

    # Process end angle relative to mid
    diff_end = end - mid_processed
    if diff_end > math.pi:
        end_processed = end - 2 * math.pi
    elif diff_end < -math.pi:
        end_processed = end + 2 * math.pi

    # First segment: start to mid
    if t <= t_mid:
        tau = t - t_start

        # Calculate quintic polynomial coefficients for first segment
        a0 = start
        a1 = 0  # v_start = 0
        a2 = 0  # a_start = 0

        T1_2 = T1 * T1
        T1_3 = T1_2 * T1
        T1_4 = T1_3 * T1
        T1_5 = T1_4 * T1

        # Estimate velocity at mid point
        v_mid = 0.5 * ((mid_processed - start) / T1 + (end_processed - mid_processed) / T2)

        a3 = (20 * (mid_processed - start) - (8 * v_mid + 12 * 0) * T1) / (2 * T1_3)
        a4 = (30 * (start - mid_processed) + (14 * v_mid + 16 * 0) * T1) / (2 * T1_4)
        a5 = (12 * (mid_processed - start) - (6 * v_mid + 6 * 0) * T1) / (2 * T1_5)

        # Evaluate quintic polynomial
        angle = (
            a0
            + a1 * tau
            + a2 * tau * tau
            + a3 * tau * tau * tau
            + a4 * tau * tau * tau * tau
            + a5 * tau * tau * tau * tau * tau
        )

    # Second segment: mid to end
    else:
        tau = t - t_mid

        # Calculate quintic polynomial coefficients for second segment
        a0 = mid_processed
        # Estimate velocity at mid point
        v_mid = 0.5 * ((mid_processed - start) / T1 + (end_processed - mid_processed) / T2)
        a1 = v_mid
        # Estimate acceleration at mid point
        a_mid = ((end_processed - mid_processed) / T2 - (mid_processed - start) / T1) / (T1 + T2) * 2
        a2 = a_mid / 2

        T2_2 = T2 * T2
        T2_3 = T2_2 * T2
        T2_4 = T2_3 * T2
        T2_5 = T2_4 * T2

        a3 = (20 * (end_processed - mid_processed) - (8 * 0 + 12 * v_mid) * T2 - (3 * a_mid - 0) * T2_2) / (2 * T2_3)
        a4 = (30 * (mid_processed - end_processed) + (14 * 0 + 16 * v_mid) * T2 + (3 * a_mid - 2 * 0) * T2_2) / (
            2 * T2_4
        )
        a5 = (12 * (end_processed - mid_processed) - (6 * 0 + 6 * v_mid) * T2 - (0 - a_mid) * T2_2) / (2 * T2_5)

        # Evaluate quintic polynomial
        angle = (
            a0
            + a1 * tau
            + a2 * tau * tau
            + a3 * tau * tau * tau
            + a4 * tau * tau * tau * tau
            + a5 * tau * tau * tau * tau * tau
        )

    # Normalize angle to [-pi, pi]
    angle = math.remainder(angle, 2 * math.pi)

    return angle

def switch_base(sim, joints, links, base="L"):
    for joint in joints:
        sim.setObjectParent(joint, -1)
    for link in links:
        sim.setObjectParent(link, -1)

    if base == "L":
        for i in range(7):
            sim.setObjectParent(joints[i], links[i])
            sim.setObjectParent(links[i+1], joints[i])

    elif base == "R":
        for i in range(7):
            sim.setObjectParent(links[i], joints[i])
            sim.setObjectParent(joints[i], links[i+1])

    else:
        ArgumentError("base must be 'L' or 'R'")