from math import cos, sin, pi, acos, atan2, sqrt
import numpy as np

# Robot geometric parameters (in mm)
LINK_LENGTH_1 = 185.0  # Link length l1
LINK_LENGTH_2 = 170.0  # Link length l2
OFFSET_WRIST = 23.0    # Wrist offset
HEIGHT_BASE = 230.0    # Base height d1
LENGTH_JOINT_5 = 77.0  # Joint 5 length d5
LENGTH_JOINT_6 = 85.5  # Joint 6 length d6
END_EFFECTOR_OFFSET = 85.5  # End-effector offset

def dh_transformation_matrix(alpha, a, d, theta):
    """
    Compute the Denavit-Hartenberg (DH) transformation matrix.
    :param alpha: Joint twist angle (radians)
    :param a: Link length (mm)
    :param d: Joint offset (mm)
    :param theta: Joint rotation angle (radians)
    :return: 4x4 homogeneous transformation matrix
    """
    # Construct the DH matrix using trigonometric functions
    return np.array([
        [cos(theta), -sin(theta), 0.0, a],
        [sin(theta) * cos(alpha), cos(theta) * cos(alpha), -sin(alpha), -sin(alpha) * d],
        [sin(theta) * sin(alpha), cos(theta) * sin(alpha), cos(alpha), cos(alpha) * d],
        [0.0, 0.0, 0.0, 1.0]
    ])

def normalize_angle(angle):
    """
    Normalize angle to the range [-π, π] using a loop to avoid recursion stack overflow.
    :param angle: Input angle (radians)
    :return: Normalized angle
    """
    # Adjust angle by subtracting or adding 2π until within range
    while angle > pi:
        angle -= 2 * pi
    while angle < -pi:
        angle += 2 * pi
    return angle

def normalize_angle_matrix(matrix):
    """
    Normalize all elements in a matrix to [-π, π].
    :param matrix: NumPy array (e.g., 4x6 or 6x4)
    :return: Normalized matrix
    """
    # Loop through each element and apply normalization
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = normalize_angle(matrix[i, j])
    return matrix

def my_solver(pose_array):
    """
    Inverse kinematics solver: Compute up to 4 sets of 6 joint angles for given end-effector pose array.
    Input is a NumPy array [x, y, z, alpha, beta, gamma].
    Output is a 6x4 matrix where each column is one solution (transposed from original 4x6).
    :param pose_array: NumPy array of shape (6,) for [x, y, z, alpha, beta, gamma] (x,y,z in meters; angles in radians)
    :return: 6x4 NumPy matrix with joint angles (radians; each column one solution)
    """
    def _solve_planar_joints_internal(transform_matrix):
        """
        Internal function: Solve for planar joint angles (theta2, theta3, theta4) treated as a 2D planar mechanism.
        :param transform_matrix: 4x4 transformation matrix
        :return: 2x3 NumPy array with solutions (theta1, theta2, theta3 per row); invalid if no solution
        """
        # Extract cosine and sine of phi from matrix
        cos_phi = transform_matrix[0, 0]
        sin_phi = -transform_matrix[0, 1]
        # Extract x and y positions
        x_pos = transform_matrix[0, 3]
        y_pos = -transform_matrix[2, 3]
        
        # Compute cos(theta2) using law of cosines
        cos_theta2 = (x_pos**2 + y_pos**2 - LINK_LENGTH_1**2 - LINK_LENGTH_2**2) / (2 * LINK_LENGTH_1 * LINK_LENGTH_2)
        # Check if cos(theta2) is within valid range [-1, 1]
        if abs(cos_theta2) > 1:
            return np.array([[100.0, 0, 0], [100.0, 0, 0]])  # Invalid solution flag
        
        # Compute positive and negative sin(theta2)
        sin_theta2_pos = sqrt(1 - cos_theta2**2)
        sin_theta2_neg = -sin_theta2_pos
        
        # Compute theta2 for both cases using atan2
        theta2_pos = atan2(sin_theta2_pos, cos_theta2)
        theta2_neg = atan2(sin_theta2_neg, cos_theta2)
        
        # Compute intermediate values k1, k2_pos, k2_neg
        k1 = LINK_LENGTH_1 + LINK_LENGTH_2 * cos_theta2
        k2_pos = LINK_LENGTH_2 * sin_theta2_pos
        k2_neg = LINK_LENGTH_2 * sin_theta2_neg
        
        # Compute theta1 for both cases
        theta1_pos = atan2(y_pos, x_pos) - atan2(k2_pos, k1)
        theta1_neg = atan2(y_pos, x_pos) - atan2(k2_neg, k1)
        
        # Compute theta3 for both cases
        theta3_pos = atan2(sin_phi, cos_phi) - theta1_pos - theta2_pos
        theta3_neg = atan2(sin_phi, cos_phi) - theta1_neg - theta2_neg
        
        # Return the two solutions
        return np.array([[theta1_pos, theta2_pos, theta3_pos], [theta1_neg, theta2_neg, theta3_neg]])

    # Unpack input array
    x, y, z, alpha, beta, gamma = pose_array
    
    # Construct rotation matrix from Euler angles
    r11 = cos(beta) * cos(gamma)
    r12 = -cos(beta) * sin(gamma)
    r13 = sin(beta)
    r21 = cos(alpha) * sin(gamma) + cos(gamma) * sin(alpha) * sin(beta)
    r22 = cos(alpha) * cos(gamma) - sin(alpha) * sin(beta) * sin(gamma)
    r23 = -cos(beta) * sin(alpha)
    r31 = sin(alpha) * sin(gamma) - cos(alpha) * cos(gamma) * sin(beta)
    r32 = cos(gamma) * sin(alpha) + cos(alpha) * sin(beta) * sin(gamma)
    r33 = cos(alpha) * cos(beta)
    
    # Convert position from meters to mm
    px = x * 1000.0
    py = y * 1000.0
    pz = z * 1000.0
    
    # Compute theta1 (two solutions: positive and negative branches)
    m_val = py - r23 * END_EFFECTOR_OFFSET
    n_val = px - r13 * END_EFFECTOR_OFFSET
    # Calculate discriminant for sqrt
    discriminant = n_val**2 + m_val**2 - OFFSET_WRIST**2
    theta1_pos = atan2(m_val, n_val) - atan2(OFFSET_WRIST, -sqrt(discriminant))
    theta1_neg = atan2(m_val, n_val) - atan2(OFFSET_WRIST, sqrt(discriminant))
    
    # Compute theta5 and its negative alternative
    theta5_pos = -acos(sin(theta1_pos) * r13 - cos(theta1_pos) * r23)
    theta5_pos_neg = -theta5_pos
    theta5_neg = -acos(sin(theta1_neg) * r13 - cos(theta1_neg) * r23)
    theta5_neg_neg = -theta5_neg
    
    # Compute theta6
    r31_temp_pos = r21 * cos(theta1_pos) - r11 * sin(theta1_pos)
    r32_temp_pos = r22 * cos(theta1_pos) - r12 * sin(theta1_pos)
    theta6_pos = atan2(-r32_temp_pos, r31_temp_pos)
    
    r31_temp_neg = r21 * cos(theta1_neg) - r11 * sin(theta1_neg)
    r32_temp_neg = r22 * cos(theta1_neg) - r12 * sin(theta1_neg)
    theta6_neg = atan2(-r32_temp_neg, r31_temp_neg)
    
    # Adjust theta6 sign if necessary
    if (r21 * cos(theta1_pos) - r11 * sin(theta1_pos)) != (cos(theta6_pos) * sin(theta5_pos)):
        theta6_pos += pi
    if (r21 * cos(theta1_neg) - r11 * sin(theta1_neg)) != (cos(theta6_neg) * sin(theta5_neg)):
        theta6_neg += pi
    
    # Build target homogeneous matrix M
    target_matrix = np.array([
        [r11, r12, r13, px],
        [r21, r22, r23, py],
        [r31, r32, r33, pz],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # Solve for theta2,3,4 using theta1_pos branch
    t01_inv_pos = np.linalg.inv(dh_transformation_matrix(0.0, 0.0, HEIGHT_BASE, theta1_pos))
    t45_inv_pos = np.linalg.inv(dh_transformation_matrix(pi / 2, 0.0, LENGTH_JOINT_5, theta5_pos))
    t56_inv_pos = np.linalg.inv(dh_transformation_matrix(pi / 2, 0.0, LENGTH_JOINT_6, theta6_pos))
    intermediate_matrix_pos = np.dot(np.dot(t01_inv_pos, target_matrix), np.dot(t56_inv_pos, t45_inv_pos))
    solutions_group1 = _solve_planar_joints_internal(intermediate_matrix_pos)
    
    # If no solution, try alternative theta5 and theta6
    if solutions_group1[0, 0] == 100:
        theta5_pos = theta5_pos_neg
        t45_inv_pos = np.linalg.inv(dh_transformation_matrix(pi / 2, 0.0, LENGTH_JOINT_5, theta5_pos))
        theta6_pos = normalize_angle(theta6_pos + pi)
        t56_inv_pos = np.linalg.inv(dh_transformation_matrix(pi / 2, 0.0, LENGTH_JOINT_6, theta6_pos))
        intermediate_matrix_pos = np.dot(np.dot(t01_inv_pos, target_matrix), np.dot(t56_inv_pos, t45_inv_pos))
        solutions_group1 = _solve_planar_joints_internal(intermediate_matrix_pos)
    
    # Solve for theta2,3,4 using theta1_neg branch
    t01_inv_neg = np.linalg.inv(dh_transformation_matrix(0.0, 0.0, HEIGHT_BASE, theta1_neg))
    t45_inv_neg = np.linalg.inv(dh_transformation_matrix(pi / 2, 0.0, LENGTH_JOINT_5, theta5_neg))
    t56_inv_neg = np.linalg.inv(dh_transformation_matrix(pi / 2, 0.0, LENGTH_JOINT_6, theta6_neg))
    intermediate_matrix_neg = np.dot(np.dot(t01_inv_neg, target_matrix), np.dot(t56_inv_neg, t45_inv_neg))
    solutions_group2 = _solve_planar_joints_internal(intermediate_matrix_neg)
    
    # If no solution, try alternative
    if solutions_group2[0, 0] == 100:
        theta5_neg = theta5_neg_neg
        t45_inv_neg = np.linalg.inv(dh_transformation_matrix(pi / 2, 0.0, LENGTH_JOINT_5, theta5_neg))
        theta6_neg = normalize_angle(theta6_neg + pi)
        t56_inv_neg = np.linalg.inv(dh_transformation_matrix(pi / 2, 0.0, LENGTH_JOINT_6, theta6_neg))
        intermediate_matrix_neg = np.dot(np.dot(t01_inv_neg, target_matrix), np.dot(t56_inv_neg, t45_inv_neg))
        solutions_group2 = _solve_planar_joints_internal(intermediate_matrix_neg)
    
    # Combine results with offsets into 4x6 matrix
    result_matrix = np.array([
        [theta1_pos, solutions_group1[0, 0] + pi / 2, solutions_group1[0, 1], solutions_group1[0, 2] - pi / 2, theta5_pos - pi / 2, theta6_pos],
        [theta1_pos, solutions_group1[1, 0] + pi / 2, solutions_group1[1, 1], solutions_group1[1, 2] - pi / 2, theta5_pos - pi / 2, theta6_pos],
        [theta1_neg, solutions_group2[0, 0] + pi / 2, solutions_group2[0, 1], solutions_group2[0, 2] - pi / 2, theta5_neg - pi / 2, theta6_neg],
        [theta1_neg, solutions_group2[1, 0] + pi / 2, solutions_group2[1, 1], solutions_group2[1, 2] - pi / 2, theta5_neg - pi / 2, theta6_neg]
    ])
    
    # Transpose to 6x4 matrix (each column one solution) and normalize
    return normalize_angle_matrix(result_matrix.T)