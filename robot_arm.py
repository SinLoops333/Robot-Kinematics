#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Robot arm lengths
l1, l2 = 2.0, 2.0

# Forward kinematics to calculate end effector position
def forward_kinematics(theta1, theta2):
    theta1_rad = np.radians(theta1)
    theta2_rad = np.radians(theta2)
    x = l1 * np.cos(theta1_rad) + l2 * np.cos(theta1_rad + theta2_rad)
    y = l1 * np.sin(theta1_rad) + l2 * np.sin(theta1_rad + theta2_rad)
    return x, y

# Function to find joint angles for a given end effector position
def inverse_kinematics(x, y):
    # Cosine law for the triangle formed by the two links and the line from the origin to the end effector
    cos_angle2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    sin_angle2 = np.sqrt(1 - cos_angle2**2)  # Take positive sqrt for elbow-up position
    theta2 = np.arctan2(sin_angle2, cos_angle2)

    k1 = l1 + l2 * cos_angle2
    k2 = l2 * sin_angle2
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    return np.degrees(theta1), np.degrees(theta2)

# Function to create a single transformation matrix using DH parameters
def dh_transformation_matrix(theta, d, a, alpha):
    theta_rad = np.radians(theta)
    alpha_rad = np.radians(alpha)
    return np.array([
        [np.cos(theta_rad), -np.sinS(theta_rad) * np.cos(alpha_rad),  np.sin(theta_rad) * np.sin(alpha_rad), a * np.cos(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad) * np.cos(alpha_rad), -np.cos(theta_rad) * np.sin(alpha_rad), a * np.sin(theta_rad)],
        [0,                 np.sin(alpha_rad),                     np.cos(alpha_rad),                    d],
        [0,                 0,                                      0,                                   1]
    ])

def calculate_jacobian(theta1, theta2):
    theta1_rad = np.radians(theta1)
    theta2_rad = np.radians(theta2)
    J = np.array([
        [-l1 * np.sin(theta1_rad) - l2 * np.sin(theta1_rad + theta2_rad), -l2 * np.sin(theta1_rad + theta2_rad)],
        [l1 * np.cos(theta1_rad) + l2 * np.cos(theta1_rad + theta2_rad), l2 * np.cos(theta1_rad + theta2_rad)]
    ])
    return J

# Function to calculate the complete transformation matrix from the base to the end effector
def calculate_complete_transformation_matrix(theta1, theta2):
    # First link transformation matrix 0^1T
    T1 = dh_transformation_matrix(theta1, d=0, a=0, alpha=0)
    
    # Second link transformation matrix 1^2T
    T2 = dh_transformation_matrix(theta2, d=0, a=l1, alpha=0)
    
    # Combine the transformations to get 0^2T
    T = np.dot(T1, T2)
    return T

# Animation function
def animate(i):
    if i < len(path):
        theta1, theta2 = path[i]
        x0, y0 = 0, 0
        joint1 = (l1 * np.cos(np.radians(theta1)), l1 * np.sin(np.radians(theta1)))
        effector = forward_kinematics(theta1, theta2)
        
        # Update the data for link 1 and link 2
        link1.set_data([x0, joint1[0]], [y0, joint1[1]])
        link2.set_data([joint1[0], effector[0]], [joint1[1], effector[1]])
        
        # Update the path line data
        path_line.set_data(path_x[:i+1], path_y[:i+1])
        J_v = calculate_jacobian(theta1, theta2)
        # Calculate and print the complete transformation matrix
        T = calculate_complete_transformation_matrix(theta1, theta2)

        theta1 = round(theta1, 2)
        theta2 = round(theta2, 2)
        effector = (round(effector[0], 2), round(effector[1], 2))
        J_v = np.round(J_v, 2)
        T = calculate_complete_transformation_matrix(theta1, theta2)
        print(f"Theta1: {theta1:.2f}, Theta2: {theta2:.2f}")
        print(f"End Effector Position: (x: {effector[0]:.2f}, y: {effector[1]:.2f})")
        print("Velocity Jacobian Matrix:")
        print(J_v)
        print("Complete Transformation Matrix from base to end effector:")
        print(T)
        print("-" * 30)
        
    return link1, link2, path_line

# Setup plot
fig, ax = plt.subplots()
ax.set_xlim(-1*(l1+l2), 1*(l1+l2))
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid()
ax.set_title("2-Link Robot Arm")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")

# Robot arm lines for link 1 and link 2 with different colors
link1, = ax.plot([], [], 'o-', lw=2, color='blue', label='Link 1')
link2, = ax.plot([], [], 'o-', lw=2, color='green', label='Link 2')

# Robot arm line
arm, = ax.plot([], [], 'o-', lw=2)

# Path line
path_line, = ax.plot([], [], 'r-', lw=1)

# Define the vertices of 'M'
vertices_M = [(-3, -1), (-3, 1), (-2, 0), (-1, 1), (-1, -1)]

# Calculate the path to draw 'M'
path_M = []
for vertex in vertices_M:
    theta1, theta2 = inverse_kinematics(*vertex)
    path_M.append((theta1, theta2))

# Approximate a small circle for 'O' with center at (0.5, 0)
circle_radius = 1.0
circle_center = (1.25, 0)
circle_points = 20  # number of points to approximate the circle

# Generate points around the circle for 'O'
path_O = []
for i in range(circle_points + 1):  # +1 to close the circle
    angle = 2 * np.pi * i / circle_points
    x, y = circle_center[0] + circle_radius * np.cos(angle), circle_center[1] + circle_radius * np.sin(angle)
    theta1, theta2 = inverse_kinematics(x, y)
    path_O.append((theta1, theta2))

# Define initial position for 'O'
initial_pos_O = inverse_kinematics(circle_center[0] + circle_radius, circle_center[1])

# Combine paths without returning to initial 'M' position and transition to 'O'
path = path_M + [initial_pos_O] + path_O

# Transform path into robot coordinates for plotting
path_x = []
path_y = []
for theta1, theta2 in path:
    x, y = forward_kinematics(theta1, theta2)
    path_x.append(x)
    path_y.append(y)

# Create animation
ani = FuncAnimation(fig, animate, frames=len(path), interval=700, blit=True)

# Add legend to distinguish link 1 and link 2
ax.legend()

# Show animation
plt.show()
