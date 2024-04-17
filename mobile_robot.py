#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.image as mpimg
from matplotlib.patches import Circle
from matplotlib.transforms import Affine2D

# ParametersS
grid_size = (10, 10)
start_position = (0, 0)
goal_position = (9, 9)
obstacle_position = (5, 5)
obstacle_radius = 1
inflation_radius = 0.35
robot_image_path = 'robot.png'
step_size = 0.1
rotate_step = 4

class Robot:
    def __init__(self, start_pos, image_path):
        self.position = np.array(start_pos, dtype=np.float64)
        self.image = mpimg.imread(image_path)
        self.orientation = 0
        self.history = [self.position.copy()]
        self.at_goal = False
        self.waypoints = [np.array(goal_position)]
        self.current_goal = self.waypoints.pop(0)
        self.linear_velocity = step_size / 0.1
        self.angular_velocity = 0
        self.icr = None

    def set_waypoints(self, waypoints):
        self.waypoints = [np.array(wp) for wp in waypoints]
        self.current_goal = self.waypoints.pop(0)

    def move_towards(self, obstacle_center, radius, inflation):
        if self.at_goal:
            return

        attraction_vector = self.current_goal - self.position
        distance_to_goal = np.linalg.norm(attraction_vector)
        if distance_to_goal < step_size:
            if not self.waypoints:
                self.position = self.current_goal
                self.at_goal = True
                return
            else:
                self.current_goal = self.waypoints.pop(0)

        attraction_vector /= distance_to_goal
        obstacle_vector = self.position - np.array(obstacle_center)
        distance_to_obstacle = np.linalg.norm(obstacle_vector) - radius
        repulsion_vector = np.zeros_like(obstacle_vector)
        if distance_to_obstacle < inflation:
            obstacle_vector /= distance_to_obstacle
            repulsion_strength = (1 / distance_to_obstacle - 1 / inflation)
            repulsion_vector = repulsion_strength * obstacle_vector

        result_vector = attraction_vector + repulsion_vector
        result_vector /= np.linalg.norm(result_vector)

        desired_orientation = np.degrees(np.arctan2(result_vector[1], result_vector[0])) % 360
        orientation_diff = (desired_orientation - self.orientation + 360) % 360
        if orientation_diff > 180:
            orientation_diff -= 360

        self.angular_velocity = np.radians(orientation_diff) / 0.1
        self.orientation += orientation_diff if abs(orientation_diff) < rotate_step else np.sign(orientation_diff) * rotate_step
        self.orientation %= 360

        forward_direction = np.array([
            np.cos(np.radians(self.orientation)), 
            np.sin(np.radians(self.orientation))
        ])
        self.position += forward_direction * step_size
        self.history.append(self.position.copy())

        if abs(self.angular_velocity) > 0.1:  # Only compute ICR if there is significant rotation
            radius_of_turn = abs(self.linear_velocity / self.angular_velocity)
            perpendicular_direction = np.array([forward_direction[1], -forward_direction[0]]) if self.angular_velocity > 0 else np.array([-forward_direction[1], forward_direction[0]])
            self.icr = self.position - perpendicular_direction * radius_of_turn
        else:
            self.icr = None

    def print_status(self):
        print(f"Robot Position: {self.position}")
        print(f"Linear Velocity: {self.linear_velocity:.2f} units/s")
        print(f"Angular Velocity: {self.angular_velocity:.2f} rad/s")
        if self.icr is not None:
            print(f"ICR Position: {self.icr}")
        else:
            print("ICR is at infinity (straight movement).")



robot = Robot(start_position, robot_image_path)
robot.set_waypoints([(3, 2), (4, 6), (9, 9)])

fig, ax = plt.subplots()
ax.set_xlim(-1, grid_size[0] + 1)
ax.set_ylim(-1, grid_size[1] + 1)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_title('Robot Navigation Simulation')
ax.grid(True, zorder=0)

goal_plot = ax.plot(*goal_position, marker='o', color='g', markersize=10, label='Goal', zorder=2)
path_plot, = ax.plot([], [], 'b-', linewidth=2, label='Path')
obstacle = Circle(obstacle_position, obstacle_radius, color='brown', label='Obstacle', zorder=2)
ax.add_patch(obstacle)
obstacle_inflated = Circle(obstacle_position, obstacle_radius + inflation_radius, color='orange', alpha=0.3, label='Inflation Zone', zorder=1)
ax.add_patch(obstacle_inflated)

robot_image_ax = ax.imshow(robot.image, aspect='auto', extent=[robot.position[0] - 0.5, robot.position[0] + 0.5, robot.position[1] - 0.5, robot.position[1] + 0.5], zorder=3)
icr_plot, = ax.plot([], [], 'x', color='red', markersize=10, label='ICR', zorder=4)

def init():
    robot_image_ax.set_extent([robot.position[0] - 0.5, robot.position[0] + 0.5, robot.position[1] - 0.5, robot.position[1] + 0.5])
    icr_plot.set_data([], [])
    path_plot.set_data([], [])
    return robot_image_ax, icr_plot, path_plot

def update(frame):
    robot.move_towards(obstacle_position, obstacle_radius, inflation_radius)
    robot.print_status()
    new_extent = [robot.position[0] - 0.5, robot.position[0] + 0.5, robot.position[1] - 0.5, robot.position[1] + 0.5]
    robot_image_ax.set_extent(new_extent)
    rotate_transform = Affine2D().rotate_deg_around(robot.position[0], robot.position[1], robot.orientation)
    robot_image_ax.set_transform(rotate_transform + ax.transData)
    
    path_plot.set_data([pos[0] for pos in robot.history], [pos[1] for pos in robot.history])

    if robot.icr is not None:
        icr_plot.set_data([robot.icr[0]], [robot.icr[1]])
    else:
        icr_plot.set_data([], [])

    return robot_image_ax, icr_plot, path_plot


anim = animation.FuncAnimation(fig, update, init_func=init, frames=200, interval=100, blit=True)
plt.legend()
plt.show()
