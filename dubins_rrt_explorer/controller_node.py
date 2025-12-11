import rclpy
import math
import numpy as np
import scipy.linalg
import tf_transformations
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TwistStamped
from dubins_rrt_explorer.utils import normalize_angle

class LQRController(Node):
    def __init__(self):
        super().__init__('controller_node')

        # Q: State Cost [Lateral Error, Heading Error]
        self.Q = np.diag([10.0, 1.5]) 
        
        # R: Input Cost [Steering]
        self.R = np.diag([2.5])      
        
        self.dt = 0.05  
        
        self.target_vel = 0.22        

        self.max_accel = 0.18   
        self.max_alpha = 1.80   
        
        self.cmd_v = 0.0
        self.cmd_w = 0.0    
    
        self.curr_pose = None 
        self.path = None
        self.last_nearest_idx = 0

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("LQR Controller Initialized.")

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        _, _, theta = tf_transformations.euler_from_quaternion(q)
        self.curr_pose = np.array([x, y, theta])

    def path_callback(self, msg):
        if not msg.poses: return
        new_path = []
        for pose in msg.poses:
            px = pose.pose.position.x
            py = pose.pose.position.y
            q = (pose.pose.orientation.x, pose.pose.orientation.y,
                 pose.pose.orientation.z, pose.pose.orientation.w)
            _, _, pyaw = tf_transformations.euler_from_quaternion(q)
            new_path.append([px, py, pyaw])
        
        self.path = np.array(new_path)
        self.last_nearest_idx = 0
        self.get_logger().info(f"Received path with {len(self.path)} points")

    def control_loop(self):
        if self.curr_pose is None or self.path is None:
            self.stop()
            return

        nearest_idx = self.calc_nearest_index()

        dist_to_end = np.linalg.norm(self.curr_pose[:2] - self.path[-1][:2])
        if dist_to_end < 0.3: 
            self.stop()
            self.path = None 
            self.get_logger().info("Goal Reached")
            return

        cx, cy, cyaw = self.path[nearest_idx]

        dx = self.curr_pose[0] - cx
        dy = self.curr_pose[1] - cy
        e = -math.sin(cyaw) * dx + math.cos(cyaw) * dy
        
        theta_e = normalize_angle(self.curr_pose[2] - cyaw)

        ff_idx = min(nearest_idx + 5, len(self.path) - 1)
        _, _, fyaw = self.path[ff_idx]
        
        dist_ff = np.linalg.norm(self.path[ff_idx,:2] - self.path[nearest_idx,:2])
        if dist_ff < 0.01: 
            curvature = 0.0
        else:
            curvature = normalize_angle(fyaw - cyaw) / dist_ff

        A = np.array([
            [1.0, self.target_vel * self.dt],
            [0.0, 1.0]
        ])
        
        B = np.array([
            [0.0],
            [self.dt]
        ])

        # Riccati Equation
        P = scipy.linalg.solve_discrete_are(A, B, self.Q, self.R)

        # K = (R + B^T P B)^-1 * (B^T P A)
        K = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)

        ff_omega = self.target_vel * curvature
        fb_omega = - (K[0, 0] * e + K[0, 1] * theta_e)
        cmd_w = ff_omega + fb_omega
        cmd_w = max(min(cmd_w, 2.4), -2.4)

        self.cmd_v = self.ramp_value(self.cmd_v, self.target_vel, self.max_accel)
        self.cmd_w = self.ramp_value(self.cmd_w, cmd_w, self.max_alpha)

        self.publish_cmd(self.cmd_v, self.cmd_w)

    def ramp_value(self, current, target, max_rate):
        step = max_rate * self.dt
        if target > current:
            return min(current + step, target)
        elif target < current:
            return max(current - step, target)
        return target

    def calc_nearest_index(self):
        search_window = 50 
        end_search = min(self.last_nearest_idx + search_window, len(self.path))
        
        dx = self.path[self.last_nearest_idx:end_search, 0] - self.curr_pose[0]
        dy = self.path[self.last_nearest_idx:end_search, 1] - self.curr_pose[1]
        d = dx**2 + dy**2
        
        min_idx = np.argmin(d)
        self.last_nearest_idx += min_idx
        return self.last_nearest_idx

    def publish_cmd(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def stop(self):
        self.publish_cmd(0.0, 0.0)

def main():
    rclpy.init()
    node = LQRController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()