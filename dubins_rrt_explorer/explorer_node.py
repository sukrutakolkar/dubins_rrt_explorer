import rclpy
import math
import numpy as np
import tf_transformations
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped
from scipy.ndimage import label, distance_transform_edt # <--- NEW IMPORT

class ExplorationNode(Node):
    def __init__(self):
        super().__init__('exploration_node')
        self.min_frontier_size = 5   
        self.frontier_safety_margin = 0.1 #
        
        self.info_gain_weight = 1.5
        self.cost_weight = 1
        self.goal_tolerance = 1.0
        self.timeout_duration = 30.0
        
        self.stuck_timeout = 10.0     
        self.last_progress_time = self.get_clock().now()
        self.last_pose_check = None

        self.banned_goals = {} 
        self.ban_duration = 30.0       

        self.switching_threshold = 1.2  
        self.goal_min_dist = 1.5
        self.recovery_active = False

        self.map_data = None
        self.map_info = None
        self.curr_pose = None
        self.current_goal = None
        self.goal_start_time = None
        self.blacklisted_goals = []

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10) 

        self.timer = self.create_timer(1.0, self.control_loop)
        self.get_logger().info(f"Exploration Node Initialized.")

    def map_callback(self, msg):
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.curr_pose = (x, y)

    def control_loop(self):
        if self.map_data is None or self.curr_pose is None: return
        if self.recovery_active: return

        frontiers = self.detect_frontiers()
        
        if not frontiers:
            if self.blacklisted_goals:
                self.get_logger().warn("Clearing blacklist")
                self.blacklisted_goals = []
            return

        best_new_frontier, best_new_score = self.get_best_frontier(frontiers)
        
        if self.current_goal:
            if self.is_stuck():
                self.trigger_recovery()
                return

            if best_new_frontier:
                goal_dist = math.dist(best_new_frontier, self.current_goal)
                
                if goal_dist > self.goal_min_dist:
                    current_goal_score = self.calculate_score(self.current_goal, is_current=True)
                
                    if best_new_score > (current_goal_score * self.switching_threshold):
                        self.get_logger().info(f"Switching! Found better goal (Score {best_new_score:.1f} vs {current_goal_score:.1f})")
                        self.publish_goal(best_new_frontier)
                        return

            if math.dist(self.curr_pose, self.current_goal) < self.goal_tolerance:
                self.get_logger().info("Goal Reached.")
                self.last_goal = self.current_goal
                self.current_goal = None
                return
        
        elif best_new_frontier:
            self.publish_goal(best_new_frontier)

    def is_stuck(self):
        if self.last_pose_check is None:
            self.last_pose_check = self.curr_pose
            self.last_progress_time = self.get_clock().now()
            return False

        dist_moved = math.dist(self.curr_pose, self.last_pose_check)
        current_time = self.get_clock().now()

        if dist_moved > 0.2:
            self.last_progress_time = current_time
            self.last_pose_check = self.curr_pose
            return False
        else:
            time_stuck = (current_time - self.last_progress_time).nanoseconds / 1e9
            
            if time_stuck > self.stuck_timeout:
                self.get_logger().warn(f"Rbt stuck? (Moved <0.2m in {time_stuck:.1f}s)")
                return True
            
        return False

    def calculate_score(self, frontier_point, is_current=False, size=10):
        fx, fy = frontier_point
        rx, ry = self.curr_pose
        
        dist = math.hypot(fx - rx, fy - ry)      
        dist_cost = dist * (self.cost_weight if not is_current else self.cost_weight * 0.8)
        info_gain = size * self.info_gain_weight
        
        return info_gain - dist_cost
    
    def get_best_frontier(self, frontiers):
        best_score = -float('inf')
        best_f = None
        
        for f in frontiers:
            is_banned = False
            for banned_goal in self.banned_goals.keys():
                if math.hypot(f['centroid'][0] - banned_goal[0], f['centroid'][1] - banned_goal[1]) < 0.5:
                    is_banned = True
                    break
            if is_banned: continue

            score = self.calculate_score(f['centroid'], size=f['size'])
            if score > best_score:
                best_score = score
                best_f = f['centroid']
                
        return best_f, best_score

    def detect_frontiers(self):
        free_mask = (self.map_data == 0)
        unknown_mask = (self.map_data == -1)
        obstacle_mask = (self.map_data == 100)
        
        dist_grid = distance_transform_edt(~obstacle_mask)
        
        u_up    = np.pad(unknown_mask[1:, :], ((0,1), (0,0)), constant_values=False)
        u_down  = np.pad(unknown_mask[:-1, :], ((1,0), (0,0)), constant_values=False)
        u_left  = np.pad(unknown_mask[:, 1:], ((0,0), (0,1)), constant_values=False)
        u_right = np.pad(unknown_mask[:, :-1], ((0,0), (1,0)), constant_values=False)

        frontier_pixels = free_mask & (u_up | u_down | u_left | u_right)
        
        structure = np.ones((3, 3), dtype=int)
        labeled_array, num_features = label(frontier_pixels, structure=structure)
        
        clusters = []
        for i in range(1, num_features + 1):
            indices = np.argwhere(labeled_array == i)
            
            if len(indices) < self.min_frontier_size:
                continue 

            safety_values = dist_grid[indices[:, 0], indices[:, 1]]
            
            max_safety_idx = np.argmax(safety_values)
            max_safety_val = safety_values[max_safety_idx]
            
            min_required_dist = 0.15 / self.map_info.resolution 
            
            if max_safety_val < min_required_dist:
                continue
            best_y = indices[max_safety_idx, 0]
            best_x = indices[max_safety_idx, 1]
            
            wx, wy = self.grid_to_world(best_x, best_y)
            
            is_blacklisted = False
            for bx, by in self.blacklisted_goals:
                if math.hypot(wx-bx, wy-by) < 0.5: 
                    is_blacklisted = True
                    break
            
            if not is_blacklisted:
                clusters.append({'centroid': (wx, wy), 'size': len(indices)})
                
        return clusters

    def select_best_frontier(self, frontiers):
        best_score = -float('inf')
        best_frontier = None
        rx, ry = self.curr_pose
        
        for f in frontiers:
            fx, fy = f['centroid']
            size = f['size']

            is_banned = False
            for banned_goal in self.banned_goals.keys():
                if math.hypot(fx - banned_goal[0], fy - banned_goal[1]) < 0.5:
                    is_banned = True
                    break
            if is_banned: continue
            
            info_gain = size * self.info_gain_weight
            dist = math.hypot(fx - rx, fy - ry)
            cost = dist * self.cost_weight
            
            score = info_gain - cost
            
            if score > best_score:
                best_score = score
                best_frontier = (fx, fy)
                
        return best_frontier
    
    def trigger_recovery(self):
        self.get_logger().warn("Robot Stuck! triggering recovery spin")
        self.recovery_active = True
        
        self.banned_goals[self.current_goal] = self.get_clock().now()
        self.current_goal = None
        
        self.recovery_timer = self.create_timer(0.1, self._spin_callback)
        self.spin_start_time = self.get_clock().now()

    def _spin_callback(self):
        elapsed = (self.get_clock().now() - self.spin_start_time).nanoseconds / 1e9
        
        msg = TwistStamped() 
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link" 

        if elapsed > 4.0:
            msg.twist.angular.z = 0.0
            self.vel_pub.publish(msg)
            
            self.recovery_active = False
            self.recovery_timer.destroy()
            self.get_logger().info("Recovery Complete. Replanning.")
            return

        if int(elapsed * 2) % 2 == 0: 
            msg.twist.angular.z = 0.5  
        else:
            msg.twist.angular.z = 0.0  

        self.vel_pub.publish(msg)

    def grid_to_world(self, gx, gy):
        wx = (gx * self.map_info.resolution) + self.map_info.origin.position.x
        wy = (gy * self.map_info.resolution) + self.map_info.origin.position.y
        return wx, wy

    def publish_goal(self, position):
        x, y = position
        rx, ry = self.curr_pose
        
        # Compute yaw to face the goal naturally
        yaw = math.atan2(y - ry, x - rx)
        q = tf_transformations.quaternion_from_euler(0, 0, yaw)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        
        self.goal_pub.publish(msg)
        self.current_goal = (x, y)
        self.goal_start_time = self.get_clock().now().seconds_nanoseconds()[0]
        self.get_logger().info(f"Published Goal: ({x:.2f}, {y:.2f})")

def main():
    rclpy.init()
    node = ExplorationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()