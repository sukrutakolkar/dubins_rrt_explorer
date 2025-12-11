import rclpy
import math
import numpy as np
import tf_transformations
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from tf2_ros import Buffer, TransformListener
import rclpy.time

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.declare_parameter('map_width', 400)
        self.declare_parameter('map_height', 400)
        self.declare_parameter('map_resolution', 0.05)
        
        self.declare_parameter('origin_x', -9999.0)
        self.declare_parameter('origin_y', -9999.0)

        self.width = self.get_parameter('map_width').value
        self.height = self.get_parameter('map_height').value
        self.resolution = self.get_parameter('map_resolution').value
        
        req_origin_x = self.get_parameter('origin_x').value
        req_origin_y = self.get_parameter('origin_y').value

        if req_origin_x > -5000.0 and req_origin_y > -5000.0:
            self.origin_x = req_origin_x
            self.origin_y = req_origin_y
            self.map_initialized = True
            self.get_logger().info(f"Custom Origin: ({self.origin_x}, {self.origin_y})")
        else:
            self.origin_x = None
            self.origin_y = None
            self.map_initialized = False
        
        self.prob_occ = 0.999
        self.prob_free = 0.25
        self.log_occ = np.log(self.prob_occ / (1 - self.prob_occ))
        self.log_free = np.log(self.prob_free / (1 - self.prob_free))
        
        self.map_log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        
        self.robot_radius = 0.08 
        self.safety_margin = 0.05
        self.inflation_radius = self.robot_radius + self.safety_margin
        self.inflation_kernel = self._precompute_kernel(self.inflation_radius)

        self.curr_pose = None # (x, y, theta)

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.c_space_pub = self.create_publisher(OccupancyGrid, '/costmap', 10)

        self.timer = self.create_timer(0.5, self.publish_maps)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(f"Perception Node Initialized. Inflation Radius: {self.inflation_radius}m")

    def _precompute_kernel(self, radius):
        r_pixels = int(math.ceil(radius / self.resolution))
        kernel = []
        for dy in range(-r_pixels, r_pixels + 1):
            for dx in range(-r_pixels, r_pixels + 1):
                if dx**2 + dy**2 <= r_pixels**2:
                    kernel.append((dy, dx))
        return np.array(kernel)

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if not self.map_initialized:
            self.origin_x = x - (self.width * self.resolution) / 2.0
            self.origin_y = y - (self.height * self.resolution) / 2.0
            
            self.map_initialized = True
            self.get_logger().info(f"Map Anchored! Origin set to: ({self.origin_x:.2f}, {self.origin_y:.2f})")

        q = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        _, _, theta = tf_transformations.euler_from_quaternion(q)
        self.curr_pose = (x, y, theta)

    def scan_callback(self, msg: LaserScan):
        try:
            t = self.tf_buffer.lookup_transform(
                'map',              # Target Frame (SLAM Map)
                'base_footprint',   # Source Frame (Robot)
                rclpy.time.Time(),  
                rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as ex:
            self.get_logger().warn("No transform")
            return

        rx = t.transform.translation.x
        ry = t.transform.translation.y
        
        q = (t.transform.rotation.x, t.transform.rotation.y,
             t.transform.rotation.z, t.transform.rotation.w)
        _, _, r_theta = tf_transformations.euler_from_quaternion(q)
        
        start_x = int((rx - self.origin_x) / self.resolution)
        start_y = int((ry - self.origin_y) / self.resolution)

        if not (0 <= start_x < self.width and 0 <= start_y < self.height):
            return

        step = 3 
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment * step)
        ranges = np.array(msg.ranges[::step])
        
        valid_mask = np.isfinite(ranges) & (ranges < msg.range_max)
        
        safe_ranges = ranges.copy()
        safe_ranges[~valid_mask] = 0.0
        
        world_angles = angles + r_theta
        
        end_xs = start_x + (safe_ranges * np.cos(world_angles) / self.resolution).astype(int)
        end_ys = start_y + (safe_ranges * np.sin(world_angles) / self.resolution).astype(int)

        for i in range(len(ranges)):
            if not valid_mask[i]:
                dist = msg.range_max * 0.9
                ex = int(start_x + (dist * math.cos(world_angles[i]) / self.resolution))
                ey = int(start_y + (dist * math.sin(world_angles[i]) / self.resolution))
                self.trace_line(start_x, start_y, ex, ey, hit=False)
            else:
                self.trace_line(start_x, start_y, end_xs[i], end_ys[i], hit=True)

        self.map_log_odds = np.clip(self.map_log_odds, -20.0, 20.0)

    def trace_line(self, x0, y0, x1, y1, hit):
        points = self.bresenham(x0, y0, x1, y1)
        
        in_bounds = (points[:, 0] >= 0) & (points[:, 0] < self.width) & \
                    (points[:, 1] >= 0) & (points[:, 1] < self.height)
        points = points[in_bounds]
        
        if len(points) == 0: return

        if hit:
            free_points = points[:-1]
            hit_point = points[-1]
            
            if len(free_points) > 0:
                self.map_log_odds[free_points[:, 1], free_points[:, 0]] += self.log_free
            
            self.map_log_odds[hit_point[1], hit_point[0]] += self.log_occ
        else:
            self.map_log_odds[points[:, 1], points[:, 0]] += self.log_free

    def bresenham(self, x0, y0, x1, y1):
        """ Returns list of coordinates in line """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        
        points = []
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((x, y))
        return np.array(points)

    def generate_c_space(self, raw_map):
        c_space = np.zeros_like(raw_map, dtype=np.int8)
        
        c_space[raw_map == -1] = -1
        
        occupied_indices = np.argwhere(raw_map >= 50)
        
        if len(occupied_indices) == 0:
            return c_space
        
        y_occ = occupied_indices[:, 0]
        x_occ = occupied_indices[:, 1]
        
        y_inflated = y_occ[:, None] + self.inflation_kernel[:, 0] 
        x_inflated = x_occ[:, None] + self.inflation_kernel[:, 1]
        
        y_flat = y_inflated.ravel()
        x_flat = x_inflated.ravel()
        
        valid_mask = (x_flat >= 0) & (x_flat < self.width) & \
                     (y_flat >= 0) & (y_flat < self.height)
        
        y_final = y_flat[valid_mask]
        x_final = x_flat[valid_mask]
        
        # mark danger
        c_space[y_final, x_final] = 100
        
        return c_space

    def publish_maps(self):
        exp_log = np.exp(self.map_log_odds)
        probs = (1.0 - (1.0 / (1.0 + exp_log))) * 100
        
        grid_data = np.full(probs.shape, -1, dtype=np.int8)
        
       
        grid_data[probs < 20] = 0     # Free
        grid_data[probs > 80] = 100   # Occupied
        
        self._publish_grid(grid_data, self.map_pub)
        
        c_space_data = self.generate_c_space(grid_data)
        self._publish_grid(c_space_data, self.c_space_pub)

    def _publish_grid(self, data, publisher):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.resolution = self.resolution
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        
        msg.data = data.flatten().tolist()
        publisher.publish(msg)

def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()