import rclpy, math
import numpy as np
import tf_transformations
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid

class MapperNode(Node):
    def __init__(self):
        super().__init__('mapper_node')

        self.map_x = 500
        self.map_y = 500
        self.map_resolution = 0.05
        self.origin_x = - (self.map_x / 2.0) * self.map_resolution
        self.origin_y = - (self.map_y / 2.0) * self.map_resolution

        self.prob_map  = np.full((self.map_y, self.map_x), 0.5)
        self.prob_occ  = 0.999
        self.prob_free = 0.2
        self.log_prior = np.log(0.5 / 0.5)

        self.log_occ  = self.log_odds_from_prob(self.prob_occ)
        self.log_free = self.log_odds_from_prob(self.prob_free)

        self.curr_pos = None

        self.odom_sub = self.create_subscription(Odometry,  '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.map_pub_timer = self.create_timer(1, self.publish_map)

        self.logger = self.get_logger()
        self.logger.info("Mapper initialized.")

    def odom_callback(self, msg: Odometry):
        self.curr_pos = msg.pose.pose

    def scan_callback(self, msg: LaserScan):
        if not self.curr_pos:
            self.logger.warn("Pose information not set, dropping message.")
            return

        rbt_x = self.curr_pos.position.x
        rbt_y = self.curr_pos.position.y
        _, _, rbt_theta = tf_transformations.euler_from_quaternion(
            (self.curr_pos.orientation.x,
             self.curr_pos.orientation.y,
             self.curr_pos.orientation.z,
             self.curr_pos.orientation.w))
        
        rbt_x_grid = int((rbt_x - self.origin_x) / self.map_resolution)
        rbt_y_grid = int((rbt_y - self.origin_y) / self.map_resolution)

        effective_max_range = msg.range_max * 0.99

        for i, dist in enumerate(msg.ranges):
            if not np.isfinite(dist) or dist > effective_max_range:
                dist = effective_max_range
                is_hit = False
            else:
                is_hit = True

            beam_angle  = msg.angle_min + (i * msg.angle_increment)
            world_angle = beam_angle + rbt_theta

            dist_grid = dist / self.map_resolution

            endpoint_x_grid = int(rbt_x_grid + (dist_grid * math.cos(world_angle)))
            endpoint_y_grid = int(rbt_y_grid + (dist_grid * math.sin(world_angle)))

            points = self.trace_beam(rbt_x_grid, rbt_y_grid, endpoint_x_grid, endpoint_y_grid)

            for j, point in enumerate(points):
                px, py = point

                if 0 <= px < self.map_x and 0 <= py < self.map_y:
                    is_last = (j == (len(points) - 1))

                    if is_hit and is_last:
                        log_ism = self.log_occ
                    else:
                        log_ism = self.log_free
                    
                    log_prev = self.log_odds_from_prob(self.prob_map[py, px])
                    log_odds = log_prev + log_ism - self.log_prior
                    
                    self.prob_map[py, px] = self.prob_from_log_odds(log_odds)
    
    def trace_beam(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        err = dx - dy

        points = []

        while True:
            points.append([x0, y0])
            
            if x0 == x1 and y0 == y1:
                break
            
            # out of map bounds
            if not (0 <= x0 < self.map_x and 0 <= y0 < self.map_y):
                return points[:-1]

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return points
    
    def log_odds_from_prob(self, p):
        p = np.clip(p, 0.001, 0.999)
        return np.log(p / (1 - p))
    
    def prob_from_log_odds(self, l):
        # l = np.clip(l, -20, 20) 
        return 1.0 - (1.0 / (1.0 + np.exp(l)))
    
    def publish_map(self):
        msg = OccupancyGrid()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        msg.info.width  = self.map_x
        msg.info.height = self.map_y
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.resolution = self.map_resolution

        map_flat = self.prob_map.flatten()

        map_data = np.zeros(map_flat.shape, np.int8)
        map_data[map_flat == 0.5] = -1
        map_data[map_flat > 0.5]  = (map_flat[map_flat > 0.5] * 100).astype(np.int8)
        map_data[map_flat < 0.5]  = (map_flat[map_flat < 0.5] * 100).astype(np.int8)

        msg.data = map_data.tolist()
        self.map_pub.publish(msg)
        self.logger.info("OccupancyGrid published")

def main():
    rclpy.init()
    node = MapperNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    



                    

