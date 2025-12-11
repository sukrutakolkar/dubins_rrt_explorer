import rclpy
import math, time
import numpy as np
import tf_transformations
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, Quaternion

from dublins_rrt_explorer.utils import DubinsSolver

class NodeRRT:
    def __init__(self, x, y, theta, parent=None, cost=0.0, edge_cost=0.0, path_x=None, path_y=None, path_yaw=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = parent
        self.cost = cost           # Total cost from root
        self.edge_cost = edge_cost # Cost from parent to this node 
        self.children = []         # List of children nodes 
        
        self.path_x = path_x if path_x is not None else []
        self.path_y = path_y if path_y is not None else []
        self.path_yaw = path_yaw if path_y is not None else []

class RRTPlanner:
    def __init__(self, map_data, map_info):
        self.map_data = map_data
        self.map_info = map_info
        self.resolution = map_info.resolution
        self.origin_x = map_info.origin.position.x
        self.origin_y = map_info.origin.position.y
        self.width = map_info.width
        self.height = map_info.height

        self.dubins = DubinsSolver(step_size=0.05)
        self.turning_radius = 0.25 
        self.goal_sample_rate = 0.10
        self.max_iter = 30000 
        self.goal_tolerance = 0.5
        
        self.node_list = []

    def plan(self, start_pose, goal_pose):
        self.node_list = [NodeRRT(start_pose[0], start_pose[1], start_pose[2])]
        
        print(f"RRT Planning: {start_pose} to {goal_pose}")

        for i in range(self.max_iter):
            rnd_node = self.get_random_node(goal_pose)
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            is_goal = (rnd_node.x == goal_pose[0] and rnd_node.y == goal_pose[1])
            if not is_goal:
                dx = rnd_node.x - nearest_node.x
                dy = rnd_node.y - nearest_node.y
                rnd_node.theta = math.atan2(dy, dx)
            
            new_node = self.steer(nearest_node, rnd_node)
            
            if new_node and self.check_collision(new_node):
                self.node_list.append(new_node)
                
                # Check goal
                if self.dist_to_goal(new_node, goal_pose) <= self.goal_tolerance:
                    print(f"Target Reached at iter {i}")
                    return self.finalize_path(new_node, goal_pose)

        print("RRT Max Iterations Reached. No Path Found.")
        return None

    def steer(self, from_node, to_node):
        start = [from_node.x, from_node.y, from_node.theta]
        end   = [to_node.x, to_node.y, to_node.theta]
        
        points, cost = self.dubins.get_shortest_path(start, end, self.turning_radius)
        
        if points is None or cost > 3.0: return None

        px = [p[0] for p in points[1:]]
        py = [p[1] for p in points[1:]]
        pyaw = [p[2] for p in points[1:]]

        if not px: return None

        new_node = NodeRRT(px[-1], py[-1], pyaw[-1])
        new_node.parent = from_node
        new_node.path_x = px
        new_node.path_y = py
        new_node.path_yaw = pyaw
        
        new_node.edge_cost = cost                     
        new_node.cost = from_node.cost + cost
        
        return new_node

    def get_random_node(self, goal):
        if np.random.random() > self.goal_sample_rate:
            rx = np.random.uniform(self.origin_x, self.origin_x + self.width * self.resolution)
            ry = np.random.uniform(self.origin_y, self.origin_y + self.height * self.resolution)
            return NodeRRT(rx, ry, 0.0)
        else:
            return NodeRRT(goal[0], goal[1], goal[2])

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        return dlist.index(min(dlist))

    def check_collision(self, node):
        if node is None: return False
        
        ix = ((np.array(node.path_x) - self.origin_x) / self.resolution).astype(int)
        iy = ((np.array(node.path_y) - self.origin_y) / self.resolution).astype(int)
        
        valid_mask = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
        
        if not np.all(valid_mask):
            return False
            
        map_vals = self.map_data[iy, ix]
        if np.any((map_vals == 100) | (map_vals == -1)):
            return False
            
        return True

    def dist_to_goal(self, node, goal):
        return math.hypot(node.x - goal[0], node.y - goal[1])

    def finalize_path(self, node, goal_pose):
        goal_node = NodeRRT(goal_pose[0], goal_pose[1], goal_pose[2])
        
        final_connect = self.steer(node, goal_node)
        
        if final_connect and self.check_collision(final_connect):
            self.node_list.append(final_connect)
            return self.generate_trajectory(len(self.node_list) - 1)
        else:
            return self.generate_trajectory(len(self.node_list) - 1)

    def generate_trajectory(self, goal_ind):
        path = []
        node = self.node_list[goal_ind]
        while node.parent is not None:
            for i in range(len(node.path_x) - 1, -1, -1):
                path.append([node.path_x[i], node.path_y[i], node.path_yaw[i]])
            node = node.parent
        path.append([node.x, node.y, node.theta])
        return path[::-1]


class RRTStarPlanner(RRTPlanner):
    def __init__(self, map_data, map_info):
        super().__init__(map_data, map_info)
        self.rewire_radius = 2.0  
        self.max_iter = 20000 

    def plan(self, start_pose, goal_pose):
        self.node_list = [NodeRRT(start_pose[0], start_pose[1], start_pose[2])]
        print(f"RRT* Planning: {start_pose} to {goal_pose}")

        for i in range(self.max_iter):
            rndm_node = self.get_random_node(goal_pose)
            nearest_ind = self.get_nearest_node_index(self.node_list, rndm_node)
            nearest_node = self.node_list[nearest_ind]

            is_goal = (rndm_node.x == goal_pose[0] and rndm_node.y == goal_pose[1])
            if not is_goal:
                rndm_node.theta = math.atan2(rndm_node.y - nearest_node.y, rndm_node.x - nearest_node.x)

            new_node = self.steer(nearest_node, rndm_node)
            
            if new_node and self.check_collision(new_node):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                
                if new_node:
                    if new_node.parent:
                        new_node.parent.children.append(new_node)   

                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)
                    
                    # returning the first path without refining because it takes too long
                    if self.dist_to_goal(new_node, goal_pose) <= self.goal_tolerance:
                        print(f"RRT* Target Reached at iter {i}")
                        return self.finalize_path(new_node, goal_pose)
             
        print("RRT* Max Iterations Reached.")
        return None
    
    def propagate_cost_to_leaves(self, parent_node):
        for child in parent_node.children:
            child.cost = parent_node.cost + child.edge_cost
            self.propagate_cost_to_leaves(child)

    def find_near_nodes(self, new_node):
        r = self.rewire_radius 
        
        dlist = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 for node in self.node_list]
        near_inds = [i for i, d in enumerate(dlist) if d <= r**2]
        return near_inds

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return new_node
        
        costs = []
        valid_paths = []
        
        for i in near_inds:
            near_node = self.node_list[i]
            
            t_node = self.steer(near_node, new_node) 
            
            if t_node and self.check_collision(t_node):
                costs.append(t_node.cost)
                valid_paths.append(t_node)
            else:
                costs.append(float('inf'))
                valid_paths.append(None)
        
        min_cost = min(costs)
        if min_cost == float('inf'):
            return new_node 
        
        if min_cost < new_node.cost:
            best_node = valid_paths[costs.index(min_cost)]
            return best_node
            
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            
            edge_node = self.steer(new_node, near_node)
            
            if edge_node is None:
                continue
                
            if edge_node.cost < near_node.cost:
                if self.check_collision(edge_node):
                    if near_node.parent and near_node in near_node.parent.children:
                        near_node.parent.children.remove(near_node)

                    near_node.parent = new_node
                    near_node.cost = edge_node.cost
                    near_node.path_x = edge_node.path_x
                    near_node.path_y = edge_node.path_y
                    near_node.path_yaw = edge_node.path_yaw
                    new_node.children.append(near_node)
                    self.propagate_cost_to_leaves(near_node)

class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')
        
        # Parameters
        self.declare_parameter('planner_type', 'rrt_star') # 'rrt', 'rrt_star'
        
        self.planner = None
        self.curr_pose = None
        self.map_received = False

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.map_sub  = self.create_subscription(OccupancyGrid, '/costmap', self.map_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.path_pub = self.create_publisher(Path, '/plan', 10)

        p_type = self.get_parameter('planner_type').get_parameter_value().string_value
        self.get_logger().info(f"Planner Node Initialized. Mode: {p_type.upper()}")

    def odom_callback(self, msg):
        self.curr_pose = msg.pose.pose

    def map_callback(self, msg):
        data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        
        p_type = self.get_parameter('planner_type').get_parameter_value().string_value
        
        if p_type == 'rrt_star':
            self.planner = RRTStarPlanner(data, msg.info)
        else:
            self.planner = RRTPlanner(data, msg.info)
            
        if not self.map_received:
            self.get_logger().info("Costmap received.")
            self.map_received = True

    def goal_callback(self, msg):
        if not self.curr_pose or not self.planner:
            self.get_logger().warn("Cannot plan: No Pose or No Map.")
            return
            
        self.get_logger().info("Received Goal. Starting Planner.")
        
        rx = self.curr_pose.position.x
        ry = self.curr_pose.position.y
        _, _, rtheta = tf_transformations.euler_from_quaternion(
            [self.curr_pose.orientation.x, self.curr_pose.orientation.y,
             self.curr_pose.orientation.z, self.curr_pose.orientation.w])
        
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        _, _, gtheta = tf_transformations.euler_from_quaternion(
            [msg.pose.orientation.x, msg.pose.orientation.y,
             msg.pose.orientation.z, msg.pose.orientation.w])
             
        path_points = self.planner.plan([rx, ry, rtheta], [gx, gy, gtheta])
        
        if path_points:
            self.publish_path(path_points)
            self.get_logger().info(f"Path Published! Length: {len(path_points)} points")
        else:
            self.get_logger().warn("Failed to find path.")

    def publish_path(self, points):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        
        for p in points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            q = tf_transformations.quaternion_from_euler(0, 0, p[2])
            pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            msg.poses.append(pose)
            
        self.path_pub.publish(msg)

def main():
    rclpy.init()
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()