import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
import sys
import select
import termios
import tty
from pynput import keyboard

MAX_LINEAR_VEL  = 2.0
MIN_LINEAR_VEL  = -1.0
MAX_ANGULAR_VEL = 1.5  # radians per second
LINEAR_ACCEL    = 0.1     # linear velocity increment per key press
ANGULAR_ACCEL   = 0.1    # angular velocity increment
DAMPING_FACTOR  = 0.95  # reduce velocity if no key pressed

ACCEL = 'w'
DECEL = 's'
LEFT  = 'a'
RIGHT = 'd'
STOP  = 'b'

class TeleopNode(Node):
    def __init__(self):
        super().__init__('kb_teleop_node')
        self.publisher = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        self.timer = self.create_timer(0.05, self.publish_cmd)

        self.applied_v = 0.0
        self.applied_w = 0.0

        self.current_v = 0.0
        self.current_w = 0.0

        # save old terminal settings, set raw
        self.old_settings = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())  # or tty.setraw

        self.logger = self.get_logger()
        self.logger.info(f"{self.__class__.__name__} termios initialized.")

    def read_input(self):
        while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            k = sys.stdin.read(1)
            if k == ACCEL:
                self.applied_v += LINEAR_ACCEL
                
            elif k == DECEL:
                self.applied_v -= LINEAR_ACCEL
                
            elif k == LEFT:
                self.applied_w += ANGULAR_ACCEL
                
            elif k == RIGHT:
                self.applied_w -= ANGULAR_ACCEL
                
            elif k == STOP:
                self.reset_control()
        
    def publish_cmd(self):
        self.read_input()
        # apply increments from key presses
        if self.applied_v:
            self.current_v += self.applied_v
        else:                                   # if no keys were pressed, apply damping
            self.current_v *= DAMPING_FACTOR

        if self.applied_w:
            self.current_w += self.applied_w
        else:
            self.current_w *= DAMPING_FACTOR

        # clamp the final values
        self.current_v = max(MIN_LINEAR_VEL, min(MAX_LINEAR_VEL, self.current_v))
        self.current_w = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, self.current_w))

        # prepare and publish the command
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = ""
        cmd.twist.linear.x  = self.current_v
        cmd.twist.angular.z = self.current_w
        
        self.logger.debug(f"Publishing v: {self.current_v}, w: {self.current_w}")
        self.publisher.publish(cmd)

        # clear applied inputs
        self.applied_v = self.applied_w = 0.0

    def destroy_node(self):
        # restore terminal
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old_settings)
        super().destroy_node()

    # control functions
    def accelerate(self):
        self.applied_v += LINEAR_ACCEL

    def decelerate(self):
        self.applied_v -= LINEAR_ACCEL

    def turn_left(self):
        self.applied_w += ANGULAR_ACCEL
    
    def turn_right(self):
        self.applied_w -= ANGULAR_ACCEL

    def reset_control(self):
        self.current_v = self.current_w = self.applied_v = self.applied_w = 0.0

class TeleopNode1(Node):
    def __init__(self):
        super().__init__('kb_teleop_node')
        self.publisher = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        self.timer = self.create_timer(0.05, self.publish_cmd) # 20 Hz

        self.current_v = 0.0
        self.current_w = 0.0

        self.keys_pressed = set()

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

        self.logger = self.get_logger()
        self.logger.info("teleop node started.")
        self.logger.info(f"Controls:\n w/s: linear | a/d: angular | b: stop")

    def on_press(self, key):
        try:
            self.keys_pressed.add(key.char)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key.char in self.keys_pressed:
                self.keys_pressed.remove(key.char)
        except AttributeError:
            pass

    def publish_cmd(self):
        applied_v_increment = 0.0
        applied_w_increment = 0.0
        self.logger.info(str(len(self.keys_pressed)))

        if ACCEL in self.keys_pressed:
            applied_v_increment = LINEAR_ACCEL
        if DECEL in self.keys_pressed:
            applied_v_increment = -LINEAR_ACCEL
        if LEFT in self.keys_pressed:
            applied_w_increment = ANGULAR_ACCEL
        if RIGHT in self.keys_pressed:
            applied_w_increment = -ANGULAR_ACCEL
        if STOP in self.keys_pressed:
            self.current_v = 0.0
            self.current_w = 0.0

        if applied_v_increment != 0.0:
            self.current_v += applied_v_increment
        else:
            self.current_v *= DAMPING_FACTOR

        if applied_w_increment != 0.0:
            self.current_w += applied_w_increment
        else:
            self.current_w *= DAMPING_FACTOR

        self.current_v = max(MIN_LINEAR_VEL, min(MAX_LINEAR_VEL, self.current_v))
        self.current_w = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, self.current_w))

        cmd_stamped = TwistStamped()
        cmd_stamped.header.stamp = self.get_clock().now().to_msg()
        cmd_stamped.header.frame_id = ""
        cmd_stamped.twist.linear.x  = self.current_v
        cmd_stamped.twist.angular.z = self.current_w
        
        self.logger.debug(f"Publishing v: {self.current_v}, w: {self.current_w}")
        self.publisher.publish(cmd_stamped)

    def destroy_node(self):
        self.logger.info("Stopping keyboard listener.")
        self.listener.stop()
        super().destroy_node()

def main():
    rclpy.init()
    node = TeleopNode1()
    try:
        rclpy.spin(node)  
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()