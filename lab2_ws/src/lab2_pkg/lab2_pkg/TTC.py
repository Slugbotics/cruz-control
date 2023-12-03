import sys
import math

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovariance

class TTC(Node):
    def __init__(self):
        super().__init__('TTC')
        self.LaserScanSubscribe = self.create_subscription(LaserScan,'scan',self.LaserScanCallback,10)
        self.OdometrySubscribe = self.create_subscription(Odometry,'ego_racecar/odom',self.OdometryCallback,10)
        self.drive_relayPublish = self.create_publisher(AckermannDriveStamped,'drive',10)
        # self.declare_parameter('Velocity',value = 0.0)
        self.Velocity = 0.0 
        # self.get_logger().info("TTC Created")       

    def LaserScanCallback(self, msg):
        # self.get_logger().info("LaserScanCallback")
        smallest_d = msg.range_max
        idx = 0
        tol = 1.6
        for dis in msg.ranges:
            # self.get_logger().info(str(dis))
            if dis < smallest_d:
                smallest_d = dis
            idx+=1
            # self.get_logger().info(str(math.degrees(msg.angle_increment*idx+msg.angle_min)))
        dR = self.Velocity*math.cos(msg.angle_increment*idx+msg.angle_min)
        if dR != 0 and smallest_d/dR > -tol:
            self.get_logger().info(str(round(smallest_d/dR,2)))
        if dR != 0 and smallest_d/dR < 0 and smallest_d/dR > -tol:
            ack_msg = AckermannDriveStamped()
            ack_msg.drive.speed = 0.0
            self.drive_relayPublish.publish(ack_msg)
            # self.get_logger().info("stoppp")
            

    def OdometryCallback(self, msg):
        # self.get_logger().info("OdometryCallback")
        self.Velocity = msg.twist.twist.linear.x
        # self.setParameter('Velocity',value = msg.twist.covariance[0])

def main(args=None):
    rclpy.init(args=args)

    TTCNode = TTC()

    rclpy.spin(TTCNode)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()