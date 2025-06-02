
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped 
import rclpy
from rclpy.node import Node

class ObstacleMover(Node): 
    def __init__(self, node_name='obstacle_mover'):
        super().__init__(node_name)
        
        # Obstalce dictionary with params that will be used for publishing and moving 
        self.obstacle = {
            'obstacle_one': { 'publisher': self.create_publisher(Twist, '/obstacle_one/cmd_vel', 1), 'pose_publisher': self.create_publisher(PoseWithCovarianceStamped, '/obstacle_pose', 1), 'position': [3.0, 3.0], 'direction': 1 },
            'obstacle_two': { 'publisher': self.create_publisher(Twist, '/obstacle_two/cmd_vel', 1), 'pose_publisher': self.create_publisher(PoseWithCovarianceStamped, '/obstacle_pose', 1),  'position': [0.0, 0.0], 'direction': 1 },
            'obstacle_three': { 'publisher': self.create_publisher(Twist, '/obstacle_three/cmd_vel', 1), 'pose_publisher': self.create_publisher(PoseWithCovarianceStamped, '/obstacle_pose', 1),  'position': [-3.0, -3.0], 'direction': 1 }
        }
        
        self.timer = self.create_timer(0.1, self.move_obstacles) 
        self.object_threshold = 3
    
    def move_obstacles(self):
        for obstacle_name, obs in self.obstacle.items():
            # Get params
            publisher = obs['publisher']
            pose_publisher = obs['pose_publisher']
            position = obs['position']
            direction = obs['direction']
            
            # Calculate the next positon
            next_position =  position[0] + direction * 0.5 * self.movement_step
            
            # Check if it is oo close too wall
            if abs(next_position) >= self.object_threshold:
                obs['direction'] *= -1
                direction = obs['direction']
                next_position =  position[0] + direction * 0.5 * self.movement_step
            
            # Set the next position
            position[0] = next_position
            
            # Publish the speed and pose that will be used in the qln_dyn
            twist = Twist()
            twist.linear.x = obs['direction'] * 0.5
            publisher.publish(twist)
    
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = obstacle_name
            pose_msg.pose.pose.position.x = position[0]
            pose_msg.pose.pose.position.y = position[1]
            pose_msg.pose.pose.position.z = 0.0
            pose_msg.pose.pose.orientation.w = 1.0
            pose_msg.pose.covariance = [0.0] * 36
            pose_publisher.publish(pose_msg)
            
    
def main(args=None):
    # 1st. initialization of node.
    rclpy.init()
    obstacle = ObstacleMover()
    rclpy.spin(obstacle)
    rclpy.shutdown()  

if __name__ == "__main__":
    main()
