import numpy as np
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
import math # use of pi.
import random
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped 
from rclpy.duration import Duration

import tf2_ros # library for transformations.
from tf2_ros import TransformException

import tf_transformations

MAP_TOPIC = 'moving_map'
USE_SIM_TIME = True
TF_BASE_LINK = 'base_link'
TF_ODOM_LINK = 'odom'
DEFAULT_CMD_VEL_TOPIC = 'cmd_vel'

RESOLUTION = 1.0  # meters
GRID_HEIGHT = 10
GRID_WIDTH = 10
FREQUENCY = 1  # Hz

UNKNOWN = -1
FREE = 0
OCCUPIED = 100

# Q-learning parameters 
LEARNING_RATE = 0.1 
DISCOUNT_FACTOR = 0.9
EPSILON = 0.3
EPISODES = 1000
MAX = 100 
# Actions 
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4
# REWARDS
GOAL = 100
COLLISION = -100
PENALTY = -1

FREQUENCY = 10

# Velocities that will be used 
LINEAR_VELOCITY = 0.1 # m/s
ANGULAR_VELOCITY = math.pi/8 # rad/s


class QLearningAlgortihm(Node):
    def __init__(self):
       super().__init__('q_learning_node')
       
       use_sim_time_param = rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            USE_SIM_TIME
       )
       self.set_parameters([use_sim_time_param])

       # Algorithm parameters 
       self.q_table = np.zeros((GRID_HEIGHT * GRID_WIDTH, 5))
       self.grid = None
       self.start = (1,1)
       self.goal = (7,8)
       self.actions = [UP, DOWN, LEFT, RIGHT, STAY]
       self.current_episode = 0
       
       self.grid = np.full((GRID_HEIGHT, GRID_WIDTH), FREE)
       
       # Grid walls
       self.grid[0, :] = OCCUPIED
       self.grid[-1, :] = OCCUPIED
       self.grid[:, 0] = OCCUPIED
       self.grid[:, -1] = OCCUPIED
       
       # Grid params
       self.resolution = RESOLUTION
       self.height = GRID_HEIGHT
       self.width = GRID_WIDTH
       self.origin_x = - (self.width / 2) * self.resolution
       self.origin_y = - (self.height / 2) * self.resolution
       
       
       # Publishers and subscriptions 
       self.path_pub = self.create_publisher(Path, '/q_learning/path', 10)
       self.bresenham_path_pub = self.create_publisher(Path, '/q_learning/bresenham_path', 10)
       self.obstacle_sub = self.create_subscription(PoseWithCovarianceStamped, '/obstacle_pose', self.obstacle_pose_callback, 10)
       self._cmd_pub = self.create_publisher(Twist, DEFAULT_CMD_VEL_TOPIC, 10)
       self.obstacle_positions = {'obstacle_one': [3.0 ,3.0], 'obstacle_two': [0.0, 0.0], 'obstacle_three': [-3.0 ,-3.0]}
       self._map_pub = self.create_publisher(OccupancyGrid, MAP_TOPIC, 1)
    
       self.timer = self.create_timer(1.0 / FREQUENCY, self.qlearn_function)
       
       # Robot params 
       self.linear_velocity = LINEAR_VELOCITY
       self.angular_velocity = ANGULAR_VELOCITY
        
       # Transformations
       self.tf_buffer = tf2_ros.Buffer()
       self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
       
    def obstacle_pose_callback(self, msg):
        """Get the obstacle positions"""
        obs_name = msg.header.frame_id 
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.obstacle_positions[obs_name] = [x, y]
        self.update_grid() 
    
    def update_grid(self): 
        """Update the grid with the obstacles"""
        
        # Redfine grid with walls
        self.grid = np.full((GRID_HEIGHT, GRID_WIDTH), FREE)
        self.grid[0, :] = OCCUPIED
        self.grid[-1, :] = OCCUPIED
        self.grid[:, 0] = OCCUPIED
        self.grid[:, -1] = OCCUPIED
        
        # Update the obstacle positions 
        for pose in self.obstacle_positions.values():
            col = int(pose[0] + 5.0)
            row = int(pose[1] + 5.0)
            if 0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH:
                self.grid[row,col] = OCCUPIED
        self.publish_map()

         
    def index_of_state(self, state):
        """Convert state to index for q table"""
        # Function to get a index for very state value based on grid size 
        row, col = state 
        return row * GRID_WIDTH + col
    
    def is_state_valid(self, state):
        """Check if state is valid if it's in bounds and not an obstacle"""
        # State bounding checking 
        row, col = state
        self.grid = self.map_node.grid
        if row < 0 or row >= GRID_HEIGHT or col < 0 or col >= GRID_WIDTH or self.grid[row, col] == OCCUPIED:
            return False 
        return True
    
    def get_new_state(self, state, action):
        """Get the new state"""
        # Figure out which state to go to next based on action 
        row, col = state
        if action == UP:
            new_state = (row - 1, col) 
        elif action == DOWN:
            new_state = (row + 1, col) 
        elif action == LEFT:
            new_state = (row, col - 1) 
        elif action == RIGHT:
            new_state = (row, col + 1) 
        else: 
            new_state = (row, col) 
        return new_state
    
    def reward_calc(self, state):
        """Get the reward"""
        # Simple reward calculation
        if state == self.goal: 
            return GOAL
        if not self.is_state_valid(state):
            return COLLISION
        return PENALTY
    
    def get_action(self, index):
        """Get the best action  based on q table"""
        if random.random() < EPSILON:
            return random.randint(UP,STAY) 
        return np.argmax(self.q_table[index])
    
    def qlearn_function(self):
        """Function to run the q learning alg""" 
        
        # Update the grid each time to have most recent obstacle positions 
        self.update_grid()
        
        # Check if the episode count is complete 
        if self.current_episode >= EPISODES:
            self.timer.cancel()
            print("Done training")
            path = self.get_best_path()
            if path:
                self.publish_path(path, self.path_pub)
                self.bresenham_algorithm(path)
                self.follow_path(path)
            else:
                print("No valid path")
            return 
     
        state = self.start 
        state_index = self.index_of_state(state) 
        for iteration in range(MAX):
            action = self.get_action(state_index)
            
            # Get new state and calcuate the reward for it 
            new_state = self.get_new_state(state, action)
            new_state_index = self.index_of_state(new_state)
            reward = self.reward_calc(new_state)
            
            # Q table updates based on whether the state is valid or not 
            if self.is_state_valid(new_state):
                self.q_table[state_index, action] = (1 - LEARNING_RATE) * self.q_table[state_index, action] +  LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(self.q_table[new_state_index]))
                state = new_state
                state_index = new_state_index
            else: 
                self.q_table[state_index, action] = (1 - LEARNING_RATE) * self.q_table[state_index, action] + LEARNING_RATE * reward
                break 
            if state == self.goal: 
                break 
            
        self.current_episode += 1
        self.publish_map()
                
    def get_best_path(self): 
        """Function to get the best q-table path""" 
        
        # Initiliaze path with the start pose and get starting params 
        self.update_grid()
        best_path = [self.start]
        state = self.start 
        state_index = self.index_of_state(state) 
        count = 0
        
        # Loop until the state is at the goal 
        while state != self.goal and count < MAX:
            
            # Get the action for the state and find the next state based on that
            action = np.argmax(self.q_table[state_index])
            new_state = self.get_new_state(state, action)
            
            if not self.is_state_valid(new_state):
                print("No path found")
                return []
            
            # Add new state to path and then proceed through iteration
            best_path.append(new_state) 
            state = new_state
            state_index = self.index_of_state(state) 
            count += 1
        if state == self.goal:
            return best_path
        else: 
            print("No path found")
            return []
    

    def bresenham_algorithm(self, path):
        """Function to compute the bresenham algorithm"""
        
        # Start and end points
        x1,y1 = path[0]
        x2,y2 = path[-1]
    
        # Difference between points
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        # Incremenent/decrement based on dirrection
        inc_x = 1 if x1 < x2 else -1
        inc_y = 1 if y1 < y2 else -1
        
        # Error term and list for the line 
        error = dx - dy 
        bres_line = [] 
          
        # Loop to append values to the list
        while True:
            bres_line.append((x1,y1))
            if x1 == x2 and y1 == y2:
                break 
            error_2 = 2*error
            
            # Error term checking 
            if error_2 > -dy: 
                error -= dy
                x1 += inc_x
            if error_2 < dx:
                error += dx
                y1 += inc_y
        
        self.publish_path(bres_line, self.bresenham_path_pub)
    
    def publish_path(self, path_list, publisher): 
        """Function to publish the poses for the path"""
        print(path_list)
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for point in path_list:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(point[1] - 5.0 + 0.5)
            pose.pose.position.y = float(point[0] - 5.0 + 0.5) 
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            
        publisher.publish(path_msg)
        self.publish_map()

    def publish_map(self):
        """Function to publish map in rviz for visualization with poses"""
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.info.resolution = self.resolution
        msg.info.height = self.height
        msg.info.width = self.width
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0

        msg.data = self.grid.flatten().tolist()
        self._map_pub.publish(msg)
        
    def follow_path(self, path):   
        """Function to move the robot along the path"""

        print("Follow path:")
        print(path)
        
        # Use transforms to get current positions 
        def get_pose():
            while not self.tf_buffer.can_transform(TF_BASE_LINK,  TF_ODOM_LINK,   rclpy.time.Time()):
                rclpy.spin_once(self)
                pass
            try:
                tf_msg = self.tf_buffer.lookup_transform( TF_BASE_LINK, TF_ODOM_LINK,   rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform: {ex}')
                return
            self.get_logger().info(
                    f'got: {tf_msg}')
            #Intializating the transformation variables that will be used to get the robot coordinates 
            translation = tf_msg.transform.translation
            quaternion = tf_msg.transform.rotation 
            roll, pitch, yaw = tf_transformations.euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w], axes='rxyz')
            return translation.x, translation.y, yaw
        
        for pose in path:
            # Get goal points and the difference between current for travelling  
            goal_x = (pose[1] - 5.0 + 0.5) 
            goal_y = (pose[0] - 5.0 + 0.5) 
            
            current_x, current_y, current_yaw = get_pose()
    
            difference_x =  goal_x - current_x
            difference_y =  goal_y - current_y
            
            #Math for the amount robot will need to turn and how far it will need to travel
            turn_angle = math.atan2(difference_y, difference_x) - current_yaw
            distance = math.sqrt(difference_x ** 2 +
                    difference_y ** 2)
            
            #Function calls to turn and travel
            self.spin(turn_angle)
            self.move_forward(distance)
        self.stop()
        
    def stop(self):
        """Stop the robot."""
        twist_msg = Twist()
        self._cmd_pub.publish(twist_msg)
    
    def move_forward(self, distance):
        """Move the robot forward in a straight line"""
        twist_msg = Twist()
        twist_msg.linear.x = self.linear_velocity 
        move_time = Duration( seconds = abs(distance/self.linear_velocity)) #Amount of time the robot needs to move for 
        rclpy.spin_once(self)
        start_time = self.get_clock().now() 
        
        # Loop.
        while rclpy.ok():
            rclpy.spin_once(self)
            # Check if traveled of given distance based on time.
            if self.get_clock().now() - start_time >= move_time:
                break
            # Publish message.
            self._cmd_pub.publish(twist_msg)
        # Traveled the required distance, stop.
        self.stop()
    
    def spin(self, spin_angle):
        """Turn the robot."""
        spin_time = Duration(seconds = abs(spin_angle / self.angular_velocity))  #Amount of time the robot needs to rotate for 
        start_time = self.get_clock().now() 
        angular_velocity = self.angular_velocity 
        
        #Conditional checking if the angular velocity should be negative 
        if spin_angle < 0: 
            angular_velocity = self.angular_velocity * -1
        
        #Loop to spin the robot 
        while rclpy.ok():
            rclpy.spin_once(self)
            if self.get_clock().now() - start_time >= spin_time:
                        break 
            twist_msg = Twist() 
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = angular_velocity
            self._cmd_pub.publish(twist_msg) 
def main(args=None):
    rclpy.init(args=args)
    
    q_node = QLearningAlgortihm()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(q_node)
    try:
        executor.spin()
    except KeyboardInterrupt: 
        pass
    finally:
        q_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

  