import numpy as np
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
import time 
import random
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from obstacle_move import MovingMap, OCCUPIED, FREE, UNKNOWN, MAP_TOPIC, GRID_HEIGHT, GRID_WIDTH

# Q-learning parameters 
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.95
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
DECAY_RATE = 0.995
EPISODES = 3000
MAX = 5000
# Actions 
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4
# REWARDS
GOAL = 1000
COLLISION = -50
TIMESTEP_REWARD = -1
CLOSER_REWARD = 5.0
STUCK_REWARD = -20.0

FREQUENCY = 20

class QLearningAlgorithm(Node):
    def __init__(self,  map_node, node_name='q_learning', context=None):
       super().__init__(node_name, context=context)
       
       # Algorithm parameters 
       self.q_table = np.zeros((GRID_HEIGHT * GRID_WIDTH, 5))
       self.grid = None
       self.start = (1,1)
       self.goal = (18,18)
       self.actions = [UP, DOWN, LEFT, RIGHT, STAY]
       self.current_episode = 0
       self.map_node = map_node
       self.grid = self.map_node.grid
       self.epsilon = INITIAL_EPSILON
       
       # Path publisher 
       self.path_pub = self.create_publisher(Path, '/q_learning/path', 10)
       
       #Timer
       self.timer = self.create_timer(1.0 / FREQUENCY, self.qlearn_function)
       
       self.training_time = time.time()
        
        
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
    
    def reward_calc(self, cur_state, prev_state):
        """Get the reward"""
        
        # If at goal return goal reward
        if cur_state == self.goal: 
            return GOAL
        
        # If a obstacle return collsion value 
        if not self.is_state_valid(cur_state):
            return COLLISION
        
        # Get point distances from goal 
        prev_dist = abs(prev_state[0] - self.goal[0]) + abs(prev_state[1] - self.goal[1]) 
        cur_dist =  abs(cur_state[0] - self.goal[0]) + abs(cur_state[1] - self.goal[1])
        diff = prev_dist - cur_dist
        
        # Increase reward for each timestep and getting closer and the differrenc
        reward = TIMESTEP_REWARD + CLOSER_REWARD * diff
        
        # Check if stuck in the place 
        if diff == 0:
            reward += STUCK_REWARD
        
        # Harsh penalty for being stuck to really discourage and fix reward system
        if cur_state == prev_state:
            reward += STUCK_REWARD * 2
            
        return reward
    
    
    def qlearn_function(self):
        """Function to run the q learning alg"""
        
        self.map_node.publish_map()
        self.grid = self.map_node.grid
        
        # Check if the episode count is complete 
        if self.current_episode >= EPISODES:
            self.timer.cancel()
            print("Done training")
            path = self.get_best_path()
            if path:
                # Get metrics if a best path is found 
                path_reward = self.calculate_path_reward(path)
                path_len = len(path) - 1
                total_time = time.time() - self.training_time
                self.publish_path(path)
                print(f"Total episode count: {self.current_episode}")
                print(f"Path reward for episodes: {path_reward}")
                print(f"Best path length: {path_len}")
                print(f"Training time: {total_time}")
            else:
                print("No valid path")
            return 
        
        state = self.start 
        state_index = self.index_of_state(state) 
        episode_reward = 0 

        for iteration in range(MAX):
            
            # Action calculation based on epsilon 
            if random.random() < self.epsilon:
                action = random.randint(UP,STAY) 
            else: 
                action = np.argmax(self.q_table[state_index])
    
            # Get new state and calcuate the reward for it 
            new_state = self.get_new_state(state, action)
            new_state_index = self.index_of_state(new_state)
            reward = self.reward_calc(new_state, state)
            episode_reward += reward
            
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
        
        # Epsilon decay
        self.epsilon = max(MIN_EPSILON, INITIAL_EPSILON * DECAY_RATE)
        
        # Periodicaly printing the episode for metrics 
        if self.current_episode % 100 == 0:
            print(f"Episode {self.current_episode}, Reward: {episode_reward:.1f}")
            
        self.current_episode += 1
                
    def get_best_path(self): 
        """Function to get the best q-table path""" 
        
        # Initiliaze path with the start pose and get starting params 
        best_path = [self.start]
        state = self.start 
        state_index = self.index_of_state(state) 
        count = 0
        visited = set()
        
        # Loop until the state is at the goal 
        while state != self.goal:
            
            # Conditional prevent looping 
            if state in visited: 
                print("Loop in path") 
                break
            visited.add(state)
            
            # Get the action for the state and find the next state based on that
            action = np.argmax(self.q_table[state_index])
            new_state = self.get_new_state(state, action)
            
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
    
    def calculate_path_reward(self, path): 
        """Function to get the reward of the path"""
        total_reward = 0 
        
        # Find the reward for each state then get the total reward to use for the path metric 
        for i in range(1, len(path)):
            cur_state = path[i]
            prev_state = path[i-1]
            reward = self.reward_calc(cur_state, prev_state)
            total_reward += reward
            if cur_state == self.goal:
                break
        return total_reward
    
    def publish_path(self, path_list): 
        """Function to publish the poses for the path"""
        
        print(f"Best path list: {path_list}")
        path_msg = Path()
        path_msg.header.frame_id = MAP_TOPIC
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for point in path_list:
            pose = PoseStamped()
            pose.header.frame_id = MAP_TOPIC
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(point[1] + 0.5)
            pose.pose.position.y = float(point[0] + 0.5) 
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            
        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    
    map_node = MovingMap()
    q_node = QLearningAlgorithm(map_node)
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(map_node)
    executor.add_node(q_node)
    try:
        executor.spin()
    except KeyboardInterrupt: 
        pass
    finally:
        q_node.destroy_node()
        map_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

  