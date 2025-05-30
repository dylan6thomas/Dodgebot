import numpy as np
from collections import defaultdict
import math
import random
from obstacle_move import MovingMap, OCCUPIED, FREE, UNKNOWN, MAP_TOPIC, GRID_HEIGHT, GRID_WIDTH
from static_map import Map

from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

FREQUENCY = 20

# TODO: Make sure goal_dir part of state is relative to robot's direction
# TODO: Add heuristic to make training better
# TODO: Improved Q-Learning?


# Direction definitions
RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3
SAME = 4

# Map from robot_facing x goal_global → relative_dir
RELATIVE_DIRECTION_MAP = [
    [UP, LEFT,  DOWN, RIGHT],  # Robot facing RIGHT (0)
    [RIGHT, UP, LEFT,  DOWN],  # Robot facing UP (1)
    [DOWN, RIGHT, UP, LEFT],   # Robot facing LEFT (2)
    [LEFT, DOWN, RIGHT, UP],   # Robot facing DOWN (3)
]

# Action definitions
FORWARD = 1
BACKWARD = -1
STOP = 0
TURN_LEFT = 2
TURN_RIGHT = 3

# Reward definitions
GOAL_REWARD = 500
CLOSER_REWARD = 10
TIMESTEP_REWARD = -0.1
COLLISION_REWARD = -600

SCAN_HISTORY_LEN = 3

class State:
    def __init__(self, scan_action_history, goal_dir, scan_history_len=3):
        self.goal_dir = goal_dir
        self.scan_history_len = scan_history_len
        self.scan_action_history = scan_action_history  # will store tuples of (radius_map, action)
    
    def __eq__(self, other):
        if self.goal_dir != other.goal_dir or len(self.scan_action_pairs) != len(other.scan_action_pairs):
            return False
        return all(np.array_equal(a[0], b[0]) and a[1] == b[1]
               for a, b in zip(self.scan_action_pairs, other.scan_action_pairs))

    def __hash__(self):
        return hash((tuple((s.tobytes(), a) for s, a in self.scan_action_pairs), self.goal_dir))
  
class Location:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __eq__(self, value):
    return self.x == value.x and self.y == value.y
  
  def __str__(self):
      return str((self.x, self.y))
  def __repr__(self):
        return self.__str__()
  def square_distance(self, other):
      return (other.x - self.x)**2 + (other.y - self.y)**2
  
class RobotInfo:
    def __init__(self, location, direction, goal_loc, scan_radius, map_height, map_width, q_table = None):
        self.location = location
        self.direction = direction
        self.goal_loc = goal_loc
        self.scan_radius = scan_radius
        self.map_height = map_height
        self.map_width = map_width

        self.previous_map = None
        self.decay_map = np.zeros(())

        self.current_state = None

        if q_table is None:
            self.q_table = defaultdict(lambda: 0)
        else:
            self.q_table = q_table

        dummy_scan = np.full((2 * self.scan_radius + 1, 2 * self.scan_radius + 1), -1, dtype=np.int8)
        self.scan_action_history = [(dummy_scan.copy(), None) for _ in range(SCAN_HISTORY_LEN)]

  # This will have to be changed when we make it actually process laser scans
    # def get_state(self, map):
    #     updated_map = map.copy()

    #     if self.previous_map is not None:
    #         updated_map, updated_decay = self.update_change_tracker(self.previous_map, map, self.decay_map)
    #     else:
    #         updated_decay = np.zeros_like(updated_map)

    #     # Pad the map with UNKNOWN (-1) around edges
    #     padded_map = np.pad(updated_map, 
    #                         pad_width=self.scan_radius, 
    #                         mode='constant', 
    #                         constant_values=-1)

    #     # Adjust robot location for padded map coordinates
    #     cx = self.location.x + self.scan_radius
    #     cy = self.location.y + self.scan_radius

    #     # Extract fixed-size patch centered at robot
    #     radius = padded_map[
    #         cy - self.scan_radius : cy + self.scan_radius + 1,
    #         cx - self.scan_radius : cx + self.scan_radius + 1
    #     ]

    #     # Rotate patch according to direction (ensure direction is int in [0..3])
    #     radius_rotated = np.rot90(radius, k=self.direction)

    #     # Scan the rotated patch
    #     scan = self.scan(radius_rotated)

    #     goal_dir = self.get_goal_dir()

    #     self.current_state = State(scan, goal_dir)

    #     self.previous_map = updated_map
    #     self.decay_map = updated_decay
    #     return self.current_state
    def get_state(self, map):
        updated_map = map.copy()

        # Padding and rotating
        padded_map = np.pad(updated_map,
                            pad_width=self.scan_radius,
                            mode='constant',
                            constant_values=-1)

        cx = self.location.x + self.scan_radius
        cy = self.location.y + self.scan_radius

        radius = padded_map[
            cy - self.scan_radius:cy + self.scan_radius + 1,
            cx - self.scan_radius:cx + self.scan_radius + 1
        ]

        radius_rotated = np.rot90(radius, k=self.direction)
        scan = self.scan(radius_rotated)
        goal_dir = self.get_goal_dir()

        # Use previously recorded history
        history = self.scan_action_history[-(SCAN_HISTORY_LEN - 1):]
        history = list(history)
        history.append((scan.copy(), None))  # Add current scan with unknown action
        self.current_state = State(history, goal_dir, SCAN_HISTORY_LEN)

        self.current_state = State(history, goal_dir, SCAN_HISTORY_LEN)
        return self.current_state
    
    def scan(self, radius,  num_rays=120):
        scan = np.ones_like(radius) * -1 # start as unknown

        cx, cy = self.location.x, self.location.y

        h, w, = radius.shape

        for angle in np.linspace(0, 2 * np.pi, num=num_rays, endpoint=False):
            dx = math.cos(angle)
            dy = math.sin(angle)

            x, y = cx, cy  # start exactly at the robot's position

            for step in range(max(h, w) * 4):  # march far enough
                ix, iy = int(x), int(y)
                if not (0 <= ix < h and 0 <= iy < w):
                    break

                if radius[iy, ix] == 100:
                    break

                scan[iy, ix] = 0  # mark visible
                x += dx * 0.2  # smaller steps for smoother rays
                y += dy * 0.2 

        return scan

    def update_change_tracker(self, prev_map, curr_map, decay_map):
        """
        Track cells that changed from 100 → 0 for 3 iterations.

        - prev_map: map from previous iteration
        - curr_map: current scan result
        - decay_map: same shape as maps, initially all zeros
        Returns:
        - new_map: copy of curr_map with 200s for tracked disappearing obstacles
        - updated_decay_map: new decay values
        """
        new_map = curr_map.copy()
        updated_decay = np.maximum(decay_map - 1, 0)

        # Detect obstacle disappearance: 100 → 0
        disappear_mask = (prev_map == 100) & (curr_map == 0)
        updated_decay[disappear_mask] = 3  # restart countdown

        # Apply visual mark
        new_map[updated_decay > 0] = 200

        return new_map, updated_decay

    
    def get_goal_dir(self):
        dx = self.goal_loc.x - self.location.x
        dy = self.goal_loc.y - self.location.y

        if dx == 0 and dy == 0:
            return SAME  # already at goal

        # Determine global goal direction
        if abs(dx) > abs(dy):
            global_dir = RIGHT if dx > 0 else LEFT
        else:
            global_dir = DOWN if dy > 0 else UP

        # Convert to relative direction using lookup table
        relative_dir = RELATIVE_DIRECTION_MAP[self.direction][global_dir]
        return relative_dir

        
    
    def turn_dir(self, turn_dir):
        self._record_action(turn_dir)
        if turn_dir == TURN_LEFT:
            self.direction = (self.direction - 1) % 4
        elif turn_dir ==TURN_RIGHT:
            self.direction = (self.direction + 1) % 4
    
    def move(self, direction):
        self._record_action(direction)
        if direction == FORWARD:
            step = 1
        elif direction ==  BACKWARD:
            step = -1
        else:
            return

        if self.direction == RIGHT:
            self.location.x = max(0, min(self.location.x + step, self.map_width - 1))
        elif self.direction == LEFT:
            self.location.x = max(0, min(self.location.x - step, self.map_width - 1))
        elif self.direction == UP:
            self.location.y = max(0, min(self.location.y + step, self.map_height - 1))
        elif self.direction == DOWN:
            self.location.y = max(0, min(self.location.y - step, self.map_height - 1))
  
    def get_best_value(self):
        actions = [FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, STOP]
        return max([self.q_table[(self.current_state, action)] for action in actions])
    
    def _record_action(self, action):
        if self.current_state is not None:
            latest_scan = self.current_state.scan_action_history[-1][0]
            self.scan_action_history.append((latest_scan.copy(), action))
            if len(self.scan_action_history) > SCAN_HISTORY_LEN:
                self.scan_action_history.pop(0)

class QLearner:

    def __init__(self, robot_info, training_rounds, map_node, alpha=0.4, gamma=0.99, epsilon=1.0, decay=(1 - 1e-3)):
        self.robot_info = robot_info
        self.training_rounds = training_rounds
        self.map_node = map_node
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.closest_distance = self.robot_info.location.square_distance(robot_info.goal_loc)

    def play_round(self, start, goal):
        print(f"Playing round from {start} to {goal} with epsilon {self.epsilon}")
        
        self.reset(start, goal)

        step_limit = 500
        step = 0
        path = []
        total_reward = 0
        while(goal != self.robot_info.location and step < step_limit and self.map_node.grid[self.robot_info.location.y, self.robot_info.location.x] != 100):

            cur_map = self.map_node.grid
            state = self.robot_info.get_state(cur_map)
            action = self.get_exploration_action(self.epsilon)
            self.take_action(action)
            self.map_node.publish_map()
            reward = self.get_reward(cur_map)

            new_distance = self.robot_info.location.square_distance(goal)
            if(new_distance < self.closest_distance):
                reward += CLOSER_REWARD * (self.closest_distance - new_distance)
                self.closest_distance = new_distance

            total_reward += reward

            prev_value = self.robot_info.q_table[(state,action)]

            self.robot_info.q_table[(state,action)] = (1 - self.alpha) * prev_value + self.alpha * (reward + self.gamma * self.robot_info.get_best_value())

            step += 1
            path.append(Location(self.robot_info.location.x, self.robot_info.location.y))
            if reward == COLLISION_REWARD:
                print("Collided!")
                break
        self.epsilon *= self.decay
        if self.robot_info.location == goal:
            print("Found goal")
        print(f"Reward: {total_reward}")
        print(len(path))
        return path

    def get_exploration_action(self, epsilon):
        if random.random() > epsilon:
            return self.get_best_action()
        else:
            return random.choice([FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, STOP])
    
    def get_reward(self, map):
        if map[self.robot_info.location.y, self.robot_info.location.x] == OCCUPIED:
            return COLLISION_REWARD
        elif self.robot_info.location == self.robot_info.goal_loc:
            return GOAL_REWARD
        else: return TIMESTEP_REWARD

    def get_best_action(self):
        actions = [FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, STOP]
        state = self.robot_info.get_state(self.map_node.grid)
        return max(actions, key=lambda a: self.robot_info.q_table[(state, a)])
  
    def reset(self, start: Location, goal: Location):
        print(f"reseting location to {start}, {goal}")
        self.robot_info.location = Location(start.x, start.y)
        self.robot_info.goal = Location(goal.x, goal.y)

    def take_action(self, action):
        if action in [FORWARD, BACKWARD]:
            self.robot_info.move(action)
        elif action in [TURN_LEFT, TURN_RIGHT]:
            self.robot_info.turn_dir(action)

class QLearningNode(Node):
    def __init__(self, robot_info, map_node, start, goal, episodes=1000):
        super().__init__('q_learning_node')
        self.sub = self.create_subscription(
            OccupancyGrid, 
            MAP_TOPIC, 
            self.map_callback, 
            1)
        
        # Path publisher
        self.path_pub = self.create_publisher(Path, '/q_learning/path', 10)
        
        self.latest_map = None
        self.robot_info = robot_info
        self.learner = QLearner(robot_info, training_rounds=episodes, map_node=map_node)
        
        self.timer = self.create_timer(1.0 / FREQUENCY, self.train_loop)
        self.start = start
        self.goal = goal

    def map_callback(self, msg: OccupancyGrid):
        grid = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.latest_map = grid

    def publish_path(self, path_locs: list):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for loc in path_locs:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(loc.x + 0.5)
            pose.pose.position.y = float(loc.y + 0.5) # flip so that y=0 is at the top
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def train_loop(self):
        # if self.latest_map is None:
        #     return
        # self.grid = self.latest_map.copy()
        path = self.learner.play_round(self.start, self.goal)

        self.publish_path(path)
        

def main(args=None):
    rclpy.init(args=args)

    map_node = MovingMap()

    start = Location(0, 0)
    goal = Location(GRID_WIDTH - 3, GRID_HEIGHT - 2)

    # map_node = Map()
    robot_info = RobotInfo(
        location=start,
        direction=RIGHT,
        goal_loc=goal,
        scan_radius=2,
        map_height=GRID_HEIGHT,
        map_width=GRID_WIDTH
    )
    learner_node = QLearningNode(robot_info, map_node, start, goal)

    executor = rclpy.executors.MultiThreadedExecutor()
    # executor.add_node(map_node)
    executor.add_node(learner_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        map_node.destroy_node()
        learner_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
