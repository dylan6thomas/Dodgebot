import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from qlearning import Location, RobotInfo

from obstacle_move import MovingMap, OCCUPIED, FREE, UNKNOWN, MAP_TOPIC, GRID_HEIGHT, GRID_WIDTH
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path

import rclpy
from rclpy.node import Node

# Constants
FREQUENCY = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
LEARNING_RATE = 1e-3

# Directions
RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3
SAME = 4

# Actions
FORWARD = 1
BACKWARD = -1
STOP = 0
TURN_LEFT = 2
TURN_RIGHT = 3
ACTIONS = [FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, STOP]

# Rewards
GOAL_REWARD = 500
CLOSER_REWARD = 10
TIMESTEP_REWARD = -2
COLLISION_REWARD = -600

DIR_TO_VEC = {
    RIGHT: (1, 0),
    UP: (0, 1),
    LEFT: (-1, 0),
    DOWN: (0, -1),
}

# Classes from your code: Location, RobotInfo, etc. — reuse as-is
# Add just the DQN-specific components below

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=5):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

class DQNLearner:
    def __init__(self, robot_info, map_node, episodes, gamma=0.99, epsilon=1.0, epsilon_min=0.1, decay=0.995):
        self.robot_info = robot_info
        self.map_node = map_node
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.closest_distance = float('inf')

        self.buffer = ReplayBuffer(MEMORY_SIZE)
        input_size = (2 * robot_info.scan_radius + 1) ** 2 + 5  # assuming 1 channel for goal_dir
        self.policy_net = DQN(input_size=input_size, output_size=len(ACTIONS)).to(DEVICE)
        self.target_net = DQN(input_size=input_size, output_size=len(ACTIONS)).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.steps_done = 0
        self.train_steps = 0

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.choice(ACTIONS)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state_tensor)
            return ACTIONS[q_values.argmax().item()]

    def optimize_model(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        transitions = self.buffer.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.stack(batch.state)).to(DEVICE)
        action_batch = torch.LongTensor([ACTIONS.index(a) for a in batch.action]).unsqueeze(1).to(DEVICE)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(DEVICE)
        next_state_batch = torch.FloatTensor(np.stack(batch.next_state)).to(DEVICE)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)

        expected_values = reward_batch + self.gamma * next_state_values
        loss = self.loss_fn(state_action_values, expected_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_episode(self, start, goal):
        print(f"Training episode with start {start} and goal {goal}")
        self.robot_info.location = Location(start.x, start.y)
        self.robot_info.goal_loc = goal
        self.closest_distance = self.robot_info.location.square_distance(goal)

        state_obj = self.robot_info.get_state(self.map_node.grid)
        state = self.get_full_state_vector(state_obj)
        total_reward = 0
        steps = 0
        step_limit = 500

        while self.robot_info.location != goal and steps < step_limit:
            action = self.select_action(state)
            self.take_action(action)

            reward = self.compute_reward()
            total_reward += reward
            next_state_obj = self.robot_info.get_state(self.map_node.grid)
            next_state = self.get_full_state_vector(next_state_obj)
            self.buffer.push(state, action, next_state, reward)
            state = next_state

            self.buffer.push(state, action, next_state, reward)
            self.optimize_model()

            state = next_state
            steps += 1

            if reward == COLLISION_REWARD:
                break

        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)
        return total_reward
    
    def get_path(self, start, goal):
        print(f"Simulating path from {start} to {goal}")
        
        # Clone robot and set up initial state
        sim_robot = self.robot_info.copy()
        sim_robot.location = Location(start.x, start.y)
        sim_robot.goal_loc = goal

        path = [Location(start.x, start.y)]
        steps = 0
        step_limit = 500

        state_obj = sim_robot.get_state(self.map_node.grid)
        state = self.get_full_state_vector(state_obj)

        while sim_robot.location != goal and steps < step_limit:
            action = self.select_action(state, eval_mode=True)  # Use greedy policy
            self.take_action(action, sim_robot)

            reward = self.compute_reward(sim_robot)
            if reward == COLLISION_REWARD:
                break

            path.append(Location(sim_robot.location.x, sim_robot.location.y))

            next_state_obj = sim_robot.get_state(self.map_node.grid)
            state = self.get_full_state_vector(next_state_obj)
            steps += 1

        return path

    def compute_reward(self, robot_info=None):
        map = self.map_node.grid
        if robot_info is None:
            robot_info = self.robot_info
        if map[robot_info.location.y, robot_info.location.x] == OCCUPIED:
            return COLLISION_REWARD
        elif robot_info.location == robot_info.goal_loc:
            print(f"Found goal {robot_info.goal_loc}, our location is {robot_info.location}")
            return GOAL_REWARD
        else:
            new_distance = robot_info.location.square_distance(robot_info.goal_loc)
            if new_distance < self.closest_distance:
                self.closest_distance = new_distance
                return CLOSER_REWARD + TIMESTEP_REWARD
            return TIMESTEP_REWARD

    def take_action(self, action, robot_info = None):
        if robot_info is None:
            if action in [FORWARD, BACKWARD]:
                self.robot_info.move(action)
            elif action in [TURN_LEFT, TURN_RIGHT]:
                self.robot_info.turn_dir(action)
        else:
            if action in [FORWARD, BACKWARD]:
                robot_info.move(action)
            elif action in [TURN_LEFT, TURN_RIGHT]:
                robot_info.turn_dir(action)

    def get_full_state_vector(self, state_obj):
        flat_map = state_obj.radius_map.flatten().astype(np.float32)
        goal_dir_onehot = np.eye(5, dtype=np.float32)[state_obj.goal_dir]
        return np.concatenate([flat_map, goal_dir_onehot])

class DQNNode(Node):
    def __init__(self, robot_info, map_node, start, goal, episodes=1000):
        super().__init__('dqn_node')
        self.path_pub = self.create_publisher(Path, '/dqn/path', 10)
        self.robot_info = robot_info
        self.map_node = map_node
        self.learner = DQNLearner(robot_info, map_node, episodes)
        self.start = start
        self.goal = goal

        self.timer = self.create_timer(1.0 / FREQUENCY, self.train_loop)

    def train_loop(self):
        reward = self.learner.train_episode(self.start, self.goal)
        print(f"Episode complete. Total reward: {reward}")
        self.publish_path()

    def publish_path(self):
        path = self.learner.get_path(self.start, self.goal)

        # Publish path as ROS message
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for p in path:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(p.x + 0.5)
            pose.pose.position.y = float(p.y + 0.5)
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def simulate_action(self, loc, direction, action):
        """
        Given a current location and direction, simulate the result of an action.
        Returns a (Location, direction) tuple. If the move would go out of bounds, location remains unchanged.
        """
        map_height = self.map_node.grid.shape[0]
        map_width = self.map_node.grid.shape[1]

        new_loc = loc
        new_direction = direction

        if action == FORWARD:
            dx, dy = DIR_TO_VEC[direction]
            candidate_loc = Location(loc.x + dx, loc.y + dy)

            # Check if candidate location is within bounds
            if 0 <= candidate_loc.x < map_width and 0 <= candidate_loc.y < map_height:
                new_loc = candidate_loc

        elif action == BACKWARD:
            dx, dy = DIR_TO_VEC[direction]
            candidate_loc = Location(loc.x - dx, loc.y - dy)

            if 0 <= candidate_loc.x < map_width and 0 <= candidate_loc.y < map_height:
                new_loc = candidate_loc

        elif action == TURN_LEFT:
            new_direction = (direction - 1) % 4

        elif action == TURN_RIGHT:
            new_direction = (direction + 1) % 4

        elif action == STOP:
            pass  # do nothing

        else:
            raise ValueError(f"Unknown action: {action}")

        return new_loc, new_direction

def main(args=None):
    rclpy.init(args=args)

    map_node = MovingMap()

    start = Location(0, 0)
    goal = Location(GRID_WIDTH - 3, GRID_HEIGHT - 2)

    robot_info = RobotInfo(
        location=start,
        direction=RIGHT,
        goal_loc=goal,
        scan_radius=2,
        map_height=GRID_HEIGHT,
        map_width=GRID_WIDTH
    )

    dqn_node = DQNNode(robot_info, map_node, start, goal)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(dqn_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        dqn_node.destroy_node()
        map_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()