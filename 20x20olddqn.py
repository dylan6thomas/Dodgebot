import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from qlearning import Location, RobotInfo, SCAN_HISTORY_LEN

from obstacle_move import MovingMap, OCCUPIED, FREE, UNKNOWN, MAP_TOPIC, GRID_HEIGHT, GRID_WIDTH
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path

import rclpy
from rclpy.node import Node

import torch.nn.functional as F

import os
import time

random.seed(0)

# Constants
FREQUENCY = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
MEMORY_SIZE = 10000
TARGET_UPDATE = 100
LEARNING_RATE = 3e-4
CHANGE_MAP_COUNT = 1

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

SCALE = 1

# GOAL_REWARD = 50     # Reduce to be less dominant
# CLOSER_REWARD = .5   # Give clear positive feedback
# FURTHER_REWARD = -.05 # Penalize for making the wrong move
# TIMESTEP_REWARD = -.01 # Encourage faster paths
# COLLISION_REWARD = -5.0
# STUCK_REWARD = -0.01    # Penalize being stuck
# REACHED_STEP_LIMIT_REWARD = -.8
GOAL_REWARD = 50     # Reduce to be less dominant
CLOSER_REWARD = 1.0   # Give clear positive feedback
FURTHER_REWARD = -.1 # Penalize for making the wrong move
TIMESTEP_REWARD = -.1 # Encourage faster paths
COLLISION_REWARD = -10.0
STUCK_REWARD = -0.01    # Penalize being stuck
REACHED_STEP_LIMIT_REWARD = -.8
STEP_LIMIT = 250
INITIAL_DISTANCE_LIMIT = 4

INC_DIFFICULTY_THRESH = 0.8

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
    
import numpy as np
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# https://arxiv.org/abs/1511.05952 Prioritiezed Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # how much prioritization is used (0 = uniform)
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.capacity = capacity
        self.pos = 0

    def push(self, *args, done=False):
        transition = Transition(*args)
        reward = args[3]

        if done:
            print("[PUSH] Prioritizing goal (done=True)")
            max_prio = self.priorities.max() if self.buffer else 1.0
            priority = min(max(5.0 * max_prio, 1.0), 100.0)
        else:
            priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Compute importance-sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        weights = np.array(weights, dtype=np.float32)
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)
    
    def reset(self):
        self.buffer = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)  # or reset sum tree structure
        self.pos = 0

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size=1024, output_size=5):
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
# .99 decay for 10x10
class DQNLearner:
    def __init__(self, robot_info, map_node, episodes, gamma=0.95, epsilon=1.0, epsilon_min=0.1, decay=0.995):
        self.robot_info = robot_info
        self.map_node = map_node
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.closest_distance = float('inf')

        self.buffer = PrioritizedReplayBuffer(MEMORY_SIZE)
        # input_size = ((2 * robot_info.scan_radius + 1) ** 2) * 2 + 5  # assuming 1 channel for goal_dir
        scan_dim = (2 * robot_info.scan_radius + 1) ** 2
        action_dim = len(ACTIONS)
        scan_hist_len = SCAN_HISTORY_LEN
        input_size = scan_hist_len * (scan_dim + action_dim) + action_dim + 1
        self.policy_net = DQN(input_size=input_size, output_size=len(ACTIONS)).to(DEVICE)
        self.target_net = DQN(input_size=input_size, output_size=len(ACTIONS)).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = torch.nn.SmoothL1Loss(beta=1.0) # Switch to huber loss
        self.steps_done = 0
        self.train_steps = 0

        self.episode_rewards = []
        self.running_avg_rewards = []
        self.running_avg_window = 50  # or any window size you prefer
        self.episode_losses = []
        self.running_avg_losses = []

        self.episode_successes = []
        self.success_window = 50
        self.success_window_avg = 0

        self.episode_count = 0

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.choice(ACTIONS)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state_tensor)
            return ACTIONS[q_values.argmax().item()]
        
    def select_action_boltzmann(self, state, eval_mode=False, temperature=1.0):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state_tensor).squeeze()

            if eval_mode:
                # In evaluation, just pick the greedy action
                return ACTIONS[q_values.argmax().item()]
            
            # Boltzmann (softmax) exploration
            # Apply softmax over Q-values divided by temperature
            scaled_qs = q_values / temperature
            action_probs = F.softmax(scaled_qs, dim=0).cpu().numpy()

            # Sample from the distribution
            action_index = np.random.choice(len(ACTIONS), p=action_probs)
            return ACTIONS[action_index]

    def optimize_model(self):
        if len(self.buffer) < BATCH_SIZE:
            return 0

        beta = 0.4  # can be annealed during training
        transitions, indices, weights = self.buffer.sample(BATCH_SIZE, beta)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.stack(batch.state)).to(DEVICE)
        action_batch = torch.LongTensor([ACTIONS.index(a) for a in batch.action]).unsqueeze(1).to(DEVICE)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(DEVICE)
        next_state_batch = torch.FloatTensor(np.stack(batch.next_state)).to(DEVICE)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(DEVICE)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)
        expected_values = reward_batch + self.gamma * next_state_values

        td_errors = (state_action_values - expected_values).detach().cpu().numpy().squeeze()
        new_priorities = np.abs(td_errors) + 1e-6  # small epsilon to avoid 0 priority

        self.buffer.update_priorities(indices, new_priorities)

        loss = (self.loss_fn(state_action_values, expected_values) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def train_episode(self, start, goal, step_limit):
        print(f"Training episode with start {start} and goal {goal} with epsilon {self.epsilon}")
        self.robot_info.location = Location(start.x, start.y)
        self.robot_info.goal_loc = Location(goal.x, goal.y)
        self.closest_distance = self.robot_info.location.square_distance(goal)
        self.prev_distance = self.closest_distance

        state_obj = self.robot_info.get_state(self.map_node.grid)
        state = self.get_full_state_vector(state_obj)
        total_reward = 0
        total_loss = 0
        steps = 0
        step_limit = step_limit

        path = [Location(self.robot_info.location.x, self.robot_info.location.y)]

        while self.robot_info.location != goal and steps < step_limit:
            # action = self.select_action(state)
            action = self.select_action_boltzmann(state, temperature=self.epsilon)
            self.take_action(action)
            self.map_node.publish_map()

            reward, done_flag, collided_flag = self.compute_reward(start, goal, steps)
            total_reward += reward
            next_state_obj = self.robot_info.get_state(self.map_node.grid)
            next_state = self.get_full_state_vector(next_state_obj)
            self.buffer.push(state, action, next_state, reward, done=done_flag)
            state = next_state

            total_loss += self.optimize_model()

            steps += 1

            path.append(Location(self.robot_info.location.x, self.robot_info.location.y))

            if collided_flag:
                break

        if self.episode_count > 200:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)
        self.prev_distance = self.robot_info.location.square_distance(self.robot_info.goal_loc)
        # print(path)

        self.episode_rewards.append(total_reward)

        # Compute running average
        if len(self.episode_rewards) >= self.running_avg_window:
            avg = np.mean(self.episode_rewards[-self.running_avg_window:])
            self.running_avg_rewards.append(avg)
        else:
            avg = np.mean(self.episode_rewards)
            self.running_avg_rewards.append(avg)

        print(f"Episode reward: {total_reward:.2f}, Running average reward: {avg:.2f}")

        self.episode_losses.append(total_loss)

        # Compute running average
        if len(self.episode_losses) >= self.running_avg_window:
            avg = np.mean(self.episode_losses[-self.running_avg_window:])
            self.running_avg_losses.append(avg)
        else:
            avg = np.mean(self.episode_losses)
            self.running_avg_losses.append(avg)

        print(f"Episode loss: {total_loss:.2f}, Running average loss: {avg:.2f}")

        self.episode_successes.append(int(done_flag))
        if len(self.episode_successes) >= self.success_window:
            self.success_window_avg = np.mean(self.episode_successes[-self.success_window:])

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        q_values = self.policy_net(state_tensor).squeeze().detach().cpu().numpy()
        mean_q = np.mean(q_values)
        max_q = np.max(q_values)
        min_q = np.min(q_values)

        print(f"[Q] Mean: {mean_q:.3f}, Max: {max_q:.3f}, Min: {min_q:.3f}")

        self.episode_count += 1
        return total_reward
    
    def get_path(self, start, goal, step_limit):
        print(f"Training episode with start {start} and goal {goal}")
        self.robot_info.location = Location(start.x, start.y)
        self.robot_info.goal_loc = Location(goal.x, goal.y)
        self.closest_distance = self.robot_info.location.square_distance(goal)
        self.prev_distance = self.closest_distance

        state_obj = self.robot_info.get_state(self.map_node.grid)
        state = self.get_full_state_vector(state_obj)
        steps = 0
        step_limit = step_limit
        done_flag = False

        path = [Location(self.robot_info.location.x, self.robot_info.location.y)]

        total_reward = 0

        while self.robot_info.location != goal and steps < step_limit:
            # action = self.select_action(state, eval_mode=True)
            action = self.select_action_boltzmann(state, eval_mode=True, temperature=self.epsilon)
            self.take_action(action)
            self.map_node.publish_map()

            reward, done_flag, collided_flag = self.compute_reward(start, goal, steps+1)
            total_reward += reward
            next_state_obj = self.robot_info.get_state(self.map_node.grid)
            next_state = self.get_full_state_vector(next_state_obj)
            state = next_state

            steps += 1

            path.append(Location(self.robot_info.location.x, self.robot_info.location.y))

            if collided_flag:
                print("SIMULATED PATH HIT OBSTACLE")
                break

        # print(path)
        print("SIMULATED PATH REWARD: ", total_reward)
        if done_flag: print("SIMULATED PATH REACHED GOAL")
        elif steps == STEP_LIMIT: print("SIMULATED PATH HIT STEP LIMIT")
        self.prev_distance = self.robot_info.location.square_distance(self.robot_info.goal_loc)
        return path
    
    def get_step(self):
        """
        Take one step towards the goal and return the new location.
        Assumes self.robot_info.location, self.robot_info.goal_loc, etc., have already been set.
        """
        state_obj = self.robot_info.get_state(self.map_node.grid)
        state = self.get_full_state_vector(state_obj)

        # Choose action (you can switch between select_action and select_action_boltzmann)
        action = self.select_action_boltzmann(state, eval_mode=True, temperature=self.epsilon)

        # Take the chosen action
        self.take_action(action)
        self.map_node.publish_map()


        self.prev_distance = self.robot_info.location.square_distance(self.robot_info.goal_loc)

        return Location(self.robot_info.location.x, self.robot_info.location.y)
    def compute_reward(self, start, goal, steps, robot_info=None):
        done_flag = False
        collided_flag = False
        reward_scale = 1 / math.sqrt(start.square_distance(goal))
        if robot_info is None:
            robot_info = self.robot_info

        if self.map_node.grid[robot_info.location.y, robot_info.location.x] == OCCUPIED:
            return COLLISION_REWARD, done_flag, collided_flag

        current_dist = robot_info.location.square_distance(robot_info.goal_loc)
        delta = self.prev_distance - current_dist

        reward = TIMESTEP_REWARD + delta * CLOSER_REWARD  # Scale appropriately

        if robot_info.location == robot_info.goal_loc:
            reward += GOAL_REWARD * reward_scale
            done_flag = True

        elif steps == STEP_LIMIT:
            reward += REACHED_STEP_LIMIT_REWARD

        elif delta == 0:
            reward += STUCK_REWARD

        self.prev_distance = current_dist
        return reward, done_flag, collided_flag
                
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
        # Flatten and stack all scans
        flat_scans = [scan.flatten().astype(np.float32) for scan, _ in state_obj.scan_action_history]
        scan_vector = np.concatenate(flat_scans) / 100

        # Optional: encode actions as one-hot if not None, else zeros
        action_vector = []
        for _, action in state_obj.scan_action_history:
            onehot = np.zeros(5, dtype=np.float32)  # FORWARD, BACKWARD, LEFT, RIGHT, STOP
            if action is not None:
                onehot[action] = 1.0
            action_vector.append(onehot)

        action_vector = np.concatenate(action_vector)

        # Goal direction as one-hot
        goal_dir_onehot = np.eye(5, dtype=np.float32)[state_obj.goal_dir]



        # Combine all
        return np.concatenate([scan_vector, action_vector, goal_dir_onehot, np.array([math.sqrt(self.robot_info.location.square_distance(self.robot_info.goal_loc))])])
    
    import os

    def save_checkpoint(self, save_path, extra_info=None):
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'train_steps': self.train_steps,
            'episode_count': self.episode_count,
            'episode_rewards': self.episode_rewards,
            'running_avg_rewards': self.running_avg_rewards,
            'episode_losses': self.episode_losses,
            'running_avg_losses': self.running_avg_losses,
            'episode_successes': self.episode_successes,
            'success_window_avg': self.success_window_avg,
            'replay_buffer': self.buffer,
            'extra_info': extra_info,
        }
        torch.save(checkpoint, save_path)
        print(f"[Checkpoint] Saved model and training state to {save_path}")

    def load_checkpoint(self, load_path):
        checkpoint = torch.load(load_path, map_location=DEVICE, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.train_steps = checkpoint['train_steps']
        self.episode_count = checkpoint['episode_count']
        self.episode_rewards = checkpoint['episode_rewards']
        self.running_avg_rewards = checkpoint['running_avg_rewards']
        self.episode_losses = checkpoint['episode_losses']
        self.running_avg_losses = checkpoint['running_avg_losses']
        self.episode_successes = checkpoint['episode_successes']
        self.success_window_avg = checkpoint['success_window_avg']
        self.buffer = checkpoint['replay_buffer']
        print(f"[Checkpoint] Loaded model and training state from {load_path}")

class DQNNode(Node):
    def __init__(self, robot_info, map_node, start, goal, episodes=1000, load_path=None, max_distance=-1, eval=False):
        super().__init__('dqn_node')
        self.path_pub = self.create_publisher(Path, '/dqn/path', 10)
        self.robot_info = robot_info
        self.map_node = map_node
        self.learner = DQNLearner(robot_info, map_node, episodes)
        self.change_map_count = 0
        self.start = start
        self.goal = goal
        if load_path is not None and os.path.exists(load_path) and max_distance > 0:
            self.learner.load_checkpoint(load_path)
            self.max_distance = max_distance
        else:
            self.max_distance = INITIAL_DISTANCE_LIMIT

        if not eval:
            self.timer = self.create_timer(1.0 / FREQUENCY, self.train_loop)
        else:
            # path = self.learner.get_path(self.start, self.goal, step_limit=self.max_distance * 2 * 4)
            self.publish_each_step()
            # print(path)
            
            

    def train_loop(self):
        reward = self.learner.train_episode(self.start, self.goal, self.max_distance*2 * 4)
        print(f"Episode complete. Total reward: {reward}")
        print(f"Node start: {self.start}")
        self.publish_path()

        print(f"Current success window average (len {len(self.learner.episode_successes)}): ", self.learner.success_window_avg)
        if len(self.learner.episode_successes) > self.learner.success_window and self.learner.success_window_avg >= INC_DIFFICULTY_THRESH:
            self.max_distance = min(GRID_WIDTH-1, GRID_HEIGHT-1, self.max_distance+1)
            print("Changed max distance to ", self.max_distance)
            self.learner.episode_successes = self.learner.episode_successes[-20:]
            self.learner.epsilon = max(self.learner.epsilon, 0.3)
            self.learner.decay = 0.997

            self.learner.save_checkpoint(f"checkpoints/model_distance_{self.max_distance}.pt", {
                'start': self.start,
                'goal': self.goal,
                'distance': self.max_distance
            })

        self.change_map_count += 1
        while self.change_map_count >= CHANGE_MAP_COUNT or self.map_node.grid[self.goal.y, self.goal.y] == OCCUPIED:
            print("Changing map")
            self.goal = Location(random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            self.start = Location(random.randint(max(0, self.goal.x - self.max_distance), min(GRID_WIDTH - 1, self.goal.x + self.max_distance)),
                                  random.randint(max(0, self.goal.y - self.max_distance), min(GRID_HEIGHT - 1, self.goal.y + self.max_distance)))
            while(self.start == self.goal):
                self.start = Location(random.randint(max(0, self.goal.x - self.max_distance), min(GRID_WIDTH - 1, self.goal.x + self.max_distance)),
                                  random.randint(max(0, self.goal.y - self.max_distance), min(GRID_HEIGHT - 1, self.goal.y + self.max_distance)))
            self.map_node._init_map()
            # self.learner.buffer.reset()
            self.change_map_count = 0

    def publish_path(self):
        path = self.learner.get_path(self.start, self.goal, self.max_distance*2 * 4)

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

    def publish_each_step(self):
        path = []
        step = self.learner.get_step()
        while step != Location(18, 18):
            path.append(step)

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

            step = self.learner.get_step()
            time.sleep(0.5)

        print("FOUND GOAL")
        path.append(step)

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

def main(args=None):
    rclpy.init(args=args)

    map_node = MovingMap()

    start = Location(1, 1)
    goal = Location(18, 18)

    robot_info = RobotInfo(
        location=start,
        direction=RIGHT,
        goal_loc=goal,
        scan_radius=5,
        # scan_radius=4
        map_height=GRID_HEIGHT,
        map_width=GRID_WIDTH
    )

    dqn_node = DQNNode(robot_info, map_node, start, goal, load_path="checkpoints/20x20_3obs/model_distance_18.pt", max_distance=18, eval=True)
    # dqn_node = DQNNode(robot_info, map_node, start, goal)

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