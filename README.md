# Dodgebot

### `obstacle_move.py`
Implements our map class with dynamic obstacle movements

### `qln_dynamic.py`
This script implements a Q-learning algorithm allowing a robot to learn the optimal path from start to goal while avoiding moving obstacles. The script maintains and updates a Q-table based on rewards and publishes the highest reward path based on the policy. 

To run:

Terminal 1: 

`docker compose up`

Terminal 2:

`docker compose exec ros bash`

Clear gazebo on docker 

`ros2 run rviz2 rviz2`

Then add the /moving_map map topic and /path path topic 

Terminal 3:

`docker compose exec ros bash`

`ros2 run tf2_ros static_transform_publisher 5 5 0 0 0 0 map odom`

Terminal 4

`docker compose exec ros bash`

`ros2 launch stage_ros2 stage.launch.py world:=/root/catkin_ws/src/Dodgebot/StageMap/qlearning_map`

Terminal 5

`docker compose exec ros bash`

`Python3 move_stage_obstacles.py`

Terminal 6

`docker compose exec ros bash`

`python3 qln_dynamic.py`



### `dqn.py`
Implementation of DQN training on a dynamic map

To run:

Terminal 1: 

`docker compose up`

Terminal 2:

`docker compose exec ros bash`

Clear gazebo on docker 

`ros2 run rviz2 rviz2`

Then add the /moving_map map topic and /path path topic 

Terminal 3:

`docker compose exec ros bash`

`ros2 run tf2_ros static_transform_publisher 5 5 0 0 0 0 map odom`

This transformation will change depending on the map size (half of width and height)

Terminal 4

`docker compose exec ros bash`

`python3 dqn.py`

### `robot_info.py`
Contains State and RobotInfo classes that are used in the processing of robot actions during DQN training and inference

### `StageMap/create_stage_map.py`
This script generates the stage map .pgm file. It also sets map parameters (grid height and width) and initializes all cells as free except the wall areas.

### `StageMap/move_stage_obstacles.py`
This script creates a ROS node to simulate movement of three virtual obstacles along the x-axis within a threshold of bounds. Obstacle positions and velocities are continuously published and used by qln_dynamic.py for real-time updates.

To run:

`Python3 move_stage_obstacles.py`

### `StageMap/qlearning_map.world`
Defines the simulation environment using the .pgm bitmap layout. It includes the rosbot and three obstacles, whose positions can be edited in this file.
