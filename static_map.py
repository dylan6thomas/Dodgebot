#!/usr/bin/env python

# import of relevant libraries.
import numpy as np
import rclpy # module for ROS APIs
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header


# Constants.
NODE_NAME = "static_obstacle_map"
DEFAULT_MAP_TOPIC = 'map'
USE_SIM_TIME = True
TF_ODOM_LINK = 'odom'

RESOLUTION = 0.05 # meters 
GRID_HEIGHT = 10
GRID_WIDTH = 10 
FREQUENCY = 10 #Hz.

class Map(Node): 
    def __init__(self, node_name=NODE_NAME, context=None):
        super().__init__(node_name, context=context)
    
        # Workaround not to use roslaunch
        use_sim_time_param = rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            USE_SIM_TIME
        )
        self.set_parameters([use_sim_time_param])
        
        # Setting up the publisher to send map messages. 
        self._map_pub = self.create_publisher(OccupancyGrid, DEFAULT_MAP_TOPIC, 1)
        
        # Initializing grid with default values 
        self.resolution = RESOLUTION 
        self.height = GRID_HEIGHT
        self.width = GRID_WIDTH
        self.origin_x = -(GRID_WIDTH * RESOLUTION ) / 2
        self.origin_y = -(GRID_HEIGHT * RESOLUTION) / 2
        self.grid = np.full((self.height, self.width), 0)
        
        # Rate at which to operate the while loop.
        self.rate = self.create_rate(FREQUENCY)

    def publish_map(self):
        """Function to publish the grid based on the grid data"""
        
        # Add the obstacles to the map 
        self.add_obstacles() 
        
        #Intializing all the grid values to publish it 
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = TF_ODOM_LINK
        
        grid.info.resolution = self.resolution
        grid.info.width = self.width
        grid.info.height = self.height
        grid.info.origin.position.x = self.origin_x
        grid.info.origin.position.y = self.origin_y
        
        grid.info.origin.orientation.x = 0.0
        grid.info.origin.orientation.y = 0.0
        grid.info.origin.orientation.z = 0.0
        grid.info.origin.orientation.w = 1.0
        
        grid.data = self.grid.flatten().tolist()
        self._map_pub.publish(grid)
        
    def add_obstacles(self):
        """Function to add static obstacles to the map"""  
        
        def rectange_dimension(row,col):
            self.draw_rectangle(row, col)
            self.draw_rectangle(row - 1, col)
            self.draw_rectangle(row - 1, col - 1)
            self.draw_rectangle(row, col - 1)
        
        # Right Obstacle 
        row,col = self.height - 2, self.width - 2
        rectange_dimension(row,col)
        
        # Middle Obstacle
        row, col = self.height // 2, self.width // 2
        rectange_dimension(row,col)
        
        # Left Obstacle 
        row, col = 2, 2
        rectange_dimension(row,col)
          
    def draw_rectangle(self, grid_row, grid_col):
        """Function change grid values based on the obstacles""" 
        self.grid[grid_row, grid_col] = 100
    
def main(args=None):
    # 1st. initialization of node.
    rclpy.init(args=args)

    # Initialization of the class for the create map
    map = Map()
        
    while rclpy.ok():
        map.publish_map()
        rclpy.spin_once(map)
            
    rclpy.shutdown()

if __name__ == "__main__":
    main()
