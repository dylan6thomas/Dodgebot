import numpy as np
import imageio

# Map metrics
GRID_HEIGHT = 10 
GRID_WIDTH = 10 
FREE = 0
OCCUPIED = 100

grid = np.full((GRID_HEIGHT, GRID_WIDTH), FREE)

# # Obstacles 
# grid[GRID_HEIGHT-2, GRID_WIDTH-2] = OCCUPIED
# grid[GRID_HEIGHT//2, GRID_WIDTH//2] = OCCUPIED
# grid[2,2] = OCCUPIED

# Wall section
grid[0, :] = OCCUPIED
grid[-1, :] = OCCUPIED
grid[:, 0] = OCCUPIED
grid[:, -1] = OCCUPIED

# Pgm that will be used with .world file
grid_pgm = np.where(grid == OCCUPIED, 0, 255).astype(np.uint8)
imageio.imwrite("qlearning_map.pgm", grid_pgm)