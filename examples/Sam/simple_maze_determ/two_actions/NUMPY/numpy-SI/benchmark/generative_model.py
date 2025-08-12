import numpy as np
from pymdp import utils


MAZE = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1]
])

LOCATIONS = np.array([
    [ 0,  9, 18, 27, 36, 45, 54, 63, 72],
    [ 1, 10, 19, 28, 37, 46, 55, 64, 73],
    [ 2, 11, 20, 29, 38, 47, 56, 65, 74],
    [ 3, 12, 21, 30, 39, 48, 57, 66, 75],
    [ 4, 13, 22, 31, 40, 49, 58, 67, 76],
    [ 5, 14, 23, 32, 41, 50, 59, 68, 77],
    [ 6, 15, 24, 33, 42, 51, 60, 69, 78],
    [ 7, 16, 25, 34, 43, 52, 61, 70, 79],
    [ 8, 17, 26, 35, 44, 53, 62, 71, 80]
])


A0 = np.zeros((81, 81))
A1 = np.zeros((2, 81))
A = np.array([A0, A1], dtype=object)
# A-matrices
A[0] = np.eye(81)
MAZE_vec = MAZE.T.flatten()
A[1] = np.vstack([(1 - MAZE_vec), MAZE_vec])

# B-matrices
B = np.empty((1,), dtype=object)
B[0] = np.zeros((81, 81, 2)) 

def construct_B_matrices(maze):
    nrows, ncols = maze.shape
    nstates = nrows * ncols
    nactions = 2  # UP, RIGHT

    B = np.zeros((nstates, nstates, nactions), dtype=float)

    for row in range(nrows):
        for col in range(ncols):
            s = col * nrows + row  # Convert 2D index to 1D index

            if maze[row, col] == 1:
                # Walls: self-loop for all actions
                for a in range(nactions):
                    B[s, s, a] = 1.0
                continue

            # --- UP (action 0) ---
            if row > 0 and maze[row - 1, col] == 0:
                s2 = col * nrows + (row - 1)
                B[s2, s, 0] = 1.0
            else:
                B[s, s, 0] = 1.0

            # --- RIGHT (action 1) ---
            if col < ncols - 1 and maze[row, col + 1] == 0:
                s2 = (col + 1) * nrows + row
                B[s2, s, 1] = 1.0
            else:
                B[s, s, 1] = 1.0

    return B

B[0] = construct_B_matrices(MAZE)

# C-vectors
C_vector = utils.obj_array_uniform([81,2]) 
C = np.empty((2,), dtype=object)
C[0] = np.array([
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  # column 1
    -1.0, -1.0, -1.0, -1.0, -0.4, -0.6, -0.7, -0.8, -1.0,  # column 2
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.75, -1.0, # column 3
    -1.0, -1.0, -1.0,  0.0, -0.2, -0.4, -0.6, -0.7, -1.0, # column 4
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.65, -1.0,  # column 5
    -1.0, -1.0,  0.4,  0.2,  0.05, -0.15, -0.35, -0.55, -1.0, # column 6
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.4, -1.0,  # column 7
    -1.0,  1.0,  0.65, 0.4,  0.25, 0.15, 0.0, -0.2, -1.0, # column 8
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0   # column 9
    ])
C[1] = np.zeros(2)

# D-vector
D = np.empty((1,), dtype=object)
D[0] = utils.onehot(17, 81)