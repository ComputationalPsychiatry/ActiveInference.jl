import jax
import jax.numpy as jnp
import numpy as np
import jax.random as random
import seaborn as sns

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

PREFERENCES = np.array([
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.700, 0.000, 0.825, 0.000],
    [0.000, 0.000, 0.000, 0.501, 0.000, 0.600, 0.000, 0.700, 0.000],
    [0.000, 0.300, 0.000, 0.400, 0.000, 0.525, 0.000, 0.625, 0.000],
    [0.000, 0.200, 0.000, 0.300, 0.000, 0.425, 0.000, 0.575, 0.000],
    [0.000, 0.150, 0.000, 0.200, 0.000, 0.325, 0.000, 0.501, 0.000],
    [0.000, 0.100, 0.125, 0.150, 0.175, 0.225, 0.300, 0.400, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
])

MAZE_vec = MAZE.T.flatten()

A0 = jnp.eye(81)  # Observation modality 0 as identity matrix
A1 = jnp.vstack([(1 - MAZE_vec), MAZE_vec])  # Observation modality 1

A = [A0, A1]

def construct_B_matrices_jax(maze):
    nrows, ncols = maze.shape
    nstates = nrows * ncols
    nactions = 2  # Only UP (0) and RIGHT (1)

    B = jnp.zeros((nstates, nstates, nactions))

    def state_idx(row, col):
        return col * nrows + row

    for row in range(nrows):
        for col in range(ncols):
            s = state_idx(row, col)

            if maze[row, col] == 1:  # Walls
                B = B.at[s, s, :].set(1.0)
                continue

            # --- UP ---
            if row > 0 and maze[row - 1, col] == 0:
                s2 = state_idx(row - 1, col)
                B = B.at[s2, s, 0].set(1.0)
            else:
                B = B.at[s, s, 0].set(1.0)

            # --- RIGHT ---
            if col < ncols - 1 and maze[row, col + 1] == 0:
                s2 = state_idx(row, col + 1)
                B = B.at[s2, s, 1].set(1.0)
            else:
                B = B.at[s, s, 1].set(1.0)

    return B

B = [construct_B_matrices_jax(MAZE)]

C = [
     jnp.array([
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  # column 1
    -1.0, -1.0, -1.0, -1.0, -0.4, -0.6, -0.7, -0.8, -1.0,  # column 2
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.75, -1.0, # column 3
    -1.0, -1.0, -1.0,  0.0, -0.2, -0.4, -0.6, -0.7, -1.0, # column 4
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.65, -1.0,  # column 5
    -1.0, -1.0,  0.4,  0.2,  0.05, -0.15, -0.35, -0.55, -1.0, # column 6
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.4, -1.0,  # column 7
    -1.0,  1.0,  0.65, 0.4,  0.25, 0.15, 0.0, -0.2, -1.0, # column 8
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0   # column 9
    ]),
    jnp.zeros(2),
    ]

def onehot(index, size):
    return jnp.eye(size)[index]

D = [onehot(17, 81)]