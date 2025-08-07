import jax
import jax.numpy as jnp
import numpy as np
import jax.random as random

def onehot(index, size):
    return jnp.eye(size)[index]

class SIMazeEnv:
    ACTIONS = {
        0: (-1, 0),  # UP
        1: (0, 1),   # RIGHT
    }

    def __init__(self, maze, preferences, start_state):
        self.maze = maze
        self.preferences = preferences
        self.nrows, self.ncols = maze.shape
        self.agent_row, self.agent_col = self._get_coordinates(start_state)

    def _get_coordinates(self, start_state):
        row = start_state % self.nrows
        col = start_state // self.nrows
        return row, col

    def _get_state_index(self):
        return self.agent_col * self.nrows + self.agent_row

    def step(self, action):
        dr, dc = self.ACTIONS[int(action[0])]

        r_new = self.agent_row + dr
        c_new = self.agent_col + dc

        if 0 <= r_new < self.nrows and 0 <= c_new < self.ncols and self.maze[r_new, c_new] == 0:
            self.agent_row, self.agent_col = r_new, c_new

        return self.get_observation()

    def get_observation(self):
        pos_index = self._get_state_index()

        valence = 0 if np.random.rand() < self.preferences[self.agent_row, self.agent_col] else 1

        # Create one-hot encoded arrays with batch dimensions
        obs_position = jnp.expand_dims(onehot(pos_index, self.nrows * self.ncols), axis=0)  
        obs_context = jnp.expand_dims(onehot(0, 2), axis=0)  
        obs_valence = jnp.expand_dims(onehot(valence, 2), axis=0)

        return [obs_position, obs_context, obs_valence]