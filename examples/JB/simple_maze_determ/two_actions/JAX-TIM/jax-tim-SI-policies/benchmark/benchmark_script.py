#import os
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import os
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from pymdp.agent import Agent
from pymdp import control
from pymdp.envs.env import Env
from tom.planning.si import rollout

import pandas as pd

import debugger
import time

folder = "./two_actions/JAX-TIM/jax-tim-SI-policies"


def simu(horizon):
    
    os.makedirs(folder, exist_ok=True) 

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


    c0 = jnp.array([
        [-1.  , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  ],
        [-1.  , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  ,  1.  , -1.  ],
        [-1.  , -1.  , -1.  , -1.  , -1.  ,  0.4 , -1.  ,  0.65, -1.  ],
        [-1.  , -1.  , -1.  ,  0.  , -1.  ,  0.2 , -1.  ,  0.4 , -1.  ],
        [-1.  , -0.4 , -1.  , -0.2 , -1.  ,  0.05, -1.  ,  0.25, -1.  ],
        [-1.  , -0.6 , -1.  , -0.4 , -1.  , -0.15, -1.  ,  0.15, -1.  ],
        [-1.  , -0.7 , -1.  , -0.6 , -1.  , -0.35, -1.  ,  0.  , -1.  ],
        [-1.  , -0.8 , -0.75, -0.7 , -0.65, -0.55, -0.4 , -0.2 , -1.  ],
        [-1.  , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  ]
    ])

    c0 = c0.T.flatten()

    C = [
        c0,
        jnp.zeros(2),
        ]

    def onehot(index, size):
        return jnp.eye(size)[index]

    D = [onehot(17, 81)]

    print([a.shape for a in A], [b.shape for b in B], [c.shape for c in C], [d.shape for d in D])

    # setting up the agents
    # default gamma value in pymdp is 16.0, reduce gamma to avoid the info gain trap
    agent = Agent(A=A, B=B, C=C, D=D, apply_batch=True, learn_A=False, learn_B=False, gamma=1.0, sampling_mode="full")

    params = {
        "A": agent.A,
        "B": agent.B,
        "D": agent.D,
    }
    dependencies = {
        "A": agent.A_dependencies,
        "B": agent.B_dependencies,
    }

    env = Env(params, dependencies)
    key = jr.PRNGKey(11)
    T = 10
    start_time = time.perf_counter()

    last, info, env = rollout(
        agent,
        env,
        T,
        key,
        max_depth=horizon,
        max_nodes = 1000,
        max_branching = 10,
        policy_prune_threshold = 1/ 16,
        observation_prune_threshold = 1 / 16,
        entropy_stop_threshold = 0.5,
        efe_stop_threshold = 5,
        kl_threshold= 1e-2,
        prune_penalty = 512,
        )

    print("\nobservations loc= \n{}\n".format(info["observation"][0][:,0,0]))
    print("\nobservations wall= \n{}\n".format(info["observation"][1][:,0,0]))

    #print(info)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"\nFunction execution time: {elapsed_time:.4f} seconds\n")


    qs = info["qs"][0][:,0,:]
    q_pi = info["qpi"][:,1,:][-1,:].reshape((1, -1))
    
    
    
    df = pd.DataFrame({
        "folder" : np.repeat(folder, 10),
        "n_sim_steps":  np.repeat(10, 10),
        "horizon": np.repeat(horizon, 10),
        "sim_time": np.repeat(elapsed_time, 10),
        
        # for graph experiments only
        "graph_final_node_count": info["size"][0],  
        
        "action": info["action"][1:,0,0],
        "observation": info["observation"][0][0:-1,0,0],
        #Symbol("G") => agent.history.G,
        #Symbol("q_pi") => agent.history.q_pi,
        #Symbol("policy") => [values(x) for x in agent.history.policy]
    }
    )
    df.to_csv(os.path.join(folder, "results_horz_{}.csv".format(horizon)))
    #assert False


def run():
    for ii in range(10,16):
        simu(ii)


#assert False
#matplotlib widget
#import tree_visualisation_nontom_optimized as vis_opt

#t = 1
#tree = jtu.tree_map(lambda x: x[t], info["tree"])
#vis_opt.explore_planning_tree_nontom(tree, fig_size=(8, 6), depth=6, min_prob=0.0)