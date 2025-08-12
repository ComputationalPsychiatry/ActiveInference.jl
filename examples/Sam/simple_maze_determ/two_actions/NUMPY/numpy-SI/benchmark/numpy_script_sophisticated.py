import os

import pymdp
from pymdp import utils
from pymdp.agent import Agent
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import numpy as np
import pandas as pd
import time
import tracemalloc

from generative_model import MAZE, A, B, C, D
from maze_env import SIMazeEnv

folder = "./two_actions/NUMPY/numpy-SI/benchmark"

def benchmark_infer_policies(agent):
    tracemalloc.start()
    start = time.perf_counter()
    q_pi, G = agent.infer_policies()
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return q_pi, G, end - start, peak / (1024 * 1024)

def run_simulation(horizon, num_iterations):
    agent = Agent(
        A=A,
        B=B, 
        C=C, 
        D=D, 
        policy_len=1, 
        sophisticated=True, 
        si_horizon=horizon, 
        sampling_mode="full", 
        gamma=1.0,
        si_state_prune_threshold=1/16
        )
    env = SIMazeEnv(MAZE, start_state=17)
    observation = [17,0] 


    planning_times = []
    planning_mibs = []
    actions = []
    observations = []
    start_sim = time.perf_counter()

    for t in range(num_iterations):
        qs = agent.infer_states(observation)

        q_pi, G, elapsed_time, mem_mib = benchmark_infer_policies(agent)
        action = agent.sample_action()
        actions.append(action[0])

        observation = env.step(action) 
        observations.append(int(observation[0]))

        planning_times.append(elapsed_time)
        planning_mibs.append(mem_mib)

    end_sim = time.perf_counter()
    sim_time = end_sim - start_sim


    print("\nobservations= {}\n".format(observations))
    print("sim time= {}\n".format(sim_time))

    last_obs = [int(observation[0]), int(observation[1])]


    df = pd.DataFrame({
        "folder" : np.repeat(folder, 10),
        "n_sim_steps":  np.repeat(10, 10),
        "horizon": np.repeat(horizon, 10),
        "sim_time": np.repeat(sim_time, 10),
        
        # for graph experiments only
        #"graph_final_node_count": node_count,  
        
        "action": actions,
        "observation": observations,
        #Symbol("G") => agent.history.G,
        #Symbol("q_pi") => agent.history.q_pi,
        #Symbol("policy") => [values(x) for x in agent.history.policy]
    }
    )
    df.to_csv(os.path.join(folder, "results_horz_{}.csv".format(horizon)))

def run():
    for horizon in range(2, 16):
        print(f"Running horizon {horizon}...")
        run_simulation(horizon, 10)
        print(f"Finished {horizon}.\n")


