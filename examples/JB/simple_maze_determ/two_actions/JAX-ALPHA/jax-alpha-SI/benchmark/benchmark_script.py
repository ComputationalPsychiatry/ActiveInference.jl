import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import jax.random as random
import seaborn as sns
import matplotlib.pyplot as plt
from pymdp.agent import Agent
from pymdp.planning.si import *
from pymdp.maths import MINVAL

import pandas as pd
import time
import tracemalloc
import os

from pymdp.agent import Agent
from pymdp.planning.si import si_policy_search
from pymdp.planning.si import tree_search
from generative_model import MAZE, PREFERENCES, A, B, C, D
from maze_env import SIMazeEnv

import debugger

folder = "./two_actions/JAX-ALPHA/jax-alpha-SI/benchmark"


def count_nodes(tree):
    return len(tree.nodes)


# search function benchmarking
def benchmark_search(search_fn, *args):
    tracemalloc.start()
    start = time.perf_counter()
    #assert False
    q_pi, tree = search_fn(*args)
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return q_pi, tree, end - start, peak / (1024 * 1024)


def run_simulation(horizon, num_iterations):
    os.makedirs(folder, exist_ok=True) 
    
    # Reinitialize everything
    search_fn = si_policy_search(
        max_depth=horizon,
        policy_prune_threshold=1/16,
        policy_prune_topk=-1,
        observation_prune_threshold=1/16,
        entropy_prune_threshold=0.5,
        prune_penalty=512,
        gamma=1,
    )

    agent = Agent(A=A, B=B, C=C, D=D, policy_len=1, sampling_mode="full", gamma=1.0, alpha=1.0)
    env = SIMazeEnv(MAZE, start_state=17)
    rng_key = random.PRNGKey(seed=42)
    empirical_prior = agent.D
    observation = env.get_observation()
    empirical_prior[0] = empirical_prior[0].reshape(1, 81)

    planning_times = []
    planning_mibs = []

    start_sim = time.perf_counter()
    observations = []
    actions = []
    for t in range(num_iterations):
        qs = agent.infer_states(observation, empirical_prior)
        q_pi, tree, elapsed_time, mem_mib = benchmark_search(search_fn, agent, qs, rng_key)
        #q_pi, tree, elapsed_time, mem_mib = benchmark_search(search_fn, agent, qs, horizon)
        q_pi = q_pi.reshape((1, -1))
        action = agent.sample_action(q_pi)
        empirical_prior = agent.update_empirical_prior(action, qs)
        empirical_prior = [jnp.asarray(empirical_prior[0])]
        empirical_prior[0] = empirical_prior[0].reshape(1, 81)
        observation = env.step(action[0])
        observations.append([int(observation[0].argmax()), int(observation[1].argmax())])
        planning_times.append(elapsed_time)
        planning_mibs.append(mem_mib)
        actions.append(action)

    end_sim = time.perf_counter()
    sim_time = end_sim - start_sim
    #assert False
    last_obs = [int(observation[0].argmax()), int(observation[1].argmax())]
    node_count = count_nodes(tree)

    df = pd.DataFrame({
        "folder" : np.repeat(folder, 10),
        "n_sim_steps":  np.repeat(10, 10),
        "horizon": np.repeat(horizon, 10),
        "sim_time": np.repeat(sim_time, 10),
        
        # for graph experiments only
        "graph_final_node_count": node_count,  
        
        "action": actions,
        "observation": observations,
        #Symbol("G") => agent.history.G,
        #Symbol("q_pi") => agent.history.q_pi,
        #Symbol("policy") => [values(x) for x in agent.history.policy]
    }
    )
    df.to_csv(os.path.join(folder, "results_horz_{}.csv".format(horizon)))
    

    print(sim_time)
    print(f"Last observation: {last_obs}")
    print("\nobservations= \n{}\n".format(observations))

    #assert False

def run():
    #for horizon in range(2, 14):
    for horizon in range(2, 6):
        print(f"Running horizon {horizon}...")
        run_simulation(horizon, 10)
        print(f"Finished {horizon}.\n")
    
    
