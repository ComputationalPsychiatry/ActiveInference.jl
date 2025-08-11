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

from pymdp.agent import Agent
from pymdp.planning.si import si_policy_search
from pymdp.planning.si import tree_search
from generative_model import MAZE, PREFERENCES, A, B, C, D
from maze_env import SIMazeEnv

import debugger

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

    end_sim = time.perf_counter()
    sim_time = end_sim - start_sim
    #assert False
    last_obs = [int(observation[0].argmax()), int(observation[1].argmax())]
    node_count = count_nodes(tree)

    # Save total simulation summary
    df_total = pd.DataFrame({
        f"si_jax_time_{horizon}": [sim_time],
        f"si_jax_nodes_{horizon}": [node_count],
        f"si_jax_last_obs_{horizon}": [str(last_obs)]
    })
    #df_total.to_csv(f"si_jax_total_time_{horizon}.csv", index=False)
    print(sim_time)
    print(f"Last observation: {last_obs}")
    print("\nobservations= \n{}\n".format(observations))

    # Save per-iteration planning times and memory
    df_planning = pd.DataFrame({
        "si_jax_planning_time": planning_times,
        "si_jax_planning_MiB": planning_mibs
    })
    #df_planning.to_csv(f"si_jax_planning_{horizon}.csv", index=False)

#for horizon in range(2, 14):
for horizon in range(7, 10):
    print(f"Running horizon {horizon}...")
    run_simulation(horizon, 10)
    print(f"Finished {horizon}.\n")