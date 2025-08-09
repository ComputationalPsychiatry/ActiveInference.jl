
# taskset --cpu-list 1 python jax_script_standard_inference.py

import os

os.environ['MKL_NUM_THREADS']='1' 
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'

os.environ["NUM_INTER_THREADS"]="1"
os.environ["NUM_INTRA_THREADS"]="1"


                         
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 xla_force_host_platform_device_count=1"
os.environ["NPROC"] = "1"


import jax
jax.config.update('jax_platform_name', 'cpu')


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
from generative_model import MAZE, PREFERENCES, A, B, C, D
from maze_env import SIMazeEnv

import debugger

# Simulated entire simulation loop (no tree, no nodes)
def run_simulation(policy_len, num_iterations):
    agent = Agent(A=A, B=B, C=C, D=D, policy_len=policy_len, sampling_mode="marginal", gamma=1.0, alpha=1.0, use_inductive=False)
    
    #assert False
    infer_states = jax.jit(lambda obs, prior: agent.infer_states(obs, prior))
    infer_policies = jax.jit(lambda qs: agent.infer_policies(qs))

    #infer_states = agent.infer_states
    #infer_policies = agent.infer_policies

    env = SIMazeEnv(MAZE, PREFERENCES, start_state=17)

    def benchmark_infer_policies(qs):
        tracemalloc.start()
        start = time.perf_counter()
        q_pi, G = infer_policies(qs)
        end = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return q_pi, G, end - start, peak / (1024 * 1024)
    
    rng_key = random.PRNGKey(seed=42)
    empirical_prior = agent.D
    observation = env.get_observation()
    empirical_prior[0] = empirical_prior[0].reshape(1, 81)

    planning_times = []
    planning_mibs = []

    start_sim = time.perf_counter()

    for t in range(num_iterations):
        qs = infer_states(observation, empirical_prior)
        q_pi, G, elapsed_time, mem_mib = benchmark_infer_policies(qs)
        
        #elapsed_time = 0
        #mem_mib = 0
        #q_pi, G = infer_policies(qs)
        q_pi = q_pi.reshape((1, -1))
        action = agent.sample_action(q_pi)
        empirical_prior = agent.update_empirical_prior(action, qs)
        empirical_prior = [jnp.asarray(empirical_prior[0])]
        empirical_prior[0] = empirical_prior[0].reshape(1, 81)
        observation = env.step(action[0])
        planning_times.append(elapsed_time)
        planning_mibs.append(mem_mib)

    end_sim = time.perf_counter()
    sim_time = end_sim - start_sim
    last_obs = [int(observation[0].argmax()), int(observation[1].argmax()), int(observation[2].argmax())]
    # Save total simulation summary (no nodes)
    df_total = pd.DataFrame({
        f"std_jax_time_{horizon}": [sim_time],
        f"std_jax_last_obs_{horizon}": [str(last_obs)]
    })
    
    #df_total.to_csv(f"std_jax_jit_total_time_{horizon}.csv", index=False)
    print(sim_time)
    print(f"Last observation: {last_obs}")
    # Save per-iteration planning times and memory
    df_planning = pd.DataFrame({
        "std_jax_planning_time": planning_times,
        "std_jax_planning_MiB": planning_mibs
    })
    #df_planning.to_csv(f"std_jax_jit_planning_{horizon}.csv", index=False)

#for horizon in range(2, 14):
for horizon in range(15, 16):
    print(f"Running horizon {horizon}...")
    run_simulation(horizon, 10)
    print(f"Finished {horizon}.\n")

for horizon in range(15, 16):
    print(f"Running horizon {horizon}...")
    run_simulation(horizon, 10)
    print(f"Finished {horizon}.\n")