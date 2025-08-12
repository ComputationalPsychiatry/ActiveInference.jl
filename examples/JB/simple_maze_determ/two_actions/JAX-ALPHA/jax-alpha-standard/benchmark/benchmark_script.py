import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import jax.random as random
import seaborn as sns
import matplotlib.pyplot as plt
from pymdp.agent import Agent
from pymdp.maths import MINVAL

import pandas as pd
import time
import tracemalloc
import os 

from pymdp.agent import Agent
from generative_model import MAZE, PREFERENCES, A, B, C, D
from maze_env import SIMazeEnv
import debugger

folder = "./two_actions/JAX-ALPHA/jax-alpha-standard/benchmark"


# Simulated entire simulation loop (no tree, no nodes)
def run_simulation(policy_len, num_iterations):
    os.makedirs(folder, exist_ok=True) 
    
    agent = Agent(A=A, B=B, C=C, D=D, policy_len=policy_len, sampling_mode="full", gamma=1.0, alpha=1.0)
    infer_states = jax.jit(lambda obs, prior: agent.infer_states(obs, prior))
    infer_policies = jax.jit(lambda qs: agent.infer_policies(qs))
    env = SIMazeEnv(MAZE, start_state=17)


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
    actions = []
    observations = []
    start_sim = time.perf_counter()

    for t in range(num_iterations):
        qs = infer_states(observation, empirical_prior)
        q_pi, G, elapsed_time, mem_mib = benchmark_infer_policies(qs)
        q_pi = q_pi.reshape((1, -1))
        action = agent.sample_action(q_pi)
        
        actions.append(int(action[0][0]))
        empirical_prior = agent.update_empirical_prior(action, qs)
        empirical_prior = [jnp.asarray(empirical_prior[0])]
        empirical_prior[0] = empirical_prior[0].reshape(1, 81)
        observation = env.step(action[0])
        observations.append(int(observation[0].argmax()))
        planning_times.append(elapsed_time)
        planning_mibs.append(mem_mib)

    end_sim = time.perf_counter()
    sim_time = end_sim - start_sim
    
    
    print("\nobservations= {}\n".format(observations))
    print("sim time= {}\n".format(sim_time))
    
    
    
    last_obs = [int(observation[0].argmax()), int(observation[1].argmax())]
    # Save total simulation summary (no nodes)
    
    df = pd.DataFrame({
        "folder" : np.repeat(folder, 10),
        "n_sim_steps":  np.repeat(10, 10),
        "horizon": np.repeat(policy_len, 10),
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
    df.to_csv(os.path.join(folder, "results_horz_{}.csv".format(policy_len)))
    
    #assert False
    
def run():
    for horizon in range(2, 16):
        print(f"Running horizon {horizon}...")
        run_simulation(horizon, 10)
        print(f"Finished {horizon}.\n")
