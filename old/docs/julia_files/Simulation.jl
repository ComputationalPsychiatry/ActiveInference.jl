# # Simulation with ActiveInference.jl
# When simulating with active inference we need a perception-action loop. This loop will perform the following steps:
# 1. The agent will infer the states of the environment based on its generative model and an observation. The inference here is optimized through the minimization of the variational free energy (see `Active Inference Theory Perception`).
# 2. The agent will infer the best action based on the minimization of the expected free energy (see `Active Inference Theory Action`).
# 3. The agent will perform the action in the environment and receive an observation for use in the next iteration.

# *Note: for learning included, look at the section below.*

# #### The Perception-Action loop:
# ```julia
# T = n_iterations

# for t = 1:T

#     infer_states!(aif_agent, observation)

#     infer_policies!(aif_agent)

#     chosen_action = sample_action!(aif_agent)

#     observation = environment!(env, chosen_action)

# end
# ```

# #### The Perception-Action-Learning loop:
# When learning is included, the loop is very similar except for the addition of the update functions, which should be implemented at different points in the loop.
# Below we will show how to include learning of the parameters. It is important that only the parameters which have been provided to the agent as a prior are being updated.
# ```julia
# T = n_iterations

# for t = 1:T

#    infer_states!(aif_agent, observation)

#    update_parameters!(aif_agent)

#    infer_policies!(aif_agent)

#    chosen_action = sample_action!(aif_agent)

#    observation = environment!(env, chosen_action)

# end
# ```

# The only addition here is the `update_parameters!(aif_agent)` function, which updates the parameters of the agent, based on which priors it has been given. 
