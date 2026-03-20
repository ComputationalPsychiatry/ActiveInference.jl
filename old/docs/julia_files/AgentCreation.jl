# # Creating the Agent

# Having created the generative model parameters in the precious section, we're not ready to intialise an active inference agent.
# Firstly, we'll have to specify some settings and hyperparameters that go into the agent struct. We'll begin with the setting:

# ### Settings
# The settings are a dictionary that contains the following keys:

# ```julia
# settings = Dict(
#     "policy_len" => 1, 
#     "use_utility" => true, 
#     "use_states_info_gain" => true, 
#     "use_param_info_gain" => false,
#     "action_selection" => "stochastic", 
#     "modalities_to_learn" => "all",
#     "factors_to_learn" => "all",
#     "FPI_num_iter" => 10,
#     "FPI_dF_tol" => 0.001
# )
# ```

# The above shown values are the default and will work in most cases. If you're unsure about what to specify in the settings, you can just use the default values by not specifying them in the settings Dict for the agent.
# Here, we'll briefly describe the keys in the settings dictionary:

# - **`policy_len`** - Is the policy length, and as described previously is the number of actions the agent should plan in the future. This is provided as an integer.
# - **`use_utility`** - Is a boolean that specifies whether the agent should use **C** in the expected free energy calculation, that guides the action selection in active inference. If set to `false`, the agent will not use the parameters specified in **C**.
# - **`use_states_info_gain`** - Is a boolean that specifies whether the agent should use the information gain over states in the expected free energy calculation. If set to `false`, the agent will not use the information gain over states.
# - **`use_param_info_gain`** - Is a boolean that specifies whether the agent should use the information gain over parameters in the expected free energy calculation. If set to `false`, the agent will not use the information gain over parameters. Only relevant when learning is included.
# - **`action_selection`** - Is a string that specifies the action selection method. The options are `"stochastic"` and `"deterministic"`. If set to `"stochastic"`, the agent will sample from the posterior over policies, and if set to `"deterministic"`, the agent will choose the most probable action.
# - **`modalities_to_learn`** - Is a vector of integers that specifies which modalities the agent should learn. If set to string `"all"`, the agent will learn all modalities. If set to `[1,2]`, the agent will only learn the first and second modality. Only relevant when learning of A is included.
# - **`factors_to_learn`** - Is a vector of integers that specifies which factors the agent should learn. If set to string `"all"`, the agent will learn all factors. If set to `[1,2]`, the agent will only learn the first and second factor. Only relevant when learning of B and D is included.
# - **`FPI_num_iter`** - Is an integer that specifies the number of fixed point iterations (FPI) to perform in the free energy minimization. It can be described as a stop function of the FPI algorithm.
# - **`FPI_dF_tol`** - Is a float that specifies the tolerance of the free energy change in the FPI algorithm over each iteration. If the change in free energy is below this value, the FPI algorithm will also stop.

# For more information on the specifics of the impact of these settings, look under the `Active Inference Theory` section in the documentation.

# ### Parameters
# The parameters are a dictionary that contains the following keys:

# ```julia
# parameters = Dict(
# "gamma" => 16.0,
# "alpha" => 16.0,
# "lr_pA" => 1.0,
# "fr_pA" => 1.0,
# "lr_pB" => 1.0,
# "fr_pB" => 1.0,
# "lr_pD" => 1.0,
# "fr_pD" => 1.0
# )
# ```

# The above shown values are the default. If you're unsure about what to specify in the parameters, you can just use the default values by not specifying them in the parameter Dict for the agent.
# Here, we'll briefly describe the keys in the parameters dictionary containing the hyperparameters:
# - **`alpha`** - Is the inverse temperature of the action selection process, and usually takes a value between 1 and 32. This is only relevant when action_selection is set to `"stochastic"`.
# - **`gamma`** - Is the inverse temperature precision of the expected free energy, and usually takes a value between 1 and 32. If the value is high, the agent will be more certain in its beliefs regarding the posterior probability over policies.
# - **`lr_pA`** - Is the learning rate of **A**, and usually takes a value between 0 and 1. Only relevant when learning is included, and this goes for all learning and forgetting rates. 
# - **`fr_pA`** - Is the forgetting rate of **A**, and usually takes a value between 0 and 1. If forgetting rate is 1 it means no forgetting.
# - **`lr_pB`** - Is the learning rate of **B**, and usually takes a value between 0 and 1.
# - **`fr_pB`** - Is the forgetting rate of **B**, and usually takes a value between 0 and 1. If forgetting rate is 1 it means no forgetting.
# - **`lr_pD`** - Is the learning rate of **D**, and usually takes a value between 0 and 1.
# - **`fr_pD`** - Is the forgetting rate of **D**, and usually takes a value between 0 and 1. If forgetting rate is 1 it means no forgetting.

# 
# Having now specified the setting and parameters, we can now initialise the active inference agent. This is done by calling the `init_aif` function, which takes the following arguments:

# ## Initilising the Agent

include("../julia_files/GenerativeModelCreation.jl") #hide
A, B, C, D, E = create_matrix_templates([4,2], [4,3,2], [4,1], 1, "uniform"); #hide
parameters = Dict( #hide
"gamma" => 16.0, #hide
"alpha" => 16.0, #hide
"lr_pA" => 1.0, #hide
"fr_pA" => 1.0, #hide
"lr_pB" => 1.0, #hide
"fr_pB" => 1.0, #hide
"lr_pD" => 1.0, #hide
"fr_pD" => 1.0 #hide
); #hide
settings = Dict( #hide
    "policy_len" => 1, #hide 
    "use_utility" => true, #hide
    "use_states_info_gain" => true,  #hide
    "use_param_info_gain" => false, #hide
    "action_selection" => "stochastic", #hide
    "modalities_to_learn" => "all", #hide
    "factors_to_learn" => "all", #hide
    "FPI_num_iter" => 10, #hide
    "FPI_dF_tol" => 0.001 #hide
); #hide
aif_agent = init_aif(
    A, B, C = C, D = D, E = E, settings = settings, parameters = parameters, verbose = false
);


# You can access the settings and parameters of the agent by calling the agent struct on the agent:
aif_agent.parameters
#-
aif_agent.settings

# Having now initialised the agent, we are ready to implement it either in a simulation with a perception-action loop, or for use in model fitting with observed data.

# ## Initialising the Agent with Learning
# If you want to include learning in the agent, you can do so by specifying the prior parameters `init_aif` function. Here is an example of how to initialise the agent with learning:

# ```julia
# aif_agent = init_aif(
#     A, B, C = C, D = D, E = E, pA = pA, pB = pB, pD = pD, settings = settings, parameters = parameters, verbose = false
# );
# ```

# Here, only the prior of the parameters that are to be learned should be specified.
