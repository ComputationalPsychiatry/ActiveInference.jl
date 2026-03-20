# # Simulation Example T-Maze

# We will start from the importing of the necessary modules.

using ActiveInference
using ActiveInference.Environments

# We will create a T-Maze environment with a probability of 0.9 for reward in the the reward condition arm.
# This is a premade environment in the ActiveInference.jl package.

# ```julia
# env = TMazeEnv(0.9)
# initialize_gp(env)
# ```

# ### Creating the Generative Model
# #### The Helper Function

# When creating the generative model we can make use of the helper function, making it convenient to create the correct structure for the generative model parameters.

# To use the helper function we need to know the following:

# - Number of states in each factor of the environment
# - Number of observations in each modality
# - Number of controls or actions in each factor
# - Policy length of the agent
# - Initial fill for the parameters

# Let's start with the factors of the environment. Let's take a look at the T-Maze environment:

# ![image1](assets/TMazeIllustrationSmaller.png)

# We here have two factors with the following number of states:

# |       | Location Factor   |       | Reward Condition Factor   |
# |:------|:------------------|:------|:------------------------- |
# | 1.    | Centre            | 1.    | Reward Condition Left     |
# | 2.    | Left Arm          | 2.    | Reward Condition Right    |
# | 3.    | Right Arm         |       |                           |
# | 4.    | Cue               |       |                           |

# We will define this as a vector the following way:

# ```julia
# n_states = [4, 2]
# ```

# We will now define the modalities:

# |       | Location Modality |       | Reward Modality           |       | Cue Modality    |
# |:------|:------------------|:------|:------------------------- |:------|:--------------- |
# | 1.    | Centre            | 1.    | No Reward                 | 1.    | Cue Left        |
# | 2.    | Left Arm          | 2.    | Reward                    | 2.    | Cue Right       |
# | 3.    | Right Arm         | 3.    | Loss                      |       |                 |
# | 4.    | Cue               |       |                           |       |                 |

# Here we have 3 modalities, with 4, 3, and 2 observations in each. We will define this as a vector the following way:

# ```julia
# n_observations = [4, 3, 2]
# ```

# Now, let's take a look at the actions, or controls:

# |       | Controls Location Factor       |    | Controls Reward Condition Factor       |
# |:------|:-------------------------------|:-- |:---------------------------------------|
# | 1.    | Go to Centre                   | 1. | No Control                                      |
# | 2.    | Go to Left Arm                 |    |                                       |
# | 3.    | Go to Right Arm                |    |                                       |
# | 4.    | Go to Cue                      |    |                                       |

# As we see here, the agent cannot control the reward condition factor, and it therefore believes that there is only one way states can transition in this factor, which is independent of the agent's actions.
# We will define this as a vector the following way:

# ```julia
# n_controls = [4, 1]
# ```

# Now we can define the policy length of the agent. In this case we will just set it to 2, meaning that the agent plans two timesteps ahead in the future.
# We will just specify this as an integer:

# ```julia
# policy_length = 2
# ```

# The last thing we need to define is the initial fill for the parameters. We will just set this to zeros for now.

# ```julia
# template_type = "zeros"
# ```

# Having defined all the arguments that go into the helper function, we can now create the templates for the generative model parameters.

# ```julia
# A, B, C, D, E = create_matrix_templates(n_states, n_observations, n_controls, policy_length, template_type);
# ```

# #### Populating the Generative Model
# ##### Populating **A**

# Let's take a look at the shape of the first modality in the A parameters:
A, B, C, D, E = create_matrix_templates([4, 2], [4, 3, 2], [4, 1], 2, "zeros");#hide
A[1]

# For this first modality we provide the agent with certain knowledge on how location observations map onto location states.
# We do this the following way:

# ```julia
# # For reward condition right
# A[1][:,:,1] = [ 1.0  0.0  0.0  0.0
#                 0.0  1.0  0.0  0.0
#                 0.0  0.0  1.0  0.0
#                 0.0  0.0  0.0  1.0 ]

# # For reward condition left
# A[1][:,:,2] = [ 1.0  0.0  0.0  0.0
#                 0.0  1.0  0.0  0.0
#                 0.0  0.0  1.0  0.0
#                 0.0  0.0  0.0  1.0 ]
# ```

# For the second modality, the reward modality, we want the agent to be able to infer "no reward" with certainty when in the centre and cue locations.
# In the left and right arm though, the agent should be agnostic as to which arm produces reward and loss. This is the modality that will be learned in this example.

# ```julia
# # For reward condition right
# A[2][:,:,1] = [ 1.0  0.0  0.0  1.0
#                 0.0  0.5  0.5  0.0
#                 0.0  0.5  0.5  0.0 ]

# # For reward condition left
# A[2][:,:,2] = [ 1.0  0.0  0.0  1.0
#                 0.0  0.5  0.5  0.0
#                 0.0  0.5  0.5  0.0 ]
# ```

# In the third modality, we want the agent to infer the reward condition state when in the cue location.
# To do this, we give it an uniform probability for all locations except the cue location, where it veridically will observe the reward condition state. 

# ```julia
# # For reward condition right
# A[3][:,:,1] = [ 0.5  0.5  0.5  1.0
#                 0.5  0.5  0.5  0.0 ]

# # For reward condition left
# A[3][:,:,2] = [ 0.5  0.5  0.5  0.0
#                 0.5  0.5  0.5  1.0 ]
# ```

# ##### Populating **B**

# For the first factor we populate the **B** with determined beliefs about how the location states change depended on its actions.
# For each action, it determines where to go, without having to go through any of the other location states.
# We encode this as:

# ```julia
# # For action "Go to Center Location"
# B[1][:,:,1] = [ 1.0  1.0  1.0  1.0 
#                 0.0  0.0  0.0  0.0
#                 0.0  0.0  0.0  0.0
#                 0.0  0.0  0.0  0.0 ]

# # For action "Go to Right Arm"
# B[1][:,:,2] = [ 0.0  0.0  0.0  0.0 
#                 1.0  1.0  1.0  1.0
#                 0.0  0.0  0.0  0.0
#                 0.0  0.0  0.0  0.0 ]

# # For action "Go to Left Arm"
# B[1][:,:,3] = [ 0.0  0.0  0.0  0.0 
#                 0.0  0.0  0.0  0.0
#                 1.0  1.0  1.0  1.0
#                 0.0  0.0  0.0  0.0 ]

# # For action "Go to Cue Location"
# B[1][:,:,4] = [ 0.0  0.0  0.0  0.0 
#                 0.0  0.0  0.0  0.0
#                 0.0  0.0  0.0  0.0
#                 1.0  1.0  1.0  1.0 ]
# ```

# For the last factor there is no control, so we will just set the **B** to be the identity matrix.

# ```julia
# # For second factor, which is not controlable by the agent
# B[2][:,:,1] = [ 1.0  0.0 
#                 0.0  1.0 ] 
# ```

# ##### Populating **C**
# For the preference parameters **C** we are not interested in the first and third modality, which we will just set to a vector of zeros for each observation in that modality.
# However, for the second modality, we want the agent to prefer the "reward observation" indexed as 2, and the dislike the "loss observation" indexed as 3.

# ```julia
# # Preference over locations modality
# C[1] = [0.0, 0.0, 0.0, 0.0]

# # Preference over reward modality
# C[2] = [0.0, 3.0, -3.0]

# # Preference over cue modality
# C[3] = [0.0, 0.0]
# ```   

# ##### Populating **D**
# For the prior over states **D** we will set the agent's belief to be correct in the location state factor and uniform, or agnostic, in the reward condition factor. 

# ```julia	
# # For the location state factor
# D[1] = [1.0, 0.0, 0.0, 0.0]

# # For the reward condition state factor
# D[2] = [0.5, 0.5]
# ```


# ##### Populating **E**
# For the prior over policies **E** we will set it to be uniform, meaning that the agent has no prior preference for any policy.

# ```julia	
# # Creating a vector of a uniform distribution over the policies. This means no preferences over policies.
# E .= 1.0/length(E)
# ```

# ##### Creating the prior over **A**
# When creating the prior over **A**, we use **A** as a template, by using 'deepcopy()'.
# Then we multiply this with a scaling parameter, setting the initial concentration parameters for the Dirichlet prior over **A**, **pA**.

# ```julia	
# pA = deepcopy(A)
# scale_concentration_parameter = 2.0
# pA .*= scale_concentration_parameter
# ```

# #### Creating Settings and Parameters Dictionary

# For the settings we set the 'use_param_info_gain' and 'use_states_info_gain' to true, meaning that the agent will take exploration and parameter learning into account when calculating the prior over policies.
# We set the policy length to 2, and specify modalities to learn, which in our case is the reward modality, indexed as 2.

# ```julia	
# settings = Dict(
#     "use_param_info_gain" => true,
#     "use_states_info_gain" => true,
#     "policy_len" => 2,
#     "modalities_to_learn" => [2]
# )
# ```

# For the parameters, we just use the default values, but specify the learning rate here, just to point it out.
# ```julia	
# parameters = Dict{String, Real}(
#     "lr_pA" => 1.0,
# )
# ```

# ### Initilising the Agent
# We can now initialise the agent with the parameters and settings we have just specified. 

# ```julia
# aif_agent = init_aif(
#     A, B, C = C, D = D, E = E, pA = pA, settings = settings, parameters = parameters
# );
# ```

# ### Simulation
# We are now ready for the perception-action-learning loop:

# ```julia
# # Settting the number of trials
# T = 100

# # Creating an initial observation and resetting environment (reward condition might change)
# obs = reset_TMaze!(Env)

# # Creating a for-loop that loops over the perception-action-learning loop T amount of times
# for t = 1:T

#     # Infer states based on the current observation
#     infer_states!(aif_agent, obs)

#     # Updates the A parameters
#     update_parameters!(aif_agent)

#     # Infer policies and calculate expected free energy
#     infer_policies!(aif_agent)

#     # Sample an action based on the inferred policies
#     chosen_action = sample_action!(aif_agent)

#     # Feed the action into the environment and get new observation.
#     obs = step_TMaze!(Env, chosen_action)
# end
# ```



