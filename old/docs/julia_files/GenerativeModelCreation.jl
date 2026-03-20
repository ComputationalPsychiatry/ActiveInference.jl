# # Creating the POMDP Generative Model

# In this section we will go through the process of creating a generative model and how it should be structured. In this part, we will show the code necessary for correct typing of the generative model.
# For a theoretical explanation of POMDPs look under the "Theory" section further down in the documentation.

# ## Typing of the POMDP parameters

# In ActiveInference.jl, it is important that the parameters describing the generative model is typed correctly.
# The correct typing of the generative model parameters, which often take the shapes of matrices, tensors and vectors.
# The collections of generative model parameters are colloquially referred to as **A**, **B**, **C**, **D**, and **E**. We will denote these parameters by their letter in bold. For a quick refresher this is the vernacular used to describe these parameter collections:

# - **A** - Observation Likelihood Model
# - **B** - Transition Likelihood Model
# - **C** - Prior over Observations
# - **D** - Prior over States
# - **E** - Prior over Policies

# These should be typed the following way in ActiveInference.jl:

# ```julia
# A = Vector{Array{Float64, 3}}(undef, n_modalities)
# B = Vector{Array{Float64, 3}}(undef, n_factors)
# C = Vector{Vector{Float64, 3}}(undef, n_modalities)
# D = Vector{Vector{Float64, 3}}(undef, n_factors)
# D = Vector{Float64, 3}(undef, n_policies)
# ```

# Each of the parameter collections are vectors, where each index in the vector contains the parameters associated with a specific modality or factor.
# However, creating these from scratch is not necessary, as we have created a helper function that can create a template for these parameters.

# ## Helper Function for GM Templates
# Luckily, there is a helper function that helps create templates for the generative model parameters. This function is called `create_matrix_templates`.

# ```julia
# A, B, C, D, E = create_matrix_templates(n_states::Vector{Int64}, n_observations::Vector{Int64}, n_controls::Vector{Int64}, policy_length::Int64, template_type::String)
# ```

# This function takes the five arguments `n_states`, `n_observations`, `n_controls`, `policy_length`, and `template_type`, which have all the necessary information to create the 
# right structure of the generative model parameters. We will go through these arguments one by one:

# \

# - **n_states** - This is the number of states in the environment. The environment can have different kinds of states, which are often referred to as factors. Could be a location factor and a reward condition factor. It takes a vector of integers, where each integer represents a factor, and the value of the integer is the number of states in that factor. E.g. if we had an environment with two factors, one location factor with 4 states and one reward condition factor with 2 states, the argument would look like this: `[4,2]`
# \

# - **n_observations** - This is the number of observations the agent can make in the environment. The observations are often referred to as modalities. Could be a location modality, a reward modality and a cue modality. Similarly to the first argument, it takes a vector of integers, where each integer represents a modality, and the value of the integer is the number of observations in that modality. E.g. if we had an environment with three modalities, one location modality with 4 observations, one reward modality with 3 observations and one cue modality with 2 observations, the argument would look like this: `[4,3,2]`
# \

# - **n_controls** - This is the number of controls the agent have in the environment. The controls are the actions the agent can take in the different factors. Could be moving left or right, or choosing between two different rewards. It has one control integer for each factor, where the integer represents the number of actions in that factor. If the agent cannot control a factor, the integer should be 1. E.g. if we had an environment with two factors, one location factor with 4 actions and one reward condition factor with 1 action, the argument would look like this: `[4,1]`
# \
  
# - **policy_length** - This is the length of the policies of the agent, and is taken as an integer. The policy is a sequence of actions the agent can take in the environment. The length of the policy describes how many actions into the future the agent is planning. For example, if the agent is planning two steps into the future, the policy length would be 2, and each policy would consist of 2 actions. In that case the argument would look like this: `2`
# \

# - **template_type** - This is a string that describes the type of template you want to create, or in other words, the initial filling of the generative model structure. There are three options; `"uniform"`, which is default, `"random"`, and `"zeros"`.


# If we were to use the arguments from the examples above, the function call would look like this:
using ActiveInference #hide
n_states = [4,2]
n_observations = [4,3,2]
n_controls = [4,1]
policy_length = 2
template_type = "zeros"

A, B, C, D, E = create_matrix_templates(n_states, n_observations, n_controls, policy_length, template_type);

# When these parameter collections have been made, each factor/modality can be accessed by indexing the collection with the factor/modality index like:

# ```julia
# A[1] # Accesses the first modality in the observation likelihood model
# B[2] # Accesses the second factor in the transition likelihood model
# C[3] # Accesses the third modality in the prior over observations
# D[1] # Accesses the first factor in the prior over states
# ```

# The E-parameters are not a divided into modalities or factors, as they are the prior over policies.

# ## Populating the Parameters
# Now that the generative model parameter templates ahave been created, they can now be filled with the desired values, ie. populating the parameters.
# Let's take the example of filling **A** with some valus. To start, let's print out the first modality of the A so we get a sense of the dimensions:
A[1]
# For a quick recap on the POMDP generative model parameteres look up the [`POMDP Theory`](@ref "The Generative Model Conceptually") section further down in the documentation.

# For now, we'll suffice to say that the first modality of **A** is a 3D tensor, where the first dimension are observations in the first modality, the second dimension the first factor, and the third dimension is the second factor.
# Remember **A** maps the agents beliefs on how states generate observations. In this case, we have two 4x4 matrices, one matrix for each state int the second factor. This could be how location observations (1st dimenstion) map onto location states (2nd dimension) and reward condition (3rd dimension).
# For the sake of simplicity, let's assume that the agent can infer location states with certainty based on location observations. In this case we could populate the first modality of **A** like this:

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

# In this case the agent would infer the location state with certainty based on the location observations. One could also make the **A** more noisy in this modality, which could look like:

# ```julia
# # For reward condition right
# A[1][:,:,1] = [ 0.7  0.1  0.1  0.1
#                 0.1  0.7  0.1  0.1
#                 0.1  0.1  0.7  0.1
#                 0.1  0.1  0.1  0.7 ]

# # For reward condition left
# A[1][:,:,2] = [ 0.7  0.1  0.1  0.1
#                 0.1  0.7  0.1  0.1
#                 0.1  0.1  0.7  0.1
#                 0.1  0.1  0.1  0.7 ]
# ```

# Importantly the columns should always add up to 1, as we are here dealing with categorical probability distributions.
# For the other parameters, the process is similar, but the dimensions of the matrices will differ. For **B** the dimensions are states to states, and for **C** and **D** the dimensions are states to observations and states to factors respectively.
# Look up the `T-Maze Simulation` (insert reference here) example for a full example of how to populate the generative model parameters.

# ## Creating Dirichlet Priors over Parameters
# When learning is included, we create Dirichlet priors over the parameters **A**, **B**, and **D**. We usually do this by taking the created **A**, **B**, and **D** parameters and multiplying them with a scalar, which is the concentration parameter of the Dirichlet distribution.
# For more information on the specifics of learning and Dirichlet priors, look under the `Active Inference Theory` section in the documentation. Note here, that when we implement learning of a parameter, the parameter is going to be defined by its prior and no longer the initial 
# parameter that we specified. This is because the agent will update the parameter based on the prior and the data it receives. An example of how we would create a Dirichlet prior over **A** could look:

# ```julia
# pA = deepcopy(A)
# scale_concentration_parameter = 2.0
# pA .*= scale_concentration_parameter
# ```

# This is not relevant if learning is not included. If learning is not included, the parameters are fixed and the agent will not update them. The value of the scaling parameter determines how much each data observation impacts the update of the parameter.
# If the scaling is high, e.g. 50, then adding one data point will have a small impact on the parameter. If the scaling is low, e.g. 0.1, then adding one data point will have a large impact on the parameter. The update function updates the parameters by normalising the concentration parameters of the Dirichlet distribution.

