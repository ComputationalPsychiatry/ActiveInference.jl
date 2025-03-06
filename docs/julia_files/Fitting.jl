# # Model Fitting 

# In many cases, we want to be able to draw conclusions about specific observed phenomena, such as behavioural differences between distinct populations. A conventional approach in this context is model fitting, which involves estimating the parameter values of a model (e.g., prior beliefs) that are most likely given the observed behavior of a participant. This approach is often used in fields such as computational psychiatry or mathematical psychology  to develop more precise models and theories of mental processes, to find mechanistic differences between clinical populations, or to investigate the relationship between computational constructs such as Bayesian beliefs and neuronal dynamics.
# ## Quick Start
# #### Model Fitting with ActionModels.jl

# Model fitting in '**ActiveInference**' is mediated through '**ActionModels**', which is our sister package for implementing and fitting various behavioural models to data. The core of '**ActionModels**' is the action model function, which takes a single observation, runs the inference scheme (updating the agent's beliefs), and calculates the probability distribution over actions from which the agent samples its actions.
# *(Check out the [ActionModels documentation](https://ilabcode.github.io/ActionModels.jl/dev/markdowns/Introduction/) for more details)*

using Pkg#hide
using ActiveInference#hide
n_states=[4]#hide
n_observations=[4]#hide
n_controls=[2]#hide
policy_length=1#hide
A,B=create_matrix_templates(n_states, n_observations, n_controls, policy_length);#hide
aif = init_aif(A, B, verbose=false);#hide
using Distributions#hide
priors = Dict("alpha" => Gamma(1, 1));#hide
using DataFrames#hide 
using ActionModels#hide
# To demonstrate this, let's define a very simple generative model with a single state factor and two possible actions, and then initialize our active inference object:
# ```julia
# # Define the number of states, observations, and controls
# n_states = [4]
# n_observations = [4]
# n_controls = [2]

# # Define the policy length
# policy_length = 1

# # Use the create_matrix_templates function to create uniform A and B matrices.
# A, B = create_matrix_templates(n_states, n_observations, n_controls, policy_length)

# # Initialize an active inference object with the created matrices
# aif = init_aif(A, B)
# ```

# We can now use the `action_pomdp!` function (which serves as our active inference "action model") to calculate the probability distribution over actions for a single observation:
# ```julia
# # Define observation
# observation = [1]

# # Calculate action probabilities
# action_distribution = action_pomdp!(aif, observation)
# ```

# #### Agent in ActionModels.jl
# Another key component of '**ActionModels**' is an `Agent`, which wraps the action model and active inference object in a more abstract structure. The `Agent` is initialized using a `substruct` to include our active inference object, and the action model is our `action_pomdp!` function.

# Let's first install '**ActionModels**' from the official Julia registry and import it:
# ```julia
# Pkg.add("ActionModels")
# using ActionModels
# ```

# We can now create an `Agent` with the `action_pomdp!` function and the active inference object:

# ```julia
# # Initialize agent with active inference object as substruct
# agent = init_agent(
#     action_pomdp!,  # The active inference action model
#     substruct = aif # The active inference object
# )
# ```
# We use an initialized `Agent` primarily for fitting; however, it can also be used with a set of convenience functions to run simulations, which are described in [Simulation with ActionModels](./SimulationActionModels.md).

# #### Fitting a Single Subject Model
# We have our `Agent` object defined as above. Next, we need to specify priors for the parameters we want to estimate. 

# For example, let's estimate the action precision parameter `α` and use a Gamma distribution as its prior.

# ```julia
# # Import the Distributions package
# using Distributions

# # Define the prior distribution for the alpha parameters inside a dictionary
# priors = Dict("alpha" => Gamma(1, 1))
# ```
# We can now use the `create_model` function to instantiate a probabilistic model object with data. This function takes the `Agent` object, the priors, and a set of observations and actions as arguments.
#
# First, let's define some observations and actions as vectors:
# ```julia
# # Define observations and actions
# observations = [1, 1, 2, 3, 1, 4, 2, 1]
# actions = [2, 1, 2, 2, 2, 1, 2, 2]
# ```

# Now we can instantiate the probabilistic model object:
# ```julia
# # Create the model object
# single_subject_model = create_model(agent, priors, observations, actions)
# ```
# The `single_subject_model` can be used as a standard Turing object. Performing inference on this model is as simple as: 
# ```julia
# results = fit_model(single_subject_model)
# ```
# #### Fitting a Model with Multiple Subjects
# Often, we have data from multiple subjects that we would like to fit simultaneously. The good news is that this can be done by instantiating our probabilisitc model on an entire dataset containing data from multiple subjects.
#
# Let's define some dataset with observations and actions for three subjects:

# ```julia
# # Import the DataFrames package
# using DataFrames
#
# # Create a DataFrame 
# data = DataFrame(
#    subjectID = [1, 1, 1, 2, 2, 2, 3, 3, 3], # Subject IDs
#    observations = [1, 1, 2, 3, 1, 4, 2, 1, 3], # Observations
#    actions = [2, 1, 2, 2, 2, 1, 2, 2, 1] # Actions
# )
# ```
data = DataFrame(subjectID = [1, 1, 1, 2, 2, 2, 3, 3, 3], observations = [1, 1, 2, 3, 1, 4, 2, 1, 3], actions = [2, 1, 2, 2, 2, 1, 2, 2, 1] )#hide
#
# To instantiate the probabilistic model on our dataset, we pass the `data` DataFrame to the `create_model` function along with the names of the columns that contain the subject identifiers, observations, and actions:
# ```julia
# # Create the model object
# multi_subject_model = create_model(
#     agent, 
#     priors, 
#     data; # Dataframe
#     grouping_cols = [:subjectID], # Column with subject IDs
#     input_cols = ["observations"], # Column with observations
#     action_cols = ["actions"] # Column with actions
# )
# ```
agent = init_agent(action_pomdp!, substruct = aif);#hide
multi_subject_model = create_model(agent, priors, data; grouping_cols = [:subjectID], input_cols = ["observations"], action_cols = ["actions"]);#hide

# To fit the model, we use the `fit_model` function as before:
# ```julia
# results = fit_model(multi_subject_model)
# ```
results=fit_model(multi_subject_model, show_progress=false);#hide
# #### Customizing the Fitting Procedure
# The `fit_model` function has several optional arguments that allow us to customize the fitting procedure. For example, you can specify the number of iterations, the number of chains, the sampling algorithm, or to parallelize over chains:

# ```julia
# results = fit_model(
#     multi_subject_model, # The model object
#     parallelization = MCMCDistributed(), # Run chains in parallel
#     sampler = NUTS(;adtype=AutoReverseDiff(compile=true)), # Specify the type of sampler
#     n_itererations = 1000, # Number of iterations, 
#     n_chains = 4, # Number of chains
# )
# ```
# '**Turing**' allows us to run distributed `MCMCDistributed()` or threaded `MCMCThreads()` parallel sampling. The default is to run chains serially `MCMCSerial()`. For information on the available samplers see the [Turing documentation](https://turing.ml/dev/docs/using-turing/samplers/). 
# 
# #### Results
# 
# The output of the `fit_model` function is an object that contains the standard '**Turing**' chains which we can use to extract the summary statistics of the posterior distribution.
#
# Let's extract the chains from the results object:
chains = results.chains
#
# Note that the parameter names in the chains are somewhat cryptic. We can use the `rename_chains` function to rename them to something more understandable:
renamed_chains = rename_chains(chains, multi_subject_model)
#
# That looks better! We can now use the '**StatsPlots**' package to plot the chain traces and density plots of the posterior distributions for all subjects:
# ```julia
# using StatsPlots # Load the StatsPlots package
#
# plot(renamed_chains)
# ```
#
# ![image2](assets/chain_traces.png)
#
# We can also visualize the posterior distributions against the priors. This can be done by first taking samples from the prior:
# ```julia
# # Sample from the prior
# prior_chains = sample(multi_subject_model, Prior(), 1000)
# # Rename parameters in the prior chains
# renamed_prior_chains = rename_chains(prior_chains, multi_subject_model)
# ```
# To plot the posterior distributions against the priors, we use the `plot_parameters` function:
# ```julia
# plot_parameters(renamed_prior_chains, renamed_chains)
# ```

# ![image3](assets/posteriors.png)
