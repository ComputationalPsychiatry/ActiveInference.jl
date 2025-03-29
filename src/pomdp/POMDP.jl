"""
    action_pomdp!(agent, obs)
This function wraps the POMDP action-perception loop used for simulating and fitting the data.

Arguments:
- `agent::Agent`: An instance of ActionModels `Agent` type, which contains AIF type object as a substruct.
- `obs::Vector{Int64}`: A vector of observations, where each observation is an integer.
- `obs::Tuple{Vararg{Int}}`: A tuple of observations, where each observation is an integer.
- `obs::Int64`: A single observation, which is an integer.
- `aif::AIF`: An instance of the `AIF` type, which contains the agent's state, parameters, and substructures.

Outputs:
- Returns a `Distributions.Categorical` distribution or a vector of distributions, representing the probability distributions for actions per each state factor.
"""

### Action Model:  Returns probability distributions for actions per factor

function action_pomdp!(agent::Agent, obs::Vector{Int64})

    ### Get parameters 
    alpha = agent.substruct.parameters.alpha
    n_factors = length(agent.substruct.settings._n_controls)

    # Initialize empty arrays for action distribution per factor
    action_p = Vector{Any}(undef, n_factors)
    action_distribution = Vector{Distributions.Categorical}(undef, n_factors)

    #If there was a previous action
    if !ismissing(agent.states["action"])

        #Extract it
        previous_action = agent.states["action"]

        # If it is not a vector, make it one
        if !(previous_action isa Vector)
            previous_action = previous_action isa Integer ? [previous_action] : collect(previous_action)
        end
        #Store the action in the AIF substruct
        agent.substruct.states.action = previous_action
    end

    ### Infer states & policies

    # Run state inference 
    infer_states!(agent.substruct, obs)

    # If action is empty, update D vectors
    if ismissing(agent.states["action"]) && !isnothing(agent.substruct.parameters.pD)
        update_D!(agent.substruct)
    end

    # If learning of the B matrix is enabled and agent has a previous action
    if !ismissing(agent.states["action"]) && !isnothing(agent.substruct.parameters.pB)
        # Update Transition Matrix
        update_B!(agent.substruct)
    end

    # If learning of the A matrix is enabled
    if !isnothing(agent.substruct.parameters.pA)
        update_A!(agent.substruct)
    end

    # Run policy inference 
    infer_policies!(agent.substruct)

    ### Retrieve log marginal probabilities of actions
    log_action_marginals = get_log_action_marginals(agent.substruct)

    ### Pass action marginals through softmax function to get action probabilities
    for factor in 1:n_factors
        action_p[factor] = softmax(log_action_marginals[factor] * alpha, dims=1)
        action_distribution[factor] = Distributions.Categorical(action_p[factor])
    end

    return n_factors == 1 ? action_distribution[1] : action_distribution
end

### Action Model where the observation is a tuple

function action_pomdp!(agent::Agent, obs::Tuple{Vararg{Int}})

    # convert observation to vector
    obs = collect(obs)

    ### Get parameters 
    alpha = agent.substruct.parameters["alpha"]
    n_factors = length(agent.substruct.settings["num_controls"])

    # Initialize empty arrays for action distribution per factor
    action_p = Vector{Any}(undef, n_factors)
    action_distribution = Vector{Distributions.Categorical}(undef, n_factors)

     #If there was a previous action
     if !ismissing(agent.states["action"])

        #Extract it
        previous_action = agent.states["action"]

        # If it is not a vector, make it one
        if !(previous_action isa Vector)
            previous_action = collect(previous_action)
        end

        #Store the action in the AIF substruct
        agent.substruct.action = previous_action
    end

    ### Infer states & policies

    # Run state inference 
    infer_states!(agent.substruct, obs)

    # If action is empty and pD is not nothing, update D vectors
    if ismissing(agent.states["action"]) && agent.substruct.pD !== nothing
        qs_t1 = get_history(agent.substruct)["posterior_states"][1]
        update_D!(agent.substruct, qs_t1)
    end

    # If learning of the B matrix is enabled and agent has a previous action
    if !ismissing(agent.states["action"]) && agent.substruct.pB !== nothing

        # Get the posterior over states from the previous time step
        states_posterior = get_history(agent.substruct)["posterior_states"][end-1]

        # Update Transition Matrix
        update_B!(agent.substruct, states_posterior)
    end

    # If learning of the A matrix is enabled
    if agent.substruct.pA !== nothing
        update_A!(agent.substruct, obs)
    end

    # Run policy inference 
    infer_policies!(agent.substruct)


    ### Retrieve log marginal probabilities of actions
    log_action_marginals = get_log_action_marginals(agent.substruct)
    
    ### Pass action marginals through softmax function to get action probabilities
    for factor in 1:n_factors
        action_p[factor] = softmax(log_action_marginals[factor] * alpha, dims=1)
        action_distribution[factor] = Distributions.Categorical(action_p[factor])
    end

    return n_factors == 1 ? action_distribution[1] : action_distribution
end

function action_pomdp!(aif::POMDPActiveInference, obs::Vector{Int64})

    ### Get parameters 
    alpha = aif.parameters.alpha
    n_factors = length(aif.settings._n_controls)

    # Initialize empty arrays for action distribution per factor
    action_p = Vector{Any}(undef, n_factors)
    action_distribution = Vector{Distributions.Categorical}(undef, n_factors)

    ### Infer states & policies

    # Run state inference 
    infer_states!(aif, obs)

    # If action is empty, update D vectors
    if isempty(get_states(aif, "action")) && !isnothing(aif.parameters.pD)
        # qs_t1 = get_history(aif, "qs_current")[end]
        # update_D(aif, qs_t1)
        update_D(aif)
    end

    # If learning of the B matrix is enabled and agent has a previous action
    if !isempty(get_states(aif, "action")) && !isnothing(aif.parameters.pB)

        # Get the posterior over states from the previous time step
        # states_posterior = get_history(aif, "qs_current")[end-1]

        # Update Transition Matrix
        # update_B(aif, states_posterior)
        update_B(aif)
    end

    # If learning of the A matrix is enabled
    if !isnothing(aif.parameters.pA)
        # update_A(aif, obs)
        update_A(aif)
    end

    # Run policy inference 
    infer_policies!(aif)


    ### Retrieve log marginal probabilities of actions
    log_action_marginals = get_log_action_marginals(aif)
    
    ### Pass action marginals through softmax function to get action probabilities
    for factor in 1:n_factors
        action_p[factor] = softmax(log_action_marginals[factor] * alpha, dims=1)
        action_distribution[factor] = Distributions.Categorical(action_p[factor])
    end

    return n_factors == 1 ? action_distribution[1] : action_distribution
end

# function action_pomdp!(aif::AIF, obs::Vector{Int64})

#     ### Get parameters 
#     alpha = aif.parameters["alpha"]
#     n_factors = length(aif.settings["num_controls"])

#     # Initialize empty arrays for action distribution per factor
#     action_p = Vector{Any}(undef, n_factors)
#     action_distribution = Vector{Distributions.Categorical}(undef, n_factors)

#     ### Infer states & policies

#     # Run state inference 
#     infer_states!(aif, obs)

#     # If action is empty, update D vectors
#     if ismissing(get_states(aif)["action"]) && aif.pD !== nothing
#         qs_t1 = get_history(aif)["posterior_states"][1]
#         update_D!(aif, qs_t1)
#     end

#     # If learning of the B matrix is enabled and agent has a previous action
#     if !ismissing(get_states(aif)["action"]) && aif.pB !== nothing

#         # Get the posterior over states from the previous time step
#         states_posterior = get_history(aif)["posterior_states"][end-1]

#         # Update Transition Matrix
#         update_B!(aif, states_posterior)
#     end

#     # If learning of the A matrix is enabled
#     if aif.pA !== nothing
#         update_A!(aif, obs)
#     end

#     # Run policy inference 
#     infer_policies!(aif)


#     ### Retrieve log marginal probabilities of actions
#     log_action_marginals = get_log_action_marginals(aif)
    
#     ### Pass action marginals through softmax function to get action probabilities
#     for factor in 1:n_factors
#         action_p[factor] = softmax(log_action_marginals[factor] * alpha, dims=1)
#         action_distribution[factor] = Distributions.Categorical(action_p[factor])
#     end

#     return n_factors == 1 ? action_distribution[1] : action_distribution
# end

function action_pomdp!(agent::Agent, obs::Int64)
    action_pomdp!(agent::Agent, [obs])
end