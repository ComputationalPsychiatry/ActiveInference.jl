""" -------- Inference Functions -------- """

#### State Inference #### 

""" Get Expected States """
function get_expected_states(qs::Vector{Vector{T}} where T <: Real, B, policy::Matrix{Int64})
    n_steps, n_factors = size(policy)

    # initializing posterior predictive density as a list of beliefs over time
    qs_pi = [deepcopy(qs) for _ in 1:n_steps+1]

    # expected states over time
    for t in 1:n_steps
        for control_factor in 1:n_factors
            action = policy[t, control_factor]
            
            qs_pi[t+1][control_factor] = B[control_factor][:, :, action] * qs_pi[t][control_factor]
        end
    end

    return qs_pi[2:end]
end

""" 
    Multiple dispatch for all expected states given all policies

Multiple dispatch for getting expected states for all policies based on the agents currently
inferred states and the transition matrices for each factor and action in the policy.

qs::Vector{Vector{Real}} \n
B: Vector{Array{<:Real}} \n
policy: Vector{Matrix{Int64}}

"""
function get_expected_states(qs::Vector{Vector{Float64}}, B, policy::Vector{Matrix{Int64}})
    
    # Extracting the number of steps (policy_length) and factors from the first policy
    n_steps, n_factors = size(policy[1])

    # Number of policies
    n_policies = length(policy)
    
    # Preparing vessel for the expected states for all policies. Has number of undefined entries equal to the
    # number of policies
    qs_pi_all = Vector{Vector{Vector{Vector{Float64}}}}(undef, length(n_policies))

    # Looping through all policies
    for (policy_idx, policy_x) in enumerate(policy)

        # initializing posterior predictive density as a list of beliefs over time
        qs_pi = [deepcopy(qs) for _ in 1:n_steps+1]

        # expected states over time
        for t in 1:n_steps
            for control_factor in 1:n_factors
                action = policy_x[t, control_factor]
                
                qs_pi[t+1][control_factor] = B[control_factor][:, :, action] * qs_pi[t][control_factor]
            end
        end
        qs_pi_all[policy_idx] = qs_pi[2:end]
    end
    return qs_pi_all
end

# """
#     process_observation(observation::Int, n_modalities::Int, n_observations::Vector{Int})

# Process a single modality observation. Returns a one-hot encoded vector. 

# # Arguments
# - `observation::Int`: The index of the observed state with a single observation modality.
# - `n_modalities::Int`: The number of observation modalities in the observation. 
# - `n_observations::Vector{Int}`: A vector containing the number of observations for each modality.

# # Returns
# - `Vector{Vector{Real}}`: A vector containing a single one-hot encoded observation.
# """
# function process_observation(observation::Int, n_modalities::Int, n_observations::Vector{Int})

#     # Check if there is only one modality
#     if n_modalities == 1
#         # Create a one-hot encoded vector for the observation
#         processed_observation = onehot(observation, n_observations[1]) 
#     end

#     # Return the processed observation wrapped in a vector
#     return [processed_observation]
# end

# """
#     process_observation(observation::Union{Array{Int}, Tuple{Vararg{Int}}}, n_modalities::Int, n_observations::Vector{Int})

# Process observation with multiple modalities and return them in a one-hot encoded format 

# # Arguments
# - `observation::Union{Array{Int}, Tuple{Vararg{Int}}}`: A collection of indices of the observed states for each modality.
# - `n_modalities::Int`: The number of observation modalities in the observation. 
# - `n_observations::Vector{Int}`: A vector containing the number of observations for each modality.

# # Returns
# - `Vector{Vector{Real}}`: A vector containing one-hot encoded vectors for each modality.
# """
# function process_observation(observation::Union{Array{Int}, Tuple{Vararg{Int}}}, n_modalities::Int, n_observations::Vector{Int})

#     # Initialize the processed_observation vector
#     processed_observation = Vector{Vector{Float64}}(undef, n_modalities)

#     # Check if the length of observation matches the number of modalities
#     if length(observation) == n_modalities
#         for (modality, modality_observation) in enumerate(observation)
#             # Create a one-hot encoded vector for the current modality observation
#             one_hot = onehot(modality_observation, n_observations[modality])
#             # Add the one-hot vector to the processed_observation vector
#             processed_observation[modality] = one_hot
#         end
#     end

#     return processed_observation
# end

""" Update Posterior States """
function update_posterior_states(
    A::Vector{Array{T,N}} where {T <: Real, N}, 
    obs::Vector{Int64}; 
    prior::Union{Nothing, Vector{Vector{T}}} where T <: Real = nothing, 
    num_iter::Int=num_iter, dF_tol::Float64=dF_tol, kwargs...)
    num_obs, num_states, num_modalities, num_factors = get_model_dimensions(A)

    obs_processed = process_observation(obs, num_modalities, num_obs)
    return fixed_point_iteration(A, obs_processed, num_obs, num_states, prior=prior, num_iter=num_iter, dF_tol = dF_tol)
end


""" Run State Inference via Fixed-Point Iteration """
function fixed_point_iteration(
    A::Vector{Array{T,N}} where {T <: Real, N}, obs::Vector{Vector{Float64}}, num_obs::Vector{Int64}, num_states::Vector{Int64};
    prior::Union{Nothing, Vector{Vector{T}}} where T <: Real = nothing, 
    num_iter::Int=num_iter, dF::Float64=1.0, dF_tol::Float64=dF_tol
)
    # Get model dimensions (NOTE Sam: We need to save model dimensions in the AIF struct in the future)
    n_modalities = length(num_obs)
    n_factors = length(num_states)

    # Get joint likelihood
    likelihood = get_joint_likelihood(A, obs, num_states)
    likelihood = capped_log(likelihood)

    # Initialize posterior and prior
    qs = Vector{Vector{Float64}}(undef, n_factors)
    for factor in 1:n_factors
        qs[factor] = ones(num_states[factor]) / num_states[factor]
    end

    # If no prior is provided, create a default prior with uniform distribution
    if prior === nothing
        prior = create_matrix_templates(num_states)
    end
    
    # Create a copy of the prior to avoid modifying the original
    prior = deepcopy(prior)
    prior = capped_log_array(prior) 

    # Initialize free energy
    prev_vfe = calc_free_energy(qs, prior, n_factors)

    # Single factor condition
    if n_factors == 1
        qL = dot_product(likelihood, qs[1])  
        return [softmax(qL .+ prior[1], dims=1)]

    # If there are more factors
    else
        ### Fixed-Point Iteration ###
        curr_iter = 0
        ### Sam NOTE: We need check if ReverseDiff might potantially have issues with this while loop ###
        while curr_iter < num_iter && dF >= dF_tol
            qs_all = qs[1]
            # Loop over each factor starting from the second one
            for factor in 2:n_factors
                # Reshape and multiply qs_all with the current factor's qs
                qs_all = qs_all .* reshape(qs[factor], tuple(ones(Real, factor - 1)..., :, 1))
            end

            # Compute the log-likelihood
            LL_tensor = likelihood .* qs_all

            # Update each factor's qs
            for factor in 1:n_factors
                # Initialize qL for the current factor
                qL = zeros(Real, size(qs[factor]))

                # Compute qL for each state in the current factor
                for i in 1:size(qs[factor], 1)
                    qL[i] = sum([LL_tensor[indices...] / qs[factor][i] for indices in Iterators.product([1:size(LL_tensor, dim) for dim in 1:n_factors]...) if indices[factor] == i])
                end

                # If qs is tracked by ReverseDiff, get the value
                if ReverseDiff.istracked(softmax(qL .+ prior[factor], dims=1))
                    qs[factor] = ReverseDiff.value(softmax(qL .+ prior[factor], dims=1))
                else
                    # Otherwise, proceed as normal
                    qs[factor] = softmax(qL .+ prior[factor], dims=1)
                end
            end

            # Recompute free energy
            vfe = calc_free_energy(qs, prior, n_factors, likelihood)

            # Update stopping condition
            dF = abs(prev_vfe - vfe)
            prev_vfe = vfe

            # Increment iteration
            curr_iter += 1
        end

        return qs
    end
end



""" Calculate Accuracy Term """
function compute_accuracy(log_likelihood, qs::Vector{Vector{T}} where T <: Real)
    n_factors = length(qs)
    ndims_ll = ndims(log_likelihood)
    dims = (ndims_ll - n_factors + 1) : ndims_ll

    # Calculate the accuracy term
    accuracy = sum(
        log_likelihood[indices...] * prod(qs[factor][indices[dims[factor]]] for factor in 1:n_factors)
        for indices in Iterators.product((1:size(log_likelihood, i) for i in 1:ndims_ll)...)
    )

    return accuracy
end


""" Calculate Free Energy """
function calc_free_energy(qs::Vector{Vector{T}} where T <: Real, prior, n_factors, likelihood=nothing)
    # Initialize free energy
    free_energy = 0.0
    
    # Calculate free energy for each factor
    for factor in 1:n_factors
        # Neg-entropy of posterior marginal
        negH_qs = dot(qs[factor], log.(qs[factor] .+ 1e-16))
        # Cross entropy of posterior marginal with prior marginal
        xH_qp = -dot(qs[factor], prior[factor])
        # Add to total free energy
        free_energy += negH_qs + xH_qp
    end
    
    # Subtract accuracy
    if likelihood !== nothing
        free_energy -= compute_accuracy(likelihood, qs)
    end
    
    return free_energy
end

#### Policy Inference #### 
""" Update Posterior over Policies """
function update_posterior_policies(
    qs::Vector{Vector{T}} where T <: Real,
    A::Vector{Array{T, N}} where {T <: Real, N},
    B::Vector{Array{T, N}} where {T <: Real, N},
    C::Vector{Vector{T}} where T <: Real,
    policies::Vector{Matrix{Int64}},
    use_utility::Bool=true,
    use_states_info_gain::Bool=true,
    use_param_info_gain::Bool=false,
    pA = nothing,
    pB = nothing,
    E::Vector{T} where T <: Real = nothing,
    gamma::Real=16.0
)
    n_policies = length(policies)
    G = zeros(n_policies)
    q_pi = Vector{Float64}(undef, n_policies)
    qs_pi = Vector{Float64}[]
    qo_pi = Vector{Float64}[]
    lnE = capped_log(E)

    for (idx, policy) in enumerate(policies)
        qs_pi = get_expected_states(qs, B, policy)
        qo_pi = get_expected_obs(qs_pi, A)

        # Calculate expected utility
        if use_utility
            # If ReverseDiff is tracking the expected utility, get the value
            if ReverseDiff.istracked(calc_expected_utility(qo_pi, C))
                G[idx] += ReverseDiff.value(calc_expected_utility(qo_pi, C))

            # Otherwise calculate the expected utility and add it to the G vector
            else
                G[idx] += calc_expected_utility(qo_pi, C)
            end
        end

        # Calculate expected information gain of states
        if use_states_info_gain
            # If ReverseDiff is tracking the information gain, get the value
            if ReverseDiff.istracked(calc_states_info_gain(A, qs_pi))
                G[idx] += ReverseDiff.value(calc_states_info_gain(A, qs_pi))

            # Otherwise calculate it and add it to the G vector
            else
                G[idx] += calc_states_info_gain(A, qs_pi)
            end
        end

        # Calculate expected information gain of parameters (learning)
        if use_param_info_gain
            if pA !== nothing

                # if ReverseDiff is tracking pA information gain, get the value
                if ReverseDiff.istracked(calc_pA_info_gain(pA, qo_pi, qs_pi))
                    G[idx] += ReverseDiff.value(calc_pA_info_gain(pA, qo_pi, qs_pi))
                # Otherwise calculate it and add it to the G vector
                else
                    G[idx] += calc_pA_info_gain(pA, qo_pi, qs_pi)
                end
            end

            if pB !== nothing
                G[idx] += calc_pB_info_gain(pB, qs_pi, qs, policy)
            end
        end

    end
    
    q_pi = softmax(G * gamma + lnE, dims=1)

    return q_pi, G
end

""" Get Expected Observations """
function get_expected_obs(qs_pi, A::Vector{Array{T,N}} where {T <: Real, N})
    n_steps = length(qs_pi)
    qo_pi = []

    for t in 1:n_steps
        qo_pi_t = Vector{Any}(undef, length(A))
        qo_pi = push!(qo_pi, qo_pi_t)
    end

    for t in 1:n_steps
        for (modality, A_m) in enumerate(A)
            qo_pi[t][modality] = dot_product(A_m, qs_pi[t])
        end
    end

    return qo_pi
end

""" Calculate Expected Utility """
function calc_expected_utility(qo_pi, C)
    n_steps = length(qo_pi)
    expected_utility = 0.0
    num_modalities = length(C)

    modalities_to_tile = [modality_i for modality_i in 1:num_modalities if ndims(C[modality_i]) == 1]

    C_tiled = Vector{Matrix{Float64}}(undef, num_modalities)

    for modality in 1:num_modalities
        modality_data = reshape(C[modality], :, 1)
        if modality in modalities_to_tile
            C_tiled[modality] = repeat(modality_data, 1, n_steps)
        else
            C_tiled[modality] = modality_data
        end
    end
    
    C_prob = softmax_array(C_tiled)
    lnC =[]
    for t in 1:n_steps
        for modality in 1:num_modalities
            lnC = capped_log(C_prob[modality][:, t])
            expected_utility += dot(qo_pi[t][modality], lnC) 
        end
    end

    return expected_utility
end

""" Calculate States Information Gain """
function calc_states_info_gain(A, qs_pi)
    n_steps = length(qs_pi)
    states_surprise = 0.0

    for t in 1:n_steps
        states_surprise += calculate_bayesian_surprise(A, qs_pi[t])
    end

    return states_surprise
end

""" Calculate observation to state info Gain """
function calc_pA_info_gain(pA, qo_pi, qs_pi)

    n_steps = length(qo_pi)
    num_modalities = length(pA)

    wA = Vector{Any}(undef, num_modalities)
    for (modality, pA_m) in enumerate(pA)
        wA[modality] = spm_wnorm(pA[modality])
    end

    pA_info_gain = 0

    for modality in 1:num_modalities
        wA_modality = wA[modality] .* (pA[modality] .> 0)

        for t in 1:n_steps
            pA_info_gain -= dot(qo_pi[t][modality], dot_product(wA_modality, qs_pi[t]))
        end
    end
    return pA_info_gain
end

""" Calculate state to state info Gain """
function calc_pB_info_gain(pB, qs_pi, qs_prev, policy)
    n_steps = length(qs_pi)
    num_factors = length(pB)

    wB = Vector{Any}(undef, num_factors)
    for (factor, pB_f) in enumerate(pB)
        wB[factor] = spm_wnorm(pB_f)
    end

    pB_info_gain = 0

    for t in 1:n_steps
        if t == 1
            previous_qs = qs_prev
        else
            previous_qs = qs_pi[t-1]
        end

        policy_t = policy[t, :]

        for (factor, a_i) in enumerate(policy_t)
            wB_factor_t = wB[factor][:,:,Int(a_i)] .* (pB[factor][:,:,Int(a_i)] .> 0)
            pB_info_gain -= dot(qs_pi[t][factor], wB_factor_t * previous_qs[factor])
        end
    end
    return pB_info_gain
end

### Action Sampling ###
""" Sample Action [Stochastic or Deterministic] """
function sample_action(q_pi, policies::Vector{Matrix{Int64}}, num_controls; action_selection="stochastic", alpha=16.0)
    num_factors = length(num_controls)
    selected_policy = zeros(Real,num_factors)
    
    eltype_q_pi = eltype(q_pi)

    # Initialize action_marginals with the correct element type
    action_marginals = create_matrix_templates(num_controls, "zeros", eltype_q_pi)

    for (pol_idx, policy) in enumerate(policies)
        for (factor_i, action_i) in enumerate(policy[1,:])
            action_marginals[factor_i][action_i] += q_pi[pol_idx]
        end
    end

    action_marginals = normalize_arrays(action_marginals)

    for factor_i in 1:num_factors
        if action_selection == "deterministic"
            selected_policy[factor_i] = select_highest(action_marginals[factor_i])
        elseif action_selection == "stochastic"
            log_marginal_f = capped_log(action_marginals[factor_i])
            p_actions = softmax(log_marginal_f * alpha, dims=1)
            selected_policy[factor_i] = action_select(p_actions)
        end
    end
    return selected_policy
end

""" Edited Compute Accuracy [Still needs to be nested within Fixed-Point Iteration] """
function compute_accuracy_new(log_likelihood, qs::Vector{Vector{Real}})
    n_factors = length(qs)
    ndims_ll = ndims(log_likelihood)
    dims = (ndims_ll - n_factors + 1) : ndims_ll

    result_size = size(log_likelihood, 1) 
    results = zeros(Real,result_size)

    for indices in Iterators.product((1:size(log_likelihood, i) for i in 1:ndims_ll)...)
        product = log_likelihood[indices...] * prod(qs[factor][indices[dims[factor]]] for factor in 1:n_factors)
        results[indices[1]] += product
    end

    return results
end

""" Calculate State-Action Prediction Error """
function calculate_SAPE(aif::POMDPActiveInference)

    qs_pi_all = get_expected_states(aif.qs_current, aif.B, aif.policies)
    qs_bma = bayesian_model_average(qs_pi_all, aif.Q_pi)

    if length(aif.states["bayesian_model_averages"]) != 0
        sape = kl_divergence(qs_bma, aif.states["bayesian_model_averages"][end])
        push!(aif.states["SAPE"], sape)
    end

    push!(aif.states["bayesian_model_averages"], qs_bma)
end
