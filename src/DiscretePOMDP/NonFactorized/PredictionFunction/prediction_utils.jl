```Util functions for prediction in Discrete POMDPs ```

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
    qs_pi_all = Vector{Vector{Vector{Vector{Float64}}}}(undef, n_policies)

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


""" Get Expected Observations """
function get_expected_obs(qs_pi::Vector{Vector{Vector{Float64}}}, A::Vector{Array{T,N}} where {T <: Real, N})
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

""" Get Expected Observations """
function get_expected_obs(qs_pi::Vector{Vector{Vector{Vector{Float64}}}}, A::Vector{Array{T,N}} where {T <: Real, N})
    n_policies = length(qs_pi)
    n_steps = length(qs_pi[1])
    
    # Predefined vector to store results for all policies
    qo_pi_all = Vector{Vector{Vector{Vector{Float64}}}}(undef, n_policies)

    # Loop over each policy
    for policy_idx in 1:n_policies
        qo_pi = []

        for t in 1:n_steps
            qo_pi_t = Vector{Any}(undef, length(A))
            qo_pi = push!(qo_pi, qo_pi_t)
        end

        for t in 1:n_steps
            for (modality, A_m) in enumerate(A)
                qo_pi[t][modality] = dot_product(A_m, qs_pi[policy_idx][t])
            end
        end
        
        qo_pi_all[policy_idx] = qo_pi
    end

    return qo_pi_all
end