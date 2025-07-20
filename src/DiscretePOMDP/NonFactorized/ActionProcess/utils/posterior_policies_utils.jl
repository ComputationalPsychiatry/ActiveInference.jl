""" Update Posterior over Policies """
function update_posterior_policies(;
    qs::Vector{Vector{T}} where T <: Real,
    A::Vector{Array{T, N}} where {T <: Real, N},
    C::Vector{Vector{T}} where T <: Real,
    qs_pi_all::Vector{Vector{Vector{Vector{Float64}}}},
    qo_pi_all::Vector{Vector{Vector{Vector{Float64}}}},
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
    lnE = capped_log(E)

    for (idx, policy) in enumerate(policies)

        # Get the expected states and observations for the current policy
        qs_pi = qs_pi_all[idx]
        qo_pi = qo_pi_all[idx]

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