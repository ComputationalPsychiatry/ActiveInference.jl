""" Sample Action [Stochastic or Deterministic] """
function sample_action(;q_pi, policies::Vector{Matrix{Int64}}, n_controls, action_selection="stochastic", alpha=16.0)
    num_factors = length(n_controls)
    selected_policy = zeros(Real,num_factors)
    
    eltype_q_pi = eltype(q_pi)

    # Initialize action_marginals with the correct element type
    action_marginals = create_matrix_templates(n_controls, "zeros", eltype_q_pi)

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
