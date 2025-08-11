``` sample action from posterior over policies ```
function ActiveInferenceCore.selection(
    model::AIFModel{GenerativeModel, PerceptualProcess{T}, ActionProcess},
    q_pi::Union{Vector{Float64}, Nothing};
    action_selection::Union{Val{:stochastic}, Val{:deterministic}} = Val(:stochastic),
    alpha::Float64 = 16.0
) where T<:AbstractOptimEngine

    n_controls = model.generative_model.info.controls_per_factor
    num_factors = length(n_controls)
    selected_policy = zeros(Real, num_factors)
    
    eltype_q_pi = eltype(q_pi)

    # Initialize action_marginals with the correct element type
    action_marginals = create_matrix_templates(n_controls, "zeros", eltype_q_pi)

    # Extract policies
    policies = model.action_process.policies

    for (pol_idx, policy) in enumerate(policies)
        for (factor_i, action_i) in enumerate(policy[1,:])
            action_marginals[factor_i][action_i] += q_pi[pol_idx]
        end
    end

    action_marginals = normalize_arrays(action_marginals)

    for factor_i in 1:num_factors
        if action_selection == Val(:deterministic)
            selected_policy[factor_i] = select_highest(action_marginals[factor_i])

        elseif action_selection == Val(:stochastic)
            log_marginal_f = capped_log(action_marginals[factor_i])
            p_actions = softmax(log_marginal_f * alpha, dims=1)
            selected_policy[factor_i] = action_select(p_actions)
        end
    end
    return selected_policy
end

