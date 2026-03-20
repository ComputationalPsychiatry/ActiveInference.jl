``` sample action from posterior over policies ```

function ActiveInferenceCore.selection(
    model::AIFModel{GenerativeModel, P, ActionProcess},
    policy_posterior::NamedTuple{(:q_pi, :G, :predictions), Tuple{Vector{Float64}, Vector{Float64}, NamedTuple{(:all_predicted_states, :all_predicted_observations), Tuple{Vector{Vector{Vector{Vector{Float64}}}}, Vector{Vector{Vector{Vector{Float64}}}}}}}};
    alpha::Float64 = 16.0
) where P <: AbstractPerceptualProcess

    n_controls = model.generative_model.info.controls_per_factor
    num_factors = length(n_controls)
    selected_policy = zeros(Real, num_factors)

    eltype_q_pi = eltype(policy_posterior.q_pi)

    # Initialize action_marginals with the correct element type
    action_marginals = create_matrix_templates(n_controls, "zeros", eltype_q_pi)

    # Extract policies
    policies = model.action_process.policies

    action_selection = Val(model.action_process.action_selection)

    for (pol_idx, policy) in enumerate(policies)
        for (factor_i, action_i) in enumerate(policy[1,:])
            action_marginals[factor_i][action_i] += policy_posterior.q_pi[pol_idx]
        end
    end

    action_marginals = normalize_arrays(action_marginals)

    return (posterior = action_marginals,)

end

function ActiveInferenceCore.selection(
    model::AIFModel{GenerativeModel, CAVI{L}, ActionProcess},
    policy_posterior::NamedTuple{(:q_pi, :G, :predictions), Tuple{Vector{Float64}, Vector{Float64}, NamedTuple{(:all_predicted_states, :all_predicted_observations), Tuple{Vector{Vector{Vector{Vector{Float64}}}}, Vector{Vector{Vector{Vector{Float64}}}}}}}};
    alpha::Float64 = 16.0
) where {L}

    n_controls = model.generative_model.info.controls_per_factor
    num_factors = length(n_controls)
    selected_policy = zeros(Real, num_factors)

    eltype_q_pi = eltype(policy_posterior.q_pi)

    # Initialize action_marginals with the correct element type
    action_marginals = create_matrix_templates(n_controls, "zeros", eltype_q_pi)

    # Extract policies
    policies = model.action_process.policies

    action_selection = Val(model.action_process.action_selection)

    for (pol_idx, policy) in enumerate(policies)
        for (factor_i, action_i) in enumerate(policy[1,:])
            action_marginals[factor_i][action_i] += policy_posterior.q_pi[pol_idx]
        end
    end

    action_marginals = normalize_arrays(action_marginals)

    return (posterior = action_marginals,)
    # for factor_i in 1:num_factors
    #     if action_selection == Val(:deterministic)
    #         selected_policy[factor_i] = select_highest(action_marginals[factor_i])

    #     elseif action_selection == Val(:stochastic)
    #         log_marginal_f = capped_log(action_marginals[factor_i])
    #         p_actions = softmax(log_marginal_f * alpha, dims=1)
    #         selected_policy[factor_i] = action_select(p_actions)
    #     end
    # end
    # return (selected_policy = selected_policy,)
end