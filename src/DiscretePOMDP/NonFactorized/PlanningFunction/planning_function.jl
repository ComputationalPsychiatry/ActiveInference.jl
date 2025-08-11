``` Planning function for action distribution in a discrete POMDP model ```

function ActiveInferenceCore.planning(
    model::AIFModel{GenerativeModel, PerceptualProcess{T}, ActionProcess}, 
    posterior_states::Vector{Vector{Float64}},
    predicted_states::Vector{Vector{Vector{Vector{Float64}}}},
    predicted_observations::Vector{Vector{Vector{Vector{Float64}}}}
) where T<:AbstractOptimEngine

    # Get posterior over policies and expected free energies
    q_pi, G = update_posterior_policies(
        qs = posterior_states,
        A = model.generative_model.A,
        C = model.generative_model.C,
        policies = model.action_process.policies,
        qs_pi_all = predicted_states,
        qo_pi_all = predicted_observations,
        use_utility = model.action_process.use_utility,
        use_states_info_gain = model.action_process.use_states_info_gain,
        use_param_info_gain = model.action_process.use_param_info_gain,
        A_learning = model.perceptual_process.A_learning,
        B_learning = model.perceptual_process.B_learning,
        E = model.action_process.E,
        gamma = model.action_process.gamma
    )

    return q_pi, G
end