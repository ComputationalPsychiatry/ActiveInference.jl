``` Planning function for action distribution in a discrete POMDP model ```

function ActiveInferenceCore.planning(
    agent::AIFModel{GenerativeModel, PerceptualProcess{T}, ActionProcess}, 
    posterior_states::Vector{Vector{Float64}},
    predicted_states::Vector{Vector{Vector{Vector{Float64}}}},
    predicted_observations::Vector{Vector{Vector{Vector{Float64}}}}
) where T<:AbstractOptimEngine

    # Get posterior over policies and expected free energies
    q_pi, G = update_posterior_policies(
        qs = posterior_states,
        A = agent.generative_model.A,
        C = agent.generative_model.C,
        policies = agent.action_process.policies,
        qs_pi_all = predicted_states,
        qo_pi_all = predicted_observations,
        use_utility = agent.action_process.use_utility,
        use_states_info_gain = agent.action_process.use_states_info_gain,
        use_param_info_gain = agent.action_process.use_param_info_gain,
        A_learning = agent.perceptual_process.A_learning,
        B_learning = agent.perceptual_process.B_learning,
        E = agent.action_process.E,
        gamma = agent.action_process.gamma
    )

    # agent.action_process.posterior_policies = q_pi
    # agent.action_process.expected_free_energy = G

    # If action is enabled, sample an action from the posterior policies
    return q_pi, G
end