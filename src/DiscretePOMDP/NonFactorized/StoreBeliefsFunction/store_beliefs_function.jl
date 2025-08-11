function ActiveInferenceCore.store_beliefs!(
    model::AIFModel{GenerativeModel, PerceptualProcess{T}, ActionProcess};
    observation::Vector{Int},
    qs::Vector{Vector{Float64}},
    prior_qs_prediction::Vector{Vector{Float64}},
    qs_pi_all::Vector{Vector{Vector{Vector{Float64}}}},
    qo_pi_all::Vector{Vector{Vector{Vector{Float64}}}},
    q_pi::Vector{Float64},
    G::Vector{Float64},
    action::Vector{N}
) where {T <: AbstractOptimEngine, N <: Real}

    # Store beliefs in the perceptual process struct
    model.perceptual_process.posterior_states = qs
    model.perceptual_process.prior_qs_prediction = prior_qs_prediction
    model.perceptual_process.observation = observation

    model.perceptual_process.predicted_states = qs_pi_all
    model.perceptual_process.predicted_observations = qo_pi_all

    # Store beliefs in the action process struct
    model.action_process.posterior_policies = q_pi
    model.action_process.expected_free_energy = G
    model.action_process.action = action

end