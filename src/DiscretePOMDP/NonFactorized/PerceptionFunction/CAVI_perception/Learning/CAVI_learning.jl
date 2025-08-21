function ActiveInferenceCore.perception(
    model::AIFModel{GenerativeModel, CAVI{Learning}, ActionProcess},
    observation::Vector{Int}
)

    if model.action_process.action !== nothing
        int_action = round.(Int, model.action_process.action)
        prior_qs_prediction = get_expected_states(model.perceptual_process.posterior_states, model.generative_model.B, reshape(int_action, 1, length(int_action)))[1]
    else
        prior_qs_prediction = model.perceptual_process.prior_qs_prediction
    end

    # make observations into a one-hot encoded vector
    processed_observation = process_observation(
        observation, 
        model.generative_model.info.n_modalities, 
        model.generative_model.info.n_observations
    )

    # perform fixed-point iteration
    posterior_states = cavi(;
        A = model.generative_model.A,
        observation = processed_observation,
        n_factors = model.generative_model.info.n_factors,
        n_states = model.generative_model.info.n_states,
        prior = prior_qs_prediction,
        num_iter = model.perceptual_process.num_iter,
        dF_tol = model.perceptual_process.dF_tol
    )

    return posterior_states, prior_qs_prediction
end

""" Update the models's beliefs over states with previous posterior states and action """
function ActiveInferenceCore.perception(
    model::AIFModel{GenerativeModel, CAVI{Learning}, ActionProcess},
    observation::Vector{Int},
    previous_posterior_states::Union{Nothing, Vector{Vector{Float64}}},
    previous_action::Union{Nothing, Vector{Int}} 
)

    int_action = round.(Int, previous_action)
    prior_qs_prediction = get_expected_states(previous_posterior_states, model.generative_model.B, reshape(int_action, 1, length(int_action)))[1]

    # make observations into a one-hot encoded vector
    processed_observation = process_observation(
        observation, 
        model.generative_model.info.n_modalities, 
        model.generative_model.info.n_observations
    )

    # perform fixed-point iteration
    posterior_states = cavi(;
        A = model.generative_model.A,
        observation = processed_observation,
        n_factors = model.generative_model.info.n_factors,
        n_states = model.generative_model.info.n_states,
        prior = prior_qs_prediction,
        num_iter = model.perceptual_process.optim_engine.num_iter,
        dF_tol = model.perceptual_process.optim_engine.dF_tol
    )

    return posterior_states, prior_qs_prediction
end