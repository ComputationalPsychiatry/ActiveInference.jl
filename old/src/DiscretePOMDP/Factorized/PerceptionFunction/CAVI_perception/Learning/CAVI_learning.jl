function ActiveInferenceCore.perception(
    model::AIFModel{GenerativeModel, CAVI{Learning}, ActionProcess},
    observation::Vector{Int},
    action::Union{Nothing, Vector{Int}} = nothing
)

    if model.action_process.previous_action !== nothing
        int_action = round.(Int, action)
        prediction_states = get_states_prediction(model.perceptual_process.posterior_states, model.generative_model.B, reshape(int_action, 1, length(int_action)))[1]
    else
        prediction_states = model.perceptual_process.prediction_states
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
        prior = prediction_states,
        num_iter = model.perceptual_process.num_iter,
        dF_tol = model.perceptual_process.dF_tol
    )

    learning_posterior = update_parameters(model, observation, posterior_states, action)

    return (posterior_states = posterior_states, prediction_states = prediction_states, learning_posterior = learning_posterior)
end

""" Update the models's beliefs over states with previous posterior states and action """
function ActiveInferenceCore.perception(
    model::AIFModel{GenerativeModel, CAVI{Learning}, ActionProcess},
    observation::Vector{Int},
    previous_posterior_states::Union{Nothing, Vector{Vector{Float64}}},
    previous_action::Union{Nothing, Vector{Int}} 
)

    int_action = round.(Int, previous_action)
    prediction_states = get_states_prediction(previous_posterior_states, model.generative_model.B, reshape(int_action, 1, length(int_action)))[1]

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
        prior = prediction_states,
        num_iter = model.perceptual_process.optim_engine.num_iter,
        dF_tol = model.perceptual_process.optim_engine.dF_tol
    )

    return posterior_states, prediction_states
end