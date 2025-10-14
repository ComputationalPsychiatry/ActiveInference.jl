using ActiveInference: AIFModel


function ActionModels.reset!(model::AIFModel)
    reset_state!(model, model.perceptual_process)
    reset_state!(model.action_process)
end



function reset_state!(model::AIFModel, perceptual_process::DiscretePOMDP.CAVI)
    perceptual_process.posterior_states = nothing
    perceptual_process.previous_posterior_states = nothing
    perceptual_process.observation = nothing
    perceptual_process.predicted_states = nothing
    perceptual_process.predicted_observations = nothing

    n_states = model.generative_model.info.n_states
    perceptual_process.prediction_states = [fill(1.0 / n, n) for n in n_states]
end



function reset_state!(action_process::DiscretePOMDP.ActionProcess)
    action_process.previous_action = nothing
    action_process.posterior_policies = nothing
    action_process.expected_free_energy = nothing
end