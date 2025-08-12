""" Function for predicting states and observations based on the agent's perceptual process and generative model. """

function ActiveInferenceCore.prediction(
    agent::AIFModel{GenerativeModel, PP, ActionProcess}, 
    posterior_states::Vector{Vector{Float64}}
) where PP <: AbstractPerceptualProcess

    all_predicted_states = get_expected_states(posterior_states, agent.generative_model.B, agent.action_process.policies)
    all_predicted_observations = get_expected_obs(all_predicted_states, agent.generative_model.A)

    return all_predicted_states, all_predicted_observations
end