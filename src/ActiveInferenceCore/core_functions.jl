
"""
    infer_environment(model::AIFModel, observation, previous_action)

    Placeholder function for inferring the environment (perception) in an active inference model. 
    This function should be extended with specific implementations based on the types of generative model and inference schemes used.

    Parameters:
    - `model`: An instance of `AIFModel` containing the generative model and inference schemes.
    - `observation`: The current observation from the environment, which will be used to update beliefs about the environment.
    - `previous_action`: The action taken on the previous timestep, which may be relevant for updating beliefs about the environment.
    - `environment_prior`: The agent's belief about the environment state before incorporating the current observation.

    Returns:
    - Updated beliefs about the environment based on the provided observation and the model's generative structure.
    - A snapshot for storing the history (optional).
"""
function infer_environment(model::AIFModel, observation, previous_action, environment_prior)
    throw(
        ArgumentError(
            """
            No implementation found for `infer_environment` with the provided model components.

            Model Details:
            $(typeof(model.generative_model))
            $(typeof(model.inference_environment))
            $(typeof(model.inference_actions))

            Observation Type:  $(typeof(observation))
            Previous action type: $(typeof(previous_action))
            Environment prior type: $(typeof(environment_prior))
            """,
        ),
    )
end


"""
    infer_actions(model::AIFModel, environment_posterior)

Placeholder function for inferring the next action in an active inference model.

# Parameters
- `model`: An instance of `AIFModel`.
- `environment_posterior`: The current belief/posterior distribution over the environment, typically the output from `infer_environment`.

# Returns
- A posterior distribution over the next action.
 - A snapshot for storing the history (optional).
"""
function infer_actions(model::AIFModel, environment_posterior)
    throw(
        ArgumentError(
            """
            No implementation found for `infer_actions` with the provided model components.

            Model details:
            $(typeof(model.generative_model))
            $(typeof(model.inference_environment))
            $(typeof(model.inference_actions))

            - Environment posterior type:    $(typeof(environment_posterior))
            """,
        ),
    )

end


"""
    predict_environment(model::AIFModel, environment_belief, action)

Compute a predictive posterior for the environment, given a generative model, current beliefs, and an action. 
The predictive posterior may or may not be over observations as well as unobservables, but the function get_expected_observations is the interface to access this.

# Arguments
- `model`: An instance of `AIFModel` containing the generative model.
- `environment_belief`: The current belief/posterior distribution over the environment states.
- `action`: The proposed action for which to predict future states.

# Returns
- The predicted distribution over future states.
"""
function predict_environment(model::AIFModel, environment_belief, action)
    throw(
        ArgumentError(
            """
            No implementation found for `predict_environment` with the provided model components.

            Model details:
            $(typeof(model.generative_model))
            $(typeof(model.inference_environment))
            $(typeof(model.inference_actions))

            - Environment belief type:    $(typeof(environment_belief))
            - Action type:      $(typeof(action))
            """,
        ),
    )

end


"""
    get_expected_observations(model::AIFModel, environment_belief, action)

Returns the distribution over expected observations, given a belief (or prediction) about the environment, dependent on an action.

# Arguments
- `model`: An instance of `AIFModel` containing the generative model.
- `environment_belief`: The current belief/posterior distribution over the environment states.
- `action`: The proposed action for which to predict future observations.

# Returns
- The predicted distribution over future observations.
"""
function get_expected_observations(model::AIFModel, environment_belief, action)
    throw(
        ArgumentError(
            """
            No implementation found for `get_expected_observations` with the provided model components.

            Model details:
            $(typeof(model.generative_model))
            $(typeof(model.inference_environment))
            $(typeof(model.inference_actions))

            - Environment belief type:    $(typeof(environment_belief))
            - Action type:      $(typeof(action))
            """,
        ),
    )

end



"""
    calculate_cost(aif_model::AIFModel, environment_belief)

Calculates the cost (e.g., Expected Free Energy) associated with a particular belief about the environment. 
This is typically used during action planning to evaluate the quality of future states.

# Arguments
- `aif_model`: An instance of `AIFModel` containing the generative model.
- `environment_belief`: A belief or distribution over environment states.

# Returns
- A scalar cost value.
"""
function calculate_cost(aif_model::AIFModel, environment_belief)
    throw(
        ArgumentError(
            """
            No implementation found for `calculate_cost` with the provided model components.

            Model details:
            $(typeof(aif_model.generative_model))
            $(typeof(aif_model.inference_environment))
            $(typeof(aif_model.inference_actions))

            - Environment belief type:    $(typeof(environment_belief))
            """,
        ),
    )
end

"""
    active_inference(model::AIFModel, observation, previous_action, environment_prior)

Main function for performing active inference. First updates beliefs about environment, then updates beliefs about actions, and finally stores relevant variables for the next timestep.

Arguments:
- `model`: An instance of `AIFModel` containing the generative model and inference schemes.
- `observation`: The current observation from the environment.
- `previous_action`: The action taken on the previous timestep.
- `environment_prior`: The agent's belief about the environment state before incorporating the current observation.

Returns:
- The posterior distribution over actions resulting from the active inference process.
- The updated belief about the environment.
- Snapshots for both the environment inference and action inference processes (optional).

"""
function active_inference(model::AIFModel, observation, previous_action, environment_prior)

    # Update beliefs about the environment
    environment_posterior, infer_environment_snapshot = infer_environment(model, observation, previous_action, environment_prior)

    # Update beliefs about which action to take
    action_posterior, infer_actions_snapshot = infer_actions(model, environment_posterior)

    # Return the action posterior 
    return action_posterior, environment_posterior, infer_environment_snapshot, infer_actions_snapshot
end

