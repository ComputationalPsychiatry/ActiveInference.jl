"""
    AbstractActionType

Marker for the type of action space in a generative model. Subtypes include:
- `DiscreteActions`: Finite, categorical action spaces.
- `ContinuousActions`: Continuous control spaces.
- `MixedActions`: Hybrid spaces with both discrete and continuous components.
- `NoActions`: Models without an action component.
"""
abstract type AbstractActionType end

"""
    DiscreteActions <: AbstractActionType

    Marker for discrete action spaces in generative models.

"""
abstract type DiscreteActions <: AbstractActionType end

"""
    ContinuousActions <: AbstractActionType

    Marker for continuous action spaces in generative models.
"""
abstract type ContinuousActions <: AbstractActionType end

"""
    MixedActions <: AbstractActionType

    Marker for mixed action spaces in generative models, which include both discrete and continuous components.
"""
abstract type MixedActions <: AbstractActionType end

"""
    NoActions <: AbstractActionType

    Marker for generative models that do not include an action component.
"""
abstract type NoActions <: AbstractActionType end

"""
    AbstractObservationType

Marker for the type of observation space in a generative model. Subtypes include:
- `DiscreteObservations`: Finite/categorical observation spaces.
- `ContinuousObservations`: Continuous observation spaces.
- `MixedObservations`: Hybrid observation spaces.
- `NoObservations`: Models with no observations.
"""
abstract type AbstractObservationType end

"""
    DiscreteObservations <: AbstractObservationType

Marker for discrete/categorical observation spaces in generative models.
"""
abstract type DiscreteObservations<:AbstractObservationType end

"""
    ContinuousObservations <: AbstractObservationType

Marker for continuous observation spaces in generative models.
"""
abstract type ContinuousObservations<:AbstractObservationType end

"""
    MixedObservations <: AbstractObservationType

Marker for mixed observation spaces containing both discrete and continuous observables.
"""
abstract type MixedObservations<:AbstractObservationType end

"""
    NoObservations <: AbstractObservationType

Marker for generative models without an observation component.
"""
abstract type NoObservations<:AbstractObservationType end


"""
    AbstractGenerativeModel{AbstractActionType, AbstractObservationType, AbstractStateType}

Abstract type for generative models in active inference. Parameterized by:
- `AbstractActionType`: The type of action space (e.g., `DiscreteActions`, `ContinuousActions`).
- `AbstractObservationType`: The type of observation space (e.g., `DiscreteObservations`, `ContinuousObservations`).

Used for type dispatch in action planning and inference algorithms.
    
"""
abstract type AbstractGenerativeModel{
    action_type<:AbstractActionType,
    observation_type<:AbstractObservationType,
} end


"""
    AbstractInferenceEnvironment

Marker for the process of inferring the environment. This can be extended to provide specific inference schemes.
This is the part of active inference where inference is made over the state, parmaeters and structure of the environment (AKA. perceptual inference)
"""
abstract type AbstractInferenceEnvironment end

"""
    AbstractInferenceActions

Marker for the process of inferring actions. This can be extended to provide specific action selection schemes.
This is the part of active inference where inference is made over the action space (AKA. active inference)
"""
abstract type AbstractInferenceActions end


"""
    AIFModel{GM, PP, AP}

    The main struct for an active inference model. Parameterized by:
    - `GM`: The generative model, which must be a subtype of `AbstractGenerativeModel`.
    - `PP`: The perceptual process, which must be a subtype of `AbstractInferEnvironment`.
    - `AP`: The action process, which must be a subtype of `AbstractInferActions`.

    This struct encapsulates the generative model and the processes for perception and action selection in active inference.
"""
struct AIFModel{
    T_generative_model<:AbstractGenerativeModel,
    T_inference_environment<:AbstractInferenceEnvironment,
    T_inference_actions<:AbstractInferenceActions,
}

    generative_model::T_generative_model

    inference_environment::T_inference_environment

    inference_actions::T_inference_actions

end

#Error constructor for abstract types
function AIFModel(generative_model, inference_environment, inference_actions)
    throw(ArgumentError("""
                        Invalid AIFModel construction. 

                        Expected:
                        <: AbstractGenerativeModel
                        <: AbstractInferenceEnvironment
                        <: AbstractInferenceActions

                        Received:
                        - $(typeof(generative_model))
                        - $(typeof(inference_environment))
                        - $(typeof(inference_actions))

                        Ensure you are passing concrete instances of the required schemes.
                        """))

end

"""
    infer_environment(model::AIFModel, observation, previous_action)

    Placeholder function for inferring the environment (perception) in an active inference model. 
    This function should be extended with specific implementations based on the types of generative model and inference schemes used.

    Parameters:
    - `model`: An instance of `AIFModel` containing the generative model and inference schemes.
    - `observation`: The current observation from the environment, which will be used to update beliefs about the environment.
    - `previous_action`: The action taken on the previous timestep, which may be relevant for updating beliefs about the environment.

    Returns:
    - Updated beliefs about the environment based on the provided observation and the model's generative structure.
"""
function infer_environment(model::AIFModel, observation, previous_action)
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
    set_variables!(model::AIFModel, observation, previous_action, environment_posterior, action_posterior)

Stores new values in the active inference model for use on later timesteps. 

# Arguments
- `model`: The `AIFModel` instance to be updated.
- `observation`: The latest observation received from the environment.
- `previous_action`: The action that was actually executed on the previous timestep..
- `environment_posterior`: The updated belief about the environment.
- `action_posterior`: The updated belief/distribution over actions.

# Returns
- The updated `model` (by convention for mutating functions).
"""
function set_variables!(
    model::AIFModel,
    observation,
    previous_action,
    environment_posterior,
    action_posterior,
)
    throw(
        ArgumentError(
            """
            No implementation found for `set_variables!` with the provided model components.


            Model Details:
            $(typeof(model.generative_model))
            $(typeof(model.inference_environment))
            $(typeof(model.inference_actions))

            Data Types Received:
            - Observation: $(typeof(observation))
            - Previous action:      $(typeof(previous_action))
            - Environment posterior: $(typeof(environment_posterior))
            - Action posterior: $(typeof(action_posterior))
            """,
        ),
    )

end

"""

    active_inference(model::AIFModel, observation, previous_action)

    Main function for performing active inference. First updates beliefs about environment, then updates beliefs about actions, and finally stores relevant variables for the next timestep.

    Arguments:
    - `model`: An instance of `AIFModel` containing the generative model and inference schemes.
    - `observation`: The current observation from the environment.
    - `previous_action`: The action taken on the previous timestep.

    Returns:
    - The posterior distribution over actions resulting from the active inference process.

"""
function active_inference!(model::AIFModel, observation, previous_action)

    # Update beliefs about the environment
    environment_posterior = infer_environment(model, observation, previous_action)

    # Update beliefs about which action to take
    action_posterior = infer_actions(model, environment_posterior)

    # Store variables
    set_variables!(
        model,
        observation,
        previous_action,
        environment_posterior,
        action_posterior,
    )

    # Return the action posterior 
    return action_posterior
end
