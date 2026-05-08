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
struct DiscreteActions <: AbstractActionType end

"""
    ContinuousActions <: AbstractActionType

    Marker for continuous action spaces in generative models.
"""
struct ContinuousActions <: AbstractActionType end

"""
    MixedActions <: AbstractActionType

    Marker for mixed action spaces in generative models, which include both discrete and continuous components.
"""
struct MixedActions <: AbstractActionType end

"""
    NoActions <: AbstractActionType

    Marker for generative models that do not include an action component.
"""
struct NoActions <: AbstractActionType end

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
struct DiscreteObservations <: AbstractObservationType end

"""
    ContinuousObservations <: AbstractObservationType

Marker for continuous observation spaces in generative models.
"""
struct ContinuousObservations <: AbstractObservationType end

"""
    MixedObservations <: AbstractObservationType

Marker for mixed observation spaces containing both discrete and continuous observables.
"""
struct MixedObservations <: AbstractObservationType end

"""
    NoObservations <: AbstractObservationType

Marker for generative models without an observation component.
"""
struct NoObservations <: AbstractObservationType end


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
} <: AbstractAIFModel

    generative_model::T_generative_model

    inference_environment::T_inference_environment

    inference_actions::T_inference_actions

end

#Error constructor for abstract types
function AIFModel(generative_model, inference_environment, inference_actions)
    throw(
        ArgumentError("""
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
                      """),
    )

end