### Generative model types ###
# Abstract types for defining the types of actions, observations, and states a generative model can handle.
abstract type AbstractActionType end
abstract type DiscreteActions<:AbstractActionType end
abstract type ContinuousActions<:AbstractActionType end
abstract type MixedActions<:AbstractActionType end
abstract type NoActions<:AbstractActionType end

abstract type AbstractObservationType end
abstract type DiscreteObservations<:AbstractObservationType end
abstract type ContinuousObservations<:AbstractObservationType end
abstract type MixedObservations<:AbstractObservationType end
abstract type NoObservations<:AbstractObservationType end

abstract type AbstractStateType end
abstract type DiscreteStates<:AbstractStateType end
abstract type ContinuousStates<:AbstractStateType end
abstract type MixedStates<:AbstractStateType end
abstract type NoStates<:AbstractStateType end

#Abstract type for generative models
abstract type AbstractGenerativeModel{
    TypeAction<:AbstractActionType,
    TypeObservation<:AbstractObservationType,
    TypeState<:AbstractStateType,
} end

# Perceptual Process abstract type
abstract type AbstractPerceptualProcess end

### Action process types ###
abstract type AbstractActionProcess end

#NOTE: when making this agent, make the prior be defined by D in the generative model as part of the initialization function
struct AIFModel{
    GM <: AbstractGenerativeModel,
    PP <: AbstractPerceptualProcess,
    AP <: AbstractActionProcess
}
    ## Generative Model
    generative_model::GM

    ## Perceptual process
    perceptual_process::PP
    
    ## Action process 
    action_process::AP

end

function AIFModel(
    generative_model::AbstractGenerativeModel{AbstractActionType, AbstractObservationType, AbstractStateType},
    perceptual_process::AbstractPerceptualProcess,
    action_process::AbstractActionProcess
)

    @error "Please create a constructor for AIFModel that utilizes concrete types.
            The current generative model is: $(typeof(generative_model)),
            the perceptual process is: $(typeof(perceptual_process)),
            and the action process is: $(typeof(action_process))"

end

function perception(
    model::AIFModel{AbstractGenerativeModel{AbstractActionType, AbstractObservationType, AbstractStateType}, AbstractPerceptualProcess, AbstractActionProcess},
    observation::Vector{Real}
) 
    @error "Please create a perception function utilizing concrete type.
            The current model is: $(typeof(model))"

end

function policy_predictions(
    model::AIFModel{AbstractGenerativeModel{AbstractActionType, AbstractObservationType, AbstractStateType}, AbstractPerceptualProcess, AbstractActionProcess}
)

    @error "Please create a policy_predictions function utilizing concrete type.
            The current model is: $(typeof(model))"

end

function planning(
    model::AIFModel{AbstractGenerativeModel{AbstractActionType, AbstractObservationType, AbstractStateType}, AbstractPerceptualProcess, AbstractActionProcess}
)

    @error "Please create an action function utilizing concrete type.
            The current model is: $(typeof(model))"

end

function selection(
    model::AIFModel{AbstractGenerativeModel{AbstractActionType, AbstractObservationType, AbstractStateType}, AbstractPerceptualProcess, AbstractActionProcess},
    selection_type::Union{Val{:stochastic}, Val{:deterministic}} = Val(:stochastic)
)

    @error "Please create an action function utilizing concrete type.
            The current model is: $(typeof(model))"

end

function store_beliefs!(
    model::AIFModel{AbstractGenerativeModel{AbstractActionType, AbstractObservationType, AbstractStateType}, AbstractPerceptualProcess, AbstractActionProcess}
)

    @error "Please create a store_beliefs! function utilizing concrete type.
            The current model is: $(typeof(model))"

end


function active_inference(model::T, observation::Vector{Int64}, previous_action::Union{Nothing, Vector{Int64}}) where T <: AIFModel

    # Perform perception process
    inference_posterior = perception(model, observation, previous_action)

    # Perform action process
    policy_posterior = planning(model, inference_posterior)

    # Perform action selection
    action_posterior = selection(model, policy_posterior)

    # Store beliefs
    store_beliefs!(model, action_posterior, policy_posterior, inference_posterior, previous_action, observation)

    return action_posterior
end

# function active_inference(model::T, observation::Vector{Int64}) where T <: AIFModel

#     # Perform perception process
#     inference_posterior = perception(model, observation)

#     # Perform action process
#     action_posterior = planning(model, inference_posterior)

#     # Store beliefs
#     store_beliefs!(model, action_posterior, inference_posterior, observation)

#     return action_posterior
# end
