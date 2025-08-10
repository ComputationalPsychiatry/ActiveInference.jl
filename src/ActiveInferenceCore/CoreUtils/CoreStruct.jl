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

### Perceptual process types ###
# Optimization engine abstract type
abstract type AbstractOptimEngine end

# Perceptual Process abstract type
abstract type AbstractPerceptualProcess{
    TypeOptimEngine<:Union{AbstractOptimEngine, Missing}
} end

### Action process types ###
abstract type AbstractActionProcess end

#NOTE: when making this agent, make the prior be defined by D in the generative model as part of the initialization function
struct AIFModel{
    GM <: AbstractGenerativeModel,
    PP <: AbstractPerceptualProcess,
    AP <: AbstractActionProcess
}
    ## Generative Model ##
    generative_model::GM

    ## Perceptual process ##
    perceptual_process::PP
    
    ## Action process ##
    action_process::AP

end

function AIFModel(
    generative_model::AbstractGenerativeModel{AbstractActionType, AbstractObservationType, AbstractStateType},
    perceptual_process::AbstractPerceptualProcess{AbstractOptimEngine},
    action_process::AbstractActionProcess
)

    @error "Please create a constructor for AIFModel that utilizes concrete types.
            The current generative model is: $(typeof(generative_model)),
            the perceptual process is: $(typeof(perceptual_process)),
            and the action process is: $(typeof(action_process))"

end

function perception(
    model::AIFModel{AbstractGenerativeModel{AbstractActionType, AbstractObservationType, AbstractStateType}, AbstractPerceptualProcess{AbstractOptimEngine}, AbstractActionProcess},
    observation::Vector{Real}
) 
    @error "Please create a perception function utilizing concrete type.
            The current model is: $(typeof(model))"

end

function prediction(
    model::AIFModel{AbstractGenerativeModel{AbstractActionType, AbstractObservationType, AbstractStateType}, AbstractPerceptualProcess{AbstractOptimEngine}, AbstractActionProcess}
)

    @error "Please create a prediction function utilizing concrete type.
            The current model is: $(typeof(model))"

end

function planning(
    model::AIFModel{AbstractGenerativeModel{AbstractActionType, AbstractObservationType, AbstractStateType}, AbstractPerceptualProcess{AbstractOptimEngine}, AbstractActionProcess}
)

    @error "Please create an action function utilizing concrete type.
            The current model is: $(typeof(model))"

end





function active_inference(agent::AIFModel, observation::Vector{Int64})

    agent.perception()


    # Perform perception process
    agent.perception(agent, observation)

    # Store new beliefs
    agent.prediction(agent)

    # Perform action process
    action_distribution, G = agent.action(agent)

    return action_distribution
end

function active_inference_action(agent::AIFModel, observation::Vector{Int64})

    # Perform perception process
    agent.perception(agent, observation)

    # Store new beliefs
    agent.prediction(agent)

    # Perform action process
    action_distribution, G, action = agent.action(agent, act = true)

    return action
end
