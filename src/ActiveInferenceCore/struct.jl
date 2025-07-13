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
abstract type AbstractPerceptualProcess end

### Action process types ###
abstract type AbstractActionProcess end

#NOTE: when making this agent, make the prior be defined by D in the generative model as part of the initialization function
struct AIFAgent 

    ## Generative Model ##
    # Struct containing a generative model of the AbstractGenerativeModel type
    generative_model::AbstractGenerativeModel

    ## Perceptual process ##
    perceptual_process::AbstractPerceptualProcess
    # Function for inference (perception)
    perception::Function
    # store_new_beliefs!::Function

    # Function for calculating the predictive posterior
    # prediction::Function

    # Action process
    # action::Function
    # action_process::AbstractActionProcess

    function AIFAgent(;
        generative_model::AbstractGenerativeModel,
        perceptual_process::AbstractPerceptualProcess,
        perception::Function
    )
        new(generative_model, perceptual_process, perception)
    end
    
end

function active_inference(agent::AIFAgent, observation::Vector{Float64})

    # Perform perception process
    new_beliefs = agent.perception(agent, observation)

    # Store new beliefs
    agent.store_new_beliefs!(agent, new_beliefs)

    # Make predictions for Sophisticated Inference
    #agent.prediction(agent) # probably doesn't belong here

    # Perform action process
    # Should we here include a differentitation between planning and action selection?
    action_distribution = agent.action(agent)

    return action_distribution
end
