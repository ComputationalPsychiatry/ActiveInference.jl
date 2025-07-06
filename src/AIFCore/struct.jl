module ActiveInferenceCore

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
abstract type GenerativeModel{
    TypeAction<:AbstractActionType,
    TypeObservation<:AbstractObservationType,
    TypeState<:AbstractStateType,
} end

### Perceptual process types ###
abstract type PerceptualInferenceProcess end

### Action process types ###
abstract type ActionSelectionProcess end


struct AIFAgent 

    ## Generative Model ##
    # Struct containing a generative model of the AbstractGenerativeModel type
    generative_model::GenerativeModel

    ## Perceptual process ##
    # Function for inference (perception)
    perception::Function
    store_new_beliefs!::Function

    # Function for calculating the predictive posterior
    prediction::Function
    perception_struct::PerceptualInferenceProcess

    # Action process
    action::Function
    action_struct::ActionSelectionProcess
    
end

function active_inference(agent::AIFAgent, obs::Vector{Float64})

    # Perform perception process
    new_beliefs = agent.perception(agent)

    # Store new beliefs
    agent.store_new_beliefs!(agent, new_beliefs)

    # Make predictions for Sophisticated Inference
    #agent.prediction(agent) # probably doesn't belong here

    # Perform action process
    # Should we here include a differentitation between planning and action selection?
    action_distribution = agent.action(agent)

    return action_distribution
end

end

