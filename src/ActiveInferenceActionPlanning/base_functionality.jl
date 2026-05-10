### THIS FILE HAS THINGS THAT ARE GENERAL ACROSS DIFFERENT TREE SEARCH TYPES, BUT SPECIFIC TO THE TREE SEARCH PLANNING APPROACH. 

### DISPATCH TYPES FOR THE NODES IN THE TREE SEARCH ###
abstract type AbstractTreeSearchNode end

abstract type AbstractActionNode <: AbstractTreeSearchNode end
abstract type AbstractObservationNode <: AbstractTreeSearchNode end

struct ActionNode{T} <: AbstractActionNode
    action::T
end

struct ObservationNode{T} <: AbstractObservationNode
    observation::T
end


### TREE SEARCH TYPES ###
abstract type AbstractTreeSearchType end




### TYPE FOR CONFIG FOR THE PLANNING ###
struct DiscreteTreeSearch{Ttreesearch, Tsophistication} <: AbstractInferenceActions
    treesearch_type::Ttreesearch
    sophistication_type::Tsophistication
end

#TOP LEVEL FUNCTION FOR DISCRETE TREE SEARCH PLANNING
function infer_actions(aif_model::AIFModel{TGenModel, TInfEnv, TInfActions}, environment_posterior) where TInfActions <: DiscreteTreeSearch

    #Do the tree search to get the scores for each action
    action_costs, intermediate_values = tree_search(aif_model.inference_actions.treesearch_type, aif_model, environment_posterior)
    #Is this costs over policies or actions?

    #Unflatten
    #TODO:

    #Make into Multivariate Categorical
    #TODO:

    #Return the action posterior and intermediate_values
    return action_posterior, intermediate_values

end




### NODE-SPECIFIC UPDATE FUNCTIONS ###
function evaluate_node(
    action_node::ActionNode, aif_model::AbstractAIFModel; 
    planned_environment_belief, kwargs...)

    #Get the list of possible observations given the expected environment belief
    observation_node_children = get_possible_observations(aif_model, planned_environment_belief)

    #Return cost, children, new kwargs, and intermediate values for storing in the history
    return 0, observation_node_children, (; planned_action = action_node.action), (;)

end

function evaluate_node(
        observation_node::ObservationNode, aif_model::AbstractAIFModel; 
        planned_action, planned_environment_belief, kwargs...
    )

    #Calculate the expected environment posterior given the expected observation and previous action
    planned_environment_posterior, infer_environment_snapshot = observation_node_update(aif_model.inference_actions.sophistication_type,  aif_model, observation_node.observation, planned_action, planned_environment_belief)

    #Calculate the expected free energy
    node_cost, cost_snapshot = calculate_cost(aif_model, planned_environment_posterior)

    #Get the list of possible actions given the expected environment posterior
    action_node_children = get_possible_actions(aif_model, planned_environment_posterior)

    #Return cost, children, new kwargs, and intermediate values for storing in the history
    return node_cost, action_node_children, (; planned_environment_belief), (; infer_environment_snapshot, cost_snapshot)

end

function observation_node_update(::SophisticatedInference, aif_model::AbstractAIFModel, planned_observation, planned_action, planned_environment_prior)

    planned_environment_posterior, infer_environment_snapshot = infer_environment(aif_model, planned_observation, planned_action, planned_environment_prior)

end
function observation_node_update(::NonSophisticatedInference, aif_model::AbstractAIFModel, planned_observation, planned_action, planned_environment_prior)

    planned_environment_posterior, predict_environment_snapshot = predict_environment(aif_model, planned_action, planned_environment_prior)

end
