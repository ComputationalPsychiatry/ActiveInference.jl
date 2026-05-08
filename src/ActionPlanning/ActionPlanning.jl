abstract type AbstractTreeSearchNode end

abstract type AbstractActionNode <: AbstractTreeSearchNode end
abstract type AbstractObservationNode <: AbstractTreeSearchNode end


struct ActionNode{T} <: AbstractActionNode
    action::T
end

struct ObservationNode{T} <: AbstractObservationNode
    observation::T
end







### TOP LEVEL RECURSIVE SEARCH FUNCTION
function recursive_search(node::AbstractActionNode, search_config, accumulated_cost; old_kwargs...)

    #Do the operation on the node to get the cost, its children, and new kwargs
    (node_cost, children, new_kwargs) in calculate_node(node, search_config; old_kwargs...)

    #Update the cost
    accumulated_cost += node_cost

    ## TODO: Store the path, include stop criteria (e.g. max depth, cost threshold, etc.), return the best path
    
    #Go through each child
    for child in children

        #And repeat
        recursive_search(child, search_config, accumulated_cost; old_kwargs..., new_kwargs...)
    end

end


function calculate_node(
    action_node::ActionNode, 
    search_config::TreeSearchConfig; 
    aif_model::AbstractAIFModel, planned_environment_belief, kwargs...)

    #Get the list of possible observations given the expected environment belief
    observation_node_children = get_possible_observations(search_config, aif_model, planned_environment_belief)

    return 0, observation_node_children, (; planned_action = action_node.action)

end


function calculate_node(
    observation_node::ObservationNode, 
    search_config::TreeSearchConfig; 
    aif_model::AbstractAIFModel, planned_action, planned_environment_belief, kwargs...
    )

    #Calculate the expected environment posterior given the expected observation and previous action
    planned_environment_posterior, infer_environment_snapshot = observation_node_update(search_config.sophistication_type, aif_model, observation_node.observation, planned_action, planned_environment_belief)

    #Calculate the expected free energy
    node_cost = calculate_cost(aif_model, planned_environment_posterior)

    #Get the list of possible actions given the expected environment posterior
    action_node_children = get_possible_actions(search_config, aif_model, planned_environment_posterior)

    return node_cost, action_node_children, (; planned_environment_belief = planned_environment_posterior)

end


### SOPHISTICATED AND NON_SOPHISTICATED UPDATES FOR THE OBSERVATION NODES
function observation_node_update(sophistication_type::SophisticatedInference, aif_model::AbstractAIFModel, planned_observation, planned_action, planned_environment_prior)

    planned_environment_posterior, infer_environment_snapshot = infer_environment(aif_model, planned_observation, planned_action, planned_environment_prior)

end
function observation_node_update(sophistication_type::NonSophisticatedInference, aif_model::AbstractAIFModel, planned_observation, planned_action, planned_environment_prior)

    planned_environment_posterior, infer_environment_snapshot = predict_environment(aif_model, planned_action, planned_environment_prior)

end