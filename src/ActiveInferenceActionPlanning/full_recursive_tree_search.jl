### THIS FILE HAS THINGS THAT ARE SPECIFIC TO THE FULL RECUSRIVE TREE SEARCH METHOD

struct FullRecursiveTreeSearch <: AbstractTreeSearchType
    planning_horizon::Int
end

### FUNCTION SPECIFICALLY FOR THE FULL RECURSIVE TREE SEARCH ###
function tree_search(::FullRecursiveTreeSearch, aif_model::AIFModel, environment_posterior)

    #Get set of possible next actions
    first_action_children = get_possible_actions(aif_model, environment_posterior)

    #Do the full recursive tree search for each possible next action
    tree_search_results = [
            full_recursive_search(aif_model, child, 0.0, 0; aif_model.inference_actions.planning_horizon; planned_environment_belief = environment_posterior) 
            for child in first_action_children
         ]
    
    #Get unpack scores and intermediate values for each action
    action_scores = [r[1] for r in tree_search_results]
    intermediate_values = [r[2] for r in tree_search_results]

    #Return the action scores, and the intermediate values for storing in the history
    return action_scores, intermediate_values

end


### TOP LEVEL RECURSIVE SEARCH FUNCTION ###
function full_recursive_search(aif_model::AIFModel, node::AbstractTreeSearchNode, current_depth, planning_horizon; old_kwargs...)
    
    #Do the operation on the node to get the cost, its children, and new kwargs
    (node_cost, children, new_kwargs, intermediate_values) = evaluate_node(node, aif_model; old_kwargs...)

    #Increase the current depth
    current_depth = increment_depth(current_depth, node)

    #If the max depth is not yet reached, continue down the tree
    if current_depth < planning_horizon

        children_values = [
                    full_recursive_search(aif_model, child, current_depth, planning_horizon; old_kwargs..., new_kwargs...) 
                    for child in children
                ]

        total_children_cost = sum(r[1] for r in children_values)
        child_intermediate_values = [r[2] for r in children_values]

    else

        total_children_cost = 0.0
        child_intermediate_values = []

    end

    #Return the cost for this node and its children, as well as the total intermediate values
    return node_cost + total_children_cost, (; node, intermediate_values, child_intermediate_values)

end


#Only observation nodes increment the depth, action nodes do not
increment_depth(current_depth, node::ObservationNode) = current_depth + 1
increment_depth(current_depth, node::ActionNode) = current_depth