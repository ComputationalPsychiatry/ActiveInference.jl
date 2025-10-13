"""

This is extends the give_inputs! function of ActionsModels.jl to work with instances of the AIF type.

    single_input!(aif::AIF, obs)
Give a single observation to an AIF agent. 


"""

#=

using ActionModels

### Give single observation to the agent
function ActionModels.single_input!(aif::POMDPActiveInference, obs::Vector)

    # Running the action model to retrieve the action distributions
    action_distributions = action_pomdp!(aif, obs)

    # Get number of factors from the action distributions
    num_factors = length(action_distributions)

    # if there is only one factor
    if num_factors == 1
        # Sample action from the action distribution
        action = rand(action_distributions)

        # If the agent has not taken any actions yet
        if isempty(aif.states.action)
            push!(aif.states.action, action)
        else
        # Put the action in the last element of the action vector
            aif.states.action[end] = action
        end

        push!(aif.history.action, aif.states.action)

    # if there are multiple factors
    else
        # Initialize a vector for sampled actions 
        sampled_actions = zeros(Real,num_factors)

        # Sample action per factor
        for factor in eachindex(action_distributions)
            sampled_actions[factor] = rand(action_distributions[factor])
        end
        # If the agent has not taken any actions yet
        if isempty(aif.states.action)
            aif.states.action = sampled_actions
        else
        # Put the action in the last element of the action vector
            aif.states.action = sampled_actions
        end
        # Push the action to agent's states
        push!(aif.history.action, aif.states.action)
    end

    return aif.states.action
end

function ActionModels.give_inputs!(aif::POMDPActiveInference, observations::Vector)
    # For each individual observation run single_input! function
    for observation in observations

        ActionModels.single_input!(aif, observation)

    end

    return aif.history.action
end

=#