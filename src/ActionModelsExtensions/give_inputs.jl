"""

This is extends the give_inputs! function of ActionsModels.jl to work with instances of the Agent type.

    single_input!(agent::Agent, obs)
Give a single observation to an Agent agent. 


"""

using ActionModels

### Give single observation to the agent
function ActionModels.single_input!(agent::Agent, obs::Vector)

    # Running the action model to retrieve the action distributions
    action_distributions = action_pomdp!(agent, obs)

    # Get number of factors from the action distributions
    num_factors = length(action_distributions)

    # if there is only one factor
    if num_factors == 1
        # Sample action from the action distribution
        action = rand(action_distributions)

        # If the agent has not taken any actions yet
        if isempty(agent.action)
            push!(agent.action, action)
        else
        # Put the action in the last element of the action vector
            agent.action[end] = action
        end

        push!(agent.states["action"], agent.action)

    # if there are multiple factors
    else
        # Initialize a vector for sampled actions 
        sampled_actions = zeros(Real,num_factors)

        # Sample action per factor
        for factor in eachindex(action_distributions)
            sampled_actions[factor] = rand(action_distributions[factor])
        end
        # If the agent has not taken any actions yet
        if isempty(agent.action)
            agent.action = sampled_actions
        else
        # Put the action in the last element of the action vector
            agent.action[end] = sampled_actions
        end
        # Push the action to agent's states
        push!(agent.states["action"], agent.action)
    end

    return agent.action
end

function ActionModels.give_inputs!(agent::Agent, observations::Vector)
    # For each individual observation run single_input! function
    for observation in observations

        ActionModels.single_input!(agent, observation)

    end

    return agent.states["action"]
end