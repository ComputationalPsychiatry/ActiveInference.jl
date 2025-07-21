"""
This extends the "get_states" function of the ActionModels package to work specifically with instances of the Agent type.

    get_states(agent::Agent, target_states::Vector{String})
Retrieves multiple states from an Agent agent. 

    get_states(agent::Agent, target_state::String)
Retrieves a single target state from an Agent agent.

    get_states(agent::Agent)
Retrieves all states from an Agent agent.
"""

using ActionModels


# Retrieve multiple states
function ActionModels.get_states(agent::Agent, target_states::Vector{String})
    states = Dict()

    for target_state in target_states
        try
            states[target_state] = get_states(agent, target_state)
        catch e
            # Catch the error if a specific state does not exist
            if isa(e, ArgumentError)
                throw(ArgumentError("The specified state $target_state does not exist"))
            else
                rethrow(e) 
            end
        end
    end

    return states
end

# Retrieve a single state
function ActionModels.get_states(agent::Agent, target_state::String)
    if haskey(agent.states, target_state)
        state_history = agent.states[target_state]
        if target_state == "policies"
            return state_history
        else
            # return the latest state or missing
            return isempty(state_history) ? missing : last(state_history)
        end
    else
        throw(ArgumentError("The specified state $target_state does not exist"))
    end
end


# Retrieve all states
function ActionModels.get_states(agent::Agent)
    all_states = Dict()
    for (key, state_history) in agent.states
        if key == "policies"
            all_states[key] = state_history
        else
            # get the latest state or missing
            all_states[key] = isempty(state_history) ? missing : last(state_history)
        end
    end
    return all_states
end



