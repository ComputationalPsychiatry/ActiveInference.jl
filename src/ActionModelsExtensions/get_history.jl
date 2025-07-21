"""
This extends the "get_history" function of the ActionModels package to work specifically with instances of the Agent type.

    get_history(agent::Agent, target_states::Vector{String})
Retrieves a history for multiple states of an Agent agent. 

    get_history(agent::Agent, target_state::String)
Retrieves a single target state history from an Agent agent.

    get_history(agent::Agent)
Retrieves history of all states from an Agent agent.
"""

using ActionModels


# Retrieve multiple states history
function ActionModels.get_history(agent::Agent, target_states::Vector{String})
    history = Dict()

    for target_state in target_states
        try
            history[target_state] = get_history(agent, target_state)
        catch e
            # Catch the error if a specific state does not exist
            if isa(e, ArgumentError)
                throw(ArgumentError("The specified state $target_state does not exist"))
            else
                rethrow(e) 
            end
        end
    end

    return history
end

# Retrieve a history from a single state
function ActionModels.get_history(agent::Agent, target_state::String)
    # Check if the state is in the Agent's states
    if haskey(agent.states, target_state)

        return agent.states[target_state]
    else
        # If the target state is not found, throw an ArgumentError
        throw(ArgumentError("The specified state $target_state does not exist"))
    end
end


# Retrieve all states history
function ActionModels.get_history(agent::Agent)
    return agent.states
end