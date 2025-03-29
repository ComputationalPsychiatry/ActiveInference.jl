"""
This extends the "get_history" function of the ActionModels package to work specifically with instances of the POMDPActiveInference type.

    get_history(aif::POMDPActiveInference, target_states::Vector{String})
Retrieves a history for multiple states of an AIF agent. 

    get_history(aif::POMDPActiveInference, target_state::String)
Retrieves a single target state history from an AIF agent.

    get_history(aif::POMDPActiveInference)
Retrieves history of all states from an AIF agent.
"""

using ActionModels


# Retrieve multiple states history
function ActionModels.get_history(aif::POMDPActiveInference, target_states::Vector{String})

    # Create dict to contain the retrieved history
    history = Dict{String, Any}()

    # Loop throught the provided target states and append them to the dictionary
    for target_state in target_states
        if hasproperty(aif.history, Symbol(target_state))
            state_history = getproperty(aif.history, Symbol(target_state))
            history[target_state] = state_history
        else 
            # Throw error if the target state does not exist
            throw(ArgumentError("The specified state '$target_state' does not exist in history."))
        end
    end

    return history
end

# Retrieve a history from a single state
function ActionModels.get_history(aif::POMDPActiveInference, target_state::String)
    # Check if the state exists in the AIF history struct
    if hasproperty(aif.history, Symbol(target_state))
        state_value = getproperty(aif.history, Symbol(target_state))

        return state_value
    else
        # If the target state is not found, throw an ArgumentError
        throw(ArgumentError("The specified state '$target_state' does not exist in states."))
    end
end


# Retrieve all states history and returning them as a dictionary
function ActionModels.get_history(aif::POMDPActiveInference)
    history_struct = aif.history
    history_dict = Dict{String, Any}()

    for field in fieldnames(typeof(history_struct))
        value = getfield(history_struct, field)
        history_dict[string(field)] = value
    end

    return history_dict
end