"""
This extends the "get_states" function of the ActionModels package to work specifically with instances of the AIF type.

    get_states(aif::AIF, target_states::Vector{String})
Retrieves multiple states from an AIF agent. 

    get_states(aif::AIF, target_state::String)
Retrieves a single target state from an AIF agent.

    get_states(aif::AIF)
Retrieves all states from an AIF agent.
"""

using ActionModels


# Retrieve multiple states
function ActionModels.get_states(aif::POMDPActiveInference, target_states::Vector{String})
    states = Dict{String, Any}()

    for target_state in target_states
        if hasproperty(aif.states, Symbol(target_state))
            value = getproperty(aif.states, Symbol(target_state))
            states[target_state] = value
        else
            throw(ArgumentError("The specified state '$target_states' does not exist in states."))
        end
    end

    return states
end

# Retrieve a single state
function ActionModels.get_states(aif::POMDPActiveInference, target_state::String)
    # Check if the state exists in the AIF history struct
    if hasproperty(aif.states, Symbol(target_state))
        state_value = getproperty(aif.states, Symbol(target_state))

        return state_value
    else
        # If the target state is not found, throw an ArgumentError
        throw(ArgumentError("The specified parameter '$target_parameter' does not exist in states."))
    end
end


# Retrieve all states
function ActionModels.get_states(aif::POMDPActiveInference)
    states_struct = aif.states
    states_dict = Dict{String, Any}()

    for field in fieldnames(typeof(states_struct))
        value = getfield(states_struct, field)
        states_dict[string(field)] = value
    end

    return states_dict
end


