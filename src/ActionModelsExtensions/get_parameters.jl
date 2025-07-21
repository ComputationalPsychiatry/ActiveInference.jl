"""
This extends the "get_parameters" function of the ActionModels package to work specifically with instances of the Agent type.

    get_parameters(agent::Agent, target_parameters::Vector{String})
Retrieves multiple target parameters from an Agent agent. 

    get_parameters(agent::Agent, target_parameter::String)
Retrieves a single target parameter from an Agent agent.

    get_parameters(agent::Agent)
Retrieves all parameters from an Agent agent.

"""

using ActionModels

# Retrieves multiple target parameters
function ActionModels.get_parameters(agent::Agent, target_parameters::Vector{String})
    parameters = Dict()

    for target_parameter in target_parameters
        try
            parameters[target_parameter] = get_parameters(agent, target_parameter)
        catch e
            if isa(e, ArgumentError)
                throw(ArgumentError("The specified parameter $target_parameter does not exist"))
            else
                rethrow(e)
            end
        end
    end

    return parameters
end

# Retrieves a single parameter
function ActionModels.get_parameters(agent::Agent, target_parameter::String)
    if haskey(agent.parameters, target_parameter)
        return agent.parameters[target_parameter]
    else
        throw(ArgumentError("The specified parameter $target_parameter does not exist"))
    end
end


# Retrieves all parameters 
function ActionModels.get_parameters(agent::Agent)
    return agent.parameters
end