"""
This extends the "get_parameters" function of the ActionModels package to work specifically with instances of the POMDPActiveInference type.

    get_parameters(aif::POMDPActiveInference, target_parameters::Vector{String})
Retrieves multiple target parameters from an AIF agent. 

    get_parameters(aif::POMDPActiveInference, target_parameter::String)
Retrieves a single target parameter from an AIF agent.

    get_parameters(aif::POMDPActiveInference)
Retrieves all parameters from an AIF agent.

"""

using ActionModels

# Retrieves multiple target parameters
function ActionModels.get_parameters(aif::POMDPActiveInference, target_parameters::Vector{String})
    parameters = Dict{String, Any}()

    for target_parameter in target_parameters
        if hasproperty(aif.parameters, Symbol(target_parameter))
            value = getproperty(aif.parameters, Symbol(target_parameter))
            parameters[target_parameter] = value
        else
            throw(ArgumentError("The specified parameter '$target_parameter' does not exist in parameters."))
        end
    end

    return parameters
end

# Retrieves a single parameter
function ActionModels.get_parameters(aif::POMDPActiveInference, target_parameter::String)
    # Check if the state exists in the aif state struct
    if hasproperty(aif.parameters, Symbol(target_parameter))
        parameter_value = getproperty(aif.parameters, Symbol(target_parameter))

        return parameter_value
    else
        # If the target parameter is not found, throw an ArgumentError
        throw(ArgumentError("The specified parameter '$target_parameter' does not exist in parameters."))
    end
end


# Retrieves all parameters 
function ActionModels.get_parameters(aif::POMDPActiveInference)
    parameter_struct = aif.parameters
    parameters_dict = Dict{String, Any}()

    for field in fieldnames(typeof(parameter_struct))
        value = getfield(parameter_struct, field)
        parameters_dict[string(field)] = value
    end

    return parameters_dict
end