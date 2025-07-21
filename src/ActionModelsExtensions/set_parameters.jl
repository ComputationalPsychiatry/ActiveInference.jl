"""
This extends the "set_parameters!" function of the ActionModels package to work with instances of the Agent type.

    set_parameters!(agent::Agent, target_param::String, param_value::Real)
Set a single parameter in the Agent agent

    set_parameters!(agent::Agent, parameters::Dict{String, Real})
Set multiple parameters in the Agent agent

"""

using ActionModels

# Setting a single parameter
function ActionModels.set_parameters!(agent::Agent, target_param::String, param_value::Real)
    # Update the parameters dictionary
    agent.parameters[target_param] = param_value

    # Update the struct's field based on the target_param
    if target_param == "alpha"
        agent.alpha = param_value
    elseif target_param == "gamma"
        agent.gamma = param_value
    elseif target_param == "lr_pA"
        agent.lr_pA = param_value
    elseif target_param == "fr_pA"
        agent.fr_pA = param_value
    elseif target_param == "lr_pB"
        agent.lr_pB = param_value
    elseif target_param == "fr_pB"
        agent.fr_pB = param_value
    elseif target_param == "lr_pD"
        agent.lr_pD = param_value
    elseif target_param == "fr_pD"
        agent.fr_pD = param_value
    else
        throw(ArgumentError("The parameter $target_param is not recognized."))
    end
end

# Setting multiple parameters
function ActionModels.set_parameters!(agent::Agent, parameters::Dict)
    # For each parameter in the input dictionary
    for (target_param, param_value) in parameters
        # Directly set each parameter
        set_parameters!(agent, target_param, param_value)
    end
end