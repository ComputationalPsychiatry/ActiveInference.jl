"""
This extends the "set_parameters!" function of the ActionModels package to work with instances of the AIF type.

    set_parameters!(aif::AIF, target_param::String, param_value::Real)
Set a single parameter in the AIF agent

    set_parameters!(aif::AIF, parameters::Dict{String, Real})
Set multiple parameters in the AIF agent

"""
#=
using ActionModels

# Setting a single parameter
function ActionModels.set_parameters!(aif::POMDPActiveInference, target_param::String, param_value::Any)

    # Update the struct's field based on the target_param
    if target_param == "A"
        aif.parameters.A = param_value
    elseif target_param == "B"
        aif.parameters.B = param_value
    elseif target_param == "C"
        aif.parameters.C = param_value
    elseif target_param == "D"
        aif.parameters.D = param_value
    elseif target_param == "E"
        aif.parameters.E = param_value
    elseif target_param == "pA"
        aif.parameters.pA = param_value
    elseif target_param == "pB"
        aif.parameters.pB = param_value
    elseif target_param == "pD"
        aif.parameters.pD = param_value
    elseif target_param == "alpha"
        aif.parameters.alpha = param_value
    elseif target_param == "gamma"
        aif.parameters.gamma = param_value
    elseif target_param == "lr_pA"
        aif.parameters.lr_pA = param_value
    elseif target_param == "fr_pA"
        aif.parameters.fr_pA = param_value
    elseif target_param == "lr_pB"
        aif.parameters.lr_pB = param_value
    elseif target_param == "fr_pB"
        aif.parameters.fr_pB = param_value
    elseif target_param == "lr_pD"
        aif.parameters.lr_pD = param_value
    elseif target_param == "fr_pD"
        aif.parameters.fr_pD = param_value
    else
        throw(ArgumentError("The parameter $target_param is not recognized."))
    end
end

# Setting multiple parameters
function ActionModels.set_parameters!(aif::POMDPActiveInference, parameters::Dict)
    # For each parameter in the input dictionary
    for (target_param, param_value) in parameters
        # Directly set each parameter
        set_parameters!(aif, target_param, param_value)
    end
end
=#