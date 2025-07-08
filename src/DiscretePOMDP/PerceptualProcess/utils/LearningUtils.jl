"""
    check_learning_structs(learning_rate, forgetting_rate, concentration_parameter = nothing, prior = nothing)
"""

function check_learning_structs(
    learning_rate::Float64,
    forgetting_rate::Float64,
    concentration_parameter::Union{Float64, Nothing} = nothing,
    prior::Union{AbstractVector, Nothing} = nothing
)

    # Validate learning rate and forgetting rate
    if (learning_rate <= 0.0 || forgetting_rate < 0.0) || (learning_rate > 1.0 || forgetting_rate > 1.0)
        throw(ArgumentError("Learning and forgetting rates are bounded by 0 and 1"))
    end

    if !isnothing(concentration_parameter) && !isnothing(prior)
        throw(ArgumentError("Cannot provide both concentration parameter and prior"))
    end

    if !isnothing(concentration_parameter) && concentration_parameter <= 0.0
        throw(ArgumentError("Concentration parameter must be positive"))
    end

    if isnothing(prior) && isnothing(concentration_parameter)
        throw(ArgumentError("Either prior or concentration parameter must be provided"))
    end

    return true

end
