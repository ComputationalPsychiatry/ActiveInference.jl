"""
In this script we define struct that go into the constructor of the PerceptualProcess struct. 
This will indicate whether learning is enabled. It will allow the user to specify a (Dirichlets) prior over the parameters (pX)
or specify a concentration parameter for the priors, which will be used to create an initial prior.
It will also allow the user to specify whether to update the parameters or not.
"""

# Learning structs for the DiscretePOMDP generative model

mutable struct Learn_A

    learning_rate::Float64
    forgetting_rate::Float64
    concentration_parameter::Union{Float64, Nothing}
    prior::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N}
    struct_name::String

    function Learn_A(;
        learning_rate::Float64 = 1.0,
        forgetting_rate::Float64 = 1.0,
        concentration_parameter::Union{Float64, Nothing} = nothing,
        prior::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing
    )

        struct_name = "Learn_A"
        # Validate learning rate and forgetting rate
        check_learning_structs(learning_rate, forgetting_rate, concentration_parameter, prior, struct_name)

        return new(learning_rate, forgetting_rate, concentration_parameter, prior)
    end

end

mutable struct Learn_B

    learning_rate::Float64
    forgetting_rate::Float64
    concentration_parameter::Union{Float64, Nothing}
    prior::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N}
    struct_name::String

    function Learn_B(;
        learning_rate::Float64 = 1.0,
        forgetting_rate::Float64 = 1.0,
        concentration_parameter::Union{Float64, Nothing} = nothing,
        prior::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing
    )
        struct_name = "Learn_B"
        # Validate learning rate and forgetting rate
        check_learning_structs(learning_rate, forgetting_rate, concentration_parameter, prior, struct_name)

        return new(learning_rate, forgetting_rate, concentration_parameter, prior)
    end

end

mutable struct Learn_D

    learning_rate::Float64
    forgetting_rate::Float64
    concentration_parameter::Union{Float64, Nothing}
    prior::Union{Vector{Vector{T}}, Nothing} where {T <: Real}
    struct_name::String

    function Learn_D(;
        learning_rate::Float64 = 1.0,
        forgetting_rate::Float64 = 1.0,
        concentration_parameter::Union{Float64, Nothing} = nothing,
        prior::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing
    )
        struct_name = "Learn_D"
        # Validate learning rate and forgetting rate
        check_learning_structs(learning_rate, forgetting_rate, concentration_parameter, prior, struct_name)

        return new(learning_rate, forgetting_rate, concentration_parameter, prior)
    end

end

### Validation function ###

"""
    check_learning_structs(learning_rate, forgetting_rate, concentration_parameter = nothing, prior = nothing)
"""
function check_learning_structs(
    learning_rate::Float64,
    forgetting_rate::Float64,
    concentration_parameter::Union{Float64, Nothing} = nothing,
    prior::Union{AbstractVector, Nothing} = nothing,
    struct_name::String = ""
)

    # Validate learning rate and forgetting rate
    if (learning_rate <= 0.0 || forgetting_rate < 0.0) || (learning_rate > 1.0 || forgetting_rate > 1.0)
        throw(ArgumentError("From $struct_name: Learning and forgetting rates are bounded by 0 and 1. Received: learning_rate = $learning_rate, forgetting_rate = $forgetting_rate"))
    end

    if !isnothing(concentration_parameter) && !isnothing(prior)
        throw(ArgumentError("From $struct_name: Cannot provide both concentration parameter and prior"))
    end

    if !isnothing(concentration_parameter) && concentration_parameter <= 0.0
        throw(ArgumentError("From $struct_name: Concentration parameter must be positive"))
    end

    if isnothing(prior) && isnothing(concentration_parameter)
        throw(ArgumentError("From $struct_name: Either prior or concentration parameter must be provided"))
    end

    if !isnothing(prior)
        @info "From $struct_name: Using a prior will override the parameter specified in the generative model. Ensure this is intended. Otherwise, use concentration_parameter, which will create a prior based on the parameter specified in the generative model."
    end

    return true

end