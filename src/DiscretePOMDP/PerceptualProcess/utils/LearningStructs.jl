"""
In this script we define struct that go into the constructor of the PerceptualProcess struct. 
This will indicate whether learning is enabled. It will allow the user to specify a (Dirichlets) prior over the parameters (pX)
or specify a concentration parameter for the priors, which will be used to create an initial prior.
It will also allow the user to specify whether to update the parameters or not.
"""

# Learning structs for the DiscretePOMDP generative model

struct Learn_A

    learning_rate::Float64
    forgetting_rate::Float64
    concentration_parameter::Union{Float64, Nothing}
    prior::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N}

    function Learn_A(;
        learning_rate::Float64 = 1.0,
        forgetting_rate::Float64 = 1.0,
        concentration_parameter::Union{Float64, Nothing} = nothing,
        prior::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing
    )
        # Validate learning rate and forgetting rate
        check_learning_structs(learning_rate, forgetting_rate, concentration_parameter, prior)

        return new(learning_rate, forgetting_rate, concentration_parameter, prior)
    end

end

struct Learn_B

    learning_rate::Float64
    forgetting_rate::Float64
    concentration_parameter::Union{Float64, Nothing}
    prior::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N}

    function Learn_B(;
        learning_rate::Float64 = 1.0,
        forgetting_rate::Float64 = 1.0,
        concentration_parameter::Union{Float64, Nothing} = nothing,
        prior::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing
    )
        # Validate learning rate and forgetting rate
        check_learning_structs(learning_rate, forgetting_rate, concentration_parameter, prior)

        return new(learning_rate, forgetting_rate, concentration_parameter, prior)
    end

end

struct Learn_D

    learning_rate::Float64
    forgetting_rate::Float64
    concentration_parameter::Union{Float64, Nothing}
    prior::Union{Vector{Vector{T}}, Nothing} where {T <: Real}

    function Learn_D(;
        learning_rate::Float64 = 1.0,
        forgetting_rate::Float64 = 1.0,
        concentration_parameter::Union{Float64, Nothing} = nothing,
        prior::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing
    )
        # Validate learning rate and forgetting rate
        check_learning_structs(learning_rate, forgetting_rate, concentration_parameter, prior)

        return new(learning_rate, forgetting_rate, concentration_parameter, prior)
    end

end
