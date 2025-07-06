"""
In this script, we define a concrete generative model for the Discrete POMDP as an AbstractGenerativeModel.
"""

### Discrete POMDP Generative Model ###
using ..ActiveInferenceCore: AbstractGenerativeModel, DiscreteActions, DiscreteObservations, DiscreteStates


"""
Discrete POMDP generative model containing the following fields:
- `A`: A-matrix (Observation Likelihood model)
- `B`: B-matrix (Transition model)
- `C`: C-vectors (Preferences over observations)
- `D`: D-vectors (Prior over states)
- `E`: E-vector (Habits)
"""
@kwdef mutable struct GenerativeModel <: AbstractGenerativeModel{DiscreteActions, DiscreteObservations, DiscreteStates}

    A::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing # A-matrix
    B::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing # B-matrix
    C::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing # C-vectors
    D::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing # D-vectors
    E::Union{Vector{T}, Nothing} where {T <: Real} = nothing # E-vector (Habits)

end

### Initialization of Generative Model ###
"""Initialize a generative model with the given parameters.
"""
function init_generative_model(; verbose::Bool = true, kwargs...)
    
    # Create an instance of the parameter struct
    GenerativeModelInstance = GenerativeModel(; kwargs...)

    # Perform checks using the fields
    check_parameters(GenerativeModelInstance)

    # Infer missing parameters that have not been provided
    infer_missing_parameters(GenerativeModelInstance; verbose=verbose)

    # Return an instance of the generative model
    return GenerativeModelInstance
end