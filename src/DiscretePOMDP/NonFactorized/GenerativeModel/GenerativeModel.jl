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
mutable struct GenerativeModel <: AbstractGenerativeModel{DiscreteActions, DiscreteObservations, DiscreteStates}

    A::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N}
    B::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N}
    C::Union{Vector{Vector{T}}, Nothing} where {T <: Real}
    D::Union{Vector{Vector{T}}, Nothing} where {T <: Real}
    info::GenerativeModelInfo

    function GenerativeModel(;
        A::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing,
        B::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing,
        C::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing,
        D::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing,
        verbose::Bool = true
    )
        # Make sure parameters are coherent
        check_generative_model(A, B, C, D)
        
        # Infer missing parameters
        C, D = infer_missing_parameters(A, B, C, D, verbose)
        
        # Create info struct with model information
        info = GenerativeModelInfo(A, B, C, D)
        
        # Show model information if verbose
        show_info(info; verbose=verbose)
        
        return new(A, B, C, D, info)
    end
end