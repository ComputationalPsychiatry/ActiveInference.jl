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

    A::Union{Nothing, NamedTuple}
    B::Union{Nothing, NamedTuple}
    C::Union{Nothing, NamedTuple}
    D::Union{Nothing, NamedTuple}
    info::GenerativeModelInfo

    function GenerativeModel(;
        A::Union{Nothing, NamedTuple} = nothing,
        B::Union{Nothing, NamedTuple} = nothing,
        C::Union{Nothing, NamedTuple} = nothing,
        D::Union{Nothing, NamedTuple} = nothing,
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