"""
In this script, we define a concrete generative model for the Discrete POMDP as an AbstractGenerativeModel.
"""

### Discrete POMDP Generative Model ###

include("../AIFCore/struct.jl")
include("../utils/create_matrix_templates.jl")
include("../utils/maths.jl")
include("../utils/utils.jl")
using .ActiveInferenceCore: GenerativeModel, DiscreteActions, DiscreteObservations, DiscreteStates


"""
Discrete POMDP generative model containing the following fields:
- `A`: A-matrix (Observation Likelihood model)
- `B`: B-matrix (Transition model)
- `C`: C-vectors (Preferences over observations)
- `D`: D-vectors (Prior over states)
- `E`: E-vector (Habits)
"""
@kwdef mutable struct DiscretePOMDP <: GenerativeModel{DiscreteActions, DiscreteObservations, DiscreteStates}

    A::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing # A-matrix
    B::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing # B-matrix
    C::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing # C-vectors
    D::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing # D-vectors
    E::Union{Vector{T}, Nothing} where {T <: Real} = nothing # E-vector (Habits)

end

A, B, C, D, E = create_matrix_templates([4, 2], [4, 3, 2], [4, 1], 2, "uniform")

DP = DiscretePOMDP(A, B, C, D, E)


supertype(typeof(DP))
fieldnames(typeof(DP))

dp = init_generative_model(A = A, B = B, D = D)


### Initialization of Generative Model ###
"""Initialize a generative model with the given parameters.
"""
function init_generative_model(; kwargs...)
    
    # Create an instance of the parameter struct
    GenerativeModel = DiscretePOMDP(; kwargs...)

    # Perform checks using the fields
    check_parameters(GenerativeModel)

    # Return an instance of the generative model
    return GenerativeModel
end

### Check parameters ###
"""
check generative model parameters

# Arguments
- `parameters::POMDPActiveInferenceParameters`

Throws an error if the generative model parameters are not valid:
- Both A and B must be provided.
- The dimensions of the matrices must be consistent.
- The values must be non-negative (except for C).
- The sum of each column or vector must be approximately 1.
- Not both the parameter and their prior can be provided.

"""
function check_parameters(GenerativeModel::DiscretePOMDP)

    # Destructures the parameters of the parameter struct
    (; A, B, C, D, E) = GenerativeModel

    if isnothing(A) || isnothing(B)
        throw(ArgumentError("A and B must be provided in order to infer structure of the generative model."))
    end

    # Check if the number of states in A, B, and D are consistent.
    # We let this check be done on either the prior or the parameter, depending on which is provided.
    check_parameter_states(GenerativeModel)

    # Check if the number of observation modalities in A and C are consistent.
    # We let this check be done on either the prior or the parameter, depending on which is provided.
    if !isnothing(GenerativeModel.C)
        check_parameter_observations(GenerativeModel)
    end

    # Check if the values are non-negative
    for name in fieldnames(typeof(GenerativeModel))
        parameter = getfield(GenerativeModel, name)

        # If parameter has not been provided, don't check.
        if !isnothing(parameter) && name != :C
            if !is_non_negative(parameter)
                throw(ArgumentError("All elements must be non-negative in parameter '$(name)'"))
            end
        else
            continue
        end
    end

    # Check if the probability distributions are normalized. Only A, B, D, and E are probability distributions.
    params_check_norm = (;GenerativeModel.A, GenerativeModel.B, GenerativeModel.D, GenerativeModel.E)

    for name in fieldnames(typeof(params_check_norm))
        parameter = getfield(params_check_norm, name)
        # If parameter has not been provided, don't check.
        if !isnothing(parameter)
            try 
                check_probability_distribution(parameter)
            catch e
                throw(ArgumentError("The parameter '$name' is not a valid probability distribution."))
            end
        else
            continue
        end
    end

end

"""
Function to check if the statefactor dimensions of the parameters are consistent.
"""
function check_parameter_states(GenerativeModel::DiscretePOMDP)

    # Destructures the parameters of the parameter struct
    (; A, B, D) = GenerativeModel

    A_states = [size(A[1], factor + 1) for factor in 1:length(size(A[1])[2:end])]
    B_states = [size(B[factor], 1) for factor in eachindex(B)]

    # Check whether to include D in the consistency check
    if !isnothing(D)
        D_states = [size(D[factor], 1) for factor in eachindex(D)]
    end

    # Check consistency between A/pA, B/pB, and D/pD

    if A_states != B_states && isnothing(D)

        throw(ArgumentError("""
        The number of states in each factor are different in A and B.

        States in A: $A_states
        States in B: $B_states
        """))
    elseif !isnothing(D)
        # Check consistency only between A and B if D is not provided
        if A_states != B_states || B_states != D_states
            throw(ArgumentError("""
            The number of states in each factor are different in A, B, and D.

            States in A: $A_states
            States in B: $B_states
            States in D: $D_states
            """))
        end
    end
end

"""
Function to check if the number of observationmodalities in the parameters are consistent.
"""
function check_parameter_observations(GenerativeModel::DiscretePOMDP)

    # Check the number of observations in A/pA and C
    A_observations = [size(GenerativeModel.A[modality], 1) for modality in eachindex(GenerativeModel.A)]
    C_observations = [size(GenerativeModel.C[modality], 1) for modality in eachindex(GenerativeModel.C)]

    # Throw an error if the number of observations are different
    if A_observations != C_observations
        throw(ArgumentError("\n\nThe number of observations are different in A and C \nNumber of observations in parameters: \n\nA: $A_observations \nC: $C_observations \n"))
    end

end

"""
Infer generative model parameters that are not provided.

# Arguments
- `parameters::POMDPActiveInferenceParameters`

If parameters C, D, or E are not provided, they are inferred from the provided parameters pA or A and pB or B.
"""
function infer_missing_parameters(parameters::POMDPActiveInferenceParameters, settings::POMDPActiveInferenceSettings, verbose::Bool = true)

    # If pA is provided, we create A based on pA
    if isnothing(parameters.A)
        parameters.A = normalize_arrays(deepcopy(parameters.pA))
    end

    # If pB is provided, we create B based on pB
    if isnothing(parameters.B)
        parameters.B = normalize_arrays(deepcopy(parameters.pB))
    end

    # If C is not provided, we create C based on the number of observations
    if isnothing(parameters.C)

        # Extracting n_observations
        n_observations = [size(A, 1) for A in parameters.A]

        # Creating C with zero vectors
        parameters.C = [zeros(observation_dimension) for observation_dimension in n_observations]

        if verbose
            @warn "No C-vector provided, no prior preferences will be used."
        end
    end
    
    # If D is not provided, we create either based on pD if provided. Otherwise, we create D based on the number of states
    if isnothing(parameters.D) && isnothing(parameters.pD)
        
        # Extracting n_states
        n_states = [size(B, 1) for B in parameters.B]

        # Uniform D vectors
        parameters.D = [fill(1.0 / state_dimension, state_dimension) for state_dimension in n_states]

        if verbose
            @warn "No D-vector provided, uniform priors over states will be used."
        end

    elseif !isnothing(parameters.pD)
        parameters.D = normalize_arrays(deepcopy(parameters.pD))
    end

    if isnothing(parameters.E)
        # Extracting n_controls and calculating the number of policies
        B_or_pB = isnothing(parameters.B) ? parameters.pB : parameters.B
        n_controls = [size(B_or_pB[factor], 3) for factor in eachindex(B_or_pB)]  
        n_policies = prod(n_controls) ^ settings.policy_length

        # Uniform E vector
        parameters.E = fill(1.0 / n_policies, n_policies)

        if verbose == true
            @warn "No E-vector provided, uniform prior over policies will be used."
        end
    end

end