""" utilities for the generative model of a DiscretePOMDP """


### Check generative model ###
"""
check generative model parameters

# Arguments
- `A::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing`: A-matrix (Observation Likelihood model)
- `B::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing`: B-matrix (Transition model)
- `C::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing`: C-vectors (Preferences over observations)
- `D::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing`: D-vectors (Prior over states)

Throws an error if the generative model parameters are not valid:
- Both A and B must be provided.
- The dimensions of the matrices must be consistent.
- The values must be non-negative (except for C).
- The sum of each column or vector must be approximately 1.
- Not both the parameter and their prior can be provided.

"""
function check_generative_model(
    A::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing,
    B::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing,
    C::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing,
    D::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing
)

    if isnothing(A) || isnothing(B)
        throw(ArgumentError("A and B must be provided in order to infer structure of the generative model."))
    end

    # Check if the number of states in A, B, and D are consistent.
    # We let this check be done on either the prior or the parameter, depending on which is provided.
    check_parameter_states(A, B, D)

    # Check if the number of observation modalities in A and C are consistent.
    # We let this check be done on either the prior or the parameter, depending on which is provided.
    if !isnothing(C)
        check_parameter_observations(A, C)
    end

    # Check if the values are non-negative
    parameters = (("A", A), ("B", B), ("D", D))
    for (name, parameter) in parameters

        # If parameter has not been provided, don't check.
        if !isnothing(parameter)
            if !is_non_negative(parameter)
                throw(ArgumentError("All elements must be non-negative in parameter '$(name)'"))
            end
        else
            continue
        end
    end

    # Check if the probability distributions are normalized. Only A, B, D, and E are probability distributions.
    params_check_norm = (("A", A), ("B", B), ("D", D))

    for (name, parameter) in params_check_norm

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
function check_parameter_states(
    A::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing,
    B::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing,
    D::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing
)

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
function check_parameter_observations(
    A::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing,
    C::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing
)

    # Check the number of observations in A/pA and C
    A_observations = [size(A[modality], 1) for modality in eachindex(A)]
    C_observations = [size(C[modality], 1) for modality in eachindex(C)]

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
function infer_missing_parameters(
    A::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing,
    B::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} = nothing,
    C::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing,
    D::Union{Vector{Vector{T}}, Nothing} where {T <: Real} = nothing,
    verbose::Bool = true
)

    # If C is not provided, we create C based on the number of observations
    if isnothing(C)

        # Extracting n_observations
        n_observations = [size(A, 1) for A in A]

        # Creating C with zero vectors
        C = [zeros(observation_dimension) for observation_dimension in n_observations]

        if verbose
            @info "No C-vector provided, no prior preferences will be used."
        end

    end
    
    # If D is not provided, we create either based on pD if provided. Otherwise, we create D based on the number of states
    if isnothing(D)
        
        # Extracting n_states
        n_states = [size(B, 1) for B in B]

        # Uniform D vectors
        D = [fill(1.0 / state_dimension, state_dimension) for state_dimension in n_states]

        if verbose
            @info "No D-vector provided, uniform priors over states will be used."
        end

    end

    return C, D
end
