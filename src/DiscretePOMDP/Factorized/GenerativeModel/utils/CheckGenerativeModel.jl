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
    A::Union{Nothing, NamedTuple} = nothing,
    B::Union{Nothing, NamedTuple} = nothing,
    C::Union{Nothing, NamedTuple} = nothing,
    D::Union{Nothing, NamedTuple} = nothing
)

    if isnothing(A) || isnothing(B)
        throw(ArgumentError("A and B must be provided in order to infer structure of the generative model."))
    end

    check_parameter_states(A, B, D)

    if !isnothing(C)
        check_parameter_observations(A, C)
    end

    # Check non-negativity
    parameters = (("A", A), ("B", B), ("D", D))
    for (name, parameter) in parameters
        if !isnothing(parameter)
            for val in values(parameter)
                if any(val .< 0)
                    throw(ArgumentError("All elements must be non-negative in parameter '$name'"))
                end
            end
        end
    end

    # Check probability normalization for A, B, D
    for (name, parameter) in parameters
        if !isnothing(parameter)
            for val in values(parameter)
                s = sum(val, dims=1)
                if any(abs.(s .- 1) .> 1e-8)
                    throw(ArgumentError("The parameter '$name' is not a valid probability distribution."))
                end
            end
        end
    end
end

### Check state factor consistency ###
function check_parameter_states(
    A::Union{Nothing, NamedTuple} = nothing,
    B::Union{Nothing, NamedTuple} = nothing,
    D::Union{Nothing, NamedTuple} = nothing
)
    # Extract number of states from A and B
    A_states = [collect(size(v)[2:end]) for v in values(A)]

    # Check that all elements in A_states are identical
    first_dims = A_states[1]
    for (i, dims) in enumerate(A_states)
        if dims != first_dims
            throw(ArgumentError("State dimensions differ between modalities.\nModalities: $(keys(A))\nState dims: $A_states"))
        end
    end

    B_states = [size(v, 1) for v in values(B)]

    if isnothing(D)
        if A_states != B_states
            throw(ArgumentError("Number of states differ in A and B.\nA_states=$A_states\nB_states=$B_states"))
        end
    else
        D_states = [length(v) for v in values(D)]
        if A_states[1] != B_states || B_states != D_states
            throw(ArgumentError("Number of states differ in A, B, and D.\nA_states=$A_states\nB_states=$B_states\nD_states=$D_states"))
        end
    end
end

### Check observation modalities ###
function check_parameter_observations(
    A::Union{Nothing, NamedTuple} = nothing,
    C::Union{Nothing, NamedTuple} = nothing
)
    A_observations = [size(v, 1) for v in values(A)]
    C_observations = [length(v) for v in values(C)]

    if A_observations != C_observations
        throw(ArgumentError("Number of observations differ in A and C.\nA=$A_observations\nC=$C_observations"))
    end
end

### Infer missing parameters ###
function infer_missing_parameters(
    A::Union{Nothing, NamedTuple} = nothing,
    B::Union{Nothing, NamedTuple} = nothing,
    C::Union{Nothing, NamedTuple} = nothing,
    D::Union{Nothing, NamedTuple} = nothing,
    verbose::Bool = true
)
    if isnothing(C)
        C = NamedTuple{keys(A)}( (zeros(size(v,1)) for v in values(A)) )
        if verbose
            @info "No C-vector provided, no prior preferences will be used."
        end
    end

    if isnothing(D)
        D = NamedTuple{keys(B)}( (fill(1.0/size(v,1), size(v,1)) for v in values(B)) )
        if verbose
            @info "No D-vector provided, uniform priors over states will be used."
        end
    end

    return C, D
end
