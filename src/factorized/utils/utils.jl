using Format
using Infiltrator
using Revise

#show(stdout, "text/plain", x)
# @infiltrate; @assert false


""" -------- Utility Functions -------- """

""" Creates an array of "Any" with the desired number of sub-arrays filled with zeros"""
function array_of_any_zeros(shape_list)
    arr = Array{Any}(undef, length(shape_list))
    for (i, shape) in enumerate(shape_list)
        arr[i] = zeros(Real, shape...)
    end
    return arr
end

""" Creates a onehot encoded vector """
function onehot(index::Int, vector_length::Int)
    vector = zeros(vector_length)
    vector[index] = 1.0
    return vector
end

""" Get Model Dimensions from either A or B Matrix """
function get_model_dimensions(
    A::Union{Nothing, Vector{Array{T}} where {T <: Real}} = nothing, 
    B::Union{Nothing, Vector{Array{T}} where {T <: Real}} = nothing
    )
    
    if A === nothing && B === nothing
        throw(ArgumentError("Must provide either `A` or `B`"))
    end
    num_obs, num_modalities, num_states, num_factors = nothing, nothing, nothing, nothing

    if A !== nothing
        num_obs = [size(a, 1) for a in A]
        num_modalities = length(num_obs)
    end

    if B !== nothing
        num_states = [size(b, 1) for b in B]
        num_factors = length(num_states)
    elseif A !== nothing
        num_states = [size(A[1], i) for i in 2:ndims(A[1])]
        num_factors = length(num_states)
    end

    return num_obs, num_states, num_modalities, num_factors
end


""" Selects the highest value from Array -- used for deterministic action sampling """
function select_highest(options_array::Vector{T}) where T <: Real
    options_with_idx = [(i, option) for (i, option) in enumerate(options_array)]
    max_value = maximum(value for (idx, value) in options_with_idx)
    same_prob = [idx for (idx, value) in options_with_idx if abs(value - max_value) <= 1e-8]

    if length(same_prob) > 1
        return same_prob[rand(1:length(same_prob))]
    else
        return same_prob[1]
    end
end


""" Selects action from computed actions probabilities -- used for stochastic action sampling """
function action_select(probabilities)
    sample_onehot = rand(Multinomial(1, probabilities))
    return findfirst(sample_onehot .== 1)
end

""" Function to get log marginal probabilities of actions """
function get_log_action_marginals(aif)
    num_factors = length(aif.num_controls)
    q_pi = get_states(aif, "posterior_policies")
    policies = get_states(aif, "policies")
    
    # Determine the element type from q_pi
    eltype_q_pi = eltype(q_pi)

    # Initialize action_marginals with the correct element type
    action_marginals = create_matrix_templates(aif.num_controls, "zeros", eltype_q_pi)
    log_action_marginals = Vector{Any}(undef, num_factors)
    
    for (pol_idx, policy) in enumerate(policies)
        for (factor_i, action_i) in enumerate(policy[1,:])
            action_marginals[factor_i][action_i] += q_pi[pol_idx]
        end
    end

    action_marginals = normalize_arrays(action_marginals)

    for factor_i in 1:num_factors
        log_marginal_f = capped_log(action_marginals[factor_i])
        log_action_marginals[factor_i] = log_marginal_f
    end

    return log_action_marginals
end

"""
Check if the vector of arrays is a proper probability distribution.

# Arguments

- (Array::Vector{<:Array{T}}) where T<:Real

Throws an error if the array is not a valid probability distribution:
- The values must be non-negative.
- The sum of the values must be approximately 1.
"""
function check_probability_distribution(Array::Vector{<:Array{T}}) where T<:Real
    for tensor in Array
        # Check for non-negativity
        if any(tensor .< 0)
            throw(ArgumentError("All elements must be non-negative."))
        end

        # Check for normalization
        if !all(isapprox.(sum(tensor, dims=1), 1.0, rtol=1e-5, atol=1e-8))
            throw(ArgumentError("The array is not normalized."))
        end
    end

    return true
end

"""
Check if the vector of vectors is a proper probability distribution.

# Arguments

- (Array::Vector{Vector{T}}) where T<:Real

Throws an error if the array is not a valid probability distribution:
- The values must be non-negative.
- The sum of the values must be approximately 1.
"""
function check_probability_distribution(Array::Vector{Vector{T}}) where T<:Real
    for vector in Array
        # Check for non-negativity
        if any(vector .< 0)
            throw(ArgumentError("All elements must be non-negative."))
        end

        # Check for normalization
        if !all(isapprox.(sum(vector, dims=1), 1.0, rtol=1e-5, atol=1e-8))
            throw(ArgumentError("The array is not normalized."))
        end
    end

    return true
end

"""
Check if the vector is a proper probability distribution.

# Arguments

- (Vector::Vector{T}) where T<:Real : The vector to be checked.

Throws an error if the array is not a valid probability distribution:
- The values must be non-negative.
- The sum of the values must be approximately 1.
"""
function check_probability_distribution(Vector::Vector{T}) where T<:Real
    # Check for non-negativity
    if any(Vector .< 0)
        throw(ArgumentError("All elements must be non-negative."))
    end

    # Check for normalization
    if !all(isapprox.(sum(Vector, dims=1), 1.0, rtol=1e-5, atol=1e-8))
        throw(ArgumentError("The array is not normalized."))
    end

    return true
end