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
function onehot(
        index::Int, 
        vector_length::Int,
        float_type::Type
    )

    vector = zeros(float_type, vector_length)
    vector[index] = 1.0
    return vector
end

function onehot(index::Int, vector_length::Int)
    vector = zeros(vector_length)
    vector[index] = 1.0
    return vector
end

""" Get Model Dimensions from either A or B Matrix """
function get_model_dimensions(A = nothing, B = nothing)
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
    num_factors = length(aif.settings._n_controls)
    q_pi = get_states(aif, "q_pi")
    policies = aif.settings.policies
    
    # Determine the element type from q_pi
    eltype_q_pi = eltype(q_pi)

    # Initialize action_marginals with the correct element type
    action_marginals = create_matrix_templates(aif.settings._n_controls, "zeros", eltype_q_pi)
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

""" Function that checks if a parameter is non-negative for single value parameter"""
is_non_negative(parameter::Number) = parameter >= 0

""" Function that checks if a parameter is non-negative for array parameter"""
is_non_negative(parameter::AbstractArray) = all(is_non_negative, parameter)


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

"""
    process_observation(observation::Int, n_modalities::Int, n_observations::Vector{Int})

Process a single modality observation. Returns a one-hot encoded vector. 

# Arguments
- `observation::Int`: The index of the observed state with a single observation modality.
- `n_modalities::Int`: The number of observation modalities in the observation. 
- `n_observations::Vector{Int}`: A vector containing the number of observations for each modality.

# Returns
- `Vector{Vector{Real}}`: A vector containing a single one-hot encoded observation.
"""
function process_observation(observation::Int, n_modalities::Int, n_observations::Vector{Int})

    # Check if there is only one modality
    if n_modalities == 1
        # Create a one-hot encoded vector for the observation
        processed_observation = onehot(observation, n_observations[1]) 
    end

    # Return the processed observation wrapped in a vector
    return [processed_observation]
end

"""
    process_observation(observation::Union{Array{Int}, Tuple{Vararg{Int}}}, n_modalities::Int, n_observations::Vector{Int})

Process observation with multiple modalities and return them in a one-hot encoded format 

# Arguments
- `observation::Union{Array{Int}, Tuple{Vararg{Int}}}`: A collection of indices of the observed states for each modality.
- `n_modalities::Int`: The number of observation modalities in the observation. 
- `n_observations::Vector{Int}`: A vector containing the number of observations for each modality.

# Returns
- `Vector{Vector{Real}}`: A vector containing one-hot encoded vectors for each modality.
"""
function process_observation(observation::Union{Array{Int}, Tuple{Vararg{Int}}}, n_modalities::Int, n_observations::Vector{Int})

    # Initialize the processed_observation vector
    processed_observation = Vector{Vector{Float64}}(undef, n_modalities)

    # Check if the length of observation matches the number of modalities
    if length(observation) == n_modalities
        for (modality, modality_observation) in enumerate(observation)
            # Create a one-hot encoded vector for the current modality observation
            one_hot = onehot(modality_observation, n_observations[modality])
            # Add the one-hot vector to the processed_observation vector
            processed_observation[modality] = one_hot
        end
    end

    return processed_observation
end