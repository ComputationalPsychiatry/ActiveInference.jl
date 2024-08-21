""" -------- Utility Functions -------- """

""" Creates an array of "Any" with the desired number of sub-arrays"""
function array_of_any(num_arr::Int) 
    return Array{Any}(undef, num_arr) #saves it as {Any} e.g. can be any kind of data type.
end

""" Creates an array of "Any" with the desired number of sub-arrays filled with zeros"""
function array_of_any_zeros(shape_list)
    arr = Array{Any}(undef, length(shape_list))
    for (i, shape) in enumerate(shape_list)
        arr[i] = zeros(Real, shape...)
    end
    return arr
end

""" Creates an array of "Any" as a uniform categorical distribution"""
function array_of_any_uniform(shape_list)
    arr = Array{Any}(undef, length(shape_list))  
    for i in eachindex(shape_list)
        shape = shape_list[i]
        arr[i] = norm_dist(ones(Real, shape))  
    end
    return arr
end

""" Creates a onehot encoded vector """
function onehot(index::Int, vector_length::Int)
    vector = zeros(Real, vector_length)
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


""" Equivalent to pymdp's "to_obj_array" """
function to_array_of_any(arr::Array)
    # Check if arr is already an array of arrays
    if typeof(arr) == Array{Array,1}
        return arr
    end
    # Create an array_out and assign squeezed array to the first element
    obj_array_out = Array{Any,1}(undef, 1)
    obj_array_out[1] = dropdims(arr, dims = tuple(findall(size(arr) .== 1)...))  
    return obj_array_out
end


""" Selects the highest value from Array -- used for deterministic action sampling """
function select_highest(options_array::Array{Float64})
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
    action_marginals = array_of_any_zeros(aif.num_controls)
    log_action_marginals = array_of_any(num_factors)
    q_pi = get_states(aif, "posterior_policies")
    policies = get_states(aif, "policies")
    
    for (pol_idx, policy) in enumerate(policies)
        for (factor_i, action_i) in enumerate(policy[1,:])
            action_marginals[factor_i][action_i] += q_pi[pol_idx]
        end
    end

    action_marginals = norm_dist_array(action_marginals)

    for factor_i in 1:num_factors
        log_marginal_f = spm_log_single(action_marginals[factor_i])
        log_action_marginals[factor_i] = log_marginal_f
    end

    return log_action_marginals
end

""" Generate Random Generative Model as A and B matrices """
function generate_random_GM(n_states::Vector{Int64}, n_obs::Vector{Int64}, n_controls::Vector{Int64})

    # Initialize A matrices:
    A_shapes = [[o_dim; n_states] for o_dim in n_obs]
    A = array_of_any_zeros(A_shapes)

    # Fill A matrices with random probabilities
    for (i, matrix) in enumerate(A)
        for idx in CartesianIndices(matrix)
            matrix[idx] = rand()
        end
        A[i] = norm_dist(matrix)
    end

    # Initialize B matrices
    B_shapes = [[ns, ns, n_controls[f]] for (f, ns) in enumerate(n_states)]
    B = array_of_any_zeros(B_shapes)

    # Fill B matrices with random probabilities
    for (i, matrix) in enumerate(B)
        for idx in CartesianIndices(matrix)
            matrix[idx] = rand()
        end
        B[i] = norm_dist(matrix)
    end

    return A, B
end

""" Check if the array is a proper probability distribution """
function check_normalization(arr)
    return all(tensor -> all(isapprox.(sum(tensor, dims=1), 1.0, rtol=1e-5, atol=1e-8)), arr)
end