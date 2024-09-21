"""Normalizes a Categorical probability distribution"""
function normalize_distribution(distribution)
    return distribution ./ sum(distribution, dims=1)
end


"""
    capped_log(x::Real)

# Arguments
- `x::Real`: A real number.

Return the natural logarithm of x, capped at the machine epsilon value of x.
"""
function capped_log(x::Real)
    return log(max(x, eps(x))) 
end

"""
    capped_log(x::AbstractArray{T}) where T <: Real

# Arguments
- `array::AbstractArray{T}`: An array of real numbers.

Return the natural logarithm of x, capped at the machine epsilon value of x.
"""
function capped_log(array::AbstractArray{T}) where T <: Real
    # convert Reals to Floats
    x_float = float.(array) 
    # return the capped log of each element in x
    return log.(max.(x_float, eps(one(eltype(x_float)))))
end

### This method will be deprecated once all types in the package have been made more strict.
"""
    capped_log(array::Array{Any})

# Arguments
- `array::Array{Any}`: An array of real numbers.

Return the natural logarithm of x, capped at the machine epsilon value of x.
"""
function capped_log(array::Array{Any})
    # convert Reals to Floats
    x_float = float.(array) 
    # return the capped log of each element in x
    return log.(max.(x_float, eps(one(eltype(x_float)))))
end

""" Apply capped_log to array of arrays """
function capped_log_array(array)
    
    return map(capped_log, array)
end


""" Get Joint Likelihood """
function get_joint_likelihood(A, obs_processed, num_states)
    ll = ones(Real, num_states...)
    for modality in eachindex(A)
        ll .*= dot_likelihood(A[modality], obs_processed[modality])
    end

    return ll
end

""" Dot-Product Function """
function dot_likelihood(A, obs)
    # Adjust the shape of obs to match A
    reshaped_obs = reshape(obs, (length(obs), 1, 1, 1))  
    # Element-wise multiplication and sum over the first axis
    LL = sum(A .* reshaped_obs, dims=1)
    # Remove singleton dimensions
    LL = dropdims(LL, dims= tuple(findall(size(LL) .== 1)...))
    if prod(size(LL)) == 1
        LL = [LL[]]  
    end
    return LL
end

""" Softmax Function for array of arrays """
function softmax_array(arr)
    output = Array{Any}(undef, length(arr))
    
    # Iterate through each index in arr and apply softmax
    for idx in eachindex(arr)
        output[idx] = softmax(arr[idx], dims=1)
        
    end
    
    return output
end


""" Multi-dimensional outer product """
function outer_product(x, y=nothing; remove_singleton_dims=true, args...)
    # If only x is provided and it is a vector of arrays, recursively call outer_product on its elements.
    if y === nothing && isempty(args)
        if x isa AbstractVector
            return reduce((a, b) -> outer_product(a, b), x)
        elseif typeof(x) <: Number || typeof(x) <: AbstractArray
            return x
        else
            throw(ArgumentError("Invalid input to outer_product (\$x)"))
        end
    end

    # If y is provided, perform the cross multiplication.
    if y !== nothing
        reshape_dims_x = tuple(size(x)..., ones(Real, ndims(y))...)
        A = reshape(x, reshape_dims_x)

        reshape_dims_y = tuple(ones(Real, ndims(x))..., size(y)...)
        B = reshape(y, reshape_dims_y)

        z = A .* B
    else
        z = x
    end

    # Recursively call outer_product for additional arguments
    for arg in args
        z = outer_product(z, arg; remove_singleton_dims=remove_singleton_dims)
    end

    # remove singleton dimension if true--
    if remove_singleton_dims
        z = dropdims(z, dims = tuple(findall(size(z) .== 1)...))
    end

    return z
end

#Multidimensional inner product
# Instead of summing over all indices, the function sums over only the last three
# dimensions of X while keeping the first dimension separate, creating a sum for each "layer" of X.

function dot_product(X, x)

    if all(isa.(x, AbstractArray))  
        n_factors = length(x)
    else
        x = [x]  
        n_factors = length(x)
    end

    ndims_X = ndims(X)
    dims = collect(ndims_X - n_factors + 1 : ndims_X)
    Y = zeros(Real, size(X, 1))

    for indices in Iterators.product((1:size(X, i) for i in 1:ndims_X)...)
        product = X[indices...] * prod(x[factor][indices[dims[factor]]] for factor in 1:n_factors)
        Y[indices[1]] += product
    end

    if prod(size(Y)) <= 1
        Y = only(Y)
        Y = [float(Y)]  
    end

    return Y
end


""" Calculate Bayesian Surprise """
function calculate_bayesian_surprise(A, x)
    qx = outer_product(x)
    G = 0.0
    qo = Real[]
    idx = [collect(Tuple(indices)) for indices in findall(qx .> exp(-16))]
    index_vector = []

    for i in idx   
        po = ones(Real, 1)
        for (_, A_m) in enumerate(A)
            index_vector = (1:size(A_m, 1),)  
            for additional_index in i  
                index_vector = (index_vector..., additional_index)  
            end
            po = outer_product(po, A_m[index_vector...])
        end
        po = vec(po) 
        if isempty(qo)
            resize!(qo, length(po))
            fill!(qo, 0.0)
        end
        qo += qx[i...] * po
        G += qx[i...] * dot(po, log.(po .+ exp(-16)))
    end
    G = G - dot(qo, capped_log(qo))
    return G
end

""" Normalizes multiple arrays """
function normalize_arrays(array::Vector{<:Array{<:Real}})
    return map(normalize_distribution, array)
end

""" SPM_wnorm """
function spm_wnorm(A)
    EPS_VAL = 1e-16

    A .+= EPS_VAL
    norm = 1.0 ./ sum(A, dims = 1)
    avg = 1 ./ A
    wA = norm .- avg

    return wA
end

"""
    Calculate Bayesian Model Average (BMA)

Calculates the Bayesian Model Average (BMA) which is used for the State Action Prediction Error (SAPE).
It is a weighted average of the expected states for all policies weighted by the posterior over policies.
The `qs_pi_all` should be the collection of expected states given all policies. Can be retrieved with the
`get_expected_states` function.

`qs_pi_all`: Vector{Any} \n
`q_pi`: Vector{Float64}

"""
function bayesian_model_average(qs_pi_all, q_pi)

    # Extracting the number of factors, states, and timesteps (policy length) from the first policy
    n_factors = length(qs_pi_all[1][1])
    n_states = [size(qs_f, 1) for qs_f in qs_pi_all[1][1]]
    n_steps = length(qs_pi_all[1])

    # Preparing vessel for the expected states for all policies. Has number of undefined entries equal to the number of 
    # n_steps with each entry having the entries equal to the number of factors
    qs_bma = [Vector{Vector{Real}}(undef, n_factors) for _ in 1:n_steps]

    # Populating the entries with zeros for each state in each factor for each timestep in policy
    for i in 1:n_steps
        for f in 1:n_factors
            qs_bma[i][f] = zeros(Real, n_states[f])
        end
    end

    # Populating the entries with the expected states for all policies weighted by the posterior over policies
    for i in 1:n_steps
        for (pol_idx, policy_weight) in enumerate(q_pi)
            for f in 1:n_factors
                qs_bma[i][f] .+= policy_weight .* qs_pi_all[pol_idx][i][f]
            end
        end
    end

    return qs_bma
end