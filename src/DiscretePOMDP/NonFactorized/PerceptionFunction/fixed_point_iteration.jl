""" Update the models's beliefs over states """
function ActiveInferenceCore.perception(
    model::AIFModel{GenerativeModel, PerceptualProcess{FixedPointIteration}, ActionProcess},
    observation::Vector{Int}
)

    if model.action_process.action !== nothing
        int_action = round.(Int, model.action_process.action)
        prior_qs_prediction = get_expected_states(model.perceptual_process.posterior_states, model.generative_model.B, reshape(int_action, 1, length(int_action)))[1]
    else
        prior_qs_prediction = model.perceptual_process.prior_qs_prediction
    end

    # make observations into a one-hot encoded vector
    processed_observation = process_observation(
        observation, 
        model.generative_model.info.n_modalities, 
        model.generative_model.info.n_observations
    )

    # perform fixed-point iteration
    posterior_states = fixed_point_iteration(;
        A = model.generative_model.A,
        observation = processed_observation,
        n_factors = model.generative_model.info.n_factors,
        n_states = model.generative_model.info.n_states,
        prior = prior_qs_prediction,
        num_iter = model.perceptual_process.optim_engine.num_iter,
        dF_tol = model.perceptual_process.optim_engine.dF_tol
    )

    return posterior_states, prior_qs_prediction
end

""" Update the models's beliefs over states with previous posterior states and action """
function ActiveInferenceCore.perception(
    model::AIFModel{GenerativeModel, PerceptualProcess{FixedPointIteration}, ActionProcess},
    observation::Vector{Int},
    previous_posterior_states::Union{Nothing, Vector{Vector{Float64}}},
    previous_action::Union{Nothing, Vector{Int}} 
)

    int_action = round.(Int, previous_action)
    prior_qs_prediction = get_expected_states(previous_posterior_states, model.generative_model.B, reshape(int_action, 1, length(int_action)))[1]

    # make observations into a one-hot encoded vector
    processed_observation = process_observation(
        observation, 
        model.generative_model.info.n_modalities, 
        model.generative_model.info.n_observations
    )

    # perform fixed-point iteration
    posterior_states = fixed_point_iteration(;
        A = model.generative_model.A,
        observation = processed_observation,
        n_factors = model.generative_model.info.n_factors,
        n_states = model.generative_model.info.n_states,
        prior = prior_qs_prediction,
        num_iter = model.perceptual_process.optim_engine.num_iter,
        dF_tol = model.perceptual_process.optim_engine.dF_tol
    )

    return posterior_states, prior_qs_prediction
end

""" Run State Inference via Fixed-Point Iteration """
function fixed_point_iteration(;
    A::Vector{Array{T,N}} where {T <: Real, N}, observation::Vector{Vector{Float64}}, n_factors::Int64, n_states::Vector{Int64},
    prior::Union{Nothing, Vector{Vector{T}}} where T <: Real = nothing, 
    num_iter::Int=num_iter, dF::Float64=1.0, dF_tol::Float64=dF_tol
)
    # Get joint likelihood
    likelihood = get_joint_likelihood(A, observation, n_states)
    likelihood = capped_log(likelihood)

    # Initialize posterior and prior
    qs = Vector{Vector{Float64}}(undef, n_factors)
    for factor in 1:n_factors
        qs[factor] = ones(n_states[factor]) / n_states[factor]
    end

    # If no prior is provided, create a default prior with uniform distribution
    if prior === nothing
        prior = create_matrix_templates(n_states)
    end
    
    # Create a copy of the prior to avoid modifying the original
    prior = deepcopy(prior)
    prior = capped_log_array(prior) 

    # Initialize free energy
    prev_vfe = calc_free_energy(qs, prior, n_factors)

    # Single factor condition
    if n_factors == 1
        qL = dot_product(likelihood, qs[1])  
        return [softmax(qL .+ prior[1], dims=1)]

    # If there are more factors
    else
        ### Fixed-Point Iteration ###
        curr_iter = 0
        ### Sam NOTE: We need check if ReverseDiff might potantially have issues with this while loop ###
        while curr_iter < num_iter && dF >= dF_tol
            qs_all = qs[1]
            # Loop over each factor starting from the second one
            for factor in 2:n_factors
                # Reshape and multiply qs_all with the current factor's qs
                qs_all = qs_all .* reshape(qs[factor], tuple(ones(Real, factor - 1)..., :, 1))
            end

            # Compute the log-likelihood
            LL_tensor = likelihood .* qs_all

            # Update each factor's qs
            for factor in 1:n_factors
                # Initialize qL for the current factor
                qL = zeros(Real, size(qs[factor]))

                # Compute qL for each state in the current factor
                for i in 1:size(qs[factor], 1)
                    qL[i] = sum([LL_tensor[indices...] / qs[factor][i] for indices in Iterators.product([1:size(LL_tensor, dim) for dim in 1:n_factors]...) if indices[factor] == i])
                end

                # If qs is tracked by ReverseDiff, get the value
                if ReverseDiff.istracked(softmax(qL .+ prior[factor], dims=1))
                    qs[factor] = ReverseDiff.value(softmax(qL .+ prior[factor], dims=1))
                else
                    # Otherwise, proceed as normal
                    qs[factor] = softmax(qL .+ prior[factor], dims=1)
                end
            end

            # Recompute free energy
            vfe = calc_free_energy(qs, prior, n_factors, likelihood)

            # Update stopping condition
            dF = abs(prev_vfe - vfe)
            prev_vfe = vfe

            # Increment iteration
            curr_iter += 1
        end

        return qs
    end
end


""" Calculate Free Energy """
function calc_free_energy(qs::Vector{Vector{T}} where T <: Real, prior, n_factors, likelihood=nothing)
    # Initialize free energy
    free_energy = 0.0
    
    # Calculate free energy for each factor
    for factor in 1:n_factors
        # Neg-entropy of posterior marginal
        negH_qs = dot(qs[factor], log.(qs[factor] .+ 1e-16))
        # Cross entropy of posterior marginal with prior marginal
        xH_qp = -dot(qs[factor], prior[factor])
        # Add to total free energy
        free_energy += negH_qs + xH_qp
    end
    
    # Subtract accuracy
    if likelihood !== nothing
        free_energy -= compute_accuracy(likelihood, qs)
    end
    
    return free_energy
end


""" Calculate Accuracy Term """
function compute_accuracy(log_likelihood, qs::Vector{Vector{T}} where T <: Real)
    n_factors = length(qs)
    ndims_ll = ndims(log_likelihood)
    dims = (ndims_ll - n_factors + 1) : ndims_ll

    # Calculate the accuracy term
    accuracy = sum(
        log_likelihood[indices...] * prod(qs[factor][indices[dims[factor]]] for factor in 1:n_factors)
        for indices in Iterators.product((1:size(log_likelihood, i) for i in 1:ndims_ll)...)
    )

    return accuracy
end

