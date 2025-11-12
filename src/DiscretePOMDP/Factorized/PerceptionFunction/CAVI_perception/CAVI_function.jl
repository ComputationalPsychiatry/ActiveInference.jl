""" Update the models's beliefs over states """

function ActiveInferenceCore.perception(
    model::AIFModel{GenerativeModel, P, ActionProcess},
    observation::NamedTuple{<:Any, <:NTuple{N, Int64} where {N}},
    action::Union{Nothing, Vector{Int}}
) where {P<:AbstractPerceptualProcess}

    if model.action_process.previous_action !== nothing
        int_action = round.(Int, action)
        prediction_states = get_states_prediction(model.perceptual_process.posterior_states, model.generative_model.B, reshape(int_action, 1, length(int_action)))[1]
    else
        prediction_states = model.perceptual_process.prediction_states
    end


    posterior_states = model.perceptual_process.inference_function(
        model,
        prediction_states,
        observation
    )

    return (posterior_states = posterior_states, prediction_states = prediction_states)
end


function ActiveInferenceCore.perception(
    model::AIFModel{GenerativeModel, CAVI{NoLearning}, ActionProcess},
    observation::NamedTuple{<:Any, <:NTuple{N, Int64} where {N}},
    action::Union{Nothing, Vector{Int}}
)
    if model.action_process.previous_action !== nothing
        int_action = round.(Int, action)
        prediction_states = get_states_prediction(model.perceptual_process.posterior_states, model.generative_model.B, reshape(int_action, 1, length(int_action)))[1]
    else
        prediction_states = model.perceptual_process.prediction_states
    end
    
    # perform fixed-point iteration
    # perform fixed-point iteration
    posterior_states = cavi_factorized(
        model,
        prediction_states,
        observation,
    )

    return (posterior_states = posterior_states, prediction_states = prediction_states)
end

""" Update the models's beliefs over states with previous posterior states and action """
function ActiveInferenceCore.perception(
    model::AIFModel{GenerativeModel, CAVI{NoLearning}, ActionProcess},
    observation::NamedTuple{<:Any, <:NTuple{N, Int64} where {N}},
    previous_posterior_states::Union{Nothing, Vector{Vector{Float64}}},
    previous_action::Union{Nothing, Vector{Int}} 
)

    int_action = round.(Int, previous_action)
    prediction_states = get_states_prediction(previous_posterior_states, model.generative_model.B, reshape(int_action, 1, length(int_action)))[1]

    # perform fixed-point iteration
    posterior_states = cavi_factorized(
        model,
        prediction_states,
        observation,
    )

    return (posterior_states = posterior_states, prediction_states = prediction_states)
end


""" Run Factorized State Inference via Fixed-Point Iteration """
function cavi_factorized(
        model::AIFModel,
        prediction_states::NamedTuple{<:Any, <:NTuple{N, Vector{T2}} where {N}}, 
        new_obs::NamedTuple{<:Any, <:NTuple{N, Int64} where {N}},
    ) where {T2<:AbstractFloat}
    
    # qs_prior called log_prior here, as it will be converted to log
    """
    Run the fixed point iteration algorithm with sparse dependencies between factors and outcomes 
    """
    
    #=
    Step 1: Compute marginal log likelihoods for each factor.
    Likelihood[ii].ndim will be size ndims(A[ii]) -1 (i.e., equal to the number of factors).
    E.g., if A[ii] is (4,25,3) and prior is (25,), then intermediate result is (4,25,3) , which
    is summed over dim=1, so (4,25,3) --> (1,25,3) --> (25,3).
    =#
    T2_type = Float64 # (Note Sam: hardcoded for now; later make generic over T2)
    model_info = model.generative_model.info

    log_prior = deepcopy(prediction_states)

    obs_names = keys(model_info.observation_modalities)
    mLLs = [zeros(T2_type, (1, x.A_dims[2:end]...)) for x in model_info.observation_modalities]
    marginal_LLs = (; zip(obs_names, mLLs)...)

    #log_likelihoods = []
    for (ii, obs) in enumerate(model_info.observation_modalities)
        obs_ii = onehot(new_obs[ii], obs.A_dims[1], T2_type) 
        dims = repeat([1], length(obs.A_dims))
        dims[1] = obs_ii.size[1]  # e.g., [9,1,1] for 3-dim A with 9 categories
        obs_ii = reshape(obs_ii, dims...)
        marginal_LLs[ii][:] = dropdims(sum(model.generative_model.A[ii] .* obs_ii, dims=1), dims=1) # obs_ii is broadcast over dims of A
        marginal_LLs[ii][:] = capped_log(marginal_LLs[ii])
    end
    
    # Step 2: Map prior to log space and create initial log-posterior
    for prior in log_prior
        prior[:] = capped_log(prior)
    end

    # create an uninformed guess at new qs
    qs_current = deepcopy(log_prior)
    for qs in qs_current
        qs[:] = ones(T2_type, length(qs)) / length(qs)
    end
    last = deepcopy(qs_current)

    #=
    Step 3: Iterate until convergence
    A[ii] will be sequentially multiplied by every prior of every state that is a dependency. 
    Four NamedTuples:
        - qs_current: holds updates and will be the NT that is eventually returned
        - last: qs from the last iteration. 
        - qL: qs, with everything marginalized out except obs ii
    =#

    for iter in 1:model.perceptual_process.num_iter
        #@infiltrate; @assert false
        #qs_new = Vector{Vector{Float64}}([zeros(x.size) for x in prior])
        #qs = [LEF.softmax(x) for x in new_log_q]
        
        for (ii, state) in enumerate(keys(model_info.state_factors))
            qL = zeros(T2_type, log_prior[ii].size)
            
            for (jj, obs) in enumerate(model_info.observation_modalities)
                
                if state in obs.A_dim_names
                    qL += all_marginal_log_likelihood(
                        qs_current, 
                        marginal_LLs[jj], 
                        ii, 
                        jj, 
                        model_info,
                    )
                end
            end
            #@infiltrate; @assert false
            qs_current[ii][:] = softmax(qL .+ log_prior[ii])
        end

        #printfmtln("err= {}", sum.([dd .^2 for dd in collect(values(qs_current)) - collect(values(last))]))

        if all(isapprox.(values(qs_current), values(last), atol=model.perceptual_process.dF_tol))
            break
        end

        last = deepcopy(qs_current) 
    end
    
    #@infiltrate; @assert false
    return qs_current
end


function all_marginal_log_likelihood(
        qs::NamedTuple{<:Any, <:NTuple{N, Vector{T2}} where {N}},
        LL::Array{T2}, 
        ii::Int64, 
        jj::Int64, 
        model_info::GenerativeModelInfo
    ) where {T2<:AbstractFloat}

    # returns a new version of qs updated by marginalizing over all dependencies but dependency ii
    T2_type = Float64 # (Note Sam: hardcoded for now; later make generic over T2)
    if ndims(LL) == 2 #&& LL.size[2] == 1)
        # nothing to do, this is just A[obs,:]
        return dropdims(LL, dims=1)
    end

    obs_deps = model_info.observation_modalities[jj].A_dim_names[2:end]
    dep_ids = [findfirst(x -> x == j, keys(model_info.state_factors)) for j in obs_deps]

    A_n = dropdims(copy(LL), dims=1) # (NOTE Sam: this needs to be tested with even more dims)

    # we want to skip dependency ii, so place ii first in A and in dep_ids
    if dep_ids[1] != ii 
        dep_ids2 = vcat([ii], findall(x -> x != ii, dep_ids))
        obs_ids = [findfirst(x -> x == j, dep_ids) for j in dep_ids2]
        A_n = permutedims(A_n, obs_ids)
        dep_ids = dep_ids2
    end

    deps = Vector{Vector{T2_type}}()
    for idep in reverse(dep_ids[2:end])   
        push!(deps, qs[idep])
    end

    A_n = dot_product1(A_n, deps)

    #@infiltrate; @assert false
    if A_n.size != qs[ii].size
        error("Size mismatch in all_marginal_log_likelihood: A size $(A_n.size), qs size $(qs[ii].size)")
    end

    return A_n
   

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

