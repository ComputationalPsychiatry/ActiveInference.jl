""" Update the model's beliefs over states and observations """
function update_parameters(
    model::AIFModel, 
    current_observation::Vector{Int},
    posterior_states::Vector{Vector{Float64}}
)

    if model.perceptual_process.info.A_learning_enabled == true
        A_updated, qA = update_A(model, current_observation, posterior_states)
    else
        A_updated = nothing
        qA = nothing
    end

    if model.perceptual_process.info.B_learning_enabled == true
        B_updated, qB = update_B(model, posterior_states)
    else
        B_updated = nothing
        qB = nothing
    end

    if model.perceptual_process.info.D_learning_enabled == true
        D_updated, qD = update_D(model, posterior_states)
    else
        D_updated = nothing
        qD = nothing
    end

    return (A_updated = A_updated, qA = qA, B_updated = B_updated, qB = qB, D_updated = D_updated, qD = qD)
end

""" Update obs likelihood matrix """
function update_obs_likelihood_dirichlet(pA, A, obs, qs, lr, fr, modalities)

    # If reverse diff is tracking the learning rate, get the value
    if ReverseDiff.istracked(lr)
        lr = ReverseDiff.value(lr)
    end
    # If reverse diff is tracking the forgetting rate, get the value
    if ReverseDiff.istracked(fr)
        fr = ReverseDiff.value(fr)
    end

    # Extracting the number of modalities and observations from the dirichlet: pA
    num_modalities = length(pA)
    num_observations = [size(pA[modality + 1], 1) for modality in 0:(num_modalities - 1)]

    obs = process_observation(obs, num_modalities, num_observations)

    # If modalities is not provided, learn all modalities
    if isempty(modalities)
        modalities = collect(1:num_modalities)
    end

    qA = deepcopy(pA)

    # Important! Takes first the cross product of the qs itself, so that it matches dimensions with the A and pA matrices
    qs_cross = outer_product(qs)

    for modality in modalities
        dfda = outer_product(obs[modality], qs_cross)
        dfda = dfda .* (A[modality] .> 0)
        qA[modality] = (fr * qA[modality]) + (lr * dfda)
    end

    return qA
end

""" Update state likelihood matrix """
function update_state_likelihood_dirichlet(pB, B, actions, qs::Vector{Vector{T}} where T <: Real, qs_prev, lr, fr, factors)

    if ReverseDiff.istracked(lr)
        lr = ReverseDiff.value(lr)
    end
    if ReverseDiff.istracked(fr)
        fr = ReverseDiff.value(fr)
    end

    num_factors = length(pB)

    qB = deepcopy(pB)

    # If factors is not provided, learn all factors
    if isempty(factors)
        factors = collect(1:num_factors)
    end

    for factor in factors
        dfdb = outer_product(qs[factor], qs_prev[factor])
        dfdb .*= (B[factor][:,:,Int(actions[factor])] .> 0)
        qB[factor][:,:,Int(actions[factor])] = qB[factor][:,:,Int(actions[factor])]*fr .+ (lr .* dfdb)
    end

    return qB
end

""" Update prior D matrix """
function update_state_prior_dirichlet(pD, qs::Vector{Vector{T}} where T <: Real, lr, fr, factors)

    num_factors = length(pD)

    qD = deepcopy(pD)

    # If factors is not provided, learn all factors
    if isempty(factors)
        factors = collect(1:num_factors)
    end

    for factor in factors
        idx = pD[factor] .> 0
        qD[factor][idx] = (fr * qD[factor][idx]) .+ (lr * qs[factor][idx])
    end  
    
    return qD
end

""" Update A-matrix """
function update_A(
    model::AIFModel, 
    current_observation::Vector{Int}, 
    posterior_states::Vector{Vector{Float64}}
)

    qA = update_obs_likelihood_dirichlet(
        model.perceptual_process.A_learning.prior, 
        model.generative_model.A, 
        current_observation, 
        posterior_states, 
        model.perceptual_process.A_learning.learning_rate, 
        model.perceptual_process.A_learning.forgetting_rate, 
        model.perceptual_process.A_learning.modalities_to_learn
    )
    
    qA = deepcopy(qA)
    A_updated = normalize_arrays(deepcopy(qA))
    
    return A_updated, qA
end

""" Update B-matrix """
function update_B(model::AIFModel, posterior_states::Vector{Vector{Float64}})

    # only update B if a previous posterior state exists or is not nothing
    if !isnothing(model.perceptual_process.previous_posterior_states)

        qB = update_state_likelihood_dirichlet(
            model.perceptual_process.B_learning.prior, 
            model.generative_model.B, 
            model.action_process.previous_action, 
            posterior_states, 
            model.perceptual_process.previous_posterior_states, 
            model.perceptual_process.B_learning.learning_rate, 
            model.perceptual_process.B_learning.forgetting_rate, 
            model.perceptual_process.B_learning.factors_to_learn
        )

        qB = deepcopy(qB)
        B_updated = normalize_arrays(deepcopy(qB))
    else
        qB = nothing
        B_updated = nothing
    end

    return B_updated, qB
end

""" Update D-matrix """
function update_D(model::AIFModel, posterior_states::Vector{Vector{Float64}})

    # only update D if a previous posterior state does not exists and is nothing
    if isnothing(model.perceptual_process.previous_posterior_states)

        qD = update_state_prior_dirichlet(
            model.perceptual_process.D_learning.prior, 
            posterior_states, 
            model.perceptual_process.D_learning.learning_rate, 
            model.perceptual_process.D_learning.forgetting_rate, 
            model.perceptual_process.D_learning.factors_to_learn
        )

        qD = deepcopy(qD)
        D_updated = normalize_arrays(deepcopy(qD))
    else
        qD = nothing
        D_updated = nothing
    end

    return D_updated, qD
end