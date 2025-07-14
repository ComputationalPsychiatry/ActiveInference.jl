""" Update the agent's beliefs over states and observations """
function update_parameters(agent::AIFAgent)

    if agent.perceptual_process.info.A_learning_enabled == true
        update_A(agent)
    end

    if agent.perceptual_process.info.B_learning_enabled == true
        update_B(agent)
    end

    if agent.perceptual_process.info.D_learning_enabled == true
        update_D(agent)
    end
    
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
function update_A(agent::AIFAgent)

    qA = update_obs_likelihood_dirichlet(
        agent.perceptual_process.A_learning.prior, 
        agent.generative_model.A, 
        agent.perceptual_process.current_observation, 
        agent.perceptual_process.posterior_states, 
        agent.perceptual_process.A_learning.learning_rate, 
        agent.perceptual_process.A_learning.forgetting_rate, 
        agent.perceptual_process.A_learning.modalities_to_learn
    )
    
    agent.perceptual_process.A_learning.prior = deepcopy(qA)
    agent.generative_model.A = deepcopy(normalize_arrays(qA))

end

""" Update B-matrix """
function update_B(agent::AIFAgent)

    # only update B if a previous posterior state exists or is not nothing
    if !isnothing(agent.perceptual_process.previous_posterior_states)

        qB = update_state_likelihood_dirichlet(
            agent.perceptual_process.B_learning.prior, 
            agent.generative_model.B, 
            agent.action_process.action, 
            agent.perceptual_process.posterior_states, 
            agent.perceptual_process.previous_posterior_states, 
            agent.perceptual_process.B_learning.learning_rate, 
            agent.perceptual_process.B_learning.forgetting_rate, 
            agent.perceptual_process.B_learning.factors_to_learn
        )

        agent.perceptual_process.B_learning.prior = deepcopy(qB)
        agent.generative_model.B = deepcopy(normalize_arrays(qB))
    else
        qB = nothing
    end
end

""" Update D-matrix """
function update_D(agent::AIFAgent)

    # only update D if a previous posterior state does not exists and is nothing
    if isnothing(agent.perceptual_process.previous_posterior_states)

        qD = update_state_prior_dirichlet(
            agent.perceptual_process.D_learning.prior, 
            agent.perceptual_process.posterior_states, 
            agent.perceptual_process.D_learning.learning_rate, 
            agent.perceptual_process.D_learning.forgetting_rate, 
            agent.perceptual_process.D_learning.factors_to_learn
        )

        agent.perceptual_process.D_learning.prior = deepcopy(qD)
        agent.generative_model.D = deepcopy(normalize_arrays(qD))
    else
        qD = nothing
    end
end