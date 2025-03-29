""" Update obs likelihood matrix """
function update_obs_likelihood_dirichlet(pA, A, obs, qs; lr = 1.0, fr = 1.0, modalities = "all")

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

    if modalities === "all"
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
function update_state_likelihood_dirichlet(pB, B, actions, qs::Vector{Vector{T}} where T <: Real, qs_prev; lr = 1.0, fr = 1.0, factors = "all")

    if ReverseDiff.istracked(lr)
        lr = ReverseDiff.value(lr)
    end
    if ReverseDiff.istracked(fr)
        fr = ReverseDiff.value(fr)
    end

    num_factors = length(pB)

    qB = deepcopy(pB)

    if factors === "all"
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
function update_state_prior_dirichlet(pD, qs::Vector{Vector{T}} where T <: Real; lr = 1.0, fr = 1.0, factors = "all")

    num_factors = length(pD)

    qD = deepcopy(pD)

    if factors == "all"
        factors = collect(1:num_factors)
    end

    for factor in factors
        idx = pD[factor] .> 0
        qD[factor][idx] = (fr * qD[factor][idx]) .+ (lr * qs[factor][idx])
    end  
    
    return qD
end

""" Update A-matrix """
function update_A(aif::POMDPActiveInference)

    qA = update_obs_likelihood_dirichlet(aif.parameters.pA, aif.parameters.A, aif.states.obs_current, aif.states.qs_current, lr = aif.parameters.lr_pA, fr = aif.parameters.fr_pA, modalities = aif.settings.modalities_to_learn)
    
    aif.parameters.pA = deepcopy(qA)
    aif.parameters.A = deepcopy(normalize_arrays(qA))

    return qA
end

""" Update B-matrix """
function update_B(aif::POMDPActiveInference)

    # if length(get_history(aif, "posterior_states")) > 1
    if length(aif.history.qs_current) > 2
        qs_prev = aif.history.qs_current[end-1]

        qB = update_state_likelihood_dirichlet(aif.parameters.pB, aif.parameters.B, aif.states.action, aif.states.qs_current, qs_prev, lr = aif.parameters.lr_pB, fr = aif.parameters.fr_pB, factors = aif.settings.factors_to_learn)

        aif.parameters.pB = deepcopy(qB)
        aif.parameters.B = deepcopy(normalize_arrays(qB))
    else
        qB = nothing
    end

    return qB
end

""" Update D-matrix """
function update_D(aif::POMDPActiveInference)

    if length(aif.history.qs_current) == 2 # need a smarter way to define this

        qs_t1 = aif.history.qs_current[end]
        qD = update_state_prior_dirichlet(aif.parameters.pD, qs_t1; lr = aif.parameters.lr_pD, fr = aif.parameters.fr_pD, factors = aif.settings.factors_to_learn)

        aif.parameters.pD = deepcopy(qD)
        aif.parameters.D = deepcopy(normalize_arrays(qD))
    else
        qD = nothing
    end

    return qD
end