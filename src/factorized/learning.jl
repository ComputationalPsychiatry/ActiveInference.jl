using Format
using Infiltrator
using Revise

#show(stdout, "text/plain", x)
# @infiltrate; @assert false


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
function update_state_likelihood_dirichlet(
                        pB, 
                        B, 
                        actions, 
                        qs::Vector{Vector{T}} where T <: Real, 
                        qs_prev,
                        metamodel; 
                        lr = 1.0, 
                        fr = 1.0, 
                        factors_to_learn = "all",  # either "all" or list of states like [:loc, :prize, ...]
                        )

    # We follow the same idea as in get_expected_states() and elsewhere by selecting out actions from 
    # each B matrix. But instead of finishing with a dot(B_new, qs), here we filter out actions from 
    # a wB matrix of the same shape as B[ii].

    #if ReverseDiff.istracked(lr)  # these are not yet implemented in factorized code
    #    lr = ReverseDiff.value(lr)
    #end
    #if ReverseDiff.istracked(fr)
    #    fr = ReverseDiff.value(fr)
    #end

    state_names = collect(keys(metamodel.state_deps))
    if factors_to_learn == "all"   
        factors_to_learn = state_names
    end
    
    action_names = collect(keys(metamodel.action_deps))
    Biis_with_action = [ii for ii in 1:length(pB) if length(intersect(metamodel.state_deps[ii], action_names)) > 0]
    null_actions = [metamodel.policies.action_contexts[action][:null_action] for action in action_names]
    null_action_ids = [findfirst(x -> x == null_actions[ii], metamodel.action_deps[ii]) for ii in 1:length(action_names)]

    qB = deepcopy(pB)
    for (Bii, state_name) in enumerate(state_names) 
        if !(state_name in factors_to_learn)
            continue
        end     

        # list of the hidden state factor indices that the dynamics of `qs[Bii]` depend on
        # potentially more than one action per B matrix
        factors = metamodel.state_deps[Bii]  # eg: [:loc, :loc, :move], with last being an action
        # potentially, factors for this B matrix could have more than one action
        
        factor_idx = [findfirst(x -> x == j, keys(metamodel.state_deps)) for j in factors]
        # eg., [1,1,nothing), where nothing signifies an action

        # select out action dimensions
        idx = nothing

        if Bii in Biis_with_action

            # here we handle potentially more than one action for this B matrix; 
            # e.g. policy = ((1,2), (4,5)) for action_names = (:move, :jump)
            # Then, selections[1] might equal 
            # (action_name = :move, action_option_id = 1, action_option_name = :UP, null_action_id = 5)
            
            selections = [
                NamedTuple{(:action_name, :action_option_id, :action_option_name, :null_action_id)}(
                    (name, pol, metamodel.action_deps[name][pol], null_action_id))
                for (name, pol, null_action_id) in zip(action_names, actions, null_action_ids)
            ]
            #printfmtln("\nstep={}, Bii={}, selections= {}", t, Bii, selections)
            
            idx = []  # index of dims of this B matrix, states always come before actions in depencency lists
            iaction = 1
            for (idep, dep) in enumerate(factors) 
                if dep in keys(metamodel.action_deps)
                    # this dim is an action
                    push!(idx, selections[iaction].action_option_id)
                    iaction += 1
                else
                    # this is a state
                    push!(idx, 1:B[Bii].size[idep])  # e.g., push!(idx, (100,100,5)[1]) if idep==1
                end
            end
            
            Bc = deepcopy(B[Bii][idx...])  # select out actions, now only state dependencies left            
        else
            Bc = deepcopy(B[Bii])  # no actions for this B matrix
            idx = [1:B[Bii].size[i] for i in 1:ndims(B[Bii])]  
        end

        if ndims(Bc) > 2
            @infiltrate; @assert false  # not yet implemented
        end

        # todo: get element type, rather than use Float64 below
        dfdb = outer_product(qs[Bii], qs_prev[Bii])
        dfdb .*= Float64.(Bc .> 0)  # only update cells where B[ii] > 0, with action selected out
        qB[Bii][idx...] = qB[Bii][idx...] * fr .+ (lr .* dfdb)
        
    end

    #printfmtln("\ntest pB, is same? {}\n", pB == qB)
    #@infiltrate; @assert false 

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