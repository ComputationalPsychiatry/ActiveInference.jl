

module Learning


import ActiveInference.ActiveInferenceFactorized as AI 

using Format
using Infiltrator
#using Revise

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

function update_state_likelihood_dirichlet!(agent::AI.Agent, qs_prev::NamedTuple{<:Any, <:NTuple{N, Vector{Float64}} where {N}})

    model = agent.model
    policy = agent.last_action
    
    for (state_i, state) in enumerate(model.states)
            
        if isnothing(state.pB) || ismissing(state.pB)
            continue
        end

        # select out actions from B matrix
        B, idx = AI.Utils.select_B_actions(state, policy)
        
        if ndims(B) > 2
            @infiltrate; @assert false  # not yet implemented
        end

        # todo: get element type, rather than use Float64 below
        dfdb = AI.Maths.outer_product(agent.qs_current[state_i], qs_prev[state_i])
        dfdb .*= Float64.(B .> 0)  # only update cells where B[ii] > 0, with action selected out
        state.pB[idx...] = state.pB[idx...] * agent.parameters.fr_pB .+ (agent.parameters.lr_pB .* dfdb)
        state.B[:] = deepcopy(AI.Maths.normalize_distribution(state.pB))

        #printfmtln("\ntest pB: {}, qs = {}, >.4= {} \n",  
        #    round(sum(state.pB), digits=3), 
        #    isapprox(agent.qs_current[state_i], qs_prev[state_i]),
        #    length(findall(x -> x > .5, state.B))
        #)
        
        if false && !isapprox(agent.qs_current[state_i], qs_prev[state_i])
            @infiltrate; @assert false 
        end
    end
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





end  # -- module