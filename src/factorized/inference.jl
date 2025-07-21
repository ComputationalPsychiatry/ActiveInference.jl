
module Inference


using Format
using Infiltrator
using Revise

#import LinearAlgebra as LA


#import IterTools
#import Statistics

include("./struct.jl")
include("./algos.jl")
include("./utils/maths.jl")


# @infiltrate; @assert false





""" -------- Inference Functions -------- """

#### State Inference #### 

""" Get Expected States """
function get_expected_states(
    qs::Vector{Vector{T}} where T <: Real, 
    B, 
    policy,
    metamodel
    )

    #az = Iterators.Stateful(join(collect('a':'z'), ""))
    #faz(n) = join(collect(Iterators.take(az, n)), "")
    
    
    n_steps = length(policy[1])  # policy length is the same for all actions
    action_names = keys(metamodel.action_deps)
    Biis_with_action = [ii for ii in 1:length(B) if length(intersect(metamodel.state_deps[ii], action_names)) > 0]
    # initializing posterior predictive density as a list of beliefs over time
    qs_pi = [deepcopy(qs) for _ in 1:n_steps+1]
    null_actions = [metamodel.policies.action_contexts[action][:null_action] for action in action_names]
    null_action_ids = [findfirst(x -> x == null_actions[ii], metamodel.action_deps[ii]) for ii in 1:length(action_names)]

    stop_early_at_t = n_steps + 10000  # should this policy stop early, at a specified time step?
    for t in 1:n_steps
            
        if t == stop_early_at_t
            # stop early
            return qs_pi[2:t]
        end

        for Bii in 1:length(B) 
            
            # list of the hidden state factor indices that the dynamics of `qs[Bii]` depend on
            # potentially more than one action per B matrix
            factors = metamodel.state_deps[Bii]  # eg: [:loc, :loc, :move], with last being an action
            # potentially, factors for this B matrix could have more than one action
            
            factor_idx = [findfirst(x -> x == j, keys(metamodel.state_deps)) for j in factors]
            # eg., [1,1,nothing), where nothing is for an action

            Bc = copy(B[Bii])
            selections = nothing

            # select out action dimensions
            if Bii in Biis_with_action

                # here we handle potentially more than one action for this B matrix; 
                # e.g. policy = ((1,2), (4,5)) for action_names = (:move, :jump)
                # Then, selections[1] might equal 
                # (action_name = :move, action_option_id = 1, action_option_name = :UP, null_action_id = 5)
                
                selections = [
                    NamedTuple{(:action_name, :action_option_id, :action_option_name, :null_action_id)}(
                        (name, pol[t], metamodel.action_deps[name][pol[t]], null_action_id))
                    for (name, pol, null_action_id) in zip(action_names, policy, null_action_ids)
                ]
                #printfmtln("\nstep={}, Bii={}, selections= {}", t, Bii, selections)
                
                # These are tests for pre-action state         
                for (i_selection, selection) in enumerate(selections)
                    if !(
                        # is this action unwanted (e.g., takes agent off the grid)?
                        metamodel.policies.action_contexts[selection.action_name][:option_context][selection.action_option_name](qs_pi[t])
                        )
                        #@infiltrate; @assert false
                        return nothing  # entire policy for all B matrices and actions, is invalid
                    end
                end

                idx = []  # index of dims of this B matrix, states always come before actions in depencency lists
                iaction = 1
                for (idep, dep) in enumerate(factors) 
                    if dep in keys(metamodel.action_deps)
                        # this dim is an action
                        push!(idx, selections[iaction].action_option_id)
                        iaction += 1
                    else
                        # this is a state
                        push!(idx, 1:Bc.size[idep])  # e.g., push!(idx, (100,100,5)[1]) if idep==1
                    end
                end
                
                Bc = Bc[idx...]  # select out actions, now only state dependencies left
            end

            # get expected states
            deps = Vector{Vector{Float64}}()
            for idep in reverse(factor_idx[2:end])  # first factor is new state, other are dependencies or actions 
                if isnothing(idep)
                    # this dependency is an action
                    continue
                end
                push!(deps, qs_pi[t][idep])
            end
            
            Bc = dot_product1(Bc, deps)

            if !isapprox(sum(Bc), 1.0)
                @infiltrate; @assert false
            end
            
            #printfmtln("    {}", Bc) 
            qs_pi[t+1][Bii] = Bc

            # now check if action result is unwanted/illegal or a stop

            if Bii in Biis_with_action
                # These are tests for post-action state 
                #println(selections)
                #println(t, "  ", policy[1])
                for (i_selection, selection) in enumerate(selections)
                    
                    #=
                    Has agent reached a stop condition for this B matrix? If so, all remaining policy
                    steps should be a null action (e.g., "stay" for a grid agent).  If this is a stop, 
                    this is the last step for this policy for all B matrices. Let tranistions 
                    continue for the reminder of this policy step.
                    =# 
                    
                    if t < n_steps && !(metamodel.policies.action_contexts[selection.action_name][:stopfx](qs_pi[t+1]))
                        # are all remaining actions a null action, like "stay"
                        if !all(policy[i_selection][t+1:end] .== selection.null_action_id)
                            #@infiltrate; @assert false    
                            return nothing  # entire policy for all B matrices and actions, is invalid
                        else
                            #@infiltrate; @assert false
                            stop_early_at_t = t+1
                        end
                    end
                end
                

            end
        end
    end
    
    return qs_pi[2:end]
end



""" Update Posterior States """
function update_posterior_states(agent::Agent, obs::NamedTuple{<:Any, <:NTuple{N, Int64} where {N}}) 
    # todo: there seems to be no need for this function, except to call run_factorized_fpi.
    # If this is just a pass through function, is that is what we want? Maybe later we will
    # have other algorithms, in addition to fpi.

    qs = run_factorized_fpi(agent, obs)  

    #@infiltrate; @assert false

    # we return qs rather than updating qs in the model to allow for calling update_posterior_states in SI
    return qs
end


#### Policy Inference #### 
""" Update Posterior over Policies """
function update_posterior_policies(agent)
    
    qs = agent.qs_current

    #n_steps = agent.policies[1].size[1]
    #n_policies = length(agent.policies)
    n_steps = agent.policy_len
    n_policies = agent.policies.number_policies
    
    
    G = zeros(n_policies)
    G_raw = zeros(n_policies)
    utility = Matrix{Union{Missing, Float64}}(undef, n_policies, n_steps)
    info_gain = Matrix{Union{Missing, Float64}}(undef, n_policies, n_steps)
    risk = Matrix{Union{Missing, Float64}}(undef, n_policies, n_steps)
    ambiguity = Matrix{Union{Missing, Float64}}(undef, n_policies, n_steps)
    
    if agent.pB == nothing
        info_gain_B = nothing
    else
        info_gain_B = Matrix{Union{Missing, Float64}}(undef, n_policies, n_steps)
    end
        
    #q_pi = Vector{Float64}(undef, n_steps)
    qs_pi = Vector{Float64}[]
    qo_pi = Vector{Float64}[]
    lnE = capped_log(agent.E)
    
    #@infiltrate; @assert false

    #printfmtln("\nqs_current= {}", vcat(argmax(qs[1]), qs[2:end]))

    for (idx, policy) in enumerate(agent.policies.policy_iterator)
        
        qs_pi = get_expected_states(qs, agent.B, policy, agent.metamodel)  
        if isnothing(qs_pi)
            # bad policy, given missing utility and info_gain, and zero EFE
            continue
        end
        qo_pi = get_expected_obs(qs_pi, agent.A, agent.metamodel)  
        
        # note: length of qs_pi and qo_pi will be less than policy length if stop reached early

        if false && idx in [39, 123, 173, 29, 36]
            printfmtln("\npolicy# {}, {}, qs_pi=", idx, policy)
            #@infiltrate; @assert false
            display([[vcat(argmax(qs_pi[iii][1]), map(x2 -> round.(x2, digits=4), qs_pi[iii][2:end]) )] for iii in 1:qs_pi.size[1]])
            
            printfmtln("\npolicy# {}, {}, qo_pi=", idx, policy)
            display([[vcat(argmax(qo_pi[iii][1]), map(x2 -> round.(x2, digits=4), qo_pi[iii][2:end])) ] for iii in 1:qo_pi.size[1]])
        end

        # Calculate expected utility
        if agent.use_utility
            # If ReverseDiff is tracking the expected utility, get the value
            if ReverseDiff.istracked(calc_expected_utility(qo_pi, agent.C))
                @infiltrate; @assert false
                G[idx] += ReverseDiff.value(calc_expected_utility(qo_pi, C))

            # Otherwise calculate the expected utility and add it to the G vector
            else
                utility_ = calc_expected_utility(qo_pi, agent.C)
                #G[idx] += utility_
                if length(utility_) == n_steps
                    if agent.use_sum_for_calculating_G && !any(ismissing.(utility_))
                        # use sum  
                        G[idx] += sum(utility_)  
                    else
                        # use extremes; todo: pass in desired function for this
                        G[idx] += (maximum(utility_) + minimum(utility_)) / 2  # due to missings
                    end
                else
                    # early stop, use extremes; todo: pass in desired function for this
                    @assert agent.use_sum_for_calculating_G == false  # a utility is short, sum cannot be used for any
                    G[idx] += (maximum(skipmissing(utility_)) + minimum(skipmissing(utility_))) / 2
                end

                utility[idx, 1:utility_.size[1]] = utility_
            end
        end

        # Calculate expected information gain of states
        if agent.use_states_info_gain
            # If ReverseDiff is tracking the information gain, get the value
            if false && ReverseDiff.istracked(calc_states_info_gain(agent.A, qs_pi))  # todo??
                @infiltrate; @assert false
                G[idx] += ReverseDiff.value(calc_states_info_gain(A, qs_pi))

            # Otherwise calculate it and add it to the G vector
            else
                #info_gain_ = calc_states_info_gain(agent.A, qs_pi)
                info_gain_, ambiguity_ = compute_info_gain(qs_pi, qo_pi, agent.A, agent.metamodel, idx)
                
                if length(utility_) == n_steps
                    if agent.use_sum_for_calculating_G && !any(ismissing.(utility_))
                        # use sum
                        G[idx] += sum(info_gain_)  
                    else
                        # use extremes; todo: pass in desired function for this
                        G[idx] += maximum(info_gain_) 
                    end
                else
                    # early stop, use extremes; todo: pass in desired function for this
                    @assert agent.use_sum_for_calculating_G == false  # a info_gain is short, sum cannot be used for any
                    G[idx] += maximum(skipmissing(info_gain_)) 
                end  
                info_gain[idx,1:info_gain_.size[1]] = info_gain_
                ambiguity[idx,1:ambiguity_.size[1]] = ambiguity_
                risk[idx,1:ambiguity_.size[1]] = (
                    utility[idx, 1:ambiguity_.size[1]]
                    .+ info_gain_
                    .- ambiguity_
                )
                @assert isapprox(info_gain_ + utility_, risk[idx,1:ambiguity_.size[1]] + ambiguity_)
                #@infiltrate; @assert false

            end
        end



        # Calculate expected information gain of parameters (learning)
        if agent.use_param_info_gain
            if agent.pA !== nothing
                @infiltrate; @assert false
                # if ReverseDiff is tracking pA information gain, get the value
                if ReverseDiff.istracked(calc_pA_info_gain(pA, qo_pi, qs_pi))
                    G[idx] += ReverseDiff.value(calc_pA_info_gain(pA, qo_pi, qs_pi))
                # Otherwise calculate it and add it to the G vector
                else
                    G[idx] += calc_pA_info_gain(agent.pA, qo_pi, qs_pi)
                end
            end

            if agent.pB !== nothing
                info_gain_B_ = calc_pB_info_gain(agent.pB, qs_pi, qs, policy, agent.metamodel)
                
                if length(info_gain_B_) == n_steps
                    if agent.use_sum_for_calculating_G && !any(ismissing.(info_gain_B_))
                        # use sum
                        G[idx] += sum(info_gain_B_)  
                    else
                        # use extremes; todo: pass in desired function for this
                        G[idx] += maximum(info_gain_B_) 
                    end
                else
                    # early stop, use extremes; todo: pass in desired function for this
                    @assert agent.use_sum_for_calculating_G == false  # a info_gain is short, sum cannot be used for any
                    G[idx] += maximum(skipmissing(info_gain_B_)) 
                end  
                info_gain_B[idx,1:info_gain_B_.size[1]] = info_gain_B_

            end
        end

    end

    # some utility are zero because the policy was rejected. Only use good policies to calc q_pi
    idx = findall(x -> !isapprox(x, 0.0), G)   
    q_pi = zeros(utility.size[1])
    
    if sum(G) == 0
        @infiltrate; @assert false  # All policies failed?
    end
    Eidx = agent.E[idx]
    lnE = capped_log(Eidx)
    q_pi[idx] .= softmax(G[idx] * agent.gamma + lnE, dims=1)  

    # note: now G and q_pi are no longer consistent
    #@infiltrate; @assert false
    
    return q_pi, G, utility, info_gain, risk, ambiguity, info_gain_B
end


""" Get Expected Observations """
#function get_expected_obs(qs_pi, A::Vector{Array{T,N}} where {T <: Real, N})

function get_expected_obs(
    qs_pi, 
    A::Union{Vector{Array{T}} where {T <: Real}, Vector{Array{T, N}} where {T <: Real, N}},
    metamodel,
    )

    n_steps = length(qs_pi)  # this might be equal to or less than policy length, if stop was reached
    qo_pi = []
    printflag = 0
    
    for t in 1:n_steps
        qo_pi_t = Vector{Any}(undef, length(A))
        qo_pi = push!(qo_pi, qo_pi_t)
    end

    for t in 1:n_steps
        for (modality, A_m) in enumerate(A)
            
            # list of the hidden state factor indices that the dynamics of `qs[control_factor]` depend on
            factors = metamodel.obs_deps[modality][2:end]
            factor_idx = [findfirst(x -> x == j, keys(metamodel.state_deps)) for j in factors]
            Am = copy(A_m)
           
            deps = Vector{Vector{Float64}}()
            for idep in reverse(factor_idx)  # first factor is new state, other are dependencies or actions 
                push!(deps, qs_pi[t][idep])
            end
            
            Am = dot_product1(Am, deps)
            @assert Am.size == (A_m.size[1], ) 
                       
            # todo: make rule to avoid doing dot product twice, and make sure dot product works for all instances in the code

            if !isapprox(sum(Am), 1.0)
                #printfmtln("\nmodailty= {}, Am={}", modality, Am)
                                
                Am = copy(A_m)
                Am = vcat(Am, zeros(1, Am.size[2:end]...))
                res = dot_product1(Am, deps)
                if printflag == 10
                    printfmtln("\nmodailty= {}, remade Am={}", modality, res)
                    printflag = 1
                end
                
                if !isapprox(sum(res), 1.0) || !isapprox(res[end], 0.0) 
                    @infiltrate; @assert false
                end    
                Am = res[1:end-1]
            end

            qo_pi[t][modality] = Am

        end

        
    end
    #@infiltrate; @assert false

    return qo_pi
end


""" Calculate Expected Utility """
function calc_expected_utility(qo_pi, C)
    n_steps = length(qo_pi)
    #expected_utility = 0.0
    expected_utility = zeros(n_steps)
    num_modalities = length(C)

    # when is C[i] not of dim=1?
    modalities_to_tile = [modality_i for modality_i in 1:num_modalities if ndims(C[modality_i]) == 1]
    C_tiled = deepcopy(C)
    for modality in modalities_to_tile
        modality_data = reshape(C_tiled[modality], :, 1)
        C_tiled[modality] = repeat(modality_data, 1, n_steps)
    end
    
    #printfmtln("\nC_tiled=")
    #display(C_tiled[1]) 
    
    C_prob = softmax_array(C_tiled)
    
    #printfmtln("\nC_prob=")
    #display(C_prob[1]) 
    

    # could expand expected_utility to be zeros(n_steps, num_modalities)
    lnC =[]
    for t in 1:n_steps
        for modality in 1:num_modalities
            lnC = capped_log(C_prob[modality][:, t])
            
            #printfmtln("\nlnC=")
            #display(lnC) 
            
            #expected_utility += dot(qo_pi[t][modality], lnC) 
            expected_utility[t] += dot(qo_pi[t][modality], lnC)

            # no log or softmax
            #expected_utility[t] += dot(qo_pi[t][modality], C_prob[modality][:, t])

            if expected_utility[t] > 0
                @infiltrate; @assert false
            end   
            #@infiltrate; @assert false
        end

    end
    
    #@infiltrate; @assert false
    return expected_utility
end

# --------------------------------------------------------------------




function compute_info_gain(qs, qo, A, metamodel, policy_id)
    """
    New version of expected information gain that takes into account sparse dependencies between observation modalities and hidden state factors.
    qs, qo are over policy steps
    """
    #@infiltrate; @assert false
    info_gain_per_step = zeros(qs.size[1])
    ambiguity_per_step = zeros(qs.size[1])
    for step in 1:qs.size[1]
        info_gains_per_modality = zeros(A.size[1])
        ambiguity_per_modality = zeros(A.size[1])
        qs_step = qs[step]
        qo_step = qo[step]
        for (qo_m, A_m, m) in zip(qo_step, A, 1:A.size[1])
            
            H_qo = stable_entropy(qo_m)
            H_A_m = - sum(stable_xlogx(A_m), dims=1)
            #deps = A_dependencies[m]
            #relevant_factors = [qs[idx] for idx in deps]
            factors = metamodel.obs_deps[m][2:end]
            factor_idx = [findfirst(x -> x == j, keys(metamodel.state_deps)) for j in factors]
            
            HAm = copy(H_A_m)
            
            deps = Vector{Vector{Float64}}()
            for idep in reverse(factor_idx)  # first factor is new state, other are dependencies or actions 
                push!(deps, qs_step[idep])
            end
            
            HAm = dot_product1(HAm, deps)
            
            if ndims(HAm) > 1
                @infiltrate; @assert false
            end
            @assert HAm.size[1] == 1
            #qs_H_A_m = factor_dot(H_A_m, relevant_factors)
            info_gains_per_modality[m] = H_qo - HAm[1]
            ambiguity_per_modality[m] =  HAm[1]
            #@infiltrate; @assert false
            
        end
        #printfmtln("\nstep={}, info_gains_per_modality:", step)
        #display(info_gains_per_modality)
        info_gain_per_step[step] = sum(info_gains_per_modality)
        ambiguity_per_step[step] = sum(ambiguity_per_modality)
        #@infiltrate; @assert false
        if info_gain_per_step[step] < 0
            @infiltrate; @assert false
        end 

    end

    return info_gain_per_step, ambiguity_per_step
end


""" Calculate States Information Gain """
function calc_states_info_gain(A, qs_pi)
    n_steps = length(qs_pi)
    #states_surprise = 0.0
    states_surprise = zeros(n_steps)

    for t in 1:n_steps
        states_surprise[t] = calculate_bayesian_surprise(A, qs_pi[t])
    end

    return states_surprise
end


""" Calculate observation to state info Gain """
function calc_pA_info_gain(pA, qo_pi, qs_pi)
    @infiltrate; @assert false  # not yet implemented
    
    n_steps = length(qo_pi)
    num_modalities = length(pA)

    wA = Vector{Any}(undef, num_modalities)
    for (modality, pA_m) in enumerate(pA)
        wA[modality] = spm_wnorm(pA[modality])
    end

    pA_info_gain = 0

    for modality in 1:num_modalities
        wA_modality = wA[modality] .* (pA[modality] .> 0)

        for t in 1:n_steps
            pA_info_gain -= dot(qo_pi[t][modality], dot_product(wA_modality, qs_pi[t]))
        end
    end
    return pA_info_gain
end


""" Calculate state to state info Gain """
function calc_pB_info_gain(pB, qs_pi, qs_prev, policy, metamodel)
    # this is the factorized version
    # policity eg: ((1, 1, 1),) --- one action, three steps
    
    # We follow the same idea as in get_expected_states() by selecting out actions from each B matrix.
    # But instead of finishing with a dot(B_new, qs), here we filter out actions from a wB matrix of
    # the same shape as B[ii].
    
    n_steps = length(qs_pi)  # this might be less than the policy length, if there was early stopping
    if n_steps != length(policy[1])
        @infiltrate; @assert false  # todo: validate that all of the following work for early stopping
    end
    
    action_names = keys(metamodel.action_deps)
    Biis_with_action = [ii for ii in 1:length(pB) if length(intersect(metamodel.state_deps[ii], action_names)) > 0]
    null_actions = [metamodel.policies.action_contexts[action][:null_action] for action in action_names]
    null_action_ids = [findfirst(x -> x == null_actions[ii], metamodel.action_deps[ii]) for ii in 1:length(action_names)]

    info_gain_per_step = zeros(n_steps)
    for t in 1:n_steps
        for Bii in 1:length(pB) 

            # list of the hidden state factor indices that the dynamics of `qs[Bii]` depend on
            # potentially more than one action per B matrix
            factors = metamodel.state_deps[Bii]  # eg: [:loc, :loc, :move], with last being an action
            # potentially, factors for this B matrix could have more than one action
            
            factor_idx = [findfirst(x -> x == j, keys(metamodel.state_deps)) for j in factors]
            # eg., [1,1,nothing), where nothing is for an action

            # wB takes the role of Bc in get_expected_states()
            wB = spm_wnorm(pB[Bii])  # W := .5 (1 ./ a - 1 ./ a_sum) 
            selections = nothing
            idx = nothing

            # select out action dimensions
            if Bii in Biis_with_action

                # here we handle potentially more than one action for this B matrix; 
                # e.g. policy = ((1,2), (4,5)) for action_names = (:move, :jump)
                # Then, selections[1] might equal 
                # (action_name = :move, action_option_id = 1, action_option_name = :UP, null_action_id = 5)
                
                selections = [
                    NamedTuple{(:action_name, :action_option_id, :action_option_name, :null_action_id)}(
                        (name, pol[t], metamodel.action_deps[name][pol[t]], null_action_id))
                    for (name, pol, null_action_id) in zip(action_names, policy, null_action_ids)
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
                        push!(idx, 1:wB.size[idep])  # e.g., push!(idx, (100,100,5)[1]) if idep==1
                    end
                end
                pBc = pB[Bii][idx...]
                wB = wB[idx...]  # select out actions, now only state dependencies left
            else
                pBc = pB[Bii]  # no actions for this B matrix
                idx = [1:B[Bii].size[i] for i in 1:ndims(B[Bii])]  
            end

            # the 'past posterior' used for the information gain about pB here is the posterior
            # over expected states at the timestep previous to the one under consideration
            # if we're on the first timestep, we just use the latest posterior in the
            # entire action-perception cycle as the previous posterior
            if t == 1
                previous_qs = qs_prev
            # otherwise, we use the expected states for the timestep previous to the timestep under consideration
            else
                previous_qs = qs_pi[t - 1]
            end

            # get expected states
            deps = Vector{Vector{Float64}}()
            for idep in reverse(factor_idx[2:end])  # first factor is new state, other are dependencies or actions 
                if isnothing(idep)
                    # this dependency is an action
                    continue
                end
                push!(deps, previous_qs[idep])
            end
            
            wB .*= Float64.(pBc .> 0)  # only consider wB if pB > 0
            Wqs = dot_product1(wB, deps)
            info_gain_per_step[t] -= dot(qs_pi[t][Bii], Wqs)

            #@infiltrate; @assert false
        end
    end

    return info_gain_per_step
end


### Action Sampling ###
""" Sample Action [Stochastic or Deterministic] """
function sample_action(
    q_pi, 
    policies, 
    num_controls; 
    action_selection="stochastic", 
    alpha=16.0,
    metamodel=metamodel
    )
    
    
    if action_selection == "deterministic"
        ii = argmax(q_pi)
        @infiltrate; @assert false
        selected_policy[factor_i] = select_highest(action_marginals[factor_i])
    elseif action_selection == "stochastic"
        @infiltrate; @assert false
        log_marginal_f = capped_log(action_marginals[factor_i])  # min capped_log(x) = -36.8
        p_actions = softmax(log_marginal_f * alpha, dims=1)
        selected_policy[factor_i] = action_select(p_actions)
        #@infiltrate; @assert false
    end
    
    return selected_policy

    
    @infiltrate; @assert false
    # todo: allow action choice based on q_pi or log marginal
    
    num_factors = length(num_controls)
    selected_policy = zeros(Real,num_factors)
    
    eltype_q_pi = eltype(q_pi)

    # Initialize action_marginals with the correct element type
    # action_marginals = len(factors), where each factor is size(available actions) 
    action_marginals = create_matrix_templates(num_controls, "zeros", eltype_q_pi)

    for (pol_idx, policy) in enumerate(policies.policy_iterator)
        @infiltrate; @assert false
        for (factor_i, action_i) in enumerate(policy[1,:])
            # only want to choose a 1-step action, regardless of later policy choices
            # but what if first action for best policy is seen only once? Others would 
            # get additive q_pi and be larger.
            action_marginals[factor_i][action_i] += q_pi[pol_idx]
        end
    end

    action_marginals = normalize_arrays(action_marginals)

    for factor_i in 1:num_factors
        if action_selection == "deterministic"
            selected_policy[factor_i] = select_highest(action_marginals[factor_i])
        elseif action_selection == "stochastic"
            log_marginal_f = capped_log(action_marginals[factor_i])  # min capped_log(x) = -36.8
            p_actions = softmax(log_marginal_f * alpha, dims=1)
            selected_policy[factor_i] = action_select(p_actions)
            #@infiltrate; @assert false
        end
    end
    return selected_policy
end


""" Calculate State-Action Prediction Error """
function calculate_SAPE(agent::Agent)

    # todo: is this function used anywhere?
    @assert false
    qs_pi_all = get_expected_states(agent.qs_current, agent.B, agent.policies)
    qs_bma = bayesian_model_average(qs_pi_all, agent.Q_pi)

    if length(agent.states["bayesian_model_averages"]) != 0
        sape = kl_divergence(qs_bma, agent.states["bayesian_model_averages"][end])
        push!(agent.states["SAPE"], sape)
    end

    push!(agent.states["bayesian_model_averages"], qs_bma)
end


end  # --- module