


module Inference

#using Distributed
#addprocs(4; exeflags="--project")

#if nworkers() < 4
#    addprocs(4, exeflags=`--project=$(Base.active_project()) -t 200`)
#end
#println(nworkers())
#@everywhere begin

import LogExpFunctions as LEF
import ActiveInference.ActiveInferenceFactorized as AI  

using Format
using Infiltrator

#Distributed.@everywhere import LogExpFunctions as LEF
#Distributed.@everywhere import ActiveInference.ActiveInferenceFactorized as AI 
#Distributed.@everywhere import Pkg
#Distributed.@everywhere Pkg.activate()


#using Revise

#include("./algos.jl")


# @infiltrate; @assert false




""" -------- Inference Functions -------- """

#### State Inference #### 

""" Get Expected States """
function get_expected_states(
    qs::NamedTuple{<:Any, <:NTuple{N, Vector{T2}} where {N}}, 
    policy::NamedTuple, 
    agent::AI.Agent{T2}
    ) where {T2<:AbstractFloat}
    
    model = agent.model

    # earlystop_tests for current location
    if isa(model.policies.earlystop_tests, Function) && !model.policies.earlystop_tests(qs, model)
        # agent already believes it is at an early stop, before action
        return missing  # missing is code for early stop
    end
    
    n_steps = length(policy[1])  # policy length is the same for all actions, in SI not the same as policy length

    # initializing posterior predictive density as a list of beliefs over time
    qs_pi = [deepcopy(qs) for _ in 1:n_steps+1]
    null_actions = [x.null_action for x in model.actions] 
    null_action_ids = Tuple([findfirst(x -> x == null_actions[ii], action.labels) for (ii, action) in enumerate(model.actions)])

    stop_early_at_step = n_steps + 1  # dummy, should this policy stop early, at a specified time step?
    for step_i in 1:n_steps
            
        if step_i == stop_early_at_step
            # stop early
            return qs_pi[2:step_i]
        end

        for (state_ii, state) in enumerate(model.states)
                        
            # select out actions from B matrix
            B, _ = AI.Utils.select_B_actions(state, policy, step_i)
            
            # collect B dependencies
            deps = AI.Utils.collect_dependencies(qs_pi, state, policy, step_i)
            
            qs_new = AI.Maths.dot_product1(B, deps)

            #if isapprox(qs_pi[step_i][state.name], qs_new)
            #    @infiltrate; @assert false
            #end

            if !isapprox(sum(qs_new), 1.0)
                #@infiltrate; @assert false
                @assert false
            end
            
            #printfmtln("    {}", qs_new) 
            qs_pi[step_i + 1][state.name][:] = qs_new
        end

        # action_tests
        
        if isa(model.policies.action_tests, Function) && !model.policies.action_tests(qs_pi[step_i + 1], model)
            return nothing # entire policy for all B matrices and actions is invalid
        end
    
        # todo: perform policy_tests 
        
        # earlystop_tests
        if step_i < n_steps 
            if isa(model.policies.earlystop_tests, Function) && !model.policies.earlystop_tests(qs_pi[step_i + 1], model)
                # are all remaining actions a null action, like "stay"
                @assert any(null_action_ids .== nothing) "If early stopping is used, each action must have a specified null action"
                
                for acts in collect(zip(policy...))[step_i+1:end]
                    if acts != null_action_ids
                        return nothing # entire policy for all B matrices and actions is invalid
                    end
                end
                
                # all remaining actions are null actions, so keep this policy but stop eary
                stop_early_at_step = step_i + 1
            end
        end
    end
    
    return qs_pi[2:end]
end



""" Update Posterior States """
function update_posterior_states(agent::AI.Agent, obs::NamedTuple{<:Any, <:NTuple{N, Int64} where {N}}) 
    # todo: there seems to be no need for this function, except to call run_factorized_fpi.
    # If this is just a pass through function, is that is what we want? Maybe later we will
    # have other algorithms, in addition to fpi.
    
    #@infiltrate; @assert false
    
    qs = AI.Algos.run_factorized_fpi(agent, obs)  

    #@infiltrate; @assert false

    # we return qs rather than updating qs in the model to allow for calling update_posterior_states in SI
    return qs
end



function eval_policy(policy_i, policy_iterator, agent, qs_current, action_names, n_steps)
    
    policy = policy_iterator[policy_i]
    policy = (; zip(action_names, policy)...)
    qs_pi = get_expected_states(qs_current, policy, agent)  
    
    if isnothing(qs_pi)
        # bad policy, given missing utility and info_gain, and zero EFE
        return
    end

    qo_pi = get_expected_obs(qs_pi, agent)  
            
    # note: length of qs_pi and qo_pi will be less than policy length if early stop

    # Calculate expected utility
    if agent.settings.use_utility
        # If ReverseDiff is tracking the expected utility, get the value
        #if ReverseDiff.istracked(calc_expected_utility(qo_pi, agent.C))
        #    @infiltrate; @assert false
        #    G[policy_i] += ReverseDiff.value(calc_expected_utility(qo_pi, C))

        # Otherwise calculate the expected utility and add it to the G vector
        utility_ = calc_expected_utility(qo_pi, agent)
        
        # initialize this G?
        if ismissing(agent.G_policies[policy_i])
            agent.G_policies[policy_i] = 0
        end

        if length(utility_) == n_steps && agent.settings.EFE_reduction == :sum
            # use sum  
            agent.G_policies[policy_i] += sum(utility_) 
        
        elseif agent.settings.EFE_reduction == :min_max
            agent.G_policies[policy_i] += (maximum(skipmissing(utility_)) + minimum(skipmissing(utility_))) / 2  # handles missings
        
        elseif agent.settings.EFE_reduction == :custom
            agent.G_policies[policy_i] += agent.model.policies.utility_reduction_fx(utility_)
        
        elseif length(utility_) < n_steps && agent.settings.EFE_reduction == :sum
            try
                error("There may be missing values in utility vectors; EFE_reduction = :sum cannot be used.")
            catch e
                # Add context and rethrow the error
                error("$(e)")
            end
        end

        agent.utility[policy_i, 1:utility_.size[1]] = utility_
        #@infiltrate; @assert false 
    end

    # Calculate expected information gain of states
    if agent.settings.use_states_info_gain
        # If ReverseDiff is tracking the information gain, get the value
        #if false && ReverseDiff.istracked(calc_states_info_gain(agent.A, qs_pi))  # todo??
        #    @infiltrate; @assert false
        #    G[policy_i] += ReverseDiff.value(calc_states_info_gain(A, qs_pi))

        # Otherwise calculate it and add it to the G vector
        
        info_gain_, ambiguity_ = calc_info_gain(qs_pi, qo_pi, agent)
        
        # initialize this G?
        if ismissing(agent.G_policies[policy_i])
            agent.G_policies[policy_i] = 0
        end

        if length(info_gain_) == n_steps && agent.settings.EFE_reduction == :sum
            # use sum  
            agent.G_policies[policy_i] += sum(info_gain_)  
        
        elseif agent.settings.EFE_reduction == :min_max
            agent.G_policies[policy_i] += maximum(skipmissing(info_gain_))   # handles missings
        
        elseif agent.settings.EFE_reduction == :custom
            agent.G_policies[policy_i] += agent.model.policies.info_gain_reduction_fx(info_gain_)
        
        elseif length(info_gain_) < n_steps && agent.settings.EFE_reduction == :sum
            try
                error("There may be missing values in info_gain vectors; EFE_reduction = :sum cannot be used.")
            catch e
                # Add context and rethrow the error
                error("$(e)")
            end
        end
        
        # todo: should risk and ambiguity include info_gain_A, etc?
        agent.info_gain[policy_i,1:info_gain_.size[1]] = info_gain_
        agent.ambiguity[policy_i,1:ambiguity_.size[1]] = ambiguity_
        agent.risk[policy_i,1:ambiguity_.size[1]] = (
            agent.utility[policy_i, 1:ambiguity_.size[1]]
            .+ info_gain_
            .- ambiguity_
        )
        @assert isapprox(info_gain_ + utility_, agent.risk[policy_i,1:ambiguity_.size[1]] + ambiguity_)
        #@infiltrate; @assert false
    end


    # Calculate expected information gain of parameters (learning)
    if agent.settings.use_param_info_gain
        
        if agent.info_gain_A !== nothing
            #@infiltrate; @assert false  # todo
            @assert false

            # if ReverseDiff is tracking pA information gain, get the value
            if ReverseDiff.istracked(calc_pA_info_gain(pA, qo_pi, qs_pi))
                agent.G_policies[policy_i] += ReverseDiff.value(calc_pA_info_gain(pA, qo_pi, qs_pi))
            # Otherwise calculate it and add it to the G vector
            else
                # initialize this G?
                if ismissing(agent.G_policies[policy_i])
                    agent.G_policies[policy_i] = 0
                end

                agent.G_policies[policy_i] += calc_pA_info_gain(agent.pA, qo_pi, qs_pi)
            end
        end


        if agent.info_gain_B !== nothing
            #@infiltrate; @assert false
            info_gain_B_ = calc_pB_info_gain(agent, qs_pi, qs_current, policy)
            # use sum  
            # initialize this G?
            if ismissing(agent.G_policies[policy_i])
                agent.G_policies[policy_i] = 0
            end
            
            if length(info_gain_B_) == n_steps && agent.settings.EFE_reduction == :sum
                agent.G_policies[policy_i] += sum(info_gain_B_)  
        
            elseif agent.settings.EFE_reduction == :min_max
                agent.G_policies[policy_i] += maximum(skipmissing(info_gain_B_))   # handles missings
            
            elseif agent.settings.EFE_reduction == :custom
                agent.G_policies[policy_i] += agent.model.policies.info_gain_reduction_fx(info_gain_B_)
            
            elseif length(info_gain_B_) < n_steps && agent.settings.EFE_reduction == :sum
                try
                    error("There may be missing values in info_gain_B vectors; EFE_reduction = :sum cannot be used.")
                catch e
                    # Add context and rethrow the error
                    error("$(e)")
                end
            end
            
            agent.info_gain_B[policy_i,1:info_gain_B_.size[1]] = info_gain_B_
            #@infiltrate; @assert false  # todo
        end


        if agent.info_gain_D !== nothing
            #@infiltrate; @assert false  # todo
            @assert false
            
            info_gain_B_ = calc_pD_info_gain(agent.pB, qs_pi, qs, policy, agent.metamodel)
            # initialize this G?
            if ismissing(agent.G_policies[policy_i])
                agent.G_policies[policy_i] = 0
            end
            
            if length(info_gain_B_) == n_steps
                if agent.use_sum_for_calculating_G && !any(ismissing.(info_gain_B_))
                    # use sum
                    agent.G_policies[policy_i] += sum(info_gain_B_)  
                else
                    # use extremes; todo: pass in desired function for this
                    agent.G_policies[policy_i] += maximum(info_gain_B_) 
                end
            else
                # early stop, use extremes; todo: pass in desired function for this
                @assert agent.use_sum_for_calculating_G == false  # a info_gain is short, sum cannot be used for any
                agent.G_policies[policy_i] += maximum(skipmissing(info_gain_B_)) 
            end  
            info_gain_B[policy_i,1:info_gain_B_.size[1]] = info_gain_B_

        end
    end
end

#end  # -- everywhere
#### Policy Inference #### 
""" Update Posterior over Policies """
function update_posterior_policies!(agent::AI.Agent{T2}) where {T2<:AbstractFloat}
    
    model = agent.model
    qs_current = agent.qs
    n_steps = model.policies.policy_length
    n_policies = model.policies.n_policies
    action_names = [x.name for x in model.actions]
        
    if !isnothing(agent.G_policies)
        agent.G_policies .= missing
    end
    
    if !isnothing(agent.G_actions)
        agent.G_actions .= missing
    end
    
    if !isnothing(agent.q_pi_policies)
        agent.q_pi_policies .= missing
    end
    
    if !isnothing(agent.q_pi_actions)
        agent.q_pi_actions .= missing
    end
    
    # todo: these can be nothing if they are not used, as per agent
    agent.utility .= missing
    agent.info_gain .= missing
    agent.risk .= missing
    agent.ambiguity  .= missing
    
    if !isnothing(agent.info_gain_A)
        agent.info_gain_A .= missing
    end

    if !isnothing(agent.info_gain_B)
        agent.info_gain_B .= missing
    end

    if !isnothing(agent.info_gain_D)
        agent.info_gain_D .= missing
    end
    policy_iterator = model.policies.policy_iterator
    #for (policy_i, policy) in enumerate(model.policies.policy_iterator)
    #for policy_i in 1:model.policies.n_policies
    #end
    


    fx(policy_i) = eval_policy(policy_i, policy_iterator, agent, qs_current, action_names, n_steps)
    #pmap(i -> fx(i), 1:model.policies.n_policies)
    map(i -> fx(i), 1:model.policies.n_policies)

    if sum(skipmissing(agent.G_policies)) == 0
        #@infiltrate; @assert false  # All policies failed?
        @warn format("\nThe sum over G is zero.\n")
    end
    
    # some utility can be missing. Only use good policies to calc q_pi
    idx = findall(x -> !ismissing(x), agent.G_policies)
    
    if agent.settings.EFE_over == :actions 
        
        # marginalize G over actions
        for (policy_i, policy) in enumerate(model.policies.policy_iterator)  
            if !(policy_i in idx)
                continue
            end 
            policy = (; zip(action_names, policy)...) 
            first_action = first.(values(policy))
            idx2 = findfirst(x -> x == first_action, agent.model.policies.action_iterator)
            
            if ismissing(agent.G_actions[idx2])
                agent.G_actions[idx2] = 0
            end
            
            agent.G_actions[idx2] += agent.G_policies[policy_i] 
        end
        
        idx3 = findall(x -> !ismissing(x), agent.G_actions) 
        Eidx = model.policies.E_actions[idx3]
        lnE = AI.Maths.capped_log(Eidx)
        
        agent.q_pi_actions[idx3] = LEF.softmax(T2.(agent.G_actions[idx3]) * agent.parameters.gamma + lnE, dims=1)  
        
        #@infiltrate; @assert false
        return
    end

    # calculate q_pi over policies   
    Eidx = model.policies.E_policies[idx]
    lnE = AI.Maths.capped_log(Eidx)
    agent.q_pi_policies[idx] .= LEF.softmax(T2.(agent.G_policies[idx]) * agent.parameters.gamma + lnE, dims=1)  

    #@infiltrate; @assert false
    
    return 
end


""" Get Expected Observations """
function get_expected_obs(
        qs_pi::Vector{T} where {T <: NamedTuple{<:Any, <:NTuple{N, Vector{T2}} where {N}}},
        agent::AI.Agent{T2}
    ) where {T2<:AbstractFloat}
    
    model = agent.model
    n_steps = length(qs_pi)  # this might be equal to or less than policy length, if stop was reached
    
    qo_pi = [deepcopy(agent.qo) for _ in 1:n_steps]

    printflag = 0
    
    for step_i in 1:n_steps
        for (obs_ii, obs) in enumerate(model.obs)
            
            # collect A dependencies
            deps = AI.Utils.collect_dependencies(qs_pi, obs, step_i)
            #A = deepcopy(obs.A)
            A = obs.A
                       
            qo = AI.Maths.dot_product1(A, deps)
            @assert qo.size == (A.size[1], ) 
                       
            # todo: make rule to avoid doing dot product twice, and make sure dot product works for all instances in the code

            if !isapprox(sum(qo), 1.0)
                printfmtln("\nmodailty= {}, qo={}", obs.name, qo)
                #@infiltrate; @assert false
                @assert false

                Am = copy(A_m)
                Am = vcat(Am, zeros(T2, (1, Am.size[2:end]...)))
                res = dot_product1(Am, deps)
                if printflag == 10
                    printfmtln("\nmodailty= {}, remade Am={}", modality, res)
                    printflag = 1
                end
                
                if !isapprox(sum(res), 1.0) || !isapprox(res[end], 0.0) 
                    #@infiltrate; @assert false
                    @assert false
                end    
                Am = res[1:end-1]
            end

            qo_pi[step_i][obs.name][:] = qo
        end
    end
    #@infiltrate; @assert false

    return qo_pi
end


""" Calculate Expected Utility """
function calc_expected_utility(
        qo_pi::Vector{T} where {T <: NamedTuple{<:Any, <:NTuple{N, Vector{T2}} where {N}}},
        agent::AI.Agent{T2}
    ) where {T2<:AbstractFloat}
    
    
    model = agent.model
    n_steps = length(qo_pi)
    expected_utility = zeros(T2, n_steps)
    
    #num_modalities = length(C)

    # when is C[i] not of dim=1?
    #modalities_to_tile = [modality_i for modality_i in 1:num_modalities if ndims(C[modality_i]) == 1]
    #C_tiled = deepcopy(C)
    #for modality in modalities_to_tile
    #    modality_data = reshape(C_tiled[modality], :, 1)
    #    C_tiled[modality] = repeat(modality_data, 1, n_steps)
    #end
    
    #printfmtln("\nC_tiled=")
    #display(C_tiled[1]) 
    
    #C_prob = softmax_array(C_tiled)
    
    #printfmtln("\nC_prob=")
    #display(C_prob[1]) 
    

    # C could be multidimensional and depend on different states
    for step_i in 1:n_steps
        
        for (pref_ii, pref) in enumerate(model.preferences)
            C_prob = LEF.softmax(pref.C, dims=1)
            
            if ndims(C_prob) > 1
                # todo: select for state dependencies
                #@infiltrate; @assert false
                @assert false
            end
            
            lnC = AI.Maths.capped_log(C_prob)
            
            expected_utility[step_i] += sum(qo_pi[step_i][pref.C_dim_names[1]] .* lnC)  # assumes 1-D pref

            if expected_utility[step_i] > 0
                #@infiltrate; @assert false
                @assert false
            end   
            #@infiltrate; @assert false
        end

    end
    
    #@infiltrate; @assert false
    return expected_utility
end


# --------------------------------------------------------------------------------------------------
function calc_info_gain(
        qs::Vector{T} where {T <: NamedTuple{<:Any, <:NTuple{N, Vector{T2}} where {N}}}, 
        qo::Vector{T} where {T <: NamedTuple{<:Any, <:NTuple{N, Vector{T2}} where {N}}}, 
        agent::AI.Agent) where {T2<:AbstractFloat}
    """
    New version of expected information gain that takes into account sparse dependencies between observation modalities and hidden state factors.
    qs, qo are over policy steps
    """

    model = agent.model
    #@infiltrate; @assert false
    info_gain_per_step = zeros(T2, qs.size[1])
    ambiguity_per_step = zeros(T2, qs.size[1])
    
    for step_i in 1:qs.size[1]
        
        info_gains_per_modality = zeros(T2, length(model.obs))
        ambiguity_per_modality = zeros(T2, length(model.obs))
        qs_step = qs[step_i]
        qo_step = qo[step_i]
        
        for (obs_i, obs) in enumerate(model.obs)
            
            H_qo = AI.Maths.stable_entropy(qo_step[obs.name])
            #H_A = - sum(AI.Maths.stable_xlogx(obs.A), dims=1)
            
            #if ndims(obs.A) == 2 && Set(unique(obs.A)) == Set([0,1])
            #    H_A = zeros(T2, (1,obs.A.size[2]))
            #else
                H_A = - sum(LEF.xlogx.(obs.A), dims=1)
            #end
            #@infiltrate; @assert false
            deps = AI.Utils.collect_dependencies(qs_step, obs)
            H_A = AI.Maths.dot_product1(H_A, deps)
            
            if ndims(H_A) > 1
                #@infiltrate; @assert false
                @assert false
            end
            @assert H_A.size[1] == 1
            
            info_gains_per_modality[obs_i] = H_qo - H_A[1]
            ambiguity_per_modality[obs_i] =  H_A[1]
            #@infiltrate; @assert false
        end
        
        info_gain_per_step[step_i] = sum(info_gains_per_modality)
        ambiguity_per_step[step_i] = sum(ambiguity_per_modality)
        
        if info_gain_per_step[step_i] < 0
            #@infiltrate; @assert false
            @assert false
        end 

    end

    return info_gain_per_step, ambiguity_per_step
end



""" Calculate States Information Gain """
function calc_states_info_gain(
        A, 
        qs_pi
    )
    #@infiltrate; @assert false  # not yet implemented
    @assert false

    n_steps = length(qs_pi)
    #states_surprise = 0.0
    states_surprise = zeros(n_steps)

    for t in 1:n_steps
        states_surprise[t] = calculate_bayesian_surprise(A, qs_pi[t])
    end

    return states_surprise
end


""" Calculate observation to state info Gain """
function calc_pA_info_gain(
        pA, 
        qo_pi, 
        qs_pi
    )
    #@infiltrate; @assert false  # not yet implemented
    @assert false
    
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
function calc_pB_info_gain(
    agent::AI.Agent{T2},
    qs_pi::Vector{T} where {T <: NamedTuple{<:Any, <:NTuple{N, Vector{T2}} where {N}}},
    qs_prev::NamedTuple{<:Any, <:NTuple{N, Vector{T2}} where {N}}, 
    policy::NamedTuple, 
    ) where {T2<:AbstractFloat}
    
    # todo: we could calculate info_gain per step and per state. Here we sum over all states.

    model = agent.model
    
    n_steps = length(qs_pi)  # this might be less than the policy length, if there was early stopping
    if n_steps != length(policy[1])
        #@infiltrate; @assert false  # todo: validate that all of the following work for early stopping
        @assert false
    end
    
    info_gain_per_step = zeros(T2, n_steps)
    for step_i in 1:n_steps
        
        for (state_ii, state) in enumerate(model.states)
            
            if isnothing(state.pB) || ismissing(state.pB)
                continue
            end

            # select out actions from B matrix
            pB, idx = AI.Utils.select_B_actions(state, policy, step_i, true)

            wB = AI.Maths.spm_wnorm(state.pB)  # W := .5 (1 ./ a - 1 ./ a_sum) 
            wB = wB[idx...]  # select out actions, now only state dependencies left

            # the 'past posterior' used for the information gain about pB here is the posterior
            # over expected states at the timestep previous to the one under consideration
            # if we're on the first timestep, we just use the latest posterior in the
            # entire action-perception cycle as the previous posterior
            if step_i == 1
                previous_qs = qs_prev
            # otherwise, we use the expected states for the timestep previous to the timestep under consideration
            else
                previous_qs = qs_pi[step_i - 1]
            end

            # collect B dependencies
            deps = AI.Utils.collect_dependencies(previous_qs, state, policy)
            
            wB .*= T2.(pB .> 0)  # only consider wB if pB > 0
            Wqs = AI.Maths.dot_product1(wB, deps)
            info_gain_per_step[step_i] -= sum(qs_pi[step_i][state.name] .* Wqs)
        end
    end

    #@infiltrate; @assert false
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
    
    #@infiltrate; @assert false
    @assert false
    
    if action_selection == "deterministic"
        ii = argmax(q_pi)
       
        selected_policy[factor_i] = select_highest(action_marginals[factor_i])
    elseif action_selection == "stochastic"
        log_marginal_f = capped_log(action_marginals[factor_i])  # min capped_log(x) = -36.8
        p_actions = softmax(log_marginal_f * alpha, dims=1)
        selected_policy[factor_i] = action_select(p_actions)
        #@infiltrate; @assert false
    end
    
    return selected_policy

    
    # todo: allow action choice based on q_pi or log marginal
    
    num_factors = length(num_controls)
    selected_policy = zeros(Real,num_factors)
    
    eltype_q_pi = eltype(q_pi)

    # Initialize action_marginals with the correct element type
    # action_marginals = len(factors), where each factor is size(available actions) 
    action_marginals = create_matrix_templates(num_controls, "zeros", eltype_q_pi)

    for (pol_idx, policy) in enumerate(policies.policy_iterator)
        #@infiltrate; @assert false
        @assert false
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
function calculate_SAPE(agent::AI.Agent)

    # todo: is this function used anywhere?
    @assert false
    qs_pi_all = get_expected_states(agent.qs, agent.B, agent.policies)
    qs_bma = bayesian_model_average(qs_pi_all, agent.Q_pi)

    if length(agent.states["bayesian_model_averages"]) != 0
        sape = kl_divergence(qs_bma, agent.states["bayesian_model_averages"][end])
        push!(agent.states["SAPE"], sape)
    end

    push!(agent.states["bayesian_model_averages"], qs_bma)
end


end  # --- module