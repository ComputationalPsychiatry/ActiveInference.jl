
using Format
using Infiltrator
using Revise

#import LinearAlgebra as LA
import OMEinsum as ein
import LogExpFunctions as LEF
#import IterTools
#import Statistics

include("./algos.jl")

#show(stdout, "text/plain", x)
# @infiltrate; @assert false


function dot_product1(X::Union{Array{Float64, N} where N, Matrix{Float64}}, xs::Vector{Vector{Float64}})
    # xs is a vector of qs vectors for each dependency, in reverse order
    
    if isa(X, Matrix{Float64})
        @assert length(xs) == 1
        return X * xs[1]
    end

    sizes = [collect(x.size) for x in xs]
    code2 = ein.EinCode([collect(X.size), sizes...], collect(X.size[1:end-length(sizes)]))
    
    return code2(X,xs...)
end



function dot_product2(X::Union{Array{Float64, N} where N, Matrix{Float64}}, xs::Vector{Vector{Float64}})
    # xs is a vector of qs vectors for each dependency, in reverse order
    
    if isa(X, Matrix{Float64})
        @assert length(xs) == 1
        return X * xs[1]
    end

    for (ii, dep) in enumerate(xs)
        printfmtln("\nii={}, X={}, dep={}", ii, X.size, dep.size)
        
        code2 = ein.EinCode([collect(X.size), [dep.size[1]]], collect(X.size[1:end-1]))
        printfmtln("code2= {}", code2)
        #@infiltrate; @assert false
        X = code2(X,dep)
    end

    return X
end


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
    
    
    n_steps = length(policy[1])  # same for all actions
    action_names = keys(metamodel.action_deps)
    Biis_with_action = [ii for ii in 1:length(B) if length(intersect(metamodel.state_deps[ii], action_names)) > 0]
    # initializing posterior predictive density as a list of beliefs over time
    qs_pi = [deepcopy(qs) for _ in 1:n_steps+1]
    null_actions = [metamodel.policies.action_contexts[action][:null_action] for action in action_names]
    null_action_ids = [findfirst(x -> x == null_actions[ii], metamodel.action_deps[ii]) for ii in 1:length(action_names)]

    # todo:
    # - watch locations over steps and filter out steps with repeating locations, except stay
    # should the iterator be filtered at all originally?
    # tests on initial state vs. post action
    
    stop_early_at_t = n_steps + 100  # should this policy stop early, at a specified time step?
    for t in 1:n_steps
            
        if t == stop_early_at_t
            # stop early
            return qs_pi[2:t]
        end

        for Bii in 1:length(B) 
            
            # list of the hidden state factor indices that the dynamics of `qs[Bii]` depend on
            factors = metamodel.state_deps[Bii]
            factor_idx = [findfirst(x -> x == j, keys(metamodel.state_deps)) for j in factors]
            
            Bc = copy(B[Bii])
            selections = nothing

            # handle action
            if Bii in Biis_with_action
                selections = [(name, pol[t], metamodel.action_deps[name][pol[t]], null_action_id) 
                    for (name, pol, null_action_id) in zip(action_names, policy, null_action_ids)
                ]
                #printfmtln("\nstep={}, Bii={}, selections= {}", t, Bii, selections)
                
                # These are tests for pre-action state         
                for (i_selection, selection) in enumerate(selections)
                    if !(
                        # is this action unwanted (e.g., takes agent off the grid)?
                        metamodel.policies.action_contexts[selection[1]][:option_context][selection[3]](qs_pi[t])
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
                        push!(idx, selections[iaction][2])
                        iaction += 1
                    else
                        # this is a state
                        push!(idx, 1:Bc.size[idep])
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
                    
                    if t < n_steps && !(metamodel.policies.action_contexts[selection[1]][:stopfx](qs_pi[t+1]))
                        # are all remaining actions "stay"
                        if !all(policy[i_selection][t+1:end] .== selection[4])
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

""" 
    Multiple dispatch for all expected states given all policies

Multiple dispatch for getting expected states for all policies based on the agents currently
inferred states and the transition matrices for each factor and action in the policy.

qs::Vector{Vector{Real}} \n
B: Vector{Array{<:Real}} \n
policy: Vector{Matrix{Int64}}

"""
function get_expected_states(
    qs::Vector{Vector{Float64}}, 
    B, 
    policy::Vector{Matrix{Int64}},
    metamodel
    )

    @infiltrate; @assert false
    
    # Extracting the number of steps (policy_length) and factors from the first policy
    n_steps, n_factors = size(policy[1])

    # Number of policies
    n_policies = length(policy)
    
    # Preparing vessel for the expected states for all policies. Has number of undefined entries equal to the
    # number of policies
    qs_pi_all = Vector{Any}(undef, n_policies)

    # Looping through all policies
    for (policy_idx, policy_x) in enumerate(policy)

        # initializing posterior predictive density as a list of beliefs over time
        qs_pi = [deepcopy(qs) for _ in 1:n_steps+1]

        # expected states over time
        for t in 1:n_steps
            for control_factor in 1:n_factors
                action = policy_x[t, control_factor]
                
                qs_pi[t+1][control_factor] = B[control_factor][:, :, action] * qs_pi[t][control_factor]
            end
        end
        qs_pi_all[policy_idx] = qs_pi[2:end]
    end
    return qs_pi_all
end

"""
    process_observation(observation::Int, n_modalities::Int, n_observations::Vector{Int})

Process a single modality observation. Returns a one-hot encoded vector. 

# Arguments
- `observation::Int`: The index of the observed state with a single observation modality.
- `n_modalities::Int`: The number of observation modalities in the observation. 
- `n_observations::Vector{Int}`: A vector containing the number of observations for each modality.

# Returns
- `Vector{Vector{Real}}`: A vector containing a single one-hot encoded observation.
"""
function process_observation(observation::Int, n_modalities::Int, n_observations::Vector{Int})

    # Check if there is only one modality
    if n_modalities == 1
        # Create a one-hot encoded vector for the observation
        processed_observation = onehot(observation, n_observations[1]) 
    end

    # Return the processed observation wrapped in a vector
    return [processed_observation]
end

"""
    process_observation(observation::Union{Array{Int}, Tuple{Vararg{Int}}}, n_modalities::Int, n_observations::Vector{Int})

Process observation with multiple modalities and return them in a one-hot encoded format 

# Arguments
- `observation::Union{Array{Int}, Tuple{Vararg{Int}}}`: A collection of indices of the observed states for each modality.
- `n_modalities::Int`: The number of observation modalities in the observation. 
- `n_observations::Vector{Int}`: A vector containing the number of observations for each modality.

# Returns
- `Vector{Vector{Real}}`: A vector containing one-hot encoded vectors for each modality.
"""
function process_observation(
    observation::Union{Array{Int}, Tuple{Vararg{Int}}}, 
    n_modalities::Int, 
    n_observations::Vector{Int},
    metamodel
    )



    # Initialize the processed_observation vector
    processed_observation = Vector{Vector{Float64}}(undef, n_modalities)

    # Check if the length of observation matches the number of modalities
    if length(observation) == n_modalities
        for (modality, modality_observation) in enumerate(observation)
            # Create a one-hot encoded vector for the current modality observation
            one_hot = onehot(modality_observation, n_observations[modality])
            # Add the one-hot vector to the processed_observation vector
            processed_observation[modality] = one_hot
        end
    end
    return processed_observation
end

""" Update Posterior States """
function update_posterior_states(
    #A::Vector{Array{T,N}} where {T <: Real, N}, 
    A::Vector{Array{T}} where {T <: Real}, 
    metamodel,
    obs::Vector{Int64}; 
    prior::Union{Nothing, Vector{Vector{T}}} where T <: Real = nothing, 
    num_iter::Int=num_iter, 
    dF_tol::Float64=dF_tol, 
    kwargs...)
    

    num_obs, num_states, num_modalities, num_factors = get_model_dimensions(A)
    
    obs_processed = process_observation(
        obs, 
        num_modalities, 
        num_obs,
        metamodel
    )
    
    qs = run_factorized_fpi(A, metamodel, obs_processed, prior, num_iter=num_iter)

    #@infiltrate; @assert false
        
    return qs
end

#=
""" Run State Inference via Fixed-Point Iteration """
function fixed_point_iteration(
    #A::Vector{Array{T,N}} where {T <: Real, N}, obs::Vector{Vector{Float64}}, num_obs::Vector{Int64}, num_states::Vector{Int64};
    A::Vector{Array{T}} where {T <: Real}, 
    metamodel::MetaModel,
    obs::Vector{Vector{Float64}}, 
    num_obs::Vector{Int64}, 
    num_states::Vector{Int64};
    prior::Union{Nothing, Vector{Vector{T}}} where T <: Real = nothing, 
    num_iter::Int=num_iter, 
    dF::Float64=1.0, 
    dF_tol::Float64=dF_tol
    )

    # Get model dimensions (NOTE Sam: We need to save model dimensions in the AIF struct in the future)
    n_modalities = length(num_obs)
    n_factors = length(num_states)

    # if metamodel, then num_states etc. might be wrong
    if !isnothing(metamodel)
        num_obs = [size(a, 1) for a in A]
        num_states = nothing
        n_modalities = length(num_obs)
        n_factors = nothing
    end
    

    # Get joint likelihood
    likelihood = get_joint_likelihood(A, metamodel, obs, num_states)
    likelihood = capped_log(likelihood)

    # Initialize posterior and prior
    qs = Vector{Vector{Float64}}(undef, n_factors)
    for factor in 1:n_factors
        qs[factor] = ones(num_states[factor]) / num_states[factor]
    end

    # If no prior is provided, create a default prior with uniform distribution
    if prior === nothing
        prior = create_matrix_templates(num_states)
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
                    qL[i] = sum([LL_tensor[indices...] / qs[factor][i] 
                        for indices in Iterators.product([1:size(LL_tensor, dim) 
                        for dim in 1:n_factors]...) if indices[factor] == i])
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
=#


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
        
    #q_pi = Vector{Float64}(undef, n_steps)
    qs_pi = Vector{Float64}[]
    qo_pi = Vector{Float64}[]
    lnE = capped_log(agent.E)
    
    use_means = true
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
                    if use_means
                        # use mean 
                        G[idx] += mean(utility_)  # mean rather than sum due to missings
                    else
                        # use extremes
                        G[idx] += (maximum(utility_) + minimum(utility_)) / 2  # mean rather than sum due to missings
                    end
                else
                    # early stop
                    if use_means
                        G[idx] += mean(skipmissing(utility_))
                    else
                        # use extremes
                        G[idx] += (maximum(skipmissing(utility_)) + minimum(skipmissing(utility_))) / 2
                    end
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
                
                #G[idx] += info_gain_
                if length(utility_) == n_steps
                    if use_means
                        # use mean 
                        G[idx] += mean(info_gain_)  # mean rather than sum due to missings
                    else
                        # use extremes
                        #G[idx] += (maximum(info_gain_) + minimum(info_gain_)) / 2  # mean rather than sum due to missings
                        G[idx] += maximum(info_gain_) 
                    end
                else
                    # early stop
                    if use_means
                        G[idx] += mean(skipmissing(info_gain_))
                    else
                        #G[idx] += (maximum(skipmissing(info_gain_)) + minimum(skipmissing(info_gain_))) / 2
                        G[idx] += maximum(skipmissing(info_gain_)) 
                    end
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
                    G[idx] += calc_pA_info_gain(pA, qo_pi, qs_pi)
                end
            end

            if agent.pB !== nothing
                #@infiltrate; @assert false
                G[idx] += calc_pB_info_gain(pB, qs_pi, qs, policy)
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
    
    return q_pi, G, utility, info_gain, risk, ambiguity
end


""" Get Expected Observations """
#function get_expected_obs(qs_pi, A::Vector{Array{T,N}} where {T <: Real, N})

function get_expected_obs(
    qs_pi, 
    A::Vector{Array{T}} where {T <: Real},
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

MINVAL = eps(Float64)
function stable_xlogx(x)
    
    zz =  [LEF.xlogy.(z, clamp.(z, MINVAL, Inf)) for z in x]
    #@infiltrate; @assert false
    return zz
end


function stable_entropy(x)
    z = stable_xlogx(x)
    return - sum(vcat(z...))  
end


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
        #states_surprise += calculate_bayesian_surprise(A, qs_pi[t])
        states_surprise[t] = calculate_bayesian_surprise(A, qs_pi[t])

    end

    return states_surprise
end

""" Calculate observation to state info Gain """
function calc_pA_info_gain(pA, qo_pi, qs_pi)
    @infiltrate; @assert false
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
function calc_pB_info_gain(pB, qs_pi, qs_prev, policy)
    @infiltrate; @assert false
    n_steps = length(qs_pi)
    num_factors = length(pB)

    wB = Vector{Any}(undef, num_factors)
    for (factor, pB_f) in enumerate(pB)
        wB[factor] = spm_wnorm(pB_f)
    end

    pB_info_gain = 0

    for t in 1:n_steps
        if t == 1
            previous_qs = qs_prev
        else
            previous_qs = qs_pi[t-1]
        end

        policy_t = policy[t, :]

        for (factor, a_i) in enumerate(policy_t)
            wB_factor_t = wB[factor][:,:,Int(a_i)] .* (pB[factor][:,:,Int(a_i)] .> 0)
            pB_info_gain -= dot(qs_pi[t][factor], wB_factor_t * previous_qs[factor])
        end
    end
    return pB_info_gain
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


""" Edited Compute Accuracy [Still needs to be nested within Fixed-Point Iteration] """
function compute_accuracy_new(log_likelihood, qs::Vector{Vector{Real}})
    n_factors = length(qs)
    ndims_ll = ndims(log_likelihood)
    dims = (ndims_ll - n_factors + 1) : ndims_ll

    result_size = size(log_likelihood, 1) 
    results = zeros(Real,result_size)

    for indices in Iterators.product((1:size(log_likelihood, i) for i in 1:ndims_ll)...)
        product = log_likelihood[indices...] * prod(qs[factor][indices[dims[factor]]] for factor in 1:n_factors)
        results[indices[1]] += product
    end

    return results
end

""" Calculate State-Action Prediction Error """
function calculate_SAPE(aif::AIF)

    qs_pi_all = get_expected_states(aif.qs_current, aif.B, aif.policies)
    qs_bma = bayesian_model_average(qs_pi_all, aif.Q_pi)

    if length(aif.states["bayesian_model_averages"]) != 0
        sape = kl_divergence(qs_bma, aif.states["bayesian_model_averages"][end])
        push!(aif.states["SAPE"], sape)
    end

    push!(aif.states["bayesian_model_averages"], qs_bma)
end
