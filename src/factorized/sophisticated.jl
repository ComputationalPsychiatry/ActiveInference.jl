
module Sophisticated


using Format
using Infiltrator
#using Revise

#import LinearAlgebra as LA
import OMEinsum as ein
import LogExpFunctions as LEF
#import IterTools
#import Statistics
import MetaGraphsNext as MGN
import MetaGraphs as MG
import AbstractTrees as AT
#import Multigraphs 
import Graphs
import LinearAlgebra as LA
import Statistics
import Dates


include("./struct.jl")
include("./utils/maths.jl")
include("./utils/utils.jl")
include("./algos.jl")

#show(stdout, "text/plain", x)
# @infiltrate; @assert false

#Graphs.is_directed(::Type{<:Multigraphs.Multigraph}) = false



""" -------- Inference Functions -------- """

#### State Inference #### 

# --------------------------------------------------------------------------------------------------
""" Get Expected States """
function get_expected_states(
    qs::Vector{Vector{T}} where T <: Real, 
    agent, 
    policy;  # just the current action. e.g., (1,) on first call, (3,) on next, etc.
    policy_full::Union{Nothing, Tuple}= nothing  # e.g., ((1,)) on first call, ((1,3),) on next, ((1,3,2),) on next, etc.
    )

    B = agent.B
    metamodel = agent.metamodel

    @infiltrate; @assert false
    n_policy_steps = length(policy[1])  # same policy len for all actions
    action_names = keys(metamodel.action_options)
    state_is_with_action = [ii for ii in 1:length(B) if length(intersect(metamodel.state_deps[ii], action_names)) > 0]
    # initializing posterior predictive density as a list of beliefs over time
    qs_pi = [deepcopy(qs) for _ in 1:n_policy_steps+1]
    
    null_actions = metamodel.null_actions
    null_action_ids = [findfirst(x -> x == null_actions[ii], metamodel.action_options[ii]) for ii in 1:length(action_names)]

    # todo:
    # - watch locations over steps and filter out steps with repeating locations, except stay
    # should the iterator be filtered at all originally?
    # tests on initial state vs. post action
    
    policy_step_i = 1  # policy len = 1 for sophisticated inference
    
    if agent.verbose
        printfmtln("policy={}, policy_full= {}, qs[1]= {}", policy, policy_full, argmax(qs[1]))
    end

    #if policy_full == (3,3,3)
    #    @infiltrate; @assert false    
    #end
        
    if !(metamodel.policy_tests.earlystop_fx(qs)) && length(policy_full) > 1
        # check if STAY places agent at stop point
        #@infiltrate; @assert false    
        return missing
    end


    for state_i in 1:length(B) 
        
        # list of the hidden state factor indices that the dynamics of `qs[state_i]` depend on
        factors = metamodel.state_deps[state_i]
        factor_idx = [findfirst(x -> x == j, keys(metamodel.state_deps)) for j in factors]
        
        Bc = copy(B[state_i])
        selections = nothing

        # handle action
        if state_i in state_is_with_action
            selections = [(name, pol[policy_step_i], metamodel.action_options[name][pol[policy_step_i]], null_action_id) 
                for (name, pol, null_action_id) in zip(action_names, policy, null_action_ids)
            ]
            #printfmtln("\nstep={}, state_i={}, selections= {}", t, state_i, selections)
            
            # These are tests for pre-action state         
            for (i_selection, selection) in enumerate(selections)
                @infiltrate; @assert false
                if !(
                    # is this action unwanted (e.g., takes agent off the grid)?
                    metamodel.policies.action_contexts[selection[1]][:option_context][selection[3]](qs_pi[policy_step_i])
                    )
                    #@infiltrate; @assert false
                    return nothing  # entire policy for all B matrices and actions, is invalid
                end
            end

            idx = []  # index of dims of this B matrix, states always come before actions in depencency lists
            iaction = 1
            for (idep, dep) in enumerate(factors) 
                if dep in keys(metamodel.action_options)
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
            push!(deps, qs_pi[policy_step_i][idep])
        end
        
        Bc = dot_product1(Bc, deps)

        if !isapprox(sum(Bc), 1.0)
            @infiltrate; @assert false
        end
        
        #printfmtln("    {}", Bc) 
        qs_pi[policy_step_i+1][state_i] = Bc

        # now check if action result is unwanted/illegal or a stop

        if state_i in state_is_with_action
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
                @infiltrate; @assert false
                if false && !(metamodel.policies.action_contexts[selection[1]][:stopfx](qs_pi[policy_step_i+1]))
                    # are all remaining actions "stay"
                    @infiltrate; @assert false    
                    if !all(policy[i_selection][policy_step_i+1:end] .== selection[4])
                        @infiltrate; @assert false    
                        return nothing  # entire policy for all B matrices and actions, is invalid
                    else
                        @infiltrate; @assert false
                        stop_early_at_t = t+1
                    end
                end
            end
            

        end
    end
    
    return qs_pi[2:end]
end


# --------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------
""" Update Posterior States """
function update_posterior_states(
    #A::Vector{Array{T,N}} where {T <: Real, N}, 
    A::Union{Vector{Array{T}} where {T <: Real}, Vector{Array{T, N}} where {T <: Real, N}}, 
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


# --------------------------------------------------------------------------------------------------
""" Get Expected Observations """
function get_expected_obs(
    qs_pi, 
    agent
    )

    A = agent.A
    metamodel = agent.metamodel
    
    n_steps = length(qs_pi)  # in general, this might be equal to or less than policy length, if stop was reached
    
    if n_steps > 1
        # n_steps must be 1 for sophisticated inference
        @infiltrate; @assert false
    end
    
    qo_pi = Vector{Vector{Float64}}[]
    printflag = 0
    
    for t in 1:n_steps
        qo_pi_t = Vector{Vector{Float64}}(undef, length(A))
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
                push!(deps, qs_pi[policy_step_i][idep])
            end
            
            Am = dot_product1(Am, deps)
            @assert Am.size == (A_m.size[1], ) 
                       
            # todo: make rule to avoid doing dot product twice, and make sure dot product works for all instances in the code

            if !isapprox(sum(Am), 1.0)
                #printfmtln("\nmodailty= {}, Am={}", modality, Am)
                                
                Am = copy(A_m)
                Am = vcat(Am, zeros(1, Am.size[2:end]...))
                res = dot_product1(Am, deps)
                if agent.verbose && printflag == 10
                    printfmtln("\nmodailty= {}, remade Am={}", modality, res)
                    printflag = 1
                end
                
                if !isapprox(sum(res), 1.0) || !isapprox(res[end], 0.0) 
                    @infiltrate; @assert false
                end    
                Am = res[1:end-1]
            end
            
            qo_pi[policy_step_i][modality] = Am

        end

        
    end
    #@infiltrate; @assert false

    return qo_pi
end


# --------------------------------------------------------------------------------------------------
""" Calculate Expected Utility """
function calc_expected_utility(qo_pi, C)
    
    n_steps = length(qo_pi)  # in general, this might be equal to or less than policy length, if stop was reached
    
    if n_steps > 1
        @infiltrate; @assert false
    end

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
            
            #expected_utility += dot(qo_pi[policy_step_i][modality], lnC) 
            expected_utility[policy_step_i] += LA.dot(qo_pi[policy_step_i][modality], lnC)

            # no log or softmax
            #expected_utility[policy_step_i] += dot(qo_pi[policy_step_i][modality], C_prob[modality][:, t])

            if expected_utility[policy_step_i] > 0
                @infiltrate; @assert false
            end   
            #@infiltrate; @assert false
        end

    end
    
    #@infiltrate; @assert false
    return expected_utility
end


# --------------------------------------------------------------------------------------------------
function compute_info_gain(qs, qo, A, metamodel)
    """
    New version of expected information gain that takes into account sparse dependencies between observation modalities and hidden state factors.
    qs, qo are over policy steps
    """

    n_policy_steps = qs.size[1]
    if n_policy_steps > 1
        @infiltrate; @assert false
    end
    
    #@infiltrate; @assert false
    info_gain_per_step = zeros(n_policy_steps)
    ambiguity_per_step = zeros(n_policy_steps)
    for step in 1:n_policy_steps
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


# --------------------------------------------------------------------------------------------------
function recurse_parents(node_label, Graph, path)
        # find all parents from leaf to root in tree graph, starting at leaf
        # returns list of node_labels
        push!(path, node_label)
        node_label = collect(MGN.inneighbor_labels(Graph, node_label))
        if node_label.size[1] == 1
            # should only be one parent
            #printfmtln("node_label= \n{}\n", node_label[1])
            recurse_parents(node_label[1], Graph, path)
        elseif node_label.size[1] == 0
            return path
        else
            @infiltrate; @assert false
        end
    end


# --------------------------------------------------------------------------------------------------
#### Policy Inference #### 
""" Update Posterior over Policies """
function update_posterior_policies(agent)
    #@infiltrate; @assert false
    qs = agent.qs_current

    #@infiltrate; @assert false
    n_policy_steps = agent.policy_len
    n_policies = agent.metamodel.number_policies
    
    qs_pi = Vector{Float64}[]
    qo_pi = Vector{Float64}[]
    lnE = capped_log(agent.E)
    
    t0 = Dates.time()

    siGraph = MGN.MetaGraph(
        Graphs.DiGraph();  # underlying graph structure
        label_type = Vector{Label},  # partial or complete policy tuple
        vertex_data_type = Union{ObsNode, ActionNode, EarlyStop, BadPath}, 
        edge_data_type = GraphEdge,  
        graph_data = "sophisticated DFS graph",  # tag for the whole graph
    )
    
    ObsLabel = [Label(0, tuple(agent.obs_current...), nothing, "Obs")]
    siGraph[ObsLabel] = ObsNode(
        copy(qs), 
        repeat([nothing], 6)..., 
        1.0, 
        1.0, 
        nothing, 
        tuple(agent.obs_current...), 
        0, 
        0
    )
    #@infiltrate; @assert false
    
    recurse(siGraph, agent, 1, ObsLabel)
    
    printfmtln("\nmake graph  time= {}\n", round((Dates.time() - t0) , digits=2))
    t0 = Dates.time()

    if Graphs.nv(siGraph) == 0
        @infiltrate; @assert false    
    end 

    # Now that we have a graph, prune bad paths and dangling early stops and ObsNodes. Then 
    # accumulate/record G etc.
    
    # collect all dfs paths from node 0 to leaves
    paths = []
    max_level = 0
    for ii in Graphs.vertices(siGraph)
        node_label = MGN.label_for(siGraph, ii)
        if isa(siGraph[node_label], ObsNode) 
            max_level = max(max_level, siGraph[node_label].level)
        end

        if false && isa(siGraph[node_label], ObsNode)
            if siGraph[node_label].ith_observation > 1
                printfmtln("full dfs subpolicy= {}, cnt= {}", siGraph[node_label].subpolicy, siGraph[node_label].i_observation)
                @infiltrate; @assert false    
            end
        end    
        
        if Graphs.outdegree(siGraph, ii) == 0
            # leaf
            #test = Graphs.a_star(siGraph, 1, ii)
            parents = recurse_parents(node_label, siGraph, [])
            
            #=
            todo: the conversion from node_label to code and then edge is unnecessary and extra work. Its 
            only done to match the result of Graphs.a_star. Keeping as node_labels will eliminate the need
            later for calls to MGN.label_for. 
            =#
            parents = [(MGN.code_for(siGraph, parents[j+1]), MGN.code_for(siGraph, parents[j])) for j in 1:length(parents)-1]
            push!(paths, Graphs.Edge.(reverse(parents)))
            #@infiltrate; @assert false
        end
    end

    printfmtln("\ncollect dfs paths time= {}\n", round((Dates.time() - t0) , digits=2))
    t0 = Dates.time()
    #@infiltrate; @assert false

    if max_level == 0
        printfmtln("\n-----\nError: No observation nodes greater than level 0 in graph.\n----\n")
        @infiltrate; @assert false    
    end
    
    if max_level < n_policy_steps-1
        printfmtln("\n-------\nWarning!! ------\nmax_level={} is < policy_length={}\n-----\n", 
        max_level, n_policy_steps
        )
        @infiltrate; @assert false    
    end
    
    # clean up dfs paths
    bad_node_labels = Set()
    for path in paths
        dst = MGN.label_for(siGraph, path[end].dst)  # leaf node_label
        
        if isa(siGraph[dst], BadPath)
            # remove BadPath node and any direct ancestors that have only one out degree
            for edge in reverse(path)
                dst_ = MGN.label_for(siGraph, edge.dst)
                if Graphs.outdegree(siGraph, edge.dst) in [0, 1]
                    push!(bad_node_labels, dst_)
                else
                    break
                end
            end
        end

        if isa(siGraph[dst], EarlyStop)
            #=
            Remove this EarlyStop node. Also remove its ObsNode parent. Do this because early stop 
            will be true for all actions associated with the ObsNode parent. This renders the parent
            of the ObsNode as the last ActionNode in the (shortened) policy.
            =#
            push!(bad_node_labels, dst)
            src = MGN.label_for(siGraph, path[end].src)  # parent
            @assert isa(siGraph[src], ObsNode)
            push!(bad_node_labels, src)
        end
    end

    printfmtln("\ncleanup dfs paths time= {}\n", round((Dates.time() - t0) , digits=2))
    t0 = Dates.time()
    
    # delete all unwanted nodes 
    for node_label in bad_node_labels
        delete!(siGraph, node_label)  # also deletes a node's 'to' edges
    end

    if Graphs.nv(siGraph) == 0
        @infiltrate; @assert false    
    end 

    printfmtln("\ndelete node time= {}, n_nodes= {}\n", round((Dates.time() - t0) , digits=2), Graphs.nv(siGraph))
    t0 = Dates.time()
    
    
    # Now all dfs paths are valid and end in an actionNode. Get a list of valid policies.
    policies = Set()
    for ii in Graphs.vertices(siGraph)
        if Graphs.outdegree(siGraph, ii) == 0
            # leaf
            #path = Graphs.a_star(siGraph, 1, ii)
            node_label = MGN.label_for(siGraph, ii)
            parents = recurse_parents(node_label, siGraph, [])
            #=
            todo: the conversion from node_label to code and then edge is unnecessary and extra work. Its 
            only done to match the result of Graphs.a_star. Keeping as node_labels will eliminate the need
            later for calls to MGN.label_for. 
            =#
            parents = [(MGN.code_for(siGraph, parents[j+1]), MGN.code_for(siGraph, parents[j])) for j in 1:length(parents)-1]
            path = Graphs.Edge.(reverse(parents))

            policy = []
            for edge in path  # skip node 0
                dst = MGN.label_for(siGraph, edge.dst)  
                if isa(siGraph[dst], ActionNode)  # for a path, only save action from ActionNodes
                    push!(policy, siGraph[dst].action)
                end
            end
            #=
            There can be many duplicate policies, as each might consider multiple observations. 
            Likewise, there can be many paths per policy.
            =#   
            push!(policies, policy)
        end
    end
    policies = collect(policies)

    if length(policies) == 0
        @infiltrate; @assert false    
    end 

    printfmtln("\nget policies list time= {}\n", round((Dates.time() - t0) , digits=2))
    t0 = Dates.time()
    

    # the returns here are for policies identified in this function, not in policy iterator
    
    # choose one of the following methods
    if agent.graph_postprocessing_method in ["G_prob_method", "G_prob_qpi_method"] 
        G, q_pi, utility, info_gain, risk, ambiguity = do_EFE_over_policies(siGraph, agent, policies)
    elseif agent.graph_postprocessing_method == "marginal_EFE_method"
        G, q_pi, utility, info_gain, risk, ambiguity = do_EFE_over_actions(siGraph, agent, policies)
    end

    printfmtln("\ndo G time= {}\n", round((Dates.time() - t0) , digits=2))
    t0 = Dates.time()
    

    # put polices in same order and size as policy iterator
    n_policies = agent.metamodel.number_policies
    q_pi_policies = zeros(n_policies)
    G_policies = zeros(n_policies)
    utility_policies = Matrix{Union{Missing, Float64}}(undef, n_policies, agent.policy_len)
    info_gain_policies = Matrix{Union{Missing, Float64}}(undef, n_policies, agent.policy_len)
    risk_policies = Matrix{Union{Missing, Float64}}(undef, n_policies, agent.policy_len)
    ambiguity_policies = Matrix{Union{Missing, Float64}}(undef, n_policies, agent.policy_len)

    # iterator policies do not have any early stops. Fill policies with null action if early stops
    policies_ = []
    for p in policies
        if length(p) == agent.policy_len
            push!(policies_, tuple(p...))
        else
            # 5 is the stay option. todo: make this generic
            push!(policies_, tuple( vcat(p, repeat([5], agent.policy_len))[1:agent.policy_len]...))
        end
    end


    for (ii, policy) in enumerate(agent.policy_iterator)
        jj = findfirst(x->x == policy[1], policies_)
        if isnothing(jj)
            continue
        end
        q_pi_policies[ii] = q_pi[jj]
        G_policies[ii] = G[jj]
        utility_policies[ii,:] = utility[jj,:]
        info_gain_policies[ii,:] = info_gain[jj,:]
        risk_policies[ii,:] = risk[jj,:]
        ambiguity_policies[ii,:] = ambiguity[jj,:]
    end

    #@infiltrate; @assert false  
    # for SI, do not use info gain from parameter learning
    return q_pi_policies, G_policies, utility_policies, info_gain_policies, risk_policies, ambiguity_policies
end 


# --------------------------------------------------------------------------------------------------
function do_EFE_over_policies(siGraph, agent, policies)
    #=
    This is the G * prob method, where prob is cascaded from node 0 to leaves. It calculates
    EFE over policies, not actions.
    =#
    
    # create matrices to hold results
    G = Matrix{Union{Missing, Float64}}(undef, length(policies), agent.policy_len)
    utility = Matrix{Union{Missing, Float64}}(undef, length(policies), agent.policy_len)
    info_gain = Matrix{Union{Missing, Float64}}(undef, length(policies), agent.policy_len)
    risk = Matrix{Union{Missing, Float64}}(undef, length(policies), agent.policy_len)
    ambiguity = Matrix{Union{Missing, Float64}}(undef, length(policies), agent.policy_len)

    if agent.verbose
        println("\nEnumerate policies, do_EFE_over_policies ----")
    end

    # consider each full policy; e.g., (1,2,2) over three policy steps
    for (policy_i, policy) in enumerate(policies)
        subpolicies = [Tuple(policy[1:n]) for n in 1:length(policy)]  # e.g., [(1,), (1,2), (1,2,2)]
        vdic = Dict()  # dict of ActionNodes and ObsNodes per subpolicy
        vlist = []  # list of all ActionNodes in whole policy
        for sp in subpolicies
            vdic[sp] = []
        end

        # iterate over all nodes in graph to find those that match subpolicies
        for node_label in MGN.labels(siGraph)
            # delete any calculated G values
            if isa(siGraph[node_label], ObsNode)
                siGraph[node_label].utility_updated = 0
                siGraph[node_label].info_gain_updated = 0
                siGraph[node_label].ambiguity_updated = 0
                siGraph[node_label].risk_updated = 0
                siGraph[node_label].G_updated = 0
                siGraph[node_label].q_pi_updated = 0
                siGraph[node_label].prob_updated = nothing
            else
                @assert isa(siGraph[node_label], ActionNode)
                siGraph[node_label].utility_updated = 0
                siGraph[node_label].info_gain_updated = 0
                siGraph[node_label].ambiguity_updated = 0
                siGraph[node_label].risk_updated = 0
                siGraph[node_label].G_updated = 0
                siGraph[node_label].q_pi_updated = 0
            end

            #=
            Collect ActionNodes in subpolicies. That is, if policy is (3,2,1), then collect nodes
            that have subpolicy=(3,) for first level, (3,2) for second level, and (3,2,1) for third
            level. 
            =#
            
            node_actions = tuple([label.action for label in node_label][2:end]...)  # first action is nothing
            if  isa(siGraph[node_label], ActionNode) && node_actions in subpolicies
                idx = findfirst(x -> x == node_actions, subpolicies)
                sp = subpolicies[idx]
                push!(vdic[sp], node_label)
                push!(vlist, node_label)
            end
            
            #if isa(siGraph[node_label], ObsNode) && siGraph[node_label].ith_observation > 1
                # curious how many ObsNodes an ActionNode might have?
                #printfmtln("    trimmed graph subpolicy= {}, cnt= {}", siGraph[node_label].subpolicy, siGraph[node_label].ith_observation)
            #end           
        end
        #printfmtln("\nsubpolicies={}, vlist size = {}", [x for x in subpolicies], length(vlist))
    
        # consider each subpolicy of this full policy
        for (sp_i, sp) in enumerate(reverse(subpolicies))
            #printfmtln("\npolicy= {}, sp= {}", policy, sp)

            level = length(sp)
            
            # leave as missing in case the graph has nothing in this level for policy_i
            #G[policy_i, level] = 0.0
            #utility[policy_i, level] = 0.0
            #info_gain[policy_i, level] = 0.0
            #risk[policy_i, level] = 0.0
            #ambiguity[policy_i, level] = 0.0
            
            node_labels =  vdic[sp]  # action nodes in subpolicy
            siblings_done = Set()
            for (node_label_i, node_label) in enumerate(node_labels)
                
                

                # get ObsNode parent
                parent_node_label = collect(MGN.inneighbor_labels(siGraph, node_label))
                @assert length(parent_node_label) == 1
                parent_node_label = parent_node_label[1]
                @assert isa(siGraph[parent_node_label], ObsNode)

                if node_label in siblings_done
                    if isnothing(siGraph[parent_node_label].prob_updated)
                        @infiltrate; @assert false 
                    end
                    @infiltrate; @assert false 
                    continue
                end

                # get all ActionNode siblings, some of which are not in subpolicy
                siblings = collect(MGN.outneighbor_labels(siGraph, parent_node_label))
                push!(siblings_done, siblings...)

                # get ObsNode children
                obs_children = collect(MGN.outneighbor_labels(siGraph, node_label))
                
                if length(obs_children) == 0
                    @assert sp_i == 1
                    
                    
                    # this is the last action node in a policy, convert G etc. for this action node to G_updated etc.
                    siGraph[node_label].utility_updated = siGraph[node_label].utility
                    siGraph[node_label].info_gain_updated = siGraph[node_label].info_gain
                    siGraph[node_label].risk_updated = siGraph[node_label].risk
                    siGraph[node_label].ambiguity_updated = siGraph[node_label].ambiguity
                    siGraph[node_label].G_updated = siGraph[node_label].G  
                    
                    # cascade prob_updated on ObsNodes starting from root to leaf
                    node_label_id = MGN.code_for(siGraph, node_label)
                    #path = Graphs.a_star(siGraph, 1, node_label_id)

                    parents = recurse_parents(node_label, siGraph, [])
            
                    #=
                    todo: the conversion from node_label to code and then edge is unnecessary and extra work. Its 
                    only done to match the result of Graphs.a_star. Keeping as node_labels will eliminate the need
                    later for calls to MGN.label_for. 
                    =#
                    parents = [(MGN.code_for(siGraph, parents[j+1]), MGN.code_for(siGraph, parents[j])) for j in 1:length(parents)-1]
                    path = Graphs.Edge.(reverse(parents))
                    
                    if length(path) != length(sp) * 2 - 1
                        @infiltrate; @assert false 
                    end
                    
                    # node_label and parent should be in path
                    if (MGN.code_for(siGraph, parent_node_label), MGN.code_for(siGraph, node_label)) != parents[1]
                        @infiltrate; @assert false 
                    end
                    
                    prob = 1
                    for (edge_i, edge) in enumerate(path)  # from root to leaf
                        label = MGN.label_for(siGraph, edge.src)
                        if isa(siGraph[label], ObsNode)
                            @assert edge_i % 2 == 1
                            if !isnothing(siGraph[label].prob_updated)
                                # prob for this node has already been updated
                                @assert siGraph[label].prob_updated ==  prob * siGraph[label].prob
                                prob = siGraph[label].prob_updated
                            else 
                                prob *= siGraph[label].prob
                                siGraph[label].prob_updated = prob
                            end
                        else
                            @assert edge_i % 2 == 0
                            @assert isa(siGraph[label], ActionNode)
                        end
                    end
                end    
                                

                # update the ObsNode parent
                prob = siGraph[parent_node_label].prob_updated
                
                if isnothing(prob)
                    #=
                    Apparently, a path was pruned and the current node_label has an ObsNode child
                    that also has no prob_updated. And none of the ActionNode children of that
                    ObsNode have an action that fits this policy.

                    If the following asserts fail, then perhaps the missing action is further down
                    the path, not just one jump down.
                    =#
                    @assert sp_i < length(policy)
                    obs_children = collect(MGN.outneighbor_labels(siGraph, node_label))
                    @assert length(obs_children) >= 1

                    expected_child_action = policy[sp_i + 1]
                    for obs_child in obs_children
                        @assert isnothing(siGraph[obs_child].prob_updated)
                        action_children = collect(MGN.outneighbor_labels(siGraph, obs_child))
                        @assert length(action_children) >= 1
                        for action_child in action_children
                            @assert siGraph[action_child].action != expected_child_action
                        end
                    end
                    
                    # if all this is true, then node_label is not on a valid path
                    #@infiltrate; @assert false 
                    continue
                end
                
                
                if agent.graph_postprocessing_method == "G_prob_method"
                    siGraph[parent_node_label].G_updated = prob * siGraph[node_label].G  
                    siGraph[parent_node_label].utility_updated = prob * siGraph[node_label].utility
                    siGraph[parent_node_label].info_gain_updated = prob * siGraph[node_label].info_gain
                    siGraph[parent_node_label].risk_updated = prob * siGraph[node_label].risk
                    siGraph[parent_node_label].ambiguity_updated = prob * siGraph[node_label].ambiguity
                    
                elseif agent.graph_postprocessing_method == "G_prob_qpi_method"
                    # unlike pymdp, which only sums over G, here we also keep track of utility, etc.
                    # so we also multiply utility etc. by q_pi_children
                    siGraph[parent_node_label].G_updated = prob * siGraph[node_label].G * siGraph[node_label].q_pi_children  
                    siGraph[parent_node_label].utility_updated = prob * siGraph[node_label].utility * siGraph[node_label].q_pi_children
                    siGraph[parent_node_label].info_gain_updated = prob * siGraph[node_label].info_gain * siGraph[node_label].q_pi_children
                    siGraph[parent_node_label].risk_updated = prob * siGraph[node_label].risk * siGraph[node_label].q_pi_children
                    siGraph[parent_node_label].ambiguity_updated = prob * siGraph[node_label].ambiguity * siGraph[node_label].q_pi_children
                end
                
                if ismissing(G[policy_i, level])
                    # there is at least one value in the graph for this level and policy, so initialize 
                    G[policy_i, level] = 0.0
                    utility[policy_i, level] = 0.0
                    info_gain[policy_i, level] = 0.0
                    risk[policy_i, level] = 0.0
                    ambiguity[policy_i, level] = 0.0
                end    
                
                # record results
                G[policy_i, level] += siGraph[parent_node_label].G_updated
                utility[policy_i, level] += siGraph[parent_node_label].utility_updated
                info_gain[policy_i, level] += siGraph[parent_node_label].info_gain_updated
                risk[policy_i, level] += siGraph[parent_node_label].risk_updated
                ambiguity[policy_i, level] += siGraph[parent_node_label].ambiguity_updated

            end
        end
    end

    # we have now calculated G, utility, etc. over each step of every policy
    if (any(ismissing.(info_gain)) || any(ismissing.(utility))) && agent.use_sum_for_calculating_G == true
        printfmtln("\n!!! Warning: Info gain or utility has missing elements, ignoring use_sum_for_calculating_G = true.\n")
    end

    if agent.use_sum_for_calculating_G == false || any(ismissing.(info_gain)) || any(ismissing.(utility))
        G_ = (
            Statistics.maximum.(skipmissing.(eachrow(info_gain)))
            +
            (
                Statistics.minimum.(skipmissing.(eachrow(utility)))
                +
                Statistics.maximum.(skipmissing.(eachrow(utility)))
            ) ./ 2
        )
    else
        # use sum, as in pymdp
        G_ = (
            Statistics.sum.(eachrow(info_gain))
            +
            Statistics.sum.(eachrow(utility))
        )
    end

    # transfer policies data to the policies of the policy iterator
    #Eidx = agent.E[idx]
    #lnE = capped_log(Eidx)
    lnE = 0  # todo: allow priors over 
    if length(G_) == 0
        @infiltrate; @assert false    
    end   
    q_pi = LEF.softmax(G_ .* agent.gamma .+ lnE, dims=1)  


    #@infiltrate; @assert false    
    
    return G_, q_pi, utility, info_gain, risk, ambiguity

end


# --------------------------------------------------------------------------------------------------
function do_EFE_over_actions(siGraph, agent, policies)
    #=
    This is the G * prob * qpi marginal EFE method (as per pymdp).
    
    iterate over nodes:
        examine all siblings of an action node
        calculate G*q_pi*prob for each action
        record siblings as processed
    
    iteraction over each action:
        find all ActionNodes with that action
        add their G values, etc. together
    =#
    
    # create matrices to hold results
    action_dims = tuple([tuple(collect(1:n[1])...) for n in collect(agent.metamodel.action_dims)]...)
    actions = collect(zip(action_dims...))  # these are all possible action combinations of multiple actions
    n_actions = length(actions)
    
    G = Vector{Union{Missing, Float64}}(undef, n_actions)
    utility = Vector{Union{Missing, Float64}}(undef, n_actions)
    info_gain = Vector{Union{Missing, Float64}}(undef, n_actions)
    risk = Vector{Union{Missing, Float64}}(undef, n_actions)
    ambiguity = Vector{Union{Missing, Float64}}(undef, n_actions)

    if agent.verbose
        println("\nEnumerate policies, do_EFE_over_actions ----")
    end

    # iterate over all nodes in graph
    processed = Set()
    for node_label in MGN.labels(siGraph)
        if isa(siGraph[node_label], ActionNode) && !(node_label in processed)
            parent_node_label = collect(MGN.inneighbor_labels(siGraph, node_label))
            @assert length(parent_node_label) == 1
            parent_node_label = parent_node_label[1]
            @assert isa(siGraph[parent_node_label], ObsNode)

            prob = siGraph[parent_node_label].prob

            siblings = collect(MGN.outneighbor_labels(siGraph, parent_node_label))
            push!(processed, siblings...)

            for sibling in siblings
                siGraph[sibling].G_updated = prob * siGraph[sibling].G * siGraph[sibling].q_pi_children  
                siGraph[sibling].utility_updated = prob * siGraph[sibling].utility * siGraph[sibling].q_pi_children
                siGraph[sibling].info_gain_updated = prob * siGraph[sibling].info_gain * siGraph[sibling].q_pi_children
                siGraph[sibling].risk_updated = prob * siGraph[sibling].risk * siGraph[sibling].q_pi_children
                siGraph[sibling].ambiguity_updated = prob * siGraph[sibling].ambiguity * siGraph[sibling].q_pi_children
                
                action = siGraph[sibling].action
                G[action]
            end
        end
    end



    @infiltrate; @assert false    
    
    return G_, q_pi, utility, info_gain, risk, ambiguity

end



# --------------------------------------------------------------------------------------------------
function recurse(siGraph, agent, level, ObsLabel)
    
    observation_prune_threshold = 1/16  #1/100  #1/16
    policy_prune_threshold = 1/16
    prune_penalty = 512
       
    qs = siGraph[ObsLabel].qs_next
    observation = siGraph[ObsLabel].observation

    if agent.verbose
        #@infiltrate; @assert false
        printfmtln("\n------\nlevel={}, observation={} \nObsLabel= {} \nprob= {} \nmax qs= {}", 
            level, 
            observation, 
            [[getfield(ObsLabel[ii], f) for f in fieldnames(typeof(ObsLabel[ii]))] for ii in 1:length(ObsLabel) ],
            siGraph[ObsLabel].prob,
            [argmax(q) for q in qs]
        )
    end

    if isnothing(observation)
        @infiltrate; @assert false
    end

    # to keep simple, use only one action here, todo: allow multiple actions
    children = []
    
    for (idx, action) in enumerate(agent.action_iterator)  
        # action is a tuple of integers
        subpolicy = tuple([x.action for x in ObsLabel if !isnothing(x.action)]..., action)  # e.g. ((3, 1),)
        
        # is this a valid subpolicy?
        found = false
        
        
        for policy in agent.policy_iterator
            policy_zip = zip(policy...)
            policies = Tuple.([first(policy_zip, i) for i in 1:length(policy_zip)] )
            n = length(subpolicy)
            if subpolicy == policies[n]
                found = true
                break
            end
        end

        ActionLabel = vcat(ObsLabel, Label(level, observation, action, "Action")) 
        if agent.verbose
            printfmtln("\nActionLabel[end]= \n{}", 
                Dict(key=>getfield(ActionLabel[end], key) for key ∈ fieldnames(Label))
            )
        end
        if ActionLabel in collect(MGN.labels(siGraph))
            # the label should not exist yet
            @infiltrate; @assert false
        end

        if !found
            # invalid policy
            siGraph[(ActionLabel)] = BadPath("BadPath")
                        
            # this edge will be unique
            siGraph[ObsLabel, ActionLabel] = GraphEdge()
            #@infiltrate; @assert false
            continue
        end
        
        qs_pi = get_expected_states(qs, agent, action, policy_full = subpolicy)  # assume one action only
        #@infiltrate; @assert false

        if isnothing(qs_pi) 
            # bad policy for given states
            siGraph[ActionLabel] = BadPath("BadPath")
            siGraph[ObsLabel, ActionLabel] = GraphEdge()
            #@infiltrate; @assert false
            continue
        end

        if ismissing(qs_pi)
            # early stop, will be true for all actions
            siGraph[ActionLabel] = EarlyStop("EarlyStop")
            siGraph[ObsLabel, ActionLabel] = GraphEdge()
            #@infiltrate; @assert false
            continue  
        end

        qo_pi = get_expected_obs(qs_pi, agent)  
        
        G = 0
        if agent.use_utility
            utility = calc_expected_utility(qo_pi, agent.C)[1]
            G += utility
        end

        # Calculate expected information gain of states
        if agent.use_states_info_gain
            info_gain, ambiguity = compute_info_gain(qs_pi, qo_pi, agent.A, agent.metamodel)
            info_gain = info_gain[1] 
            ambiguity = ambiguity[1]
            risk = (
                utility
                + info_gain
                - ambiguity
            )
            @assert isapprox(info_gain + utility, risk + ambiguity)
            G += info_gain
            #@infiltrate; @assert false

        end
        #@infiltrate; @assert false
        
        # add to graph if not already present
        siGraph[ActionLabel] = ActionNode(
            qs,  # size number of state variables [number of categories per variable]
            qs_pi, # size number of actions (=1) [number of state variables [ number of categories per variable]]
            qo_pi, # size number of actions (=1) [number of observation variables [ number of categories per variable]]
            utility, # scalar
            info_gain, # scalar
            ambiguity, # scalar
            risk, # scalar
            G, # scalar
            false,  # are childen pruned out due to prune penalty?
            nothing,  # q_pi_children
                                   
            nothing, # utility_updated
            nothing, # info_gain_updated
            nothing, # ambiguity_updated
            nothing, # risk_updated
            nothing, # G_updated
            nothing, # q_pi_updated
           
            action,
            subpolicy,
            observation,
            level,
        )
        
        # edge will always be unique
        siGraph[ObsLabel, ActionLabel] = GraphEdge()
        push!(children, ActionLabel)  # an action might not result in a child
        
    end 
    
    if length(children) == 0
        #e.g., loc=9, stay move indicates early stop
        #@infiltrate; @assert false
        return  # no change to graph
    end

        
    #=
    We need a qs_pi vector across actions to potentially filter out unlikely actions, but some 
    actions might have been skipped for BadPath or EarlyStop. For convienience, we base qs_pi only 
    on the valid actions. But this means magnitudes of qs_pi over actions can vary based on number of 
    children. 
    Todo: adjust threshold based on number of children.

    Also, we are only pruning once, based on the initial G vector. It would also be possible to
    prune in a while loop, until no more children are pruned out. E.g., the first prune removes
    child_1, it is deleted, q_pi is recalculated, and now child_2 is below the threshold.
    Todo: allow for multiple pruning rounds.
    =#
    
    orig_children = deepcopy(children)
    for prune_round in 1:length(orig_children)
        
        G_children = [siGraph[child].G for child in children]
        
        if any(ismissing.(G_children)) || any(isnothing.(G_children))
            @infiltrate; @assert false
        end
        
        # todo: add indicive cost to G (record inductive cost)

        q_pi_children = LEF.softmax(G_children * agent.gamma)  # this is qs_pi over all viable children of ObsNode 
        
        # prune out low q_pi_children
        good_children = []
        for (ii, child) in enumerate(children)
            
            if q_pi_children[ii] < policy_prune_threshold
                siGraph[child].pruned = true  # record that child is pruned 
                
                if agent.verbose
                    printfmtln("---- prune round= {}: q_pi= {}, action= {} \nObsLabel={}\n", 
                        prune_round,    
                        q_pi_children[ii],
                        [getfield(child[end], f) for f in fieldnames(typeof(child[end]))],
                        [[getfield(ObsLabel[ii], f) for f in fieldnames(typeof(ObsLabel[ii]))] for ii in 1:length(ObsLabel) ]
                    )
                end
                # attach BadPath node to child
                BadPathLabel = vcat(deepcopy(child), Label(level, observation, siGraph[child].action, "PrunedPath"))  
                if BadPathLabel in collect(MGN.labels(siGraph))
                    # the label should not exist yet
                    @infiltrate; @assert false
                end

                siGraph[BadPathLabel] = BadPath("PrunedPath")
                siGraph[child, BadPathLabel] = GraphEdge()
                #@infiltrate; @assert false
            else
                push!(good_children, child)
            end
        end

        if length(good_children) == 0
            @infiltrate; @assert false
        end

        if length(good_children) == length(children)
            # all children are good, no more pruning needed
            break
        end

        children = good_children
    end

    # record results for q_pi on viable ActionNodes
    G_children = [siGraph[child].G for child in children]  # This is G over all viable children of ObsNode
    q_pi_children = LEF.softmax(G_children * agent.gamma)  # this is qs_pi over all viable children of ObsNode 
    for (ii, child) in enumerate(children)
        siGraph[child].q_pi_children = q_pi_children[ii]
        @assert !isnothing(siGraph[child].q_pi_children)
    end
    #@infiltrate; @assert false
    

    if level == agent.policy_len
        if agent.verbose
            println("returning terminal path")
        end

        return # just return the (pruned) graph
    end


    # make observation iterator for new observations
    qo_pi_sizes = [1:x.size[1] for x in siGraph[children[1]].qo_pi[1]]
    observation_iterator = Iterators.product(qo_pi_sizes...) # every possible combination of observations e.g., (23, 2, 1) for 3 observations
    

    for (idx, ActionLabel) in enumerate(children)
        
        qo_next = siGraph[ActionLabel].qo_pi[1]
        skip_observations = true
        
        # do standard inference?
        if agent.use_SI_graph_for_standard_inference
            # test without belief updates due to likely observations
            # use this for non-sophisticated (standard) inference
            skip_observations = false
            
            # make Obs node and link to parent, overwrite ObsLabel, make only a single Obs node
            ObsLabel = copy(ActionLabel)
            ObsLabel[end] = Label(level, observation, siGraph[ActionLabel].action, "Obs")
            
            if ObsLabel in collect(MGN.labels(siGraph))
                # the label should not exist yet
                @infiltrate; @assert false
            end
            
            # do not call update_posterior_states to get qs_next, use original
            qs_next = siGraph[ActionLabel].qs_pi[1]  
            prob = 1.0
            cnt = 1
            observation = nothing  # this is a dummy ObsNode

            siGraph[ObsLabel] = ObsNode(
                qs_next,
                nothing,  # utility_updated
                nothing,  # info_gain_updated
                nothing,  # ambiguity_updated
                nothing,  # risk_updated
                nothing,  # G_updated
                nothing,  # q_pi_updated
                prob,
                nothing,  # prob_updated
                siGraph[ActionLabel].subpolicy,
                observation,
                level,
                cnt
            )

            siGraph[ActionLabel, ObsLabel] = GraphEdge()
            
            if agent.verbose
                printfmtln("\n        level= {}, calling sophisticated", level)
            end
            #@infiltrate; @assert false
            recurse(siGraph, agent, level+1, ObsLabel)
            
            continue
        end
            
        # do regular SI
        cnt = 0
        for (i_observation, observation) in enumerate(observation_iterator)    
            n = length(observation)
            probs = zeros(n)
            for i in 1:length(observation)
                probs[i] = qo_next[i][observation[i]]
            end
            prob  = prod(vcat(1.0, probs))
            
            #@infiltrate; @assert false
            # ignore low probability states in the search tree
            if prob < observation_prune_threshold
                #    printfmtln("        level= {}, observation= {}, probs= {}, prob= {}", 
                #    level, observation, round.(probs, sigdigits=3), round(prob, sigdigits=3)
                #)
                continue
            end

            skip_observations = false
            cnt += 1

            if agent.verbose
                printfmtln("        level= {}, observation= {}, probs= {}, prob= {}", 
                    level, observation, round.(probs, sigdigits=3), round(prob, sigdigits=3)
                )
            end

            qs_next = update_posterior_states(
                agent.A, 
                agent.metamodel,
                collect(observation), 
                prior = siGraph[ActionLabel].qs_pi[1], 
                num_iter = agent.FPI_num_iter, 
                dF_tol = agent.FPI_dF_tol
            )
        
            # make Obs node and link to parent, overwrite ObsLabel
            # ObsLabel = vcat(ActionLabel, Label(level, observation, siGraph[ActionLabel].action, "Obs"))  # longer
            ObsLabel = copy(ActionLabel)
            ObsLabel[end] = Label(level, observation, siGraph[ActionLabel].action, "Obs")
            
            if ObsLabel in collect(MGN.labels(siGraph))
                # the label should not exist yet
                @infiltrate; @assert false
            end
        
            siGraph[ObsLabel] = ObsNode(
                qs_next,
                nothing,  # utility_updated
                nothing,  # info_gain_updated
                nothing,  # ambiguity_updated
                nothing,  # risk_updated
                nothing,  # G_updated
                nothing,  # q_pi_updated
                prob,
                nothing,  # prob_updated
                siGraph[ActionLabel].subpolicy,
                observation,
                level,
                cnt
            )

            siGraph[ActionLabel, ObsLabel] = GraphEdge()
            
            if agent.verbose
                printfmtln("\n        level= {}, calling sophisticated", level)
            end
            #@infiltrate; @assert false
            recurse(siGraph, agent, level+1, ObsLabel)
            
            #@infiltrate; @assert false
            
            #G_weighted = np.dot(qs_pi_next, G_next) * prob
            #G[idx] += G_weighted
        end

        if agent.verbose
            printfmtln("        level= {}, ending observation loop, idx={}, obs cnt= {}", level, idx, cnt)            
        end
        if cnt > 1 && agent.verbose
            printfmtln("\nsubpolicy= {}", siGraph[ActionLabel].subpolicy)
            #@infiltrate; @assert false
        end
        if skip_observations == true
            # all observation's were skipped; could be true in first sim steps if learning a B matrix
            printfmtln("\n---- Warning ---\nNo observations, not extending branch. level= {}, observation= {}, actions= {}\n-------\n", 
                level, observation, [x.action for x in ActionLabel]
            )
            #@infiltrate; @assert false
        end
        
        #@infiltrate; @assert false

    end
    if agent.verbose
        printfmtln("    level = {}, ending idx loop", level)
    end
        
end


end  # -- module