
module Sophisticated

using Format
using Infiltrator
using Revise

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



""" -------- Inference Functions -------- """

#### State Inference #### 

""" Get Expected States """
function get_expected_states(
    qs::Vector{Vector{T}} where T <: Real, 
    agent, 
    policy;  # just the current action. e.g., (1,) on first call, (3,) on next, etc.
    policy_full::Union{Nothing, Tuple}= nothing  # e.g., (1,) on first call, (1,3) on next, (1,3,2) on next, etc.
    )

    B = agent.B
    metamodel = agent.metamodel

    # simple one-action policy considred here. todo: test for multiple actions
    
    n_steps = length(policy[1])  # same policy len for all actions
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
    
    t = 1  # policy len = 1 for sophisticated inference
    
    if agent.verbose
        printfmtln("policy={}, policy_full= {}, qs[1]= {}", policy, policy_full, argmax(qs[1]))
    end

    #if policy_full == (3,3,3)
    #    @infiltrate; @assert false    
    #end
    #@infiltrate; @assert false
    
    if !(metamodel.policies.action_contexts[:move][:stopfx](qs)) && length(policy_full) > 1
        # check if STAY places agent at stop point
        #@infiltrate; @assert false    
        return missing
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
                
                if false && !(metamodel.policies.action_contexts[selection[1]][:stopfx](qs_pi[t+1]))
                    # are all remaining actions "stay"
                    @infiltrate; @assert false    
                    if !all(policy[i_selection][t+1:end] .== selection[4])
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


""" Get Expected Observations """
function get_expected_obs(
    qs_pi, 
    agent
    )

    A = agent.A
    metamodel = agent.metamodel
    
    n_steps = length(qs_pi)  # this might be equal to or less than policy length, if stop was reached
    
    if n_steps > 1
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
                if agent.verbose && printflag == 10
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
            
            #expected_utility += dot(qo_pi[t][modality], lnC) 
            expected_utility[t] += LA.dot(qo_pi[t][modality], lnC)

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


function compute_info_gain(qs, qo, A, metamodel)
    """
    New version of expected information gain that takes into account sparse dependencies between observation modalities and hidden state factors.
    qs, qo are over policy steps
    """

    n_steps = qs.size[1]
    if n_steps > 1
        @infiltrate; @assert false
    end
    
    #@infiltrate; @assert false
    info_gain_per_step = zeros(n_steps)
    ambiguity_per_step = zeros(n_steps)
    for step in 1:n_steps
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
function recurse_parents(node, Graph, path)
        # find all parents from leaf to root in tree graph, starting at leaf
        # returns list of nodes
        push!(path, node)
        node = collect(MGN.inneighbor_labels(Graph, node))
        if node.size[1] == 1
            # should only be one parent
            #printfmtln("node= \n{}\n", node[1])
            recurse_parents(node[1], Graph, path)
        elseif node.size[1] == 0
            return path
        else
            @infiltrate; @assert false
        end
    end


# --------------------------------------------------------------------------------------------------
#### Policy Inference #### 
""" Update Posterior over Policies """
function update_posterior_policies(agent)
    
    qs = agent.qs_current

    n_steps = agent.policy_len
    n_policies = agent.policies.number_policies
    
    qs_pi = Vector{Float64}[]
    qo_pi = Vector{Float64}[]
    lnE = capped_log(agent.E)
    
    t0 = Dates.time()

    Graph = MGN.MetaGraph(
        Graphs.DiGraph();  # underlying graph structure
        label_type = Vector{Label},  # partial or complete policy tuple
        vertex_data_type = Union{ObsNode, ActionNode, EarlyStop, BadPath}, 
        edge_data_type = GraphEdge,  
        graph_data = "sophisticated DFS graph",  # tag for the whole graph
    )
    
    ObsLabel = [Label(0, nothing, nothing, "Obs")]
    Graph[ObsLabel] = ObsNode(copy(qs), repeat([nothing], 6)..., 1.0, 1.0, nothing, nothing, 0, 0)
    #@infiltrate; @assert false
    
    recurse(Graph, agent, 1, ObsLabel)
    
    printfmtln("\nmake graph  time= {}\n", round((Dates.time() - t0) , digits=2))
    t0 = Dates.time()

    if Graphs.nv(Graph) == 0
        @infiltrate; @assert false    
    end 

    # Now that we have a graph, prune bad paths and dangling early stops and ObsNodes. Then 
    # accumulate/record G etc.
    
    function bfs_traversal(g, s)
        b = Graphs.bfs_tree(g, s)
        x = [s]
        i = 1
        while i <= length(x)
            append!(x, Graphs.neighbors(b, x[i]))
            i += 1
        end
        return x
    end

    # collect all dfs paths from node 0 to leaves
    paths = []
    max_level = 0
    for ii in Graphs.vertices(Graph)
        node = MGN.label_for(Graph, ii)
        if isa(Graph[node], ObsNode) 
            max_level = max(max_level, Graph[node].level)
        end
        if false && isa(Graph[node], ObsNode)
            if Graph[node].ith_observation > 1
                #printfmtln("full dfs subpolicy= {}, cnt= {}", Graph[node].subpolicy, Graph[node].i_observation)
                #@infiltrate; @assert false    
            end
        end    
        if Graphs.outdegree(Graph, ii) == 0
            # leaf
            #test = Graphs.a_star(Graph, 1, ii)
            parents = recurse_parents(node, Graph, [])
            
            #=
            todo: the conversion from node to code and then edge is unnecessary and extra work. Its 
            only done to match the result of Graphs.a_star. Keeping as nodes will eliminate the need
            later for calls to MGN.label_for. 
            =#
            parents = [(MGN.code_for(Graph, parents[j+1]), MGN.code_for(Graph, parents[j])) for j in 1:length(parents)-1]
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
    
    if max_level < n_steps-1
        printfmtln("\n-------\nWarning!! ------\nmax_level={} is < policy_length={}\n-----\n", 
        max_level, n_steps
        )
        @infiltrate; @assert false    
    end
    
    # clean up dfs paths
    bad_nodes = Set()
    for path in paths
        dst = MGN.label_for(Graph, path[end].dst)  # terminal node
        if isa(Graph[dst], BadPath)
            # remove end node and any direct ancestors that have only one out degree
            for edge in reverse(path)
                dst_ = MGN.label_for(Graph, edge.dst)
                if Graphs.outdegree(Graph, edge.dst) in [0, 1]
                    push!(bad_nodes, dst_)
                else
                    break
                end
            end
        end

        if isa(Graph[dst], EarlyStop)
            # remove this node only and its parent, if ObsNode
            push!(bad_nodes, dst)
            dst_ = MGN.label_for(Graph, path[end].src)  # parent
            @assert isa(Graph[dst_], ObsNode)
            push!(bad_nodes, dst_)
        end

    end

    printfmtln("\ncleanup dfs paths time= {}\n", round((Dates.time() - t0) , digits=2))
    t0 = Dates.time()
    
    # delete all unwanted nodes 
    for node in bad_nodes
        delete!(Graph, node)  # also deletes a node's 'to' edges
    end

    if Graphs.nv(Graph) == 0
        @infiltrate; @assert false    
    end 

    printfmtln("\ndelete nodes time= {}\n", round((Dates.time() - t0) , digits=2))
    t0 = Dates.time()
    
    
    # Now all dfs paths are valid and end in an actionNode. Get a list of valid policies.
    policies = Set()
    for ii in Graphs.vertices(Graph)
        if Graphs.outdegree(Graph, ii) == 0
            # leaf
            #path = Graphs.a_star(Graph, 1, ii)
            node = MGN.label_for(Graph, ii)
            parents = recurse_parents(node, Graph, [])
            #=
            todo: the conversion from node to code and then edge is unnecessary and extra work. Its 
            only done to match the result of Graphs.a_star. Keeping as nodes will eliminate the need
            later for calls to MGN.label_for. 
            =#
            parents = [(MGN.code_for(Graph, parents[j+1]), MGN.code_for(Graph, parents[j])) for j in 1:length(parents)-1]
            path = Graphs.Edge.(reverse(parents))

            policy = []
            for edge in path  # skip node 0
                dst = MGN.label_for(Graph, edge.dst)  
                if isa(Graph[dst], ActionNode)  # for a path, only save action from ActionNodes
                    push!(policy, Graph[dst].action)
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
    if agent.graph_postprocessing_method == "G_prob_method"
        G, q_pi, utility, info_gain, risk, ambiguity = do_G_prob_method(Graph, agent, policies)
    elseif agent.graph_postprocessing_method == "G_prob_qpi_method"
        G, q_pi, utility, info_gain, risk, ambiguity = do_G_prob_qpi_method(Graph, agent, policies)
    elseif agent.graph_postprocessing_method == "marginal_EFE_method"
        G, q_pi, utility, info_gain, risk, ambiguity = do_marginal_EFE_method(Graph, agent, policies)
    end

    printfmtln("\ndo G time= {}\n", round((Dates.time() - t0) , digits=2))
    t0 = Dates.time()
    

    # put polices in same order and size as policy iterator
    n_policies = agent.policies.number_policies
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


    for (ii, policy) in enumerate(agent.policies.policy_iterator)
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


function do_G_prob_method(Graph, agent, policies)
    # this is the G * prob method, where prob is cascaded from node 0 to leaves
    
    # create matrices to hold results
    G = Matrix{Union{Missing, Float64}}(undef, length(policies), agent.policy_len)
    utility = Matrix{Union{Missing, Float64}}(undef, length(policies), agent.policy_len)
    info_gain = Matrix{Union{Missing, Float64}}(undef, length(policies), agent.policy_len)
    risk = Matrix{Union{Missing, Float64}}(undef, length(policies), agent.policy_len)
    ambiguity = Matrix{Union{Missing, Float64}}(undef, length(policies), agent.policy_len)

    println("\nEnumerate policies ----")
    for (i_policy, policy) in enumerate(policies)
        subpolicies = [Tuple(policy[1:n]) for n in 1:length(policy)]  # e.g., [(1,), (1,2), (1,2,2)]
        vdic = Dict()  # dict of ActionNodes and ObsNodes per subpolicy
        vlist = []  # list of all ActionNodes in whole policy
        for sp in subpolicies
            vdic[sp] = []
        end

        # iterate over all nodes in graph to find those that match subpolicies
        for node in MGN.labels(Graph)
            # delete any calculated G values
            if isa(Graph[node], ObsNode)
                Graph[node].utility_updated = 0
                Graph[node].info_gain_updated = 0
                Graph[node].ambiguity_updated = 0
                Graph[node].risk_updated = 0
                Graph[node].G_updated = 0
                Graph[node].q_pi_updated = 0
                Graph[node].prob_updated = nothing
            else
                @assert isa(Graph[node], ActionNode)
                Graph[node].utility_updated = 0
                Graph[node].info_gain_updated = 0
                Graph[node].ambiguity_updated = 0
                Graph[node].risk_updated = 0
                Graph[node].G_updated = 0
                Graph[node].q_pi_updated = 0
            end

            # collect ActionNodes in subpolicies (i.e., in graph levels)
            for sp in subpolicies
                sp_action = sp[end]
                if Graph[node].subpolicy == sp 
                    if (
                        isa(Graph[node], ActionNode)  # start calculations with action
                        &&
                        Graph[node].action == sp_action  # for G*prob method, only need action nodes of exact policy 
                        )
                        push!(vdic[sp], node)
                        push!(vlist, node)
                    end
                end
            end

            #if isa(Graph[node], ObsNode) && Graph[node].ith_observation > 1
                # curious how many ObsNodes an ActionNode might have?
                #printfmtln("    trimmed graph subpolicy= {}, cnt= {}", Graph[node].subpolicy, Graph[node].ith_observation)
            #end           
        end
        #printfmtln("\nsubpolicies={}, vlist size = {}", [x for x in subpolicies], length(vlist))
    
        for sp in reverse(subpolicies)
            #printfmtln("\npolicy= {}, sp= {}", policy, sp)

            level = length(sp)
            G[i_policy, level] = 0.0
            utility[i_policy, level] = 0.0
            info_gain[i_policy, level] = 0.0
            risk[i_policy, level] = 0.0
            ambiguity[i_policy, level] = 0.0
            
            nodes =  vdic[sp]  # action nodes in subpolicy
            siblings_done = Set()
            for node in nodes
                
                if node in siblings_done
                    continue
                end

                # get ObsNode parent
                parent_node = collect(MGN.inneighbor_labels(Graph, node))
                @assert length(parent_node) == 1
                parent_node = parent_node[1]
                @assert isa(Graph[parent_node], ObsNode)

                # get all ActionNode siblings, some of which are not in subpolicy
                siblings = collect(MGN.outneighbor_labels(Graph, parent_node))
                push!(siblings_done, siblings...)

                # get ObsNode children
                children = collect(MGN.outneighbor_labels(Graph, node))
                
                if length(children) == 0
                    # this is the last action in a policy, convert G etc. for this action node to G_updated etc.
                    Graph[node].utility_updated = Graph[node].utility
                    Graph[node].info_gain_updated = Graph[node].info_gain
                    Graph[node].risk_updated = Graph[node].risk
                    Graph[node].ambiguity_updated = Graph[node].ambiguity
                    Graph[node].G_updated = Graph[node].G
                    
                    # cascade prob_updated on ObsNodes starting from root to leaf
                    node_id = MGN.code_for(Graph, node)
                    #path = Graphs.a_star(Graph, 1, node_id)

                    parents = recurse_parents(node, Graph, [])
            
                    #=
                    todo: the conversion from node to code and then edge is unnecessary and extra work. Its 
                    only done to match the result of Graphs.a_star. Keeping as nodes will eliminate the need
                    later for calls to MGN.label_for. 
                    =#
                    parents = [(MGN.code_for(Graph, parents[j+1]), MGN.code_for(Graph, parents[j])) for j in 1:length(parents)-1]
                    path = Graphs.Edge.(reverse(parents))

                    prob = 1
                    for edge in path
                        label = MGN.label_for(Graph, edge.src)
                        if isa(Graph[label], ObsNode)
                            if !isnothing(Graph[label].prob_updated)
                                # prob for this node has already been updated
                                @assert Graph[label].prob_updated ==  prob * Graph[label].prob
                                prob = Graph[label].prob_updated
                            else 
                                prob *= Graph[label].prob
                                Graph[label].prob_updated = prob
                            end
                        end
                    end
                end
                
                # first update node from child, if any children
                #for child in children
                #    @infiltrate; @assert false    
                #    # sum G etc. over children (ObsNodes)
                #    Graph[node].G_updated += Graph[child].G_updated
                #    Graph[node].utility_updated += Graph[child].utility_updated
                #    Graph[node].info_gain_updated += Graph[child].info_gain_updated
                #    Graph[node].risk_updated += Graph[child].risk_updated
                #    Graph[node].ambiguity_updated += Graph[child].ambiguity_updated
                #end
                    
                # update parent
                prob = Graph[parent_node].prob_updated
                #@infiltrate; @assert false    
                Graph[parent_node].G_updated = prob * Graph[node].G
                Graph[parent_node].utility_updated = prob * Graph[node].utility
                Graph[parent_node].info_gain_updated = prob * Graph[node].info_gain
                Graph[parent_node].risk_updated = prob * Graph[node].risk
                Graph[parent_node].ambiguity_updated = prob * Graph[node].ambiguity
                    
                # record results
                G[i_policy, level] += Graph[parent_node].G_updated
                utility[i_policy, level] += Graph[parent_node].utility_updated
                info_gain[i_policy, level] += Graph[parent_node].info_gain_updated
                risk[i_policy, level] += Graph[parent_node].risk_updated
                ambiguity[i_policy, level] += Graph[parent_node].ambiguity_updated

            end

            
        end
    end

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


function do_G_prob_qpi_method(Graph, agent, policies)
    # this is the G * prob method, where prob is cascaded from node 0 to leaves
    
    # create matrices to hold results
    G = Matrix{Union{Missing, Float64}}(undef, length(policies), agent.policy_len)
    utility = Matrix{Union{Missing, Float64}}(undef, length(policies), agent.policy_len)
    info_gain = Matrix{Union{Missing, Float64}}(undef, length(policies), agent.policy_len)
    risk = Matrix{Union{Missing, Float64}}(undef, length(policies), agent.policy_len)
    ambiguity = Matrix{Union{Missing, Float64}}(undef, length(policies), agent.policy_len)

    println("\nEnumerate policies ----")
    for (i_policy, policy) in enumerate(policies)
        subpolicies = [Tuple(policy[1:n]) for n in 1:length(policy)]  # e.g., [(1,), (1,2), (1,2,2)]
        vdic = Dict()  # dict of ActionNodes and ObsNodes per subpolicy
        vlist = []  # list of all ActionNodes in whole policy
        for sp in subpolicies
            vdic[sp] = []
        end

        # iterate over all nodes in graph to find those that match subpolicies
        for node in MGN.labels(Graph)
            # delete any calculated G values
            if isa(Graph[node], ObsNode)
                Graph[node].utility_updated = 0
                Graph[node].info_gain_updated = 0
                Graph[node].ambiguity_updated = 0
                Graph[node].risk_updated = 0
                Graph[node].G_updated = 0
                Graph[node].q_pi_updated = 0
                Graph[node].prob_updated = nothing
            else
                @assert isa(Graph[node], ActionNode)
                Graph[node].utility_updated = 0
                Graph[node].info_gain_updated = 0
                Graph[node].ambiguity_updated = 0
                Graph[node].risk_updated = 0
                Graph[node].G_updated = 0
                Graph[node].q_pi_updated = 0
            end

            # collect ActionNodes in subpolicies (i.e., in graph levels)
            for sp in subpolicies
                sp_action = sp[end]
                if Graph[node].subpolicy == sp 
                    if (
                        isa(Graph[node], ActionNode)  # start calculations with action
                        &&
                        Graph[node].action == sp_action  # for G*prob method, only need action nodes of exact policy 
                        )
                        push!(vdic[sp], node)
                        push!(vlist, node)
                    end
                end
            end

            if isa(Graph[node], ObsNode) && Graph[node].ith_observation > 1
                # curious how many ObsNodes an ActionNode might have?
                #printfmtln("    trimmed graph subpolicy= {}, cnt= {}", Graph[node].subpolicy, Graph[node].ith_observation)
            end           
        end
        #printfmtln("\nsubpolicies={}, vlist size = {}", [x for x in subpolicies], length(vlist))
    
        for sp in reverse(subpolicies)
            #printfmtln("\npolicy= {}, sp= {}", policy, sp)

            level = length(sp)
            G[i_policy, level] = 0.0
            utility[i_policy, level] = 0.0
            info_gain[i_policy, level] = 0.0
            risk[i_policy, level] = 0.0
            ambiguity[i_policy, level] = 0.0
            
            nodes =  vdic[sp]  # action nodes in subpolicy
            siblings_done = Set()
            for node in nodes
                
                if node in siblings_done
                    continue
                end

                # get ObsNode parent
                parent_node = collect(MGN.inneighbor_labels(Graph, node))
                @assert length(parent_node) == 1
                parent_node = parent_node[1]
                @assert isa(Graph[parent_node], ObsNode)

                # get all ActionNode siblings, some of which are not in subpolicy
                siblings = collect(MGN.outneighbor_labels(Graph, parent_node))
                push!(siblings_done, siblings...)

                # get ObsNode children
                children = collect(MGN.outneighbor_labels(Graph, node))
                
                if length(children) == 0
                    # this is the last action in a policy, convert G etc. to G_updated etc.
                    for sibling in siblings
                        Graph[sibling].utility_updated = Graph[sibling].utility
                        Graph[sibling].info_gain_updated = Graph[sibling].info_gain
                        Graph[sibling].risk_updated = Graph[sibling].risk
                        Graph[sibling].ambiguity_updated = Graph[sibling].ambiguity
                        Graph[sibling].G_updated = Graph[sibling].G
                    end
                end
                
                # first update node from child, if any children
                for child in children
                    @infiltrate; @assert false "todo: fully implement this method"

                    # sum G etc. over children (ObsNodes)
                    Graph[node].G_updated += Graph[child].G_updated
                    Graph[node].utility_updated += Graph[child].utility_updated
                    Graph[node].info_gain_updated += Graph[child].info_gain_updated
                    Graph[node].risk_updated += Graph[child].risk_updated
                    Graph[node].ambiguity_updated += Graph[child].ambiguity_updated
                end
                    
                # update parent
                prob = Graph[parent_node].prob
                sibling_Gs = [Graph[sibling].G_updated for sibling in siblings]
                sibling_q_pi = LEF.softmax(sibling_Gs .* agent.gamma)  
                
                #@infiltrate; @assert false    
                Graph[parent_node].G_updated = prob * Graph[node].G
                Graph[parent_node].utility_updated = prob * Graph[node].utility
                Graph[parent_node].info_gain_updated = prob * Graph[node].info_gain
                Graph[parent_node].risk_updated = prob * Graph[node].risk
                Graph[parent_node].ambiguity_updated = prob * Graph[node].ambiguity
                    
                # record results
                G[i_policy, level] += Graph[parent_node].G_updated
                utility[i_policy, level] += Graph[parent_node].utility_updated
                info_gain[i_policy, level] += Graph[parent_node].info_gain_updated
                risk[i_policy, level] += Graph[parent_node].risk_updated
                ambiguity[i_policy, level] += Graph[parent_node].ambiguity_updated

            end

            if false
                # average G etc. over all nodes in this level (??) If not, last layer will have high G etc.
                G[i_policy, level] = G[i_policy, level] / length(nodes)
                utility[i_policy, level] = utility[i_policy, level] / length(nodes)
                info_gain[i_policy, level] = info_gain[i_policy, level] / length(nodes)
                risk[i_policy, level] = risk[i_policy, level] / length(nodes)
                ambiguity[i_policy, level] += Graph[node].ambiguity / length(nodes)
            end
        end
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
function do_marginal_EFE_method(Graph, agent, policies)
    # this is the G * prob * qpi marginal EFE method (as per pymdp)
    @infiltrate; @assert false  # not yet implemented

end


# --------------------------------------------------------------------------------------------------
function recurse(Graph, agent, level, ObsLabel)
    
    observation_prune_threshold = 1/100  #1/16
    policy_prune_threshold = 1/16
    prune_penalty = 512
       
    qs = Graph[ObsLabel].qs_next
    observation = Graph[ObsLabel].observation

    if agent.verbose
        printfmtln("\n------\nlevel={}, observation={}", level, observation)
    end

    # to keep simple, use only one action here, todo: allow multiple actions
    children = []
    for (idx, action) in enumerate(1:agent.metamodel.action_dims[:move][1])
        # action is integer
        
        subpolicy = tuple([x.action for x in ObsLabel if !isnothing(x.action)]..., action) 
        
        # is this a valid subpolicy?
        found = false
        for policy in agent.policies.policy_iterator
            if subpolicy == policy[1][1:length(subpolicy)]
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
        if ActionLabel in collect(MGN.labels(Graph))
            # the label should not exist yet
            @infiltrate; @assert false
        end

        if !found
            # invalid policy
            Graph[(ActionLabel)] = BadPath("BadPath")
                        
            # this edge will be unique
            Graph[ObsLabel, ActionLabel] = GraphEdge()
            continue
        end
        
        qs_pi = get_expected_states(qs, agent, (action,), policy_full = subpolicy)  # assume one action only
        #@infiltrate; @assert false

        if isnothing(qs_pi) 
            # bad policy for given states
            Graph[ActionLabel] = BadPath("BadPath")
            Graph[ObsLabel, ActionLabel] = GraphEdge()
            continue
        end

        if ismissing(qs_pi)
            # early stop, will be true for all actions
            Graph[ActionLabel] = EarlyStop("EarlyStop")
            Graph[ObsLabel, ActionLabel] = GraphEdge()
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
        
        # add to graph if not already present
        Graph[ActionLabel] = ActionNode(
            qs,
            qs_pi,
            qo_pi,
            utility,
            info_gain,
            ambiguity,
            risk,
            G,
            
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
        Graph[ObsLabel, ActionLabel] = GraphEdge()
        push!(children, ActionLabel)
        
    end 
    
    if length(children) == 0
        #e.g., loc=9, stay move indicates early stop
        #@infiltrate; @assert false
        return  # no change to graph
    end

    if level == agent.policy_len
        if agent.verbose
            println("returning terminal path")
        end
        return # just return the graph
    end
    
    #=
    We need a q_pi vector to potentially filter out unlikely actions, but some actions might have
    been skipped for BadPath or EarlyStop. For convienience, we base q_pi only on the valid actions.
    But this means magnitudes of q_pi can vary based on number of children. 
    Todo: adjust threshold based on number of children.
    =#
    n = length(children)
    G = zeros(n)
    for (ii, child) in enumerate(children)
        G[ii] = Graph[child].G
    end
    
    if any(ismissing.(G)) || any(isnothing.(G))
        @infiltrate; @assert false
    end
    
    q_pi = LEF.softmax(G * agent.gamma)
    
    # filter out low q_pi?
    good = []
    for (ii, child) in enumerate(children)
        if q_pi[ii] < policy_prune_threshold
            # we will just ignore/delete this child
            #@infiltrate; @assert false
            # penalize G[ichild]  
            #G[ii] -= prune_penalty
            MGN.delete!(Graph, child)
        else
            push!(good, ii)
        #    # update q_pi on child
        #    Graph[child].q_pi = q_pi[ii]  
        end
    end
    good_children = [children[ii] for ii in good]
    
    if length(good_children) == 0
        @infiltrate; @assert false
    end

    qo_pi_sizes = [1:x.size[1] for x in Graph[good_children[1]].qo_pi[1]]
    observation_iterator = Iterators.product(qo_pi_sizes...) # every possible combination of observations e.g., (23, 2, 1) for 3 observations
    

    for (idx, ActionLabel) in enumerate(good_children)
        
        qo_next = Graph[ActionLabel].qo_pi[1]
        skip_observations = true
        
        # do standard inference?
        if agent.use_SI_graph_for_standard_inference
            # test without belief updates due to likely observations
            # use this for non-sophisticated (standard) inference
            skip_observations = false
            
            # make Obs node and link to parent, overwrite ObsLabel, make only a single Obs node
            ObsLabel = copy(ActionLabel)
            ObsLabel[end] = Label(level, observation, Graph[ActionLabel].action, "Obs")
            
            if ObsLabel in collect(MGN.labels(Graph))
                # the label should not exist yet
                @infiltrate; @assert false
            end
            
            # do not call update_posterior_states to get qs_next, use original
            qs_next = Graph[ActionLabel].qs_pi[1]  
            prob = 1.0
            cnt = 1
            observation = nothing  # this is a dummy ObsNode

            Graph[ObsLabel] = ObsNode(
                qs_next,
                nothing,  # utility_updated
                nothing,  # info_gain_updated
                nothing,  # ambiguity_updated
                nothing,  # risk_updated
                nothing,  # G_updated
                nothing,  # q_pi_updated
                prob,
                nothing,  # prob_updated
                Graph[ActionLabel].subpolicy,
                observation,
                level,
                cnt
            )

            Graph[ActionLabel, ObsLabel] = GraphEdge()
            
            if agent.verbose
                printfmtln("\n        level= {}, calling sophisticated", level)
            end
            #@infiltrate; @assert false
            recurse(Graph, agent, level+1, ObsLabel)
            
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
                prior = Graph[ActionLabel].qs_pi[1], 
                num_iter = agent.FPI_num_iter, 
                dF_tol = agent.FPI_dF_tol
            )
        
            # make Obs node and link to parent, overwrite ObsLabel
            # ObsLabel = vcat(ActionLabel, Label(level, observation, Graph[ActionLabel].action, "Obs"))  # longer
            ObsLabel = copy(ActionLabel)
            ObsLabel[end] = Label(level, observation, Graph[ActionLabel].action, "Obs")
            
            if ObsLabel in collect(MGN.labels(Graph))
                # the label should not exist yet
                @infiltrate; @assert false
            end
        
            Graph[ObsLabel] = ObsNode(
                qs_next,
                nothing,  # utility_updated
                nothing,  # info_gain_updated
                nothing,  # ambiguity_updated
                nothing,  # risk_updated
                nothing,  # G_updated
                nothing,  # q_pi_updated
                prob,
                nothing,  # prob_updated
                Graph[ActionLabel].subpolicy,
                observation,
                level,
                cnt
            )

            Graph[ActionLabel, ObsLabel] = GraphEdge()
            
            if agent.verbose
                printfmtln("\n        level= {}, calling sophisticated", level)
            end
            #@infiltrate; @assert false
            recurse(Graph, agent, level+1, ObsLabel)
            
            #@infiltrate; @assert false
            
            #G_weighted = np.dot(q_pi_next, G_next) * prob
            #G[idx] += G_weighted
        end

        if agent.verbose
            printfmtln("        level= {}, ending observation loop, idx={}, obs cnt= {}", level, idx, cnt)            
        end
        if cnt > 1 && agent.verbose
            printfmtln("\nsubpolicy= {}", Graph[ActionLabel].subpolicy)
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