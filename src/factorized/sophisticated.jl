
module Sophisticated


import LogExpFunctions as LEF
import MetaGraphsNext as MGN
#import MetaGraphs as MG
import Graphs
import Statistics: maximum, minimum, sum
import Dates
import Random
import UUIDs

import ActiveInference.ActiveInferenceFactorized as AI 

using Format
using Infiltrator
#using Revise


# @infiltrate; @assert false


""" -------- Inference Functions -------- """

#### State Inference #### 


# --------------------------------------------------------------------------------------------------
function recurse_parents(node_label::Base.UUID, Graph, path::Vector{Base.UUID})
    # find all parents from leaf to root in tree graph, starting at leaf
    # returns list of node_labels
    push!(path, node_label)
    parents = collect(MGN.inneighbor_labels(Graph, node_label))
    if parents.size[1] == 1
        # should only be one parent
        #printfmtln("node_label= \n{}\n", node_label[1])
        recurse_parents(parents[1], Graph, path)
    elseif parents.size[1] == 0
        return path
    else
        @infiltrate; @assert false
    end
end


#=
Important info from Graphs.jl:
Note that [the vertes ID (integer values)] is NOT persistent if vertices added earlier are removed. 
When rem_vertex!(g, v) is called, v is "switched" with the last vertex before being deleted. 
As edges are identified by vertex indices, one has to be careful with edges as well. 

This means that any list of deletions has to be managed via labels. For each new deletion the current
vertex ID has to be found via its label!

Also, it appears that deleting a node will delete any outgoing edges, and whatever nodes they connect
to that have one or zero out edges. 
=#


# --------------------------------------------------------------------------------------------------
#### Policy Inference #### 
""" Update Posterior over Policies """
function update_posterior_policies!(
        agent::AI.Agent{T2}, 
        obs_current::NamedTuple{<:Any, <:NTuple{N, T} where {N,T}}
    ) where {T2<:AbstractFloat}
    
    #@infiltrate; @assert false
    
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

    t0 = Dates.time()

    if !isnothing(agent.settings.random_seed)
        rng = Random.Xoshiro(agent.settings.random_seed) 
    else
        rng = Random.Xoshiro()  
    end

    ObsParent = AI.ObsNode{T2}(
        deepcopy(qs_current), 
        repeat([nothing], 6)..., 
        T2.(1.0), 
        T2.(1.0), 
        obs_current, 
        1,
        nothing,
    )

    if agent.settings.graph == :explicit
        siGraph = MGN.MetaGraph(
            Graphs.DiGraph();  # underlying graph structure
            label_type = Base.UUID,  # partial or complete policy tuple
            vertex_data_type = Union{AI.ObsNode, AI.ActionNode, AI.EarlyStop, AI.BadPath}, 
            edge_data_type = AI.GraphEdge,  
            graph_data = "sophisticated inference graph",  # tag for the whole graph
        )
        ObsLabel = UUIDs.uuid1(rng) 
        siGraph[ObsLabel] = ObsParent
    else
        siGraph = nothing
        ObsLabel = nothing
    end
    
    #@infiltrate; @assert false
    G_grandchildren, q_pi_grandchildren = recurse(siGraph, ObsParent, agent, ObsLabel, action_names, rng)
    
    printfmtln("\nmake graph  time= {}\n", round((Dates.time() - t0) , digits=2))
    t0 = Dates.time()

    if agent.settings.graph == :implicit
        # there is no graph and EFE is over actions
        agent.q_pi_actions[:] = q_pi_grandchildren
        agent.G_actions[:] = G_grandchildren
        return
    end



    #@infiltrate; @assert false
    nv = Graphs.nv(siGraph) 
    push!(agent.history.graph_initial_node_count, nv)

    if nv == 0
        @infiltrate; @assert false    
    end 

    # Now that we have a graph, prune bad paths and dangling early stops and ObsNodes. Then 
    # accumulate/record G etc.
    
    # collect all dfs paths from node 0 to leaves
    paths = []
    max_level = 0
    for ii in Graphs.vertices(siGraph)
        
        node_label = MGN.label_for(siGraph, ii)
        if isa(siGraph[node_label], AI.ObsNode) 
            max_level = max(max_level, siGraph[node_label].level)
        end
        
        if ii > 1
            try
                @assert Graphs.indegree(siGraph, ii) == 1
                if isa(siGraph[node_label], AI.ActionNode)
                    @assert length(siGraph[node_label].policy[1]) == siGraph[node_label].level
                end
            catch e
                @infiltrate; @assert false
            end
        end
        
        if Graphs.outdegree(siGraph, ii) == 0
            # leaf
            #test = Graphs.a_star(siGraph, 1, ii)
            parents = recurse_parents(node_label, siGraph, Vector{Base.UUID}())
            
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
    
    if max_level < n_steps-1
        s = format("\nmax_level={} is < policy_length={}. Setting all G to zero.\n", max_level, n_steps)
        @warn s
        if !isnothing(agent.G_policies)
            agent.G_policies .= 0
            agent.q_pi_policies .= LEF.softmax(T2.(agent.G_policies), dims=1) 
        end
        agent.G_actions .= 0
        agent.q_pi_actions .= LEF.softmax(T2.(agent.G_actions), dims=1)
        return
           
    end
    
    # clean up dfs paths
    bad_node_labels = Set()
    for path in paths
        dst = MGN.label_for(siGraph, path[end].dst)  # leaf node_label
        
        if isa(siGraph[dst], AI.BadPath)
            # start at leaf, remove BadPath node and any parents in path that have only one out degree
            for edge in reverse(path)
                dst_ = MGN.label_for(siGraph, edge.dst)
                if Graphs.outdegree(siGraph, edge.dst) in [0, 1]
                    push!(bad_node_labels, dst_)
                else
                    break  # keep remaining path
                end
            end
            # delete all unwanted nodes in path 
            for node_label in bad_node_labels
                delete!(siGraph, node_label)  # also deletes a node's 'to' edges
            end
            bad_node_labels = Set()
        end

        if isa(siGraph[dst], AI.EarlyStop)
            #=
            Remove this EarlyStop node. Also remove its ObsNode parent. Do this because early stop 
            will be true for all actions associated with the ObsNode parent. This renders the parent
            of the ObsNode as the last ActionNode in the (shortened) policy.
            =#
            push!(bad_node_labels, dst)
            src = MGN.label_for(siGraph, path[end].src)  # parent
            @assert isa(siGraph[src], AI.ObsNode)
            push!(bad_node_labels, src)
            
            # delete all unwanted nodes in path 
            for node_label in bad_node_labels
                delete!(siGraph, node_label)  # also deletes a node's 'to' edges
            end
            bad_node_labels = Set()

        end
    end

    # handle pruned nodes and path
    bad_node_labels = Set()
    if !agent.settings.SI_use_pymdp_methods
        for ii in Graphs.vertices(siGraph)
            node_label = MGN.label_for(siGraph, ii)
            if isa(siGraph[node_label], AI.ActionNode)
                
                if siGraph[node_label].pruned
                    push!(bad_node_labels, node_label)
                    
                    parents = recurse_parents(node_label, siGraph, Vector{Base.UUID}())
                    #=
                    todo: the conversion from node_label to code and then edge is unnecessary and extra work. Its 
                    only done to match the result of Graphs.a_star. Keeping as node_labels will eliminate the need
                    later for calls to MGN.label_for. 
                    =#
                    parents = [(MGN.code_for(siGraph, parents[j+1]), MGN.code_for(siGraph, parents[j])) for j in 1:length(parents)-1]
                    path = Graphs.Edge.(reverse(parents))
                    
                    # if an ActionNode is pruned, also remove any parents that have only one out degree
                    for edge in reverse(path)
                        dst_ = MGN.label_for(siGraph, edge.dst)
                        if Graphs.outdegree(siGraph, edge.dst) in [0, 1]
                            push!(bad_node_labels, dst_)
                        else
                            break # keep remaining path
                        end
                    end
                end
            end
        end

        # delete all unwanted nodes in path 
        if length(bad_node_labels) > 0
            printfmtln("\nlen bad node labels pruned = {}\n", length(bad_node_labels))
            #@infiltrate; @assert false  
        end
        for node_label in bad_node_labels
            delete!(siGraph, node_label)  # also deletes a node's 'to' edges
        end
        bad_node_labels = Set()
    end

    printfmtln("\ncleanup dfs paths time= {}, start nv= {}, final nv= {}\n", 
        round((Dates.time() - t0) , digits=2), nv, Graphs.nv(siGraph)
    )
    push!(agent.history.graph_final_node_count, Graphs.nv(siGraph))
    t0 = Dates.time()


    if Graphs.nv(siGraph) == 0
        @infiltrate; @assert false    
    end 

    #=
    Now all paths are valid and end in an actionNode (at arbitrary some level). We will need a list of
    all valid paths, including path nodes, but that list could be large. So instead, we collect here 
    only valid leaves for a unique policy. Later we will examine each policy separately.
    =#
    leaves = Dict{NamedTuple, Vector}()  # todo: make the type exact.  todo: preinitialize each vector to some large size, for memory management
    max_level = 0
    min_level = n_steps + 1
    cnt_action_nodes = 0
    for ii in Graphs.vertices(siGraph)
        
        # todo: this try-catch is for testing, take it out
        node_label = MGN.label_for(siGraph, ii)
        if isa(siGraph[node_label], AI.ActionNode)
            cnt_action_nodes += 1
        end
        if ii > 1
            try
                @assert Graphs.indegree(siGraph, ii) == 1
                if isa(siGraph[node_label], AI.ActionNode)
                    @assert length(siGraph[node_label].policy[1]) == siGraph[node_label].level
                end
            catch e
                @infiltrate; @assert false
            end
        end
        
        if Graphs.outdegree(siGraph, ii) == 0
            # leaf
            @assert isa(siGraph[node_label], AI.ActionNode)  
            policy = siGraph[node_label].policy
            if policy in keys(leaves)
                push!(leaves[policy], node_label)
            else
                leaves[policy] = [node_label]
            end
            max_level = max(max_level, length(policy[1]))
            min_level = min(min_level, length(policy[1]))
        end
    end
    
    if length(leaves) == 0
        @infiltrate; @assert false    
    end 

    printfmtln("\nget leaves list time= {}, final action nodes={}, min_level= {}, max_level= {}\n", 
        round((Dates.time() - t0) , digits=2), cnt_action_nodes, min_level, max_level
    )
    push!(agent.history.graph_min_level, min_level)
    push!(agent.history.graph_max_level, max_level)
    
    t0 = Dates.time()

    if max_level < n_steps
        printfmtln("\nWarning: max level ({}) < n_steps ({})\n", max_level, n_steps)
    end
    
    # choose a postprocessing method
    if agent.settings.EFE_over == :actions
        # marginal EFE
        agent.q_pi_actions[:] = q_pi_grandchildren
        agent.G_actions[:] = G_grandchildren
    else
        do_EFE_over_policies(siGraph, agent, leaves, T2)
    end

    printfmtln("\ndo G time= {}\n", round((Dates.time() - t0) , digits=2))
    t0 = Dates.time()
    
    
    # @infiltrate; @assert false  

    return 
end 


# --------------------------------------------------------------------------------------------------
function do_EFE_over_policies(siGraph, agent, leaves, T2)
    #=
    This is the G * prob method, where prob is cascaded from node 0 to leaves. It calculates
    EFE over policies, not actions.
    =#
    
    if agent.settings.verbose
        println("\nEnumerate policies, do_EFE_over_policies ----")
    end

    leaves_keys = keys(leaves)
    action_names = [x.name for x in agent.model.actions]

    #=
    Consider each policy in graph; e.g., (1,2,2) over three policy steps. As an alternative, we could
    consider each policy in policy_iterator, but many of these might not appear in the graph.
    =#
    for policy in leaves_keys

        # todo: if policies are created as per a pattern, only a subset needs to be searched over to 
        # find the policy_i corresponding to the current policy.
        policy_i = findfirst(x -> x == values(policy), agent.model.policies.policy_iterator) 
        
        if isnothing(policy_i) 
            s = format("Error: isnothing(policy={}). If this fails, no valid instances of a policy exist. Try ", policy) 
            s *= "reducing prunning thresholds or set :SI_use_pymdp_methods to false."
            @warn s
            continue
        end
        #=
        todo: either fail the above assert as is or add null actions to fill up a subpolicy to make
        a full policy. For example, (move=(1,2),) --> (move=(1,2,5),) if 5 is the null action and
        policy len is three. This requires specification of a null action, which is currently not enforced.
        And it requires that all possible subpolicies ending in one or more null actions must be 
        added to policy iterator. The latter could greatly expand the size of the policy iterator. 
        =#

        paths = Vector{Vector{Base.UUID}}()
        done = Set()
        for leaf in leaves[policy]
            
            # last parent is top node, first parent in parents is leaf
            parents = recurse_parents(leaf, siGraph, Vector{Base.UUID}())  
            push!(paths, parents)
            
            # reset all updated values on this path to 0. EFE calculations are per path.
            for parent in parents
                if parent in done
                    # if a parent is in done with updating, so is its parent
                    break
                end
                if isa(siGraph[parent], AI.ObsNode)
                    siGraph[parent].utility_updated = 0
                    siGraph[parent].info_gain_updated = 0
                    siGraph[parent].ambiguity_updated = 0
                    siGraph[parent].risk_updated = 0
                    siGraph[parent].G_updated = 0
                    siGraph[parent].q_pi_updated = 0
                    siGraph[parent].prob_updated = nothing
                else
                    @assert isa(siGraph[parent], AI.ActionNode)
                    siGraph[parent].utility_updated = 0
                    siGraph[parent].info_gain_updated = 0
                    siGraph[parent].ambiguity_updated = 0
                    siGraph[parent].risk_updated = 0
                    siGraph[parent].G_updated = 0
                    siGraph[parent].q_pi_updated = 0
                end
                push!(done, parent)
            end

            # this is the last action node in this policy, convert G etc. to G_updated etc.
            siGraph[leaf].utility_updated = siGraph[leaf].utility
            siGraph[leaf].info_gain_updated = siGraph[leaf].info_gain
            siGraph[leaf].risk_updated = siGraph[leaf].risk
            siGraph[leaf].ambiguity_updated = siGraph[leaf].ambiguity
            siGraph[leaf].G_updated = siGraph[leaf].G  
            siGraph[leaf].q_pi_updated = siGraph[leaf].q_pi
        end

        #=
        Set updated probabilities on each ObsNode of a path, with prob=1 for node 1 and probabilities
        for later nodes a product of earlier probabilities. Suppose the obs nodes in a path of three 
        have prob = [1, .5, .6]. Then the updated probs for these nodes, from root to leaf, are 
        [1, .5, .3]. Thus, the probability of an ObsNode in a path depends on the probabilities of all 
        parent ObsNodes. 
        =# 
        done = Set()
        for path in paths
            obs_nodes = reverse(path[2:2:end])  # root ObsNode is last, first is ActionNode at last level 
            prob = 1
            for node in reverse(obs_nodes)  # root ObsNode first
                prob = prob * siGraph[node].prob 
                if node in done
                    continue  # todo: is it faster just to overwrite nodes that are already done?
                end
                siGraph[node].prob_updated = prob
                push!(done, node)
            end
        end

        # Now we make updates for the nodes in each path, level by level, starting with the leaf.
        for level in reverse(1: length(policy[1]))
            
            for path in paths  # path is leaf first, root last
                #todo: selecting current obs and action would be faster without using reverse
                obs_node = reverse(path)[level*2-1]  
                action_node = reverse(path)[level*2]

                # todo: if we are certain method is ok, this test could be skipped
                # get all ActionNode siblings, some of which are not in subpolicy
                siblings = collect(MGN.outneighbor_labels(siGraph, obs_node))
                q_pi = [siGraph[sibling].q_pi for sibling in siblings]
                @assert isapprox(sum(q_pi), 1)
                
                # update the ObsNode parent
                prob = siGraph[obs_node].prob_updated
                if isnothing(prob)
                    @infiltrate; @assert false 
                end
                
                if agent.settings.policy_postprocessing_method == :G_prob
                    siGraph[obs_node].G_updated = prob * siGraph[action_node].G  
                    siGraph[obs_node].utility_updated = prob * siGraph[action_node].utility
                    siGraph[obs_node].info_gain_updated = prob * siGraph[action_node].info_gain
                    siGraph[obs_node].risk_updated = prob * siGraph[action_node].risk
                    siGraph[obs_node].ambiguity_updated = prob * siGraph[action_node].ambiguity
                    
                elseif agent.settings.policy_postprocessing_method == :G_prob_q_pi
                    # unlike pymdp, which only sums over G, here we also keep track of utility, etc.
                    # so we also multiply utility etc. by q_pi
                    qpi = siGraph[action_node].q_pi
                    siGraph[obs_node].G_updated = prob * siGraph[action_node].G * qpi
                    siGraph[obs_node].utility_updated = prob * siGraph[action_node].utility * qpi
                    siGraph[obs_node].info_gain_updated = prob * siGraph[action_node].info_gain * qpi
                    siGraph[obs_node].risk_updated = prob * siGraph[action_node].risk * qpi
                    siGraph[obs_node].ambiguity_updated = prob * siGraph[action_node].ambiguity * qpi
                end
                   
                if ismissing(agent.utility[policy_i, level])
                    # G etc. are initially filled with missings. Now there is at least one value in 
                    # the graph for this level and policy, so initialize to zero to allow for addition.
                    agent.utility[policy_i, level] = 0.0
                    agent.info_gain[policy_i, level] = 0.0
                    agent.risk[policy_i, level] = 0.0
                    agent.ambiguity[policy_i, level] = 0.0
                end    
                
                # Sum G etc. values for this policy over paths, at each level. 
                agent.utility[policy_i, level] += siGraph[obs_node].utility_updated
                agent.info_gain[policy_i, level] += siGraph[obs_node].info_gain_updated
                agent.risk[policy_i, level] += siGraph[obs_node].risk_updated
                agent.ambiguity[policy_i, level] += siGraph[obs_node].ambiguity_updated
            end
        end
    end
    #@infiltrate; @assert false    


    if all(ismissing.(agent.utility))
        @warn "\nall(ismissing.(agent.utility)). Setting all G to zero.\n"
        agent.G_policies .= 0
        agent.q_pi_policies .= LEF.softmax(T2.(agent.G_policies), dims=1)  
        agent.G_actions .= 0
        agent.q_pi_actions .= LEF.softmax(T2.(agent.G_actions), dims=1)
        return
        #@infiltrate; @assert false    
    end


    #=
    We have now calculated G, utility, etc. over each step of every policy. But if there are early
    stops or if there are paths that have been filtered out, and as a result if there are no action
    nodes at all for a given level over all the paths of this policy, then some values in G etc.
    will be missing. In that case, we cannot use sum to reduce them to a single value.
    =#
    
    # find rows that do not have all missings (rows with all missings cannot be used in reduction functions)
    idx = findall(row -> !all(ismissing, row), eachrow(agent.utility))

    if agent.settings.EFE_reduction == :custom
        fx_utility = agent.model.policies.utility_reduction_fx
        fx_info_gain = agent.model.policies.info_gain_reduction_fx

        if isnothing(fx_utility) || ismissing(fx_utility) 
            @assert false
        elseif isnothing(fx_info_gain) || ismissing(fx_info_gain)
            @assert false
        end

        agent.G_policies[idx] = (
            fx_info_gain.(skipmissing.(eachrow(agent.info_gain[idx,:])))
            +
            fx_utility.(skipmissing.(eachrow(agent.utility[idx,:])))
        )
    
    elseif agent.settings.EFE_reduction == :min_max || any(ismissing.(agent.info_gain)) || any(ismissing.(agent.utility))
        
        if  agent.settings.EFE_reduction == :sum && (any(ismissing.(agent.info_gain)) || any(ismissing.(agent.utility)))
            printfmtln("\n!!! Warning: Info gain or utility has missing elements, ignoring :sum for EFE reduction. Using :min_max.\n")
        end
        try
            agent.G_policies[idx] = (
                maximum.(skipmissing.(eachrow(agent.info_gain[idx,:])))
                +
                (
                    minimum.(skipmissing.(eachrow(agent.utility[idx,:])))
                    +
                    maximum.(skipmissing.(eachrow(agent.utility[idx,:])))
                ) ./ 2.
            )
        catch e
            @infiltrate; @assert false 
        end
    else
        # use sum, as in pymdp
        agent.G_policies[idx] = (
            sum.(eachrow(agent.info_gain[idx,:])) 
            + 
            sum.(eachrow(agent.utility[idx,:]))
        )
    end

    # some utility can be missing. Only use good policies to calc q_pi
    idx = findall(x -> !ismissing(x), agent.G_policies)   
    
    if sum(skipmissing(agent.G_policies)) == 0
        #@infiltrate; @assert false  # All policies failed?
        @warn format("\nThe sum over G is zero.\n")
    end

    Eidx = agent.model.policies.E_policies[idx]
    lnE = AI.Maths.capped_log(Eidx)
    agent.q_pi_policies[idx] .= LEF.softmax(agent.G_policies[idx] * agent.parameters.gamma + lnE, dims=1)  


    #if agent.history.sim_step == 2
    #    @infiltrate; @assert false  
    #end  

    #@infiltrate; @assert false    
    
    return 
end


# --------------------------------------------------------------------------------------------------
function recurse(
        siGraph::Union{Nothing, MGN.MetaGraph}, 
        ObsParent::AI.ObsNode, 
        agent::AI.Agent{T2}, 
        ObsLabel::Union{Nothing, Base.UUID}, 
        action_names::Vector{Symbol}, 
        rng::Random.Xoshiro
    ) where {T2<:AbstractFloat}
    
    # todo: induction, info_gain_A, info_gain_B, info_gain_D are not yet handled

    model = agent.model
    observation_prune_threshold = agent.settings.SI_observation_prune_threshold
    policy_prune_threshold = agent.settings.SI_policy_prune_threshold
    prune_penalty = agent.settings.SI_prune_penalty

    level = ObsParent.level
       
    qs = ObsParent.qs_next
    observation = ObsParent.observation

    if agent.settings.verbose
        #@infiltrate; @assert false
        printfmtln("\n------\nlevel={}, observation={} \nprob= {} \nmax qs= {}\n", 
            level, 
            observation, 
            ObsParent.prob,
            [argmax(q) for q in qs]
        )
    end

    if isnothing(observation)
        @infiltrate; @assert false
    end

    children = []
        
    for (idx, action) in enumerate(model.policies.action_iterator)  
        # action is a tuple of integers
        policy1 = (; zip(action_names, Tuple.(action))...)  # a policy of length 1
        
        # make policy
        if isnothing(ObsParent.policy)
            policy = policy1
        else
            policy = ObsParent.policy
            policy = (; zip(keys(policy), Tuple.([vcat(policy[ii]..., act) for (ii,act) in enumerate(action)]))...)
        end
        
        # qs_pi is the qs_prior for what the node belives it will see after an action, before seeing the obs
        earlystop, filtered, qs_pi = AI.Inference.get_expected_states(qs, policy1, agent)  # use a policy of one action

        if !isnothing(siGraph)
            # add any nodes for earlystop or filtered
            if filtered
                # bad policy for given states
                ActionLabel = UUIDs.uuid1(rng) 
                siGraph[ActionLabel] = AI.BadPath("BadPath")
                siGraph[ObsLabel, ActionLabel] = AI.GraphEdge()
                #@infiltrate; @assert false
                continue
            end

            if earlystop
                # early stop, will be true for all actions
                ActionLabel = UUIDs.uuid1(rng) 
                siGraph[ActionLabel] = AI.EarlyStop("EarlyStop")
                siGraph[ObsLabel, ActionLabel] = AI.GraphEdge()
                #@infiltrate; @assert false  
                continue  
            end
        else
            # for an implicit graph there should be no early stops or bad paths
            if earlystop || filtered
                @infiltrate; @assert false #"Should be no earlystops or filtered paths."
            end
        end

        qo_pi = AI.Inference.get_expected_obs(qs_pi, agent)  
        
        G = 0
        if agent.settings.use_utility
            utility = AI.Inference.calc_expected_utility(qo_pi, agent)[1]
            G += utility
        end

        # Calculate expected information gain of states
        if agent.settings.use_states_info_gain
            info_gain, ambiguity = AI.Inference.calc_info_gain(qs_pi, qo_pi, agent)
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

        # todo: hack for now, just add parameter info gain to existing info gain. Later, these should be separated.
        if agent.settings.use_param_info_gain
            info_gain_B = AI.Inference.calc_pB_info_gain(agent, qs_pi, qs, policy1)[1]
            
            info_gain += info_gain_B
            G += info_gain_B
            #@infiltrate; @assert false
        end
        
        #if policy == (move=(1,2),) 
        #    @infiltrate; @assert false
        #end

        # add to graph 
        #@infiltrate; @assert false
        ActionChild = AI.ActionNode{T2}(
            qs,  # parent qs
            qs_pi[1], 
            qo_pi[1].qo, 
            utility, # scalar
            info_gain, # scalar
            ambiguity, # scalar
            risk, # scalar
            G, # scalar
            false,  # are childen pruned out due to prune penalty?
            nothing,  # q_pi
                                   
            nothing, # utility_updated
            nothing, # info_gain_updated
            nothing, # ambiguity_updated
            nothing, # risk_updated
            nothing, # G_updated
            nothing, # q_pi_updated
           
            observation,
            level,
            policy,
        )
        
        #printfmtln("level= {}, policy= {}", ActionChild.level, ActionChild.policy)
        @assert length(policy[1]) == level

        push!(children, ActionChild)  # children are ActionNodes

        #if agent.sim_step == 2 && level == 2 && policy == (move = (2, 1),)
        #    @infiltrate; @assert false
        #end        
        
    end 
    
    if isnothing(siGraph)
        # should be no bad paths, early stops, etc. for implicit graph
        @assert length(children) == length(model.policies.action_iterator)
    end  
    
    if length(children) == 0
        #@infiltrate; @assert false
        return  (G_children=[], q_pi_children=[])  # no children, no change to graph
    end


    #=
    We need a qs_pi vector across actions to potentially filter out unlikely actions, but some 
    actions might have been skipped for BadPath or EarlyStop. For convienience, we base qs_pi only 
    on the valid actions. But this means magnitudes of qs_pi over actions can vary based on number of 
    children. 
    Todo: adjust threshold based on number of children.

    Here we prune in a while loop, until no more children are pruned out. E.g., the first prune removes
    child_1, it is deleted, q_pi is recalculated, and now child_2 is below the threshold.
    =#
    
    for prune_round in 1:length(children)

        if agent.settings.SI_use_pymdp_methods && level == model.policies.policy_length 
            # do not do any pruning on the last level of the graph
            break
        end
        
        G_children = [child.G for child in children]
        
        # todo: add inducive cost to G (record inductive cost)

        q_pi_children = LEF.softmax(G_children * agent.parameters.gamma)  

        # prune out low q_pi_children
        pruned = false
        for (ii, child) in enumerate(children)
            if child.pruned 
                continue
            end
                        
            if q_pi_children[ii] < policy_prune_threshold
                pruned = true
                child.pruned = true  # record that child is pruned 
                child.G -= prune_penalty

                if agent.settings.verbose
                    
                    state_names = [x.name for x in agent.model.states]
                    printfmtln("---- prune round= {}: argmax(q_pi)= {}, max(q_pi)= {}, child policy= {}\n", 
                        prune_round,    
                        (; zip(state_names, [argmax(x) for x in q_pi_children[ii]])...),
                        (; zip(state_names, [max(x) for x in q_pi_children[ii]])...),
                        child.policy,
                    )
                end
                #@infiltrate; @assert false
            end
        end

        if all(G_children .<= -prune_penalty/2)
            @infiltrate; @assert false
        end

        if agent.settings.SI_use_pymdp_methods
            # use only one prune round
            break
        end
        
        if !pruned
            # all children are good, no more pruning needed
            break
        end
    end

    # record results for q_pi on viable ActionNodes
    G_children = [child.G for child in children]  # This is G over all viable children of ObsNode
    q_pi_children = LEF.softmax(G_children * agent.parameters.gamma)  # this is q_pi over all children, with G penalties 
    for (ii, child) in enumerate(children)
        child.q_pi = q_pi_children[ii]  # this is the action node's effective q_pi going forward
    end
    #@infiltrate; @assert false
    
    
    # there are no more changes to children, so we can store good children in graph if :explicit
    if !isnothing(siGraph)
        action_labels = []
        for child in children
            ActionLabel = UUIDs.uuid1(rng) 
            push!(action_labels, ActionLabel)  # keep record of node labels
            siGraph[ActionLabel] = child  # copy first?
        
            # edge will always be unique
            siGraph[ObsLabel, ActionLabel] = AI.GraphEdge()
        end
    end


    if level == model.policies.policy_length
        if agent.settings.verbose
            println("returning terminal path")
        end

        return (G_children=G_children, q_pi_children=q_pi_children)  # just return the (pruned) children
    end


    #=
    Make an observation iterator for new observations. It can be very large, and for that reason do 
    not pre-calculate and store it.
    =#
    qo_pi_sizes = [1:x.size[1] for x in children[1].qo_pi]
    observation_iterator = Iterators.product(qo_pi_sizes...) # every possible combination of observations e.g., (23, 2, 1) for 3 observations

    obs_names = [x.name for x in model.obs]
    
    for (idx, ActionChild) in enumerate(children)
        if ActionChild.pruned
            # make no ObsNode children for this ActionNode
            continue
        end

        # do standard inference?
        if agent.settings.policy_inference_method == :standard
            # use this for :standard inference with :explicit or :implicit graph
            # make only a single Obs node
            # do not call update_posterior_states to get qs_next, use original
            qs_next = ActionChild.qs_pi  
            prob = 1.0
            obs = (; zip(obs_names, zeros(length(obs_names)))...)  # dummy obs
            
            ObsParent2 = AI.ObsNode{T2}(
                qs_next,
                nothing,  # utility_updated
                nothing,  # info_gain_updated
                nothing,  # ambiguity_updated
                nothing,  # risk_updated
                nothing,  # G_updated
                nothing,  # q_pi_updated
                prob,
                nothing,  # prob_updated
                obs,  # dummy obs
                level +1 ,
                ActionChild.policy,
            )

            ObsLabel = UUIDs.uuid1(rng) 
            if !isnothing(siGraph)
                siGraph[ObsLabel] = ObsParent2
                siGraph[action_labels[idx], ObsLabel] = AI.GraphEdge()
            end

            if agent.settings.verbose
                printfmtln("\n        level= {}, calling sophisticated", level)
            end
            
            #@infiltrate; @assert false
            G_grandchildren, q_pi_grandchildren  = recurse(siGraph, ObsParent2, agent, ObsLabel, action_names, rng)
            G_weighted = sum(G_grandchildren .* q_pi_grandchildren) * prob
            G_children[idx] += G_weighted

            continue
        end


        # do regular SI
        qo_next = ActionChild.qo_pi
        skip_observations = true
        
        cnt = 0
        max_prob = 0
        for (i_observation, observation) in enumerate(observation_iterator)    
            n = length(observation)
            probs = zeros(n)
            for i in 1:length(observation)
                probs[i] = qo_next[i][observation[i]]
            end
            prob  = prod(vcat(1.0, probs))

            if prob > max_prob
                max_prob = prob
            end
            
            # ignore low probability observations in the search tree
            if prob < observation_prune_threshold
                #    printfmtln("        level= {}, observation= {}, probs= {}, prob= {}", 
                #    level, observation, round.(probs, sigdigits=3), round(prob, sigdigits=3)
                #)
                continue
            end

            skip_observations = false
            cnt += 1

            if agent.settings.verbose
                printfmtln("        level= {}, observation= {}, probs= {}, prob= {}", 
                    level, observation, round.(probs, sigdigits=3), round(prob, sigdigits=3)
                )
            end

            obs = (; zip(obs_names, observation)...)
            
            #=
            We want an updated qs, given qs_prior, after seeing a (potential) observation. Here, 
            for this obs node:
                qs_prior = ActionChild.qs_pi 
            =#
            qs_next = AI.Inference.update_posterior_states(
                ActionChild.qs_pi,
                obs, 
                agent
            )
                
            # make Obs node and link to parent, overwrite ObsLabel
            # ObsLabel = vcat(ActionLabel, Label(level, observation, siGraph[ActionLabel].action, "Obs"))  # longer
            ObsLabel = UUIDs.uuid1(rng) 

            ObsParent2 = AI.ObsNode{T2}(
                qs_next,
                nothing,  # utility_updated
                nothing,  # info_gain_updated
                nothing,  # ambiguity_updated
                nothing,  # risk_updated
                nothing,  # G_updated
                nothing,  # q_pi_updated
                prob,
                nothing,  # prob_updated
                obs,
                level +1 ,
                ActionChild.policy,
            )

            #if obs == (loc_obs = 17, wall_obs = 1) && ActionChild.policy == (move = (1,),)
            #    @infiltrate; @assert false
            #end
 
                        
            #printfmtln("    level= {}, policy= {}", ObsParent.level, ObsParent.policy)
            
            if !isnothing(siGraph)
                siGraph[ObsLabel] = ObsParent2
                siGraph[action_labels[idx], ObsLabel] = AI.GraphEdge()
            end
            
            if agent.settings.verbose
                printfmtln("\n        level= {}, calling sophisticated", level)
            end
        
            #@infiltrate; @assert false
            G_grandchildren, q_pi_grandchildren  = recurse(siGraph, ObsParent2, agent, ObsLabel, action_names, rng)
            G_weighted = sum(G_grandchildren .* q_pi_grandchildren) * prob
            G_children[idx] += G_weighted
        end

        if agent.settings.verbose
            printfmtln("        level= {}, ending observation loop, idx={}, obs cnt= {}", level, idx, cnt)            
        end
        if cnt > 1 && agent.settings.verbose
            printfmtln("\ncnt= {}, policy= {}", cnt, ActionChild.policy)
            #@infiltrate; @assert false
        end
        if skip_observations == true
            # all observation's were skipped; could be true in first sim steps if learning a B matrix
            printfmtln("\n---- Warning ---\nNo observations, not extending branch. level= {}, observation= {}, max prob over obs= {}\n-------\n", 
                level, observation, max_prob
            )
            #@infiltrate; @assert false
        end
        
        #@infiltrate; @assert false

    end
    if agent.settings.verbose
        printfmtln("    level = {}, ending idx loop", level)
    end

    # todo: no logE??
    q_pi_children = LEF.softmax(G_children * agent.parameters.gamma)
    return (G_children=G_children, q_pi_children=q_pi_children)  # return the (pruned) children after generating futher children
        
end


end  # -- module