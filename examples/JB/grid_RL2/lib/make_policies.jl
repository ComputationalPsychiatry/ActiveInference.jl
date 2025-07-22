
# include("./grid.jl")
#show(stdout, "text/plain", x)
# @infiltrate; @assert false

include("./structs.jl")

#using Format
#using Infiltrator
#using Revise


# --------------------------------------------------------------------------------------------------
function policy_filter(policy, null_action_ids)
    
    
    # if two actions, at least one must be a null action (can't move diagonal)
    if length(policy) == 2 && isa(policy[1], Tuple)
        for i in 1:length(policy[1])
            if !(policy[1][i] == null_action_ids[1] || policy[2][i] == null_action_ids[2])
                return false 
            end
        end
    end
    
    if length(policy) == 2 && isa(policy[1], Int64)
        if !(policy[1] == null_action_ids[1] || policy[2] == null_action_ids[2])
            return false 
        end
    end

    # another example filter is not starting a policy with all null actions
    
    #@infiltrate; @assert false
    return true
    
end



# --------------------------------------------------------------------------------------------------
function make_iterator(model, CONFIG, policy_length)
    action_dims = [length(x.values) for x in model.actions]
    
    products = [Iterators.product(
        repeat([1:n[1]], policy_length)...) for n in action_dims
    ]
        
    
    null_actions = [x.null_action for x in model.actions]
    null_action_ids = [findfirst(x -> x == null_actions[i], x.labels) for (i,x) in enumerate(model.actions)]
    
    if policy_length == 1
        if length(model.actions) == 2
            products = ((x1[1][1], x1[2][1]) for x1 in Iterators.product(products...))
        else
            products = (x1[1] for x1 in Iterators.product(products...))
        end
    else
        products = (x1 for x1 in Iterators.product(products...))
    end
     
    if CONFIG.use_filtering || length(model.actions) == 2
        policy_iterator = []
        for p in products
            if CONFIG.use_filtering || length(model.actions) == 2
                if policy_filter(p, null_action_ids)
                    push!(policy_iterator, p)
                end
            end    
        end
    else
        policy_iterator = products
    end

    policy_iterator = Tuple(policy_iterator)
    return policy_iterator
end



# --------------------------------------------------------------------------------------------------
function make_policies(model, CONFIG, env)
    
    policy_iterator = make_iterator(model, CONFIG, CONFIG.policy_length)
    @assert isa(policy_iterator, NTuple{N1, NTuple{N2, NTuple{N3, Int64}}} where {N1,N2,N3})
    
    action_iterator = make_iterator(model, CONFIG, 1)
    @assert isa(action_iterator, NTuple{N1,NTuple{N2, Int64}} where {N1,N2})
    
    number_policies = length(policy_iterator)
    #printfmtln("\nnumber of policies= {}\n", number_policies)
    
    model = @set model.policies.policy_iterator = policy_iterator
    model = @set model.policies.action_iterator = action_iterator
    model = @set model.policies.n_policies = number_policies


    # policy functions
    function earlystop_tests(qs, model)
        loc = model.states.loc.labels[argmax(qs.loc)]  
        if loc in model.states.loc.extra.stop_locations
            return false
        end
        
        # more tests here
        return true
    end


    function action_tests(qs_pi, model, env)
        # qs_pi, after some action is considered
        # this is only good if agent knows its states and knowledge is correct

        # could instead test for prob > x of being in a prohibited cell
        new_cell = model.states.loc.labels[argmax(qs_pi.loc)]  
        
        if false && any(new_cell .< [1,1]) || any(new_cell .> [env.nY, env.nX])
            return false
        end
        
        # more tests here
        return true
    end

    action_tests(qs_pi, model) = action_tests(qs_pi, model, env)

    model = @set model.policies.action_tests = action_tests
    model = @set model.policies.earlystop_tests = earlystop_tests

    #@infiltrate; @assert false
    return model
    
end