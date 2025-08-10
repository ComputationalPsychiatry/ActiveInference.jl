
# @infiltrate; @assert false

#using Format
#using Infiltrator
#using Revise


# --------------------------------------------------------------------------------------------------
function make_iterator(model, CONFIG, policy_length)
    action_dims = [length(x.values) for x in model.actions]
    
    products = [Iterators.product(
        repeat([1:n[1]], policy_length)...) for n in action_dims
    ]
        
    if policy_length == 1
        products = (x1[1] for x1 in Iterators.product(products...))
    else
        products = (x1 for x1 in Iterators.product(products...))
    end
     
    policy_iterator = products
    

    policy_iterator = Tuple(policy_iterator)
    
    #@infiltrate; @assert false
    return policy_iterator
end



# --------------------------------------------------------------------------------------------------
function make_policies(model, CONFIG)
    
    policy_iterator = make_iterator(model, CONFIG, CONFIG.policy_length)
    @assert isa(policy_iterator, NTuple{N1, NTuple{N2, NTuple{N3, Int64}}} where {N1,N2,N3})
    
    action_iterator = make_iterator(model, CONFIG, 1)
    @assert isa(action_iterator, NTuple{N1,NTuple{N2, Int64}} where {N1,N2})
    
    number_policies = length(policy_iterator)
    #printfmtln("\nnumber of policies= {}\n", number_policies)
    

    function action_tests(qs_pi_next, qs_pi_last, model)
        idx = argmax(qs_pi_next.loc)
        idx_last = argmax(qs_pi_last.loc)
        if vec(model.states.loc.extra.MAZE)[idx] == 1 && qs_pi_next.loc[idx] > .95  
            # action has taken agent off the paths
            return false
        elseif (qs_pi_next.loc[idx] > .95) && (qs_pi_last.loc[idx_last] > .95) && (qs_pi_next.loc[idx] == qs_pi_last.loc[idx_last])
            # sure about locations, but move made no change (hit a wall)
            # todo: account for end of maze prize hit before last policy action
            return false
        else
            return true
        end
    end


    model = @set model.policies.policy_iterator = policy_iterator
    model = @set model.policies.action_iterator = action_iterator
    model = @set model.policies.n_policies = number_policies
    #model = @set model.policies.action_tests = action_tests


    #@infiltrate; @assert false
    return model
    
end