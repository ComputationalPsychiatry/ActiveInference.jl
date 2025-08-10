
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
    
    model = @set model.policies.policy_iterator = policy_iterator
    model = @set model.policies.action_iterator = action_iterator
    model = @set model.policies.n_policies = number_policies


    #@infiltrate; @assert false
    return model
    
end