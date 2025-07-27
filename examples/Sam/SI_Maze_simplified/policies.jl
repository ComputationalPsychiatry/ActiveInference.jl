function policy_tests(multiaction_policy, istay)
    policy1 =  multiaction_policy[1]
    if (
        policy1[1] == istay  # initial option is stay
        #||
        #any([policy1[ii] == istay == policy1[ii+1] for ii in 1:length(policy1)-1])  # stay repeats
        )
        return false
    else
        return true
    end
end

function move_stop(
    qs::Vector{Vector{Float64}}, 
    cells::Vector{Tuple{Int64, Int64}}, 
    stop_locations::Vector{Tuple{Int64, Int64}}
    )
    return true
end

function up_context(
    qs::Vector{Vector{Float64}}, 
    cells::Vector{Tuple{Int64, Int64}}, 
    )
    return true
end


function down_context(
    qs::Vector{Vector{Float64}}, 
    cells::Vector{Tuple{Int64, Int64}}, 
    nrows::Int64
    )
    return true
end


function left_context(
    qs::Vector{Vector{Float64}}, 
    cells::Vector{Tuple{Int64, Int64}}, 
    )
    return true
end


function right_context(
    qs::Vector{Vector{Float64}}, 
    cells::Vector{Tuple{Int64, Int64}}, 
    ncols::Int64
    )
    return true
end
# --------------------------------------------------------------------------------------------------
    
function make_policies(policy_len::Int)
    istay = findfirst(x -> x == :STAY, action_deps.move)

    # Create policy iterator based on policy length
    products = [Iterators.product(
        repeat([1:n[1]], policy_len)...) for n in collect(action_dims)
    ]
    policy_iterator = Iterators.product(products...)

    # Define contextual helper functions
    move_stop2(qs::Vector{Vector{Float64}}) = move_stop(
        qs,
        [(j, i) for i in 1:(9,9)[1] for j in 1:(9,9)[2]], 
        [(9,9)]
    )
    
    up_context2(qs::Vector{Vector{Float64}}) = up_context(
        qs,
        [(j, i) for i in 1:(9,9)[1] for j in 1:(9,9)[2]]
    )

    right_context2(qs::Vector{Vector{Float64}}) = right_context(
        qs,
        [(j, i) for i in 1:(9,9)[1] for j in 1:(9,9)[2]],
        (9,9)[2]
    )

    # Define action contexts
    action_contexts = Dict(
        :move => Dict(
            :policy_context => x -> true,  # is policy legal?
            :option_context => Dict(
                :UP => up_context2, 
                :RIGHT => right_context2, 

                ),
            :null_action => :STAY,
            :stopfx => move_stop2
        )
    )

    # Count policies
    number_policies = 0
    for _ in policy_iterator 
        number_policies += 1
    end
    println("\nnumber of policies= {}", number_policies)


    # Return Policies object
    Policies(
        policy_iterator,
        action_contexts,
        number_policies
    )
end
