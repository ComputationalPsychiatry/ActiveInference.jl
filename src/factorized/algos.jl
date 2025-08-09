

module Algos

import LogExpFunctions as LEF
import ActiveInference.ActiveInferenceFactorized as AI  #todo: is this OK?

using Format
using Infiltrator
#using Revise


#include("./utils/maths.jl")




#show(stdout, "text/plain", x)
# @infiltrate; @assert false


function all_marginal_log_likelihood(
        qs::NamedTuple{<:Any, <:NTuple{N, Vector{T2}} where {N}},
        LL::Matrix{T2}, 
        ii::Int64, 
        jj::Int64, 
        agent::AI.Agent{T2}
    ) where {T2<:AbstractFloat}

    # returns a new version of qs updated by marginalizing over all dependencies but dependency ii
    
    if ndims(LL) == 2 #&& LL.size[2] == 1)
        # nothing to do, this is just A[obs,:]
        return dropdims(LL, dims=1)
    end

    @infiltrate; @assert false  # ?? LL is e.g., 1x9 by design, never 9x1
    
    state = keys(metamodel.state_deps)[ii]
    obs_deps = metamodel.obs_deps[jj][2:end]
    dep_ids = [findfirst(x -> x == j, keys(metamodel.state_deps)) for j in obs_deps]
    
    A = copy(LL)
    
    # we want to skip dependency ii, so place ii first in A and in dep_ids
    if dep_ids[1] != ii 
        dep_ids2 = vcat([ii], findall(x -> x != ii, dep_ids))
        obs_ids = [findfirst(x -> x == j, dep_ids) for j in dep_ids2]
        A = permutedims(A, obs_ids)
        dep_ids = dep_ids2
    end

    deps = Vector{Vector{T2}}()
    for idep in reverse(dep_ids[2:end])   
        push!(deps, qs[idep])
    end
    #print(A.size)
    A = dot_product1(A, deps)
    #@infiltrate; @assert false
    if A.size != qs[ii].size
        @infiltrate; @assert false
    end

    return A
   

end


function run_factorized_fpi(
        log_prior::NamedTuple{<:Any, <:NTuple{N, Vector{T2}} where {N}}, 
        new_obs::NamedTuple{<:Any, <:NTuple{N, Int64} where {N}},
        agent::AI.Agent{T2},
    ) where {T2<:AbstractFloat}
    
    # qs_prior called log_prior here, as it will be converted to log
    """
    Run the fixed point iteration algorithm with sparse dependencies between factors and outcomes 
    """
    
    model = agent.model

    #=
    Step 1: Compute marginal log likelihoods for each factor.
    Likelihood[ii].ndim will be size ndims(A[ii]) -1 (i.e., equal to the number of factors).
    E.g., if A[ii] is (4,25,3) and prior is (25,), then intermediate result is (4,25,3) , which
    is summed over dim=1, so (4,25,3) --> (1,25,3) --> (25,3).
    =#
    
    obs_names = [x.name for x in model.obs]
    mLLs = [zeros(T2, (1, x.A_dims[2:end]...)) for x in model.obs]
    marginal_LLs = (; zip(obs_names, mLLs)...)

    #log_likelihoods = []
    for (ii, obs) in enumerate(agent.model.obs)
        obs_ii = AI.Maths.onehot(new_obs[ii], obs.A_dims[1], T2)
        dims = repeat([1], length(obs.A_dims))
        dims[1] = obs_ii.size[1]  # e.g., [9,1,1] for 3-dim A with 9 categories
        obs_ii = reshape(obs_ii, dims...)
        marginal_LLs[ii][:] = dropdims(sum(obs.A .* obs_ii, dims=1), dims=1) # obs_ii is broadcast over dims of A
        marginal_LLs[ii][:] = AI.Maths.capped_log(marginal_LLs[ii])
    end
    
    # Step 2: Map prior to log space and create initial log-posterior
    for prior in log_prior
        prior[:] = AI.Maths.capped_log(prior)
    end

    # create an uninformed guess at new qs
    qs_current = deepcopy(log_prior)
    for qs in qs_current
        qs[:] = ones(T2, length(qs)) / length(qs)
    end
    last = deepcopy(qs_current)
    
    #=
    Step 3: Iterate until convergence
    A[ii] will be sequentially multiplied by every prior of every state that is a dependency. 
    Four NamedTuples:
        - qs_current: holds updates and will be the NT that is eventually returned
        - last: qs from the last iteration. 
        - qL: qs, with everything marginalized out except obs ii
    =#

    for iter in 1:agent.settings.FPI_num_iter
        #@infiltrate; @assert false
        #qs_new = Vector{Vector{Float64}}([zeros(x.size) for x in prior])
        #qs = [LEF.softmax(x) for x in new_log_q]
        
        for (ii, state) in enumerate(model.states)
            qL = zeros(T2, log_prior[ii].size)
            
            for (jj, obs) in enumerate(model.obs)
                if state.name in obs.A_dim_names
                    qL += all_marginal_log_likelihood(
                        qs_current, 
                        marginal_LLs[jj], 
                        ii, 
                        jj, 
                        agent,
                    )
                end
            end
            #@infiltrate; @assert false
            qs_current[ii][:] = LEF.softmax(qL .+ log_prior[ii])
        end

        #printfmtln("err= {}", sum.([dd .^2 for dd in collect(values(qs_current)) - collect(values(last))]))

        if all(isapprox.(values(qs_current), values(last), atol=1e-6))
            break
        end

        last = deepcopy(qs_current) 
    end
    
    #@infiltrate; @assert false
    return qs_current
end


end  # -------- module


