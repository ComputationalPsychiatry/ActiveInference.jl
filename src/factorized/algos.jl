
using Format
using Infiltrator
using Revise

#import OMEinsum as einsum
import LogExpFunctions as LEF

#show(stdout, "text/plain", x)
# @infiltrate; @assert false


function all_marginal_log_likelihood(qs, LL, ii, jj, metamodel)
    
    if (ndims(LL) == 2 && LL.size[2] == 1)
        # nothing to do, this is just A[obs,:]
        return dropdims(LL, dims=2)
    end
        
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

    deps = Vector{Vector{Float64}}()
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

    #=
    kk = findfirst(x-> x == state, obs_dep) - 1  
    
    # ii=state, jj=observation, kk=location of state in observation dependencies  

    for (ii, state_dep) in enumerate(metamodel.state_deps)
        @infiltrate; @assert false
        
        Aii = copy(A[ii])
        deps = [findfirst(x->x==sn, keys(metamodel.state_deps)) for sn in meta_obs[2:end]]
        for (jj, dep) in enumerate(deps)  # one for each state dependency of A[ii]
            dims = repeat([1], ndims(Aii))
            prior_jj = prior[dep] 
            dims[jj+1] = prior_jj.size[1]
            prior_jj = reshape(prior_jj, dims...)
            Aii = Aii .* prior_jj
        end
        marginal_ll = sum(Aii, dims=1)
        push!(log_likelihoods, capped_log(dropdims(marginal_ll, dims=1)))  # log of squeezed matrix
    
    end
    =#    

end


function run_factorized_fpi(
    A, 
    metamodel,
    obs, 
    prior; 
    num_iter=5
    )
    """
    Run the fixed point iteration algorithm with sparse dependencies between factors and outcomes 
    """

    #=
    Step 1: Compute marginal log likelihoods for each factor.
    Likelihood[ii].ndim will be size ndims(A[ii]) -1 (i.e., equal to the number of factors).
    E.g., if A[ii] is (4,25,3) and prior is (25,), then intermediate result is (4,25,3) , which
    is summed over dim=1, so (4,25,3) --> (1,25,3) --> (25,3).
    =#
    log_likelihoods = []
    for ii in 1:length(metamodel.obs_deps)
        obs_ii = obs[ii]
        dims = repeat([1], ndims(A[ii]))
        dims[1] = obs_ii.size[1]
        obs_ii = reshape(obs_ii, dims...)
        marginal = dropdims(sum(A[ii] .* obs_ii, dims=1), dims=1)  # obs_ii is a one-hot vector
        
        log_marginal = capped_log(marginal)
        if ndims(log_marginal) == 1
            log_marginal = reshape(log_marginal, (:,1))
        end
        push!(log_likelihoods, log_marginal)  # log of squeezed matrix
    end
    
    # Step 2: Map prior to log space and create initial log-posterior
    log_prior = [capped_log(x) for x in prior]
    qs = Vector{Vector{Float64}}([ones(x.size) ./ x.size for x in prior])  # uniform marginal posterior beliefs at current timepoint
    last = copy(qs)
    #=
    Step 3: Iterate until convergence


    A[ii] will be sequentially multiplied by every prior of every state that is a dependency. 
    =#

    for iter in 1:num_iter

        qs_new = Vector{Vector{Float64}}([zeros(x.size) for x in prior])
        #qs = [LEF.softmax(x) for x in new_log_q]
        
        for (ii, state) in enumerate(keys(metamodel.state_deps)) 
            qL = zeros(prior[ii].size)
            
            for (jj, obs_dep) in enumerate(metamodel.obs_deps)
                if state in obs_dep
                    #printfmtln("\nstate={}, obs={}", ii, jj)
                    qL += all_marginal_log_likelihood(
                        qs, 
                        log_likelihoods[jj], 
                        ii, 
                        jj, 
                        metamodel
                    )
                else

                end
            end
            #@infiltrate; @assert false
            qs_new[ii] = LEF.softmax(qL .+ log_prior[ii])
        end

        qs = deepcopy(qs_new)

        if isapprox(qs, last, atol=1e-6)
            break
        end

        last = qs 
    end
    
    #@infiltrate; @assert false
    return qs
end




