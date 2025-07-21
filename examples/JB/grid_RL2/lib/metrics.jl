# https://pymdp-rtd.readthedocs.io/en/latest/notebooks/clue_chaining_demo.html
# Active Inference Demo: Epistemic Chaining

# include("./grid.jl")
#show(stdout, "text/plain", x)
# @infiltrate; @assert false


module Metrics

using Format
using Infiltrator
using Revise

using PyCall
using FreqTables
using Statistics
import LinearAlgebra as LA

####################################################################################################

function calc_metrics(history, model)
    """
    https://stats.stackexchange.com/questions/358741/meaning-and-interpretation-of-transfer-entropy

    TE(X↦Y) = 0.624 means that the history of the X process has 0.624 bits of additional information 
    for predicting the next value of Y. (i.e., it provides information about the future of Y, in 
    addition to what we know from the history of Y). Since it is non-zero, you can conclude that X 
    influences Y in some way.

    The TE should not be negative. To see if the TE is actually significant, you could compare it 
    with the entropy of Y. If H(Y) is, say 200 bits per sample, 0.624 may not be significant. TE 
    should never be more than the entropy of Y, so that's the range you expect.
    
    transfer_entropy(source_series, target_series, k=1), with k = history length
    T_{J→I} is measures the degree of dependence of I on J. 
    """
    
    
    
    pyinform = pyimport("pyinform")

    #y = [0,1,1,1,1,0,0,0,0]
    #x = [0,0,1,1,1,1,0,0,0]

    #y = [0,1,2,2,3,3,2,2,3,3,2,2]  # obs
    #x =  [1,2,2,0,2,2,1,1,2,0,1,1] # actions
    results = Dict()
    actions = history[:actions] # actions

    for ii in 1:history[:obs].size[1]  # iterate over outcomes
    
        obs = history[:obs][ii]  # obs
        TE = pyinform.transfer_entropy(obs, actions, k=1)  # each obs[i] is the result of action[i]
        TE_reverse = pyinform.transfer_entropy(actions, obs, k=1)  
        empowerment = nothing

        if false
            # first idea, empowerment calc from timeseries of actions and observations. 
            @infiltrate; @assert false
            Pyx = collect(freqtable(obs, actions) )
            
            if Pyx.size[2] == 1
                printfmtln("/n*** size of Pyx is {} ***\n", Pyx.size)
                return TE, TE_reverse, missing
            end

            Pyx = Pyx ./ sum(Pyx, dims=1)
            empowerment, _ = blahut_arimoto(permutedims(Pyx) .+ 1e-9)
        elseif ii != 1
            # can only calculate empowerment for location, as actions only change location
            empowerment = missing
        else
            # use method from https://github.com/Mchristos/empowerment/tree/master
            
            @assert isapprox(sort(unique(model.B[1])), [0.0,1.0])  # only handle deterministic for now
                        
            T = model.B[1]  # (next_state, curr_state, action_id)
            E = zeros(model.grid_dims)
            for y in 1:model.grid_dims[1]
                for x in 1:model.grid_dims[2]
                    s = findfirst(z -> z == (y,x), model.cells)
                    E[y,x] = compute_empowerment(T, true, model.policy_length, s)
                end
            end
            
            # sum empowerment by the locations or other states that the agent visited
            empowerment = sum(E[obs])
        end

        results[ii] = TE, TE_reverse, empowerment 
    end
    
    #@infiltrate; @assert false
    return results
end


function compute_empowerment(T, det, n_step, state; n_samples=1000, epsilon = 1e-6)
    """
    Compute the empowerment of a state in a grid world  
    T : numpy array, shape (n_states, n_actions, n_states)
        Transition matrix describing the probabilistic dynamics of a markov decision process
        (without rewards). Taking action a in state s, T describes a probability distribution
        over the resulting state as T[:,a,s]. In other words, T[s',a,s] is the probability of
        landing in state s' after taking action a in state s. The indices may seem "backwards"
        because this allows for convenient matrix multiplication.   
    det : bool
        True if the dynamics are deterministic.
    n_step : int 
        Determines the "time horizon" of the empowerment computation. The computed empowerment is
        the influence the agent has on the future over an n_step time horizon. 
    n_samples : int
        Number of samples for approximating the empowerment in the deterministic case.
    state : int 
        State for which to compute the empowerment.
    
    https://github.com/Mchristos/empowerment/tree/master
    """
    
    
    n_states, _, n_actions  = T.size
    if det
        # JB: blahut_arimoto not used, empowerment = log2 of number of reachable states
        # only sample if too many actions sequences to iterate through
        if n_actions^n_step < 5000
            # for 5 actions and 5 steps, nstep_samples is (3125,5) == (5**5, 5)
            # [0,0,0,0,0] to [4,4,4,4,4]
            
            sizes = [1:n_actions for x in 1:n_step]
            nstep_samples = collect(Iterators.product(sizes...))[:] 
        else
            @infiltrate; @assert false
            nstep_samples = np.random.randint(0,n_actions, [n_samples,n_step] )
        end

        # fold over each nstep actions, get unique end states
        tmap = (s,a) -> argmax(T[:,s,a]) 
        seen = Set()
        for (ii, aseq) in enumerate(nstep_samples)
            
            """
            JB: given current state, and some action from a series of actions, what is 
                most likely new state? 
            
            examples starting from cell 0
            tmap(0, 0) = np.int64(10) up  
            (Pdb) tmap(0, 1) = np.int64(0) down
            (Pdb) tmap(0, 2) = np.int64(1) right
            (Pdb) tmap(0, 3) = np.int64(0) left
            (Pdb) tmap(0, 4) = np.int64(0) stay
            reduce(tmap, [state,*[0,1,2,3,4]]) = 0 because first 10, then down to 0
                then right, then left, then stay -> 0 
            """
            push!(seen, reduce(tmap, [state, aseq...]))  # JB: ending location after series of steps
        end

        # empowerment = log # of reachable states 
        return log2(length(seen))
    
    else
        @infiltrate; @assert false
        
        nstep_actions = list(itertools.product(range(n_actions), repeat = n_step))  # e.g., [(1,1,1) to (5,5,5)]
        Bn = np.zeros([n_states, len(nstep_actions), n_states])
        for (i, an) in enumerate(nstep_actions)
            # JB: given the (100,100) transition matrix for each in a series of actions,
            # calculate the probability that an agent starts at any cell and lands at
            # any cell. For example, if deterministic, agent starts at loc=0, for actions
            # [0,0,0,0,0] and ends in loc=10. Then Bn[:,i,0] is a one-hot at loc=10.
            Bn[:, i , :] = reduce(((x,y) -> dot(y, x)), map(a -> T[:,a,:]), an)
        end
        # JB: Bn e.g., (100,3125), sum(0) = ones. 
        # empowerment calculated via (loc, policy) for some starting state 
        return blahut_arimoto(Bn[:,:,state], epsilon=epsilon)
    end
end


function blahut_arimoto(p_y_x) 
    """
    Maximize the capacity between I(X;Y)
    p_y_x: each row represnets probability assinmnet
    log_base: the base of the log when calaculating the capacity
    thresh: the threshold of the update, finish the calculation when gettting to it.
    max_iter: the maximum iterations of the calculation
    """

    log_base = 2
    thresh = 1e-12
    max_iter = 1e3

    # Input test
    
    @assert abs(mean(sum(p_y_x, dims=2)) .- 1) < 1e-6
    @assert p_y_x.size[1] > 1

    # The number of inputs: size of |X|
    m = p_y_x.size[1]

    # The number of outputs: size of |Y|
    n = p_y_x.size[2]

    # Initialize the prior uniformly
    R = ones((1, m)) / m
    #print("\nB shape= {}".format(R.shape))
    # Compute the R(x) that maximizes the capacity
    Q = nothing
    for iteration in 1:Int(max_iter)
        
        Q = permutedims(R) .* p_y_x
        Q = Q ./ sum(Q, dims=1)

        R1 = permutedims(prod(Q.^p_y_x, dims=2))
        R1 = R1 / sum(R1)

        tolerance = LA.norm(R1 - R)
        R = R1
        if tolerance < thresh
            break
        end
    end

    # Calculate the capacity
    R = R[:]  # flatten
    C = 0
    for i in 1:m
        if R[i] > 0
            C += sum(R[i] .* p_y_x[i, :] .* log.(Q[i, :] ./ R[i] .+ 1e-16))
        end
    end
    C = C / log(log_base)
    
    return C, R

end



end  #  module