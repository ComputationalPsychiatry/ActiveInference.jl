

# scratch file, just to try a few things out



    




function test()


    function forward_backward(A, L, p)
        N = size(L, 3) # Number of time steps
        M = size(L, 2) # Number of hidden states
        Q = size(L, 1) # Number of hidden states
        
        alpha = zeros(Q, N)
        beta = zeros(M, N)
        
        for t in 1:N
            if t == 1
                #@infiltrate; @assert false
                alpha[:, t] = exp.(L[:,:,t]) ./ sum(exp.(L[:,:,t])) * p # Forward pass
            else
                #@infiltrate; @assert false
                alpha[:, t] = exp.(L[:,:,t] .+ log.(alpha[:, t-1])) ./ sum(exp.(L[:,:,t] .+ log.(alpha[:, t-1])))  * p # Forward pass
            end
        end
        
        for t in reverse(2:N)
            @infiltrate; @assert false
            beta[:, t] = exp.(L[:,:,t] .+ log.(reshape(A' * beta[:, t], (1,3)))) ./ sum(exp.(L[:,:,t] .+ log.(reshape(beta[:, t-1], (1,3))))) * p # Backward pass
        end
        
        log_Z = logsumexp(alpha[N,:])
        b = alpha[1,:] ./ (alpha[1,:] + beta[1,:] .- log_Z) # MAP estimate
    
        return alpha, beta, b
    end

    L = rand(2,3,4)
    A = rand(3,3)
    p = rand(3)
    
    res = forward_backward(A, L, p)

    # https://github.com/maxmouchet/HMMBase.jl/blob/master/src/likelihoods.jl
    function forwardlog!(α::AbstractMatrix, c::AbstractVector, a::AbstractVector, A::AbstractMatrix, LL::AbstractMatrix)
        (size(LL, 1) == 0) && return
        fill!(α, 0.0)
        fill!(c, 0.0)
        m = vec_maximum(view(LL, 1, :))
        for j in eachindex(a)
            α[1, j] = a[j] * exp(LL[1, j] - m)
            c[1] += α[1, j]
        end
        for j in eachindex(c)
            α[1, j] /= c[1]
        end
        c[1] = log(c[1]) + m
        @inbounds for t in 2:size(LL, 1)
            m = vec_maximum(view(LL, t, :))
            for j in eachindex(α[t-1, :])
                α[t, j] += α[t-1, j] * A[j]
                α[t, j] *= exp(LL[t, j] - m)
                c[t] += α[t, j]
            end
            for j in eachindex(c[t])
                α[t, j] /= c[t]
            end
            c[t] = log(c[t]) + m
        end
    end

end
