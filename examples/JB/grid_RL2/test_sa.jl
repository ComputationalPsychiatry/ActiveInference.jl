

using BenchmarkTools
using StaticArrays
using Infiltrator
using Revise


module TestSA

using Format
using Infiltrator
using Revise
using StaticArrays


#import LinearAlgebra as LA
import OMEinsum as ein

#show(stdout, "text/plain", x)
# @infiltrate; @assert false

#Graphs.is_directed(::Type{<:Multigraphs.Multigraph}) = false


# --------------------------------------------------------------------------------------------------
struct Agent
    A
    v1
end



function dot_product1(X, xs)
#function dot_product1(X::MArray{Tuple{20, 20, 20}, Float64, 3, 8000}, xs::Vector{MVector{20, Float64}})
    # xs is a vector of qs vectors for each dependency, in reverse order
    
    if isa(X, Matrix{Float64})
        @assert length(xs) == 1
        return X * xs[1]
    end
    #@infiltrate; @assert false
    sizes = [collect(size(x)) for x in xs]
    code2 = ein.EinCode([collect(size(X)), sizes...], collect(size(X)[1:end-length(sizes)]))
    
    return code2(X,xs...)
end


function run(agent)
    for i in 1:200
        z = dot_product1(agent.A, [agent.v1, agent.v1])
    end
end    



end  # -- module
    

A = randn(5,5,5)
v1 = randn(5)

#@infiltrate; @assert false
B = MArray{Tuple{5,5,5}}(A)
v2 = MArray{Tuple{5}}(v1)


agent = TestSA.Agent(A, v1)
@btime TestSA.run(agent) 

agent = TestSA.Agent(B, v2)
@btime TestSA.run(agent) 
