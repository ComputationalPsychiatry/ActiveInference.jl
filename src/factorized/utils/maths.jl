
module Maths



import OMEinsum as ein
import LogExpFunctions as LEF

#using Format
#using Infiltrator
#using Revise

# This utils modeule collects all the math utils functions that JB uses in the factorized version.
# todo: Perhaps other math utils functions, in other math utils files, are not used anywhere. If so,
# they can be deleted.

MINVAL = eps(Float64)


function dot_product1(X::Union{Array{Float64, N} where N, Matrix{Float64}}, xs::Vector{Vector{Float64}})
    # xs is a vector of qs vectors for each dependency, in reverse order
    
    if isa(X, Matrix{Float64})
        @assert length(xs) == 1
        return X * xs[1]
    end

    sizes = [collect(x.size) for x in xs]
    code2 = ein.EinCode([collect(X.size), sizes...], collect(X.size[1:end-length(sizes)]))
    
    #=
    tried code2 = ein.EinCode([collect(X.size), [dep.size[1]]], collect(X.size[1:end-1]))
    but that did not work
    =#
    return code2(X,xs...)
end


""" Creates a onehot encoded vector """
function onehot(index::Int, vector_length::Int)
    vector = zeros(vector_length)
    vector[index] = 1.0
    return vector
end



function stable_xlogx(x)
    zz =  [LEF.xlogy.(z, clamp.(z, MINVAL, Inf)) for z in x]
    #@infiltrate; @assert false
    return zz
end


function stable_entropy(x)
    z = stable_xlogx(x)
    return - sum(vcat(z...))  
end


"""
    capped_log(x::Real)

# Arguments
- `x::Real`: A real number.

Return the natural logarithm of x, capped at the machine epsilon value of x.
"""
function capped_log(x::Real)
    return log(max(x, eps(x))) 
end

"""
    capped_log(array::Array{Float64})
"""
function capped_log(array::Array{Float64}) 

    epsilon = oftype(array[1], 1e-16)
    # Return the log of the array values capped at epsilon
    array = log.(max.(array, epsilon))

    return array
end

"""
    capped_log(array::Array{T}) where T <: Real 
"""
function capped_log(array::Array{T}) where T <: Real 

    epsilon = oftype(array[1], 1e-16)
    # Return the log of the array values capped at epsilon
    array = log.(max.(array, epsilon))

    return array
end

"""
    capped_log(array::Vector{Real})
"""
function capped_log(array::Vector{Real})
    epsilon = oftype(array[1], 1e-16)

    array = log.(max.(array, epsilon))
    # Return the log of the array values capped at epsilon
    return array
end

""" Apply capped_log to array of arrays """
function capped_log_array(array)
    
    return map(capped_log, array)
end


end  # -- module