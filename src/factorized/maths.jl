#@infiltrate; @assert false


module Maths



import OMEinsum as ein
import LogExpFunctions as LEF

#using Format
using Infiltrator
#using Revise

# This utils modeule collects all the math utils functions that JB uses in the factorized version.
# todo: Perhaps other math utils functions, in other math utils files, are not used anywhere. If so,
# they can be deleted.


function dot_product1(
        X::Union{Array{T, N}, Matrix{T}} where {N}, 
        xs::Vector{Vector{T}} 
    ) where {T<:AbstractFloat}

    # xs is a vector of qs vectors for each dependency, in reverse order
    
    if isa(X, Matrix{T})
        @assert length(xs) == 1
        return X * xs[1]
    end

    sizes = [collect(x.size) for x in xs]
    code2 = ein.EinCode([collect(X.size), sizes...], collect(X.size[1:end-length(sizes)]))
    
    #=
    tried code2 = ein.EinCode([collect(X.size), [dep.size[1]]], collect(X.size[1:end-1]))
    but that did not work
    =#
    
    #@infiltrate; @assert false
    return code2(X,xs...)
end


""" Creates a onehot encoded vector """
function onehot(
        index::Int, 
        vector_length::Int,
        float_type::Type
    )

    vector = zeros(float_type, vector_length)
    vector[index] = 1.0
    return vector
end



function xstable_xlogx(x::Matrix{T}) where {T<:AbstractFloat}
    

    MINVAL = eps(T)
    #z1 =  [LEF.xlogy.(z, clamp.(z, MINVAL, Inf)) for z in x]
    #z2 =  [LEF.xlogy.(z, z) for z in x]
    z3 = LEF.xlogx.(x)
    #@infiltrate; @assert false
    return zz
end


function stable_entropy(x::Vector{T}) where {T<:AbstractFloat}
    z =  LEF.xlogx.(x)
    return - sum(z)  
end


function stable_entropy(x::Matrix{T}) where {T<:AbstractFloat}
    @infiltrate; @assert false
    z =  [LEF.xlogy.(z, z) for z in x]
    @infiltrate; @assert false
    return - sum(vcat(z...))  
end


"""
    capped_log(x::Real)

# Arguments
- `x::Real`: A real number.

Return the natural logarithm of x, capped at the machine epsilon value of x.
"""
function capped_log(x::T) where {T<:AbstractFloat}
    return log(max(x, eps(x))) 
end

#=
"""
    capped_log(array::Array{Float64})
"""
function capped_log(array::Array{T}) where {T<:AbstractFloat} 

    epsilon = oftype(array[1], 1e-16)
    # Return the log of the array values capped at epsilon
    array = log.(max.(array, epsilon))

    return array
end
=#

"""
    capped_log(array::Array{T}) where T <: Real 
"""
function capped_log(array::Array{T}) where {T<:AbstractFloat} 

    epsilon = oftype(array[1], 1e-16)
    # Return the log of the array values capped at epsilon
    array = log.(max.(array, epsilon))

    return array
end

"""
    capped_log(array::Vector{Real})
"""
function capped_log(array::Vector{T}) where {T<:AbstractFloat}
    epsilon = oftype(array[1], 1e-16)

    array = log.(max.(array, epsilon))
    # Return the log of the array values capped at epsilon
    return array
end

""" Apply capped_log to array of arrays """
function capped_log_array(array)
    
    return map(capped_log, array)
end


""" SPM_wnorm """
function spm_wnorm(A::Array{T2, N} where {N}) where {T2<:AbstractFloat}
    
    EPS_VAL = T2.(1e-16)

    A .+= EPS_VAL
    norm = T2.(1.0) ./ sum(A, dims = 1)
    avg = 1 ./ A
    wA = norm .- avg
    
    #@infiltrate; @assert false
    return wA
end


""" Multi-dimensional outer product """
function outer_product(
        x::Vector{T2}, 
        y::Union{Nothing, Vector{T2}} =nothing; 
        remove_singleton_dims::Bool =true, 
        args...
    ) where {T2<:AbstractFloat}
    
    #@infiltrate; @assert false

    # If only x is provided and it is a vector of arrays, recursively call outer_product on its elements.
    if y === nothing && isempty(args)
        if x isa AbstractVector
            return reduce((a, b) -> outer_product(a, b), x)
        elseif typeof(x) <: Number || typeof(x) <: AbstractArray
            return x
        else
            throw(ArgumentError("Invalid input to outer_product (\$x)"))
        end
    end

    # If y is provided, perform the cross multiplication.
    if y !== nothing
        reshape_dims_x = tuple(size(x)..., ones(Real, ndims(y))...)
        A = reshape(x, reshape_dims_x)
        
        reshape_dims_y = tuple(ones(Real, ndims(x))..., size(y)...)
        B = reshape(y, reshape_dims_y)

        z = A .* B

    else
        z = x
    end

    # Recursively call outer_product for additional arguments
    for arg in args
        z = outer_product(z, arg; remove_singleton_dims=remove_singleton_dims)
    end

    # Remove singleton dimensions if true
    if remove_singleton_dims
        z = dropdims(z, dims = tuple(findall(size(z) .== 1)...))
    end

    return z
end


"""Normalizes a Categorical probability distribution"""
function normalize_distribution(distribution)
    distribution .= distribution ./ sum(distribution, dims=1)
    return distribution
end


""" Normalizes multiple arrays """
function normalize_arrays(array::Vector{<:Array{<:T}}) where {T<:AbstractFloat}
    return map(normalize_distribution, array)
end


""" Normalizes multiple arrays """
function normalize_arrays(array::Vector{Any})
    return map(normalize_distribution, array)
end


end  # -- module