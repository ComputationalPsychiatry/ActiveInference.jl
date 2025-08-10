""" FixedPointIteration struct containing the settings for FPI. """
@kwdef struct FixedPointIteration <: AbstractOptimEngine 
    num_iter::Int = 10
    dF_tol::Float64 = 1e-3
end
