"""
In this script, we define a the perceptual inference process for the DiscretePOMDP.
"""

using ..ActiveInferenceCore: AbstractPerceptualProcess

#Struct for containing current beliefs and optimization engine
mutable struct PerceptualProcess <: AbstractPerceptualProcess
    #beliefs about states
    qs::Union{Vector{Vector{Float64}}, Nothing}
    #beliefs about parameters
    pA::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N}
    pB::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N}
    pD::Union{Vector{Vector{T}}, Nothing} where {T <: Real}

    #Struct for containing the "meta" information, such as whether to update parameters etc
    # model_info::POMDPPerceptionModelInfo

    optim_engine::Function #Default is handwritten optim CAVI

end



