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

    info::InfoStruct #Struct for containing the "meta" information, such as whether to update parameters etc

    optim_engine::Function #Default is handwritten optim CAVI

    function PerceptualProcess(
        qs::Union{Vector{Vector{Float64}}, Nothing} = nothing,
        A_learning::Union{Nothing, Learn_A} = nothing,
        B_learning::Union{Nothing, Learn_B} = nothing,
        D_learning::Union{Nothing, Learn_D} = nothing,
        info::InfoStruct = InfoStruct(),
        optim_engine::Function = fixed_point_iteration
    )


        new(qs, pA, pB, pD, info, optim_engine)
    end
end



