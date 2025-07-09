"""
In this script, we define a the perceptual inference process for the DiscretePOMDP.
"""

using ..ActiveInferenceCore: AbstractPerceptualProcess

#Struct for containing current beliefs and optimization engine
mutable struct PerceptualProcess <: AbstractPerceptualProcess
    #beliefs about states
    qs::Union{Vector{Vector{Float64}}, Nothing}
    prior::Union{Vector{Float64}, Nothing}
    #beliefs about parameters
    A_learning::Union{Nothing, Learn_A}
    B_learning::Union{Nothing, Learn_B}
    D_learning::Union{Nothing, Learn_D}

    info::PerceptualProcessInfo #Struct for containing the "meta" information, such as whether to update parameters etc

    optim_engine::Function #Default is handwritten optim CAVI

    function PerceptualProcess(;
        generative_model::GenerativeModel,
        qs::Union{Vector{Vector{Float64}}, Nothing} = nothing,
        A_learning::Union{Nothing, Learn_A} = nothing,
        B_learning::Union{Nothing, Learn_B} = nothing,
        D_learning::Union{Nothing, Learn_D} = nothing,
        optim_engine::Function = fixed_point_iteration
    )

        # compare_generative_perceptual(generative_model, A_learning, B_learning, D_learning)
        create_learning_priors(generative_model, A_learning, B_learning, D_learning)
        @show typeof(A_learning)
        info_struct = PerceptualProcessInfo(A_learning, B_learning, D_learning, optim_engine)

        new(qs, nothing, A_learning, B_learning, D_learning, info_struct, optim_engine)
    end
end

function perception(;
    generative_model::GenerativeModel,
    perceptual_process::PerceptualProcess,
    observation::Vector{Int}
)
    new_posterior_states = perceptual_process.optim_engine(perceptual_process, generative_model, observation)


end



