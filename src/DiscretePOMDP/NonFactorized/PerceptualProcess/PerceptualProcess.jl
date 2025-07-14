"""
In this script, we define a the perceptual inference process for the DiscretePOMDP.
"""

using ..ActiveInferenceCore: AbstractPerceptualProcess, AIFAgent, AbstractOptimEngine


#Struct for containing current beliefs and optimization engine
mutable struct PerceptualProcess <: AbstractPerceptualProcess
    #beliefs about states
    posterior_states::Union{Vector{Vector{Float64}}, Nothing}
    prior::Union{Vector{Float64}, Nothing}
    #beliefs about parameters
    A_learning::Union{Nothing, Learn_A}
    B_learning::Union{Nothing, Learn_B}
    D_learning::Union{Nothing, Learn_D}

    info::PerceptualProcessInfo #Struct for containing the "meta" information, such as whether to update parameters etc

    optim_engine::AbstractOptimEngine

    function PerceptualProcess(;
        posterior_states::Union{Vector{Vector{Float64}}, Nothing} = nothing,
        A_learning::Union{Nothing, Learn_A} = nothing,
        B_learning::Union{Nothing, Learn_B} = nothing,
        D_learning::Union{Nothing, Learn_D} = nothing,
        optim_engine::AbstractOptimEngine = FixedPointIteration()
    )

        # compare_generative_perceptual(generative_model, A_learning, B_learning, D_learning)
        # create_learning_priors(A_learning, B_learning, D_learning) # use in agent creation instead

        info_struct = PerceptualProcessInfo(A_learning, B_learning, D_learning, optim_engine)

        new(posterior_states, nothing, A_learning, B_learning, D_learning, info_struct, optim_engine)
    end
end

function perception(
    agent::AIFAgent,
    observation::Vector{Int}
)

    # Infer states using the specified optimization engine
    new_posterior_states = infer_states(agent, observation, agent.perceptual_process.optim_engine)
    agent.perceptual_process.posterior_states = new_posterior_states

    # If learning is enabled, update the beliefs about the parameters
    


end



