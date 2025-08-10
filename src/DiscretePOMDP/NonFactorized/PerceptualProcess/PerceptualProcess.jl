"""
In this script, we define a the perceptual inference process for the DiscretePOMDP.
"""

using ..ActiveInferenceCore: AbstractPerceptualProcess, AbstractOptimEngine

#Struct for containing current beliefs and optimization engine
mutable struct PerceptualProcess{T<:Union{AbstractOptimEngine, Missing}} <: AbstractPerceptualProcess{T}

    # beliefs about states, prior and observation
    posterior_states::Union{Vector{Vector{Float64}}, Nothing}
    previous_posterior_states::Union{Vector{Vector{Float64}}, Nothing}
    prior::Union{Vector{Vector{Float64}}, Nothing}
    current_observation::Union{Vector{Int}, Nothing}

    # learning structs
    A_learning::Union{Nothing, Learn_A}
    B_learning::Union{Nothing, Learn_B}
    D_learning::Union{Nothing, Learn_D}

    # Struct for containing the "meta" information, such as whether to update parameters etc
    info::PerceptualProcessInfo

    # Optimization engine for state inference
    optim_engine::Union{T, Missing}

    function PerceptualProcess(;
        A_learning::Union{Nothing, Learn_A} = nothing,
        B_learning::Union{Nothing, Learn_B} = nothing,
        D_learning::Union{Nothing, Learn_D} = nothing,
        optim_engine::Union{AbstractOptimEngine, Missing} = FixedPointIteration(),
        verbose::Bool = true
    )

        info_struct = PerceptualProcessInfo(A_learning, B_learning, D_learning, optim_engine)

        # Show process information if verbose
        show_info(info_struct; verbose=verbose)

        new{typeof(optim_engine)}(nothing, nothing, nothing, nothing, A_learning, B_learning, D_learning, info_struct, optim_engine)
    end
end

# function perception(
#     agent::AIFModel{GenerativeModel, PerceptualProcess{FixedPointIteration}, ActionProcess},
#     observation::Vector{Int}
# )
#     println("Hello, this is FPI")
#     # # Set the current observation in the perceptual process
#     # agent.perceptual_process.current_observation = observation

#     # # Set the current posterior_states to the previous_posterior states
#     # agent.perceptual_process.previous_posterior_states = agent.perceptual_process.posterior_states

#     # # Infer states with muætiple dispatch on the optimization engine
#     # new_posterior_states = infer_states(agent, agent.perceptual_process.optim_engine)
#     # agent.perceptual_process.posterior_states = new_posterior_states

#     # # If learning is enabled, update the beliefs about the parameters
#     # if agent.perceptual_process.info.learning_enabled
#     #     update_parameters(agent)
#     # end

# end

