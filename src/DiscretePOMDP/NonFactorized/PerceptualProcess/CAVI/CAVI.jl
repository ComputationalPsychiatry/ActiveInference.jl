"""
In this script, we define a the perceptual inference process for the DiscretePOMDP.
"""

using ..ActiveInferenceCore: AbstractPerceptualProcess

# Learning marker structs
abstract type LearningState end
struct Learning <: LearningState end
struct NoLearning <: LearningState end

#Struct for containing current beliefs and optimization engine
mutable struct CAVI{L <: LearningState} <: AbstractPerceptualProcess

    # beliefs about states, prior and observation
    posterior_states::Union{Vector{Vector{Float64}}, Nothing}
    previous_posterior_states::Union{Vector{Vector{Float64}}, Nothing}
    prediction_states::Union{Vector{Vector{Float64}}, Nothing}
    observation::Union{Vector{Int}, Nothing}

    # Fields containing predictions from the prediction function
    predicted_states::Union{Vector{Vector{Vector{Vector{Float64}}}}, Nothing}
    predicted_observations::Union{Vector{Vector{Vector{Vector{Float64}}}}, Nothing}

    # learning structs
    A_learning::Union{Nothing, Learn_A}
    B_learning::Union{Nothing, Learn_B}
    D_learning::Union{Nothing, Learn_D}

    # Struct for containing the "meta" information, such as whether to update parameters etc
    info::CAVIInfo

    # Settings
    num_iter::Int
    dF_tol::Float64

    function CAVI(;
        A_learning::Union{Nothing, Learn_A} = nothing,
        B_learning::Union{Nothing, Learn_B} = nothing,
        D_learning::Union{Nothing, Learn_D} = nothing,
        num_iter::Int = 10,
        dF_tol::Float64 = 1e-3,
        verbose::Bool = true
    )

        info_struct = CAVIInfo(A_learning, B_learning, D_learning)

        # Show process information if verbose
        show_info(info_struct; verbose=verbose)

        L = info_struct.learning_enabled ? Learning : NoLearning
        

        new{L}(nothing, nothing, nothing, nothing, nothing, nothing, A_learning, B_learning, D_learning, info_struct, num_iter, dF_tol)
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

