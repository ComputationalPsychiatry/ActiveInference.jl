"""
In this script, we define a the action_process process for the DiscretePOMDP.
"""

using ..ActiveInferenceCore: AbstractActionProcess, AIFAgent

# Struct for containing the action process
mutable struct ActionProcess <: AbstractActionProcess

    # Function for calculating the predictive posterior
    # prediction::Function

    # Function for calculating the action
    # get_action_distribution::Function

    # Struct for containing the "meta" information, such as whether to update parameters etc
    # info::ActionProcessInfo

    # Field containing prior over policies 'E'. Also called 'habits'.
    E::Union{Vector{T}, Nothing} where {T <: Real}

    # Fields for containing information about the action process
    policy_length::Int
    policies::Union{Vector{Matrix{Int64}}, Nothing}
    predicted_states::Union{Vector{Vector{Vector{Vector{Float64}}}}, Nothing}
    predicted_observations::Union{Vector{Vector{Vector{Vector{Float64}}}}, Nothing}
    previous_action::Union{Vector{Int}, Nothing}

    function ActionProcess(;
        # prediction::Function = predict_states_observations,
        # get_action_distribution::Function = get_action_distribution,
        E::Union{Vector{T}, Nothing} where {T <: Real} = nothing,
        policy_length::Int = 2,
        policies::Union{Vector{Matrix{Int64}}, Nothing} = nothing,
        predicted_states::Union{Vector{Vector{Float64}}, Nothing} = nothing,
        predicted_observations::Union{Vector{Int}, Nothing} = nothing,
        previous_action::Union{Vector{Int}, Nothing} = nothing
    )
        new(E, policy_length, policies, predicted_states, predicted_observations, previous_action)
    end
end

function predict_states_observations(agent::AIFAgent)

    all_predicted_states = get_expected_states(agent.perceptual_process.posterior_states, agent.generative_model.B, agent.action_process.policies)
    all_predicted_observations = get_expected_obs(all_predicted_states, agent.generative_model.A)

    @show typeof(all_predicted_states)
    @show typeof(all_predicted_observations)

    agent.action_process.predicted_states = all_predicted_states
    agent.action_process.predicted_observations = all_predicted_observations

end


function get_action_distribution(agent::AIFAgent)

    # Get the action distribution from the action process
    return action_process.get_action_distribution(agent, action_process)
end


