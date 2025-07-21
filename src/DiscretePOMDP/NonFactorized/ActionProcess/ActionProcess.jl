"""
In this script, we define a the action_process process for the DiscretePOMDP.
"""

using ..ActiveInferenceCore: AbstractActionProcess, AIFAgent

# Struct for containing the action process
mutable struct ActionProcess <: AbstractActionProcess

    # Struct for containing the "meta" information, such as whether to update parameters etc
    info::ActionProcessInfo

    # Settings for the actions process
    use_utility::Bool
    use_states_info_gain::Bool
    use_param_info_gain::Bool
    gamma::Real #? should not be here?

    # Field containing prior over policies 'E'. Also called 'habits'.
    E::Union{Vector{T}, Nothing} where {T <: Real}

    # Fields for containing information about the action process
    policy_length::Int
    policies::Union{Vector{Matrix{Int64}}, Nothing}

    # Fields containing predictions, actions, and posterior policies
    predicted_states::Union{Vector{Vector{Vector{Vector{Float64}}}}, Nothing}
    predicted_observations::Union{Vector{Vector{Vector{Vector{Float64}}}}, Nothing}
    previous_action::Union{Vector{Int}, Nothing}
    posterior_policies::Union{Vector{Float64}, Nothing}
    expected_free_energy::Union{Vector{Float64}, Nothing}

    # Field for action selection
    action_selection::String
    alpha::Real

    function ActionProcess(;
        use_utility::Bool = true,
        use_states_info_gain::Bool = true,
        use_param_info_gain::Bool = false,
        gamma::Real = 16.0,
        E::Union{Vector{T}, Nothing} where {T <: Real} = nothing,
        policy_length::Int = 2,
        policies::Union{Vector{Matrix{Int64}}, Nothing} = nothing,
        predicted_states::Union{Vector{Vector{Float64}}, Nothing} = nothing,
        predicted_observations::Union{Vector{Int}, Nothing} = nothing,
        previous_action::Union{Vector{Int}, Nothing} = nothing,
        posterior_policies::Union{Vector{Float64}, Nothing} = nothing,
        expected_free_energy::Union{Vector{Float64}, Nothing} = nothing,
        action_selection::String = "stochastic",
        alpha::Real = 16.0,
        verbose::Bool = true
    )

        # Create the ActionProcessInfo struct
        info = ActionProcessInfo(
            use_utility,
            use_states_info_gain,
            use_param_info_gain,
            policy_length,
            policies,
            E,
            gamma,
            action_selection,
            alpha
        )

        show_info(info, verbose=verbose)

        if action_selection != "stochastic" && action_selection != "deterministic"
            error("action_selection must be either 'stochastic' or 'deterministic'")
        end

        new(
            info, 
            use_utility, 
            use_states_info_gain, 
            use_param_info_gain, 
            gamma, 
            E, 
            policy_length, 
            policies, 
            predicted_states, 
            predicted_observations, 
            previous_action, 
            posterior_policies, 
            expected_free_energy, 
            action_selection,
            alpha
        )
    end
end

function predict_states_observations(agent::AIFAgent)

    all_predicted_states = get_expected_states(agent.perceptual_process.posterior_states, agent.generative_model.B, agent.action_process.policies)
    all_predicted_observations = get_expected_obs(all_predicted_states, agent.generative_model.A)

    agent.action_process.predicted_states = all_predicted_states
    agent.action_process.predicted_observations = all_predicted_observations

end

function action_distribution(agent::AIFAgent; act::Bool = false)

    # Get posterior over policies and expected free energies
    q_pi, G = update_posterior_policies(
        qs = agent.perceptual_process.posterior_states,
        A = agent.generative_model.A,
        C = agent.generative_model.C,
        policies = agent.action_process.policies,
        qs_pi_all = agent.action_process.predicted_states,
        qo_pi_all = agent.action_process.predicted_observations,
        use_utility = agent.action_process.use_utility,
        use_states_info_gain = agent.action_process.use_states_info_gain,
        use_param_info_gain = agent.action_process.use_param_info_gain,
        A_learning = agent.perceptual_process.A_learning,
        B_learning = agent.perceptual_process.B_learning,
        E = agent.action_process.E,
        gamma = agent.action_process.gamma
    )

    agent.action_process.posterior_policies = q_pi
    agent.action_process.expected_free_energy = G

    # If action is enabled, sample an action from the posterior policies
    if act
        # Sample action from the posterior policies
        action = sample_action(
            q_pi = q_pi, 
            policies = agent.action_process.policies, 
            n_controls = agent.generative_model.info.controls_per_factor; 
            action_selection=agent.action_process.info.action_selection, 
            alpha=agent.action_process.info.alpha
        )

        agent.action_process.previous_action = action
        return q_pi, G, action
    else
        return q_pi, G
    end

end


