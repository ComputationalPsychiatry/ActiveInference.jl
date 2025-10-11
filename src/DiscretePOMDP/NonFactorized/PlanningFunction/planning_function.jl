``` Planning function for action distribution in a discrete POMDP model ```

function ActiveInferenceCore.planning(
    model::AIFModel{GenerativeModel, CAVI{NoLearning}, ActionProcess},
    posterior::NamedTuple{(:posterior_states, :prior_qs_prediction), Tuple{Vector{Vector{Float64}}, Vector{Vector{Float64}}}}
)

    predictions = prediction(model, posterior)

    # Get posterior over policies and expected free energies
    q_pi, G =  update_posterior_policies(
        qs = posterior.posterior_states,
        A = model.generative_model.A,
        C = model.generative_model.C,
        policies = model.action_process.policies,
        qs_pi_all = predictions.all_predicted_states,
        qo_pi_all = predictions.all_predicted_observations,
        use_utility = model.action_process.use_utility,
        use_states_info_gain = model.action_process.use_states_info_gain,
        use_param_info_gain = model.action_process.use_param_info_gain,
        A_learning = nothing,
        B_learning = nothing,
        E = model.action_process.E,
        gamma = model.action_process.gamma
    )

    return (q_pi = q_pi, G = G, predictions = predictions)
end

function ActiveInferenceCore.planning(
    model::AIFModel{GenerativeModel, CAVI{Learning}, ActionProcess},
    posterior::NamedTuple{
        (:posterior_states, :prior_qs_prediction, :learning_posterior),
        Tuple{
            Vector{Vector{Float64}},
            Vector{Vector{Float64}},
            NamedTuple{
                (:A_updated, :qA, :B_updated, :qB, :D_updated, :qD),
                Tuple{T_A, T_qA, T_B, T_qB, T_D, T_qD}
            }
        }
    }
)  where {
    T_A  <: Union{Vector{<:Array{Float64}}, Nothing},
    T_qA <: Union{Vector{<:Array{Float64}}, Nothing},
    T_B  <: Union{Vector{<:Array{Float64}}, Nothing},
    T_qB <: Union{Vector{<:Array{Float64}}, Nothing},
    T_D  <: Union{Vector{Vector{Float64}}, Nothing},
    T_qD <: Union{Vector{Vector{Float64}}, Nothing}
}

    predictions = prediction(model, posterior)

    A = posterior.learning_posterior.A_updated !== nothing ? posterior.learning_posterior.A_updated : model.generative_model.A

    # Get posterior over policies and expected free energies
    q_pi, G =  update_posterior_policies(
        qs = posterior.posterior_states,
        A = A,
        C = model.generative_model.C,
        policies = model.action_process.policies,
        qs_pi_all = predictions.all_predicted_states,
        qo_pi_all = predictions.all_predicted_observations,
        use_utility = model.action_process.use_utility,
        use_states_info_gain = model.action_process.use_states_info_gain,
        use_param_info_gain = model.action_process.use_param_info_gain,
        A_learning = model.perceptual_process.A_learning,
        B_learning = model.perceptual_process.B_learning,
        E = model.action_process.E,
        gamma = model.action_process.gamma
    )

    return (q_pi = q_pi, G = G, predictions = predictions)
end