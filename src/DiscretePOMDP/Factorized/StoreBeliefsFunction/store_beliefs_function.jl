# Without Learning + With action selection

function ActiveInferenceCore.store_beliefs!(
    model::AIFModel{GenerativeModel, P, ActionProcess},
    action_posterior::NamedTuple{(:posterior,), Tuple{Vector{Vector{T}}}},
    policy_posterior::NamedTuple{(:q_pi, :G, :predictions), Tuple{Vector{Float64}, Vector{Float64}, NamedTuple{(:all_predicted_states, :all_predicted_observations), Tuple{Vector{Vector{Vector{Vector{Float64}}}}, Vector{Vector{Vector{Vector{Float64}}}}}}}},
    posterior::NamedTuple{(:posterior_states, :prediction_states), Tuple{Vector{Vector{Float64}}, Vector{Vector{Float64}}}},
    previous_action::Union{Nothing, Vector{Int}},
    observation::Vector{Int}
) where {T <: Real, P <: AbstractPerceptualProcess}

    # Store beliefs in the perceptual process struct
    model.perceptual_process.posterior_states = posterior.posterior_states
    model.perceptual_process.prediction_states = posterior.prediction_states
    model.perceptual_process.observation = observation

    model.perceptual_process.predicted_states = policy_posterior.predictions.all_predicted_states
    model.perceptual_process.predicted_observations = policy_posterior.predictions.all_predicted_observations

    # Store beliefs in the action process struct
    model.action_process.posterior_policies = policy_posterior.q_pi
    model.action_process.expected_free_energy = policy_posterior.G
    model.action_process.action_posterior = action_posterior.posterior
    model.action_process.previous_action = previous_action

end

function ActiveInferenceCore.store_beliefs!(
    model::AIFModel{GenerativeModel, CAVI{NoLearning}, ActionProcess},
    action_posterior::NamedTuple{(:posterior,), Tuple{Vector{Vector{T}}}},
    policy_posterior::NamedTuple{(:q_pi, :G, :predictions), Tuple{Vector{Float64}, Vector{Float64}, NamedTuple{(:all_predicted_states, :all_predicted_observations), Tuple{Vector{Vector{Vector{Vector{Float64}}}}, Vector{Vector{Vector{Vector{Float64}}}}}}}},
    posterior::NamedTuple{(:posterior_states, :prediction_states), Tuple{Vector{Vector{Float64}}, Vector{Vector{Float64}}}},
    previous_action::Union{Nothing, Vector{Int}},
    observation::Vector{Int}
) where T <: Real

    # Store beliefs in the perceptual process struct
    model.perceptual_process.posterior_states = posterior.posterior_states
    model.perceptual_process.prediction_states = posterior.prediction_states
    model.perceptual_process.observation = observation

    model.perceptual_process.predicted_states = policy_posterior.predictions.all_predicted_states
    model.perceptual_process.predicted_observations = policy_posterior.predictions.all_predicted_observations

    # Store beliefs in the action process struct
    model.action_process.posterior_policies = policy_posterior.q_pi
    model.action_process.expected_free_energy = policy_posterior.G
    model.action_process.action_posterior = action_posterior.posterior
    model.action_process.previous_action = previous_action

end

# With Learning + With action selection
function ActiveInferenceCore.store_beliefs!(
    model::AIFModel{GenerativeModel, CAVI{Learning}, ActionProcess},
    action_posterior::NamedTuple{(:posterior,), Tuple{Vector{Vector{T}}}},
    policy_posterior::NamedTuple{(:q_pi, :G, :predictions), Tuple{Vector{Float64}, Vector{Float64}, NamedTuple{(:all_predicted_states, :all_predicted_observations), Tuple{Vector{Vector{Vector{Vector{Float64}}}}, Vector{Vector{Vector{Vector{Float64}}}}}}}},
        posterior::NamedTuple{
        (:posterior_states, :prediction_states, :learning_posterior),
        Tuple{
            Vector{Vector{Float64}},
            Vector{Vector{Float64}},
            NamedTuple{
                (:A_updated, :qA, :B_updated, :qB, :D_updated, :qD),
                Tuple{T_A, T_qA, T_B, T_qB, T_D, T_qD}
            }
        }
    },
    previous_action::Union{Nothing, Vector{Int}},
    observation::Vector{Int}
)  where {
    T_A  <: Union{Vector{<:Array{Float64}}, Nothing},
    T_qA <: Union{Vector{<:Array{Float64}}, Nothing},
    T_B  <: Union{Vector{<:Array{Float64}}, Nothing},
    T_qB <: Union{Vector{<:Array{Float64}}, Nothing},
    T_D  <: Union{Vector{Vector{Float64}}, Nothing},
    T_qD <: Union{Vector{Vector{Float64}}, Nothing},
    T   <: Real
}

    # Store beliefs in the perceptual process struct
    model.perceptual_process.posterior_states = posterior.posterior_states
    model.perceptual_process.prediction_states = posterior.prediction_states
    model.perceptual_process.observation = observation

    model.perceptual_process.predicted_states = policy_posterior.predictions.all_predicted_states
    model.perceptual_process.predicted_observations = policy_posterior.predictions.all_predicted_observations

    # Store learning beliefs
    if model.perceptual_process.A_learning !== nothing && posterior.learning_posterior.A_updated !== nothing
        model.perceptual_process.A_learning.prior = posterior.learning_posterior.qA
        model.generative_model.A = posterior.learning_posterior.A_updated
    end
    if model.perceptual_process.B_learning !== nothing && posterior.learning_posterior.B_updated !== nothing
        model.perceptual_process.B_learning.prior = posterior.learning_posterior.qB
        model.generative_model.B = posterior.learning_posterior.B_updated
    end
    if model.perceptual_process.D_learning !== nothing && posterior.learning_posterior.D_updated !== nothing
        model.perceptual_process.D_learning.prior = posterior.learning_posterior.qD
        model.generative_model.D = posterior.learning_posterior.D_updated
    end

    # Store beliefs in the action process struct
    model.action_process.posterior_policies = policy_posterior.q_pi
    model.action_process.expected_free_energy = policy_posterior.G
    model.action_process.action_posterior = action_posterior.posterior
    model.action_process.previous_action = previous_action

end

# Without Learning - minus action selection
function ActiveInferenceCore.store_beliefs!(
    model::AIFModel{GenerativeModel, CAVI{NoLearning}, ActionProcess},
    action_posterior::NamedTuple{(:q_pi, :G, :predictions), Tuple{Vector{Float64}, Vector{Float64}, NamedTuple{(:all_predicted_states, :all_predicted_observations), Tuple{Vector{Vector{Vector{Vector{Float64}}}}, Vector{Vector{Vector{Vector{Float64}}}}}}}},
    posterior::NamedTuple{(:posterior_states, :prediction_states), Tuple{Vector{Vector{Float64}}, Vector{Vector{Float64}}}},
    observation::Vector{Int}
)

    # Store beliefs in the perceptual process struct
    model.perceptual_process.posterior_states = posterior.posterior_states
    model.perceptual_process.prediction_states = posterior.prediction_states
    model.perceptual_process.observation = observation

    model.perceptual_process.predicted_states = action_posterior.predictions.all_predicted_states
    model.perceptual_process.predicted_observations = action_posterior.predictions.all_predicted_observations

    # Store beliefs in the action process struct
    model.action_process.posterior_policies = action_posterior.q_pi
    model.action_process.expected_free_energy = action_posterior.G

end

# With Learning - minus action selection
function ActiveInferenceCore.store_beliefs!(
    model::AIFModel{GenerativeModel, CAVI{Learning}, ActionProcess},
    action_posterior::NamedTuple{(:q_pi, :G, :predictions), Tuple{Vector{Float64}, Vector{Float64}, NamedTuple{(:all_predicted_states, :all_predicted_observations), Tuple{Vector{Vector{Vector{Vector{Float64}}}}, Vector{Vector{Vector{Vector{Float64}}}}}}}},
    # predictions::NamedTuple{(:all_predicted_states, :all_predicted_observations), Tuple{Vector{Vector{Vector{Vector{Float64}}}}, Vector{Vector{Vector{Vector{Float64}}}}}},
        posterior::NamedTuple{
        (:posterior_states, :prediction_states, :learning_posterior),
        Tuple{
            Vector{Vector{Float64}},
            Vector{Vector{Float64}},
            NamedTuple{
                (:A_updated, :qA, :B_updated, :qB, :D_updated, :qD),
                Tuple{T_A, T_qA, T_B, T_qB, T_D, T_qD}
            }
        }
    },
    observation::Vector{Int}
)  where {
    T_A  <: Union{Vector{<:Array{Float64}}, Nothing},
    T_qA <: Union{Vector{<:Array{Float64}}, Nothing},
    T_B  <: Union{Vector{<:Array{Float64}}, Nothing},
    T_qB <: Union{Vector{<:Array{Float64}}, Nothing},
    T_D  <: Union{Vector{Vector{Float64}}, Nothing},
    T_qD <: Union{Vector{Vector{Float64}}, Nothing}
}

    # Store beliefs in the perceptual process struct
    model.perceptual_process.posterior_states = posterior.posterior_states
    model.perceptual_process.prediction_states = posterior.prediction_states
    model.perceptual_process.observation = observation

    model.perceptual_process.predicted_states = action_posterior.predictions.all_predicted_states
    model.perceptual_process.predicted_observations = action_posterior.predictions.all_predicted_observations

    # Store learning beliefs
    if model.perceptual_process.A_learning !== nothing
        model.perceptual_process.A_learning.prior = posterior.learning_posterior.qA
        model.generative_model.A = posterior.learning_posterior.A_updated
    end
    if model.perceptual_process.B_learning !== nothing
        model.perceptual_process.B_learning.prior = posterior.learning_posterior.qB
        model.generative_model.B = posterior.learning_posterior.B_updated
    end
    if model.perceptual_process.D_learning !== nothing
        model.perceptual_process.D_learning.prior = posterior.learning_posterior.qD
        model.generative_model.D = posterior.learning_posterior.D_updated
    end

    # Store beliefs in the action process struct
    model.action_process.posterior_policies = action_posterior.q_pi
    model.action_process.expected_free_energy = action_posterior.G

end

