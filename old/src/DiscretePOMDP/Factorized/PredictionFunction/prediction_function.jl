# """ Function for predicting states and observations based on the agent's perceptual process and generative model. """

# function ActiveInferenceCore.prediction(
#     model::AIFModel{GenerativeModel, CAVI{NoLearning}, ActionProcess}, 
#     posterior::NamedTuple{(:posterior_states, :prediction_states), Tuple{Vector{Vector{Float64}}, Vector{Vector{Float64}}}}
# )

#     all_predicted_states = get_states_prediction(posterior.posterior_states, model.generative_model.B, model.action_process.policies)
#     all_predicted_observations = get_expected_obs(all_predicted_states, model.generative_model.A)

#     return (all_predicted_states = all_predicted_states, all_predicted_observations = all_predicted_observations)
# end

# function ActiveInferenceCore.prediction(
#     model::AIFModel{GenerativeModel, CAVI{Learning}, ActionProcess}, 
#     posterior::NamedTuple{
#         (:posterior_states, :prediction_states, :learning_posterior),
#         Tuple{
#             Vector{Vector{Float64}},
#             Vector{Vector{Float64}},
#             NamedTuple{
#                 (:A_updated, :qA, :B_updated, :qB, :D_updated, :qD),
#                 Tuple{T_A, T_qA, T_B, T_qB, T_D, T_qD}
#             }
#         }
#     }
# ) where {
#     T_A  <: Union{Vector{<:Array{Float64}}, Nothing},
#     T_qA <: Union{Vector{<:Array{Float64}}, Nothing},
#     T_B  <: Union{Vector{<:Array{Float64}}, Nothing},
#     T_qB <: Union{Vector{<:Array{Float64}}, Nothing},
#     T_D  <: Union{Vector{Vector{Float64}}, Nothing},
#     T_qD <: Union{Vector{Vector{Float64}}, Nothing}
# }

#     A = posterior.learning_posterior.A_updated !== nothing ? posterior.learning_posterior.A_updated : model.generative_model.A
#     B = posterior.learning_posterior.B_updated !== nothing ? posterior.learning_posterior.B_updated : model.generative_model.B

#     all_predicted_states = get_states_prediction(posterior.posterior_states, B, model.action_process.policies)
#     all_predicted_observations = get_expected_obs(all_predicted_states, A)

#     return (all_predicted_states = all_predicted_states, all_predicted_observations = all_predicted_observations)
# end
 