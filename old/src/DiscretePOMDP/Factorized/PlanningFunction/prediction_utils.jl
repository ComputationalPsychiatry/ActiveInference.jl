""" Function for predicting states and observations based on the agent's perceptual process and generative model. """

function ActiveInferenceCore.policy_predictions(
    model::AIFModel{GenerativeModel, P, ActionProcess}, 
    posterior::NamedTuple{(:posterior_states, :prediction_states), Tuple{Vector{Vector{Float64}}, Vector{Vector{Float64}}}}
) where {P <: AbstractPerceptualProcess}

    all_predicted_states = get_states_prediction(posterior.posterior_states, model.generative_model.B, model.action_process.policies)
    all_predicted_observations = get_expected_obs(all_predicted_states, model.generative_model.A)

    return (all_predicted_states = all_predicted_states, all_predicted_observations = all_predicted_observations)
end

function ActiveInferenceCore.policy_predictions(
    model::AIFModel{GenerativeModel, CAVI{NoLearning}, ActionProcess}, 
    posterior::NamedTuple{
        (:posterior_states, :prediction_states)
    }
)
    all_predicted_states = get_states_prediction(posterior.posterior_states, model.generative_model.B, model.action_process.policies)
    all_predicted_observations = get_expected_obs(all_predicted_states, model.generative_model.A)

    return (all_predicted_states = all_predicted_states, all_predicted_observations = all_predicted_observations)
end

function ActiveInferenceCore.policy_predictions(
    model::AIFModel{GenerativeModel, CAVI{Learning}, ActionProcess}, 
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
    }
) where {
    T_A  <: Union{Vector{<:Array{Float64}}, Nothing},
    T_qA <: Union{Vector{<:Array{Float64}}, Nothing},
    T_B  <: Union{Vector{<:Array{Float64}}, Nothing},
    T_qB <: Union{Vector{<:Array{Float64}}, Nothing},
    T_D  <: Union{Vector{Vector{Float64}}, Nothing},
    T_qD <: Union{Vector{Vector{Float64}}, Nothing}
}

    A = posterior.learning_posterior.A_updated !== nothing ? posterior.learning_posterior.A_updated : model.generative_model.A
    B = posterior.learning_posterior.B_updated !== nothing ? posterior.learning_posterior.B_updated : model.generative_model.B

    all_predicted_states = get_states_prediction(posterior.posterior_states, B, model.action_process.policies)
    all_predicted_observations = get_expected_obs(all_predicted_states, A)

    return (all_predicted_states = all_predicted_states, all_predicted_observations = all_predicted_observations)
end


########################################
#### Utils for prediction function #####
########################################

```Util functions for prediction in Discrete POMDPs ```

""" Get Expected States """
function get_states_prediction(qs::Vector{Vector{T}} where T <: Real, B, policy::Matrix{Int64})
    n_steps, n_factors = size(policy)

    # initializing posterior predictive density as a list of beliefs over time
    qs_pi = [deepcopy(qs) for _ in 1:n_steps+1]

    # expected states over time
    for t in 1:n_steps
        for control_factor in 1:n_factors
            action = policy[t, control_factor]
            
            qs_pi[t+1][control_factor] = B[control_factor][:, :, action] * qs_pi[t][control_factor]
        end
    end

    return qs_pi[2:end]
end

""" 
    Multiple dispatch for all expected states given all policies

Multiple dispatch for getting expected states for all policies based on the agents currently
inferred states and the transition matrices for each factor and action in the policy.

qs::Vector{Vector{Real}} \n
B: Vector{Array{<:Real}} \n
policy: Vector{Matrix{Int64}}

"""
function get_states_prediction(qs::NamedTuple{<:Any, <:NTuple{N, Vector{Float64}} where {N}}, B, policy::Vector{Matrix{Int64}}
    ) 
    
    # Extracting the number of steps (policy_length) and factors from the first policy
    n_steps, n_factors = size(policy[1])

    # Number of policies
    n_policies = length(policy)
    
    # Preparing vessel for the expected states for all policies. Has number of undefined entries equal to the
    # number of policies
    qs_pi_all = Vector{Vector{Vector{Vector{Float64}}}}(undef, n_policies)

    # Looping through all policies
    for (policy_idx, policy_x) in enumerate(policy)

        # initializing posterior predictive density as a list of beliefs over time
        qs_pi = [deepcopy(qs) for _ in 1:n_steps+1]

        # expected states over time
        for t in 1:n_steps
            for control_factor in 1:n_factors
                action = policy_x[t, control_factor]
                
                qs_pi[t+1][control_factor] = B[control_factor][:, :, action] * qs_pi[t][control_factor]
            end
        end
        qs_pi_all[policy_idx] = qs_pi[2:end]
    end
    return qs_pi_all
end


""" Get Expected Observations """
function get_expected_obs(qs_pi::Vector{Vector{Vector{Float64}}}, A::Vector{Array{T,N}} where {T <: Real, N})
    n_steps = length(qs_pi)
    qo_pi = []

    for t in 1:n_steps
        qo_pi_t = Vector{Any}(undef, length(A))
        qo_pi = push!(qo_pi, qo_pi_t)
    end

    for t in 1:n_steps
        for (modality, A_m) in enumerate(A)
            qo_pi[t][modality] = dot_product(A_m, qs_pi[t])
        end
    end

    return qo_pi
end

""" Get Expected Observations """
function get_expected_obs(qs_pi::Vector{Vector{Vector{Vector{Float64}}}}, A::Vector{Array{T,N}} where {T <: Real, N})
    n_policies = length(qs_pi)
    n_steps = length(qs_pi[1])
    
    # Predefined vector to store results for all policies
    qo_pi_all = Vector{Vector{Vector{Vector{Float64}}}}(undef, n_policies)

    # Loop over each policy
    for policy_idx in 1:n_policies
        qo_pi = []

        for t in 1:n_steps
            qo_pi_t = Vector{Any}(undef, length(A))
            qo_pi = push!(qo_pi, qo_pi_t)
        end

        for t in 1:n_steps
            for (modality, A_m) in enumerate(A)
                qo_pi[t][modality] = dot_product(A_m, qs_pi[policy_idx][t])
            end
        end
        
        qo_pi_all[policy_idx] = qo_pi
    end

    return qo_pi_all
end



