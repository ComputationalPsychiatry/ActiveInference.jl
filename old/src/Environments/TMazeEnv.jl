

mutable struct TMazeEnv
    reward_prob::Float64
    reward_idx::Int64
    loss_idx::Int64
    location_factor_id::Int64
    trial_factor_id::Int64
    location_modality_id::Int64
    reward_modality_id::Int64
    cue_modality_id::Int64
    num_states::Vector{Int64}
    num_locations::Int64
    num_controls::Vector{Int64}
    num_reward_conditions::Int64
    num_cues::Int64
    num_obs::Vector{Int64}
    num_factors::Int64
    num_modalities::Int64
    reward_probs::Vector{Float64}
    transition_dist::Array{Any, 1}
    likelihood_dist::Array{Any, 1}
    _state::Array{Any, 1}
    _reward_condition_idx::Int64
    reward_condition::Vector{Int64}
    state::Array{Any, 1}

    function TMazeEnv(reward_prob::Float64;

        reward_idx::Int64 = 2,
        loss_idx::Int64 = 3,
        location_factor_id::Int64 = 1,
        trial_factor_id::Int64 = 2,
        location_modality_id::Int64 = 1,
        reward_modality_id::Int64 = 2,
        cue_modality_id::Int64 = 3,
        )
        num_states = [4, 2]
        num_locations = num_states[location_factor_id]
        num_controls = [num_locations, 1]
        num_reward_conditions = num_states[trial_factor_id]
        num_cues = num_reward_conditions
        num_obs = [num_locations, num_reward_conditions + 1, num_cues]
        num_factors = length(num_states)
        num_modalities = length(num_obs)

        reward_probs = [reward_prob, round(1-reward_prob, digits = 6)]

        new(reward_prob, reward_idx, loss_idx, location_factor_id, trial_factor_id, 
        location_modality_id, reward_modality_id, cue_modality_id, num_states, num_locations, num_controls, num_reward_conditions, num_cues, num_obs, num_factors, 
        num_modalities, reward_probs)
    end
end

function initialize_gp(env::TMazeEnv)
    env.transition_dist = construct_transition_dist(env)
    env.likelihood_dist = construct_likelihood_dist(env)
end

function step_TMaze!(env::TMazeEnv, actions)
    prob_states = Array{Any}(undef, env.num_factors)

    # Calculate the state probabilities based on actions and current state
    for factor = 1:env.num_factors
        transition_matrix = env.transition_dist[factor][:, :, Int(actions[factor])]
        current_state_vector = env._state[factor]
        prob_states[factor] = transition_matrix * current_state_vector
    end

    # Sample the next state from the probability distributions
    state = [sample_dist(ps_i) for ps_i in prob_states]

    # Construct the new state
    env._state = construct_state(env, state) 

    # Generate and return the current observation
    return get_observation(env)
end

function reset_TMaze!(env::TMazeEnv; state=nothing)
    if state === nothing
        # Initialize location state
        loc_state = ActiveInference.onehot(1, env.num_locations)

        # Randomly select a reward condition
        env._reward_condition_idx = rand(1:env.num_reward_conditions)
        env.reward_condition = ActiveInference.onehot(env._reward_condition_idx, env.num_reward_conditions)

        # Initialize the full state array
        full_state = Vector{Any}(undef, env.num_factors)
        full_state[env.location_factor_id] = loc_state
        full_state[env.trial_factor_id] = env.reward_condition

        env._state = full_state
    else
        env._state = state
    end

    # Return the current observation
    return get_observation(env)
end

function construct_transition_dist(env::TMazeEnv)

    B_locs = reshape(Matrix{Float64}(I, env.num_locations, env.num_locations), env.num_locations, env.num_locations, 1)
    B_locs = repeat(B_locs, 1, 1, env.num_locations) 
    B_locs = permutedims(B_locs, [1, 3, 2])
    
    B_trials = reshape(Matrix{Float64}(I, env.num_reward_conditions, env.num_reward_conditions), env.num_reward_conditions, env.num_reward_conditions, 1)

    B = Array{Any}(undef, env.num_factors)
    B[env.location_factor_id] = B_locs
    B[env.trial_factor_id] = B_trials

    return B
end

function construct_likelihood_dist(env::TMazeEnv)

    A_dims = [[obs_dim; env.num_states...] for obs_dim in env.num_obs]
    A = ActiveInference.array_of_any_zeros(A_dims)

    for loc in 1:env.num_states[env.location_factor_id]
        for reward_condition in 1:env.num_states[env.trial_factor_id]

            if loc == 1 # When in the center location
                A[env.reward_modality_id][1, loc, reward_condition] = 1.0

                A[env.cue_modality_id][:, loc, reward_condition] .= 1.0 / env.num_obs[env.cue_modality_id]

            elseif loc == 4  # When in the cue location
                A[env.reward_modality_id][1, loc, reward_condition] = 1.0

                A[env.cue_modality_id][reward_condition, loc, reward_condition] = 1.0

            else  # In one of the (potentially) rewarding arms
                if loc == (reward_condition + 1)
                    high_prob_idx = env.reward_idx
                    low_prob_idx = env.loss_idx
                else
                    high_prob_idx = env.loss_idx
                    low_prob_idx = env.reward_idx
                end

                # Assign probabilities based on the reward condition
                A[env.reward_modality_id][high_prob_idx, loc, reward_condition] = env.reward_probs[1]
                A[env.reward_modality_id][low_prob_idx, loc, reward_condition] = env.reward_probs[2]

                # Cue is ambiguous in the reward location
                A[env.cue_modality_id][:, loc, reward_condition] .= 1.0 / env.num_obs[env.cue_modality_id]
            end

            # Location is always observed correctly
            A[env.location_modality_id][loc, loc, reward_condition] = 1.0
        end
    end

    return A
end


function sample_dist(probabilities)

    probabilities = convert(Vector{Float64}, probabilities)

    # Ensure probabilities sum to 1
    probabilities /= sum(probabilities)

    # Julia's Categorical returns a 1-based index
    sample_onehot = rand(Multinomial(1, probabilities))
    return findfirst(sample_onehot .== 1)
end

function get_observation(env::TMazeEnv)

    # Calculate the probability of observations based on the current state and the likelihood distribution
    prob_obs = [ActiveInference.dot_product(A_m, env._state) for A_m in env.likelihood_dist]

    # Sample from the probability distributions to get actual observations
    obs = [sample_dist(po_i) for po_i in prob_obs]

    return obs
end

function construct_state(env::TMazeEnv, state_tuple)
    # Create an array of any
    state = Vector{Any}(undef, env.num_factors)

    # Populate the state array with one-hot encoded vectors
    for (f, ns) in enumerate(env.num_states)
        state[f] = ActiveInference.onehot(state_tuple[f], ns)
    end

    return state
end