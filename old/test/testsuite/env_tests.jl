using IterTools
using LinearAlgebra
using ActiveInference
using Test
using ActiveInference.Environments

@testset "Testing T-Maze Environment - Default Settings" begin

    # Initializse States, Observations, and Controls
    states = [4, 2]
    observations = [4, 3, 2]
    controls = [4, 1]
    policy_length = 2

    # Generate random Generative Model 
    A, B, C, D, E = create_matrix_templates(states, observations, controls, policy_length, "random");

    # Initialize the parameters struct
    parameters = init_pomdp_aif_parameters(A = A, B = B, C = C, D = D, E = E);

    # Initialize the settings struct
    settings = init_pomdp_aif_settings(policy_length = policy_length)

    # Initialize agent with default settings/parameters
    aif = init_pomdp_aif(
        parameters = parameters, 
        settings = settings
    );

    # Initialize T-Maze Environment
    Env = TMazeEnv(0.8)
    initialize_gp(Env)

    # set to run for 10 steps
    T = 10

    # provide initial observation
    obs = reset_TMaze!(Env)

    # Creating a for-loop that loops over the perception-action-learning loop T amount of times
    for t = 1:T

        # Infer states based on the current observation
        infer_states!(aif, obs)

        # Infer policies and calculate expected free energy
        infer_policies!(aif)

        # Sample an action based on the inferred policies
        chosen_action = sample_action!(aif)

        # Feed the action into the environment and get new observation.
        obs = step_TMaze!(Env, chosen_action)

    end

end


@testset "Testing Epistemic Chaining Environment - Default Settings" begin

    # Setting number of states, observations and controls for the generative model
    n_states = [35, 4, 2]
    n_obs = [35, 5, 3, 3]
    n_controls = [5, 1, 1]
    policy_length = 1

    # Using function for generating A and B matrices with random inputs
    A, B = create_matrix_templates(n_states, n_obs, n_controls, policy_length, "random");

    settings = init_pomdp_aif_settings(
        policy_length = 1,
        use_utility = false,
        use_states_info_gain = false,
        use_param_info_gain = false
    )

    parameters = init_pomdp_aif_parameters(A = A, B = B)


    aif = ActiveInference.init_pomdp_aif(
        parameters = parameters, 
        settings = settings
    );


    # Initializing environment
    start_loc = (1,1)
    cue1_location = (3, 1)
    cue2_loc = "L4"
    reward_cond = ("BOTTOM")
    actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    grid_locations = collect(Iterators.product(1:5, 1:7))

    Env = EpistChainEnv(start_loc, cue1_location, cue2_loc, reward_cond, grid_locations)

    # Generate random observation
    obs = Int[]
    for (i, j) in enumerate(n_obs)
        observation = rand(1:j)
        push!(obs, observation) 
    end

    # Set timesteps
    T = 5

    # Run simulation
    for t in 1:T

        qs = infer_states!(aif, obs)

        q_pi, G = infer_policies!(aif)

        chosen_action_id = sample_action!(aif)
        choice_action = actions[chosen_action_id[1]]

        loc_obs, cue1_obs, cue2_obs, reward_obs = step!(Env, choice_action)

    end

end
