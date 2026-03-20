using IterTools
using LinearAlgebra
using ActiveInference
using Test

""" Test Agent """

@testset "Single Factor Condition - Default Settings" begin

    # Initializse States, Observations, and Controls
    states = [25]
    observations = [25]
    controls = [2]
    policy_length = 1

    # Generate random Generative Model 
    A, B = create_matrix_templates(states, observations, controls, policy_length, "random");

    # Initialize the parameters struct
    parameters = init_pomdp_aif_parameters(A = A, B = B)

    # Initialize the settings struct
    settings = init_pomdp_aif_settings()

    # Initialize agent with default settings/parameters
    aif = init_pomdp_aif(
        parameters = parameters, 
        settings = settings
    );

    # Give observation to agent and run state inference
    observation = [rand(1:observations[i]) for i in axes(observations, 1)]
    qs = infer_states!(aif, observation)

    # Run policy inference
    q_pi, G = infer_policies!(aif)

    # Sample action
    action = sample_action!(aif)

    @test sum(aif.states.q_pi) == 1.0
    @test length(aif.parameters.D[1]) == states[1]
    @test length(aif.parameters.B[1][1,1,:]) == controls[1]

end


@testset "If There are more factors - Default Settings" begin

    # Initializse States, Observations, and Controls
    states = [64,2]
    observations = [64,2]
    controls = [5,1]
    policy_length = 1

    # Generate random Generative Model 
    A, B = create_matrix_templates(states, observations, controls, policy_length, "random");

    # Initialize the parameters struct
    parameters = init_pomdp_aif_parameters(A = A, B = B)

    # Initialize the settings struct
    settings = init_pomdp_aif_settings()

    # Initialize agent with default settings/parameters
    aif = init_pomdp_aif(
        parameters = parameters, 
        settings = settings
    );

    # Give observation to agent and run state inference
    observation = [rand(1:observations[i]) for i in axes(observations, 1)]
    qs = infer_states!(aif, observation)

    # Run policy inference
    q_pi, G = infer_policies!(aif)

    # Sample action
    action = sample_action!(aif)

    @test round(sum(aif.states.q_pi), digits = 6) == 1.0
    @test length(aif.parameters.D[1]) == states[1]
    @test length(aif.parameters.D[2]) == states[2]
    @test length(aif.parameters.B[1][1,1,:]) == controls[1]

end


@testset "Provide custom settings" begin

    # Initializse States, Observations, and Controls
    states = [64,2]
    observations = [64,2]
    controls = [5,2]
    policy_length = 3

    # Generate random Generative Model 
    A, B, C, D = create_matrix_templates(states, observations, controls, policy_length, "random");


    # Initialize the parameters struct
    parameters = init_pomdp_aif_parameters(A = A, B = B, C = C, D = D)

    # Initialize the settings struct
    settings = init_pomdp_aif_settings(
        policy_length = 3,
        use_states_info_gain = true,
        action_selection = "deterministic",
        use_utility = true,
    )

    # Initialize agent with default settings/parameters
    aif = init_pomdp_aif(
        parameters = parameters, 
        settings = settings
    );

    # Give observation to agent and run state inference
    observation = [rand(1:observations[i]) for i in axes(observations, 1)]
    qs = infer_states!(aif, observation)

    # Run policy inference
    q_pi, G = infer_policies!(aif)

    # Sample action deterministically 
    action = sample_action!(aif)

    # And infer new state
    observation = [rand(1:observations[i]) for i in axes(observations, 1)]
    qs_2 = infer_states!(aif, observation)

    @test round(sum(aif.states.q_pi), digits = 6) == 1.0
    @test length(aif.parameters.D[1]) == states[1]
    @test length(aif.parameters.D[2]) == states[2]
    @test length(aif.parameters.B[1][1,1,:]) == controls[1]
    @test length(aif.parameters.B[2][1,1,:]) == controls[2]
end


@testset "Learning with custom parameters" begin

    # Initializse States, Observations, and Controls
    states = [64,2]
    observations = [64,2]
    controls = [5,1]
    policy_length = 2

    # Generate random Generative Model 
    A, B, C, D = create_matrix_templates(states, observations, controls, policy_length, "random");

    # pA concentration parameter
    pA = deepcopy(A)
    for i in eachindex(pA)
        pA[i] .= 1.0
    end

    # pB concentration parameter
    pB = deepcopy(B)
    for i in eachindex(pB)
        pB[i] .= 1.0
    end

    # pD concentration parameter
    pD = deepcopy(D)
    for i in 1:length(D)
        pD[i] .= 1.0
    end
    
    # Initialize the parameters struct
    parameters = init_pomdp_aif_parameters(pA = pA, pB = pB, C = C, pD = pD,
        lr_pA = 0.5, fr_pA = 1.0,
        lr_pB = 0.6, fr_pB = 1.0,
        lr_pD = 0.7, fr_pD = 1.0,
        alpha = 2.0,
        gamma = 2.0,)

    # Initialize the settings struct
    settings = init_pomdp_aif_settings(
        policy_length = 2,
        use_param_info_gain = true
    )

    # Initialize agent with default settings/parameters
    aif = init_pomdp_aif(
        parameters = parameters, 
        settings = settings
    );

    ## Run inference with Learning
    for t in 1:2
        # Give observation to agent and run state inference
        observation = [rand(1:observations[i]) for i in axes(observations, 1)]
        qs = infer_states!(aif, observation)

        update_parameters!(aif)
    
        # Run policy inference
        q_pi, G = infer_policies!(aif)
    
        # Sample action
        action = sample_action!(aif)
    end

    @test round(sum(aif.states.q_pi), digits = 6) == 1.0
    @test round(sum(aif.parameters.A[1]), digits = 6) == states[1] * states[2]
    @test round(sum(aif.parameters.A[2]), digits = 6) == states[1] * states[2]
    @test round(sum(aif.parameters.B[1]), digits = 6) == states[1] * controls[1]
    @test round(sum(aif.parameters.B[2]), digits = 6) == states[2] * controls[2]


end
