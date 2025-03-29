using IterTools
using LinearAlgebra
using ActiveInference
using Test

""" Quick tests """

@testset "Multiple Factors/Modalities Default Settings" begin

    # Initializse States, Observations, and Controls
    states = [5,2]
    observations = [5, 4, 2]
    controls = [2,1]
    policy_length = 1

    # Generate random Generative Model 
    A,B = create_matrix_templates(states, observations, controls, policy_length, "random");

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
    infer_states!(aif, observation)

    # Run policy inference
    infer_policies!(aif)

    # Sample action
    sample_action!(aif)

    @test round(sum(aif.states.q_pi), digits = 6) == 1.0
    @test length(aif.parameters.D[1]) == states[1]
    @test length(aif.parameters.D[2]) == states[2]
    @test length(aif.parameters.B[1][1,1,:]) == controls[1]
    @test length(aif.parameters.B[2][1,1,:]) == controls[2]

end