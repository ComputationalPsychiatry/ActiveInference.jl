using Pkg
Pkg.activate(".")
#Pkg.develop(path="AIF_Julia//ActiveInference.jl")
using Revise
using ActiveInference.ActiveInferenceFactorized
using LinearAlgebra
using IterTools
using Serialization, DataFrames
using Plots
include("generative_model.jl")
include("policies.jl")
include("animate.jl")


###################################################################
######################### SIMULATION ##############################
###################################################################
policy_length = 2

policies = make_policies(policy_length)

settings = Dict(
    "policy_len" => policy_length,
    "action_selection" => "deterministic"
)

metamodel = MetaModel(obs_deps, obs_dims, state_deps, state_dims, action_deps, action_dims, policies)
agent = init_aif(A, B, C=C, D=D,
    parameters=parameters,
    settings=settings,
    sophisticated_inference=true,
    use_SI_graph_for_standard_inference=true,
    graph_postprocessing_method = "G_prob_method",
    use_sum_for_calculating_G = true,
    metamodel=metamodel
)

environment = init_env(MAZE, PREFERENCES, 18)

observation = [18, 1, 2]
T = 13
for t in 1:T
    infer_states!(agent, observation)
    infer_policies!(agent)
    action = sample_action!(agent)
    observation = step!(environment, action)
end



###################################################################
####################### Benchmark Script ##########################
###################################################################

#=
println("Starting Simulation...")

# Function to run simulation for a given policy length
function run_simulation(policy_length::Int)
    policies = make_policies(policy_length)
    
    settings = Dict(
        "policy_len" => policy_length,
        "action_selection" => "deterministic"
    )

    metamodel = MetaModel(obs_deps, obs_dims, state_deps, state_dims, action_deps, action_dims, policies)

    agent = init_aif(A, B, C=C, D=D,
        parameters=parameters,
        settings=settings,
        sophisticated_inference=true,
        use_SI_graph_for_standard_inference=true,
        graph_postprocessing_method = "G_prob_method",
        use_sum_for_calculating_G = true,
        metamodel=metamodel
    )

    environment = init_env(MAZE, PREFERENCES, 18)
    observation = [18, 1, 2]
    T = 13

    # Initialize DataFrames
    #infer_policies_results = DataFrame(timestep=Int[], time_jl=Float64[], MB_jl=Float64[])
    #total_results = DataFrame(total_time_jl = [0.0])

    # Run simulation
    total_simulation = @timed begin
        for t in 1:T
            infer_states!(agent, observation)

            # Time and memory for infer_policies!
            infer_stats = @timed infer_policies!(agent)
            action = sample_action!(agent)
            observation = step!(environment, action)

            # Save infer_policies timing and memory (bytes to MiB to MB)
            infer_time = infer_stats.time
            infer_MB = (infer_stats.bytes / 1024 / 1024)

            #push!(infer_policies_results, (timestep=t, time_jl=infer_time, MB_jl=infer_MB))
        end
    end
    println("FINAL OBSERVATION: ", observation)
    #total_results[1, :total_time_jl] = total_simulation.time

    # Save results to .jls files
    #serialize("Sophisticated/Benchmark_Script/Julia_data/infer_policies_results_$(policy_length).jls", infer_policies_results)
    #serialize("Sophisticated/Benchmark_Script/Julia_data/total_results_$(policy_length).jls", total_results)

    println("Simulation for policy_length=$policy_length completed and saved.")
end

for policy_length in 1:8
    println("Now running sim_$policy_length")
    run_simulation(policy_length)
end

=#