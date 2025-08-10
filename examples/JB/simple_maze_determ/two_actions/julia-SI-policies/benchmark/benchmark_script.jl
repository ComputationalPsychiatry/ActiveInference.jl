
# @infiltrate; @assert false



module Benchmark

folder = "./two_actions/julia-SI-policies"


#using Pkg
#Pkg.activate(".")
import Plots
import Random
import Setfield: @set
import Statistics
using CSV, DataFrames

using Format
using Infiltrator
using Revise

import ActiveInference.ActiveInferenceFactorized as AI

include("lib/make_model.jl")
include("lib/environment.jl")

include("lib/make_A.jl")
include("lib/make_B.jl")
include("lib/simulate.jl")
include("lib/make_policies.jl")
include("lib/custom_sample_action.jl")
include("lib/animate.jl")

Random.seed!(51233)

# Saved as @eval global Main.nv_from_package = $nv
#function count_nodes()
#    return nv_from_package
#end

# Benchmarking function 
function benchmark_planning(agent, observation)
    
    result = @timed begin
        AI.infer_policies!(agent, observation)
    end
    
    # Extract timing and memory info
    elapsed_time = result.time  # seconds
    allocated_bytes = result.bytes  # bytes allocated
    allocated_mib = allocated_bytes / (1024 * 1024)  # Convert to MiB
    
    return elapsed_time, allocated_mib
end

# Main simulation function
function run_simulation(horizon, num_iterations)
    policy_length = horizon 
    grid_dims = (9,9)
    grid_size = prod(grid_dims)
    grid_s = reshape(1:grid_size, (grid_dims[2], grid_dims[1]))
    cells = [[i, j] for j in 1:grid_dims[2] for i in 1:grid_dims[1] ]

    global CONFIG = (
        experiment = format("siMaze{}", 1),
        grid_dims = grid_dims,
        grid = grid_s,
        cells = cells,
        start_cell = [9,2],
        policy_length = policy_length,
        number_simulation_steps = num_iterations,
        number_simulations = 1,
        float_type = Float32,
    )
    
    # Reinitialize everything
    model = make_model(CONFIG)

    make_A(model, CONFIG);
    make_B(model, CONFIG);

    model.preferences.loc_pref.C[:] = [
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  # column 1
    -1.0, -1.0, -1.0, -1.0, -0.4, -0.6, -0.7, -0.8, -1.0,  # column 2
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.75, -1.0, # column 3
    -1.0, -1.0, -1.0,  0.0, -0.2, -0.4, -0.6, -0.7, -1.0, # column 4
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.65, -1.0,  # column 5
    -1.0, -1.0,  0.4,  0.2,  0.05, -0.15, -0.35, -0.55, -1.0, # column 6
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.4, -1.0,  # column 7
    -1.0,  1.0,  0.65, 0.4,  0.25, 0.15, 0.0, -0.2, -1.0, # column 8
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0   # column 9
    ]
    start_idx = start_id = findfirst(x -> x == CONFIG.start_cell, model.states.loc.labels)
    model.states.loc.D[start_idx] = 1
    parameters = AI.get_parameters()
    parameters = @set parameters.gamma = 1.0
    model = make_policies(model, CONFIG)
    
    settings = AI.get_settings()
    settings = @set settings.EFE_over = [:policies,  :actions][1]
    settings = @set settings.policy_postprocessing_method = [:G_prob,  :G_prob_q_pi][1] 
    settings = @set settings.policy_inference_method = [:standard,  :sophisticated][2]
    settings = @set settings.graph = [:none, :explicit, :implicit][2]
    settings = @set settings.EFE_reduction = [:sum, :min_max, :custom][2]
    
    settings = @set settings.use_param_info_gain = false
    settings = @set settings.SI_observation_prune_threshold = 1/16  
    settings = @set settings.SI_policy_prune_threshold = 1/16
    settings = @set settings.verbose = false
    settings = @set settings.SI_use_pymdp_methods = false
    settings = @set settings.action_selection = :deterministic
    
    
    
    
    global env = init_env(model, CONFIG.start_cell)
    agent = AI.create_agent(model, settings, CONFIG.float_type, parameters)
    observation = step_env!(env, nothing)

    planning_times = Float64[]
    planning_mibs = Float64[]
    
    # Time the entire simulation
    sim_result = @timed begin
        for t in 1:num_iterations
            AI.infer_states!(agent, observation)
            printfmtln("\n-----------\nt={}", t)
            for (k,v) in zip(keys(agent.qs), agent.qs)
                printfmtln("    {}: argmax= {}", k, argmax(v))
            end

            elapsed_time, mem_mib = benchmark_planning(agent, observation)
            
            action = custom_sample_action!(agent, t)
            observation = step_env!(env, action)
            
            push!(planning_times, elapsed_time)
            push!(planning_mibs, mem_mib)

            #cell_obs = model.states.loc.labels[obs.loc_obs]

            printfmtln("Grid location at time {}, after action: {}", t, observation)
            
            
        end
    end
    
    sim_time = sim_result.time
    
    last_obs = observation  
    #node_count = count_nodes()
    
    # Save total simulation summary (matching Python column names exactly)
    df_total = DataFrame(
        Symbol("std_julia_time_$(horizon)") => [sim_time],
        #Symbol("std_julia_nodes_$(horizon)") => [node_count],
        Symbol("std_julia_last_obs_$(horizon)") => [string(last_obs)]
    )
    #CSV.write("std_actions_none_julia_total_$(horizon).csv", df_total)

    df_planning = DataFrame(
        std_julia_planning_time = planning_times,
        std_julia_planning_MiB = planning_mibs
    )
    println("Time:", sim_time)
    println("Last Observation", last_obs)
    #CSV.write("std_actions_none_julia_planning_$(horizon).csv", df_planning)
    return env.history
end



# Make env global for plotting
global env = nothing

function run()
    # Main execution loop
    
    
    # writes results to folder where REPL is opened
    if ispath(folder)
        for file in readdir(folder)
            rm(joinpath(folder, file))
        end
    else
        mkpath(folder)
    end
    
    
    n_sim_steps = 10
    for horizon in 12:12
        println("Running horizon $(horizon)...")
        run_simulation(horizon, n_sim_steps)
        println("\nFinished policy len $(horizon).\n")
    end

    animate!(env.history[2:n_sim_steps+2], CONFIG.policy_length)

    @infiltrate; @assert false

end

######################################################################################################################


#=
standard inference, no graph, horizon 6: policy goes to cell 31, actions goes to cell 14
=#


end  # -- module