
# @infiltrate; @assert false
module Benchmark

folder = "./paper_maze/JULIA/julia-SI-actions-explicit-graph"


import Plots
import Random
import Setfield: @set
import Statistics
import Dates
import Distributions

using CSV, DataFrames

using Format
using Infiltrator
using Revise

import ActiveInference.ActiveInferenceFactorized as AI

include("lib/make_model.jl")
include("lib/environment.jl")

include("lib/make_A.jl")
include("lib/make_B.jl")
include("lib/make_policies.jl")
include("lib/custom_sample_action.jl")
include("lib/animate.jl")



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
    grid_dims = (8,8)
    grid_size = prod(grid_dims)
    grid_s = reshape(1:grid_size, (grid_dims[2], grid_dims[1]))
    cells = [[i, j] for j in 1:grid_dims[2] for i in 1:grid_dims[1] ]

    global CONFIG = (
        experiment = format("siMaze{}", 1),
        grid_dims = grid_dims,
        grid = grid_s,
        cells = cells,
        start_cell = [8,2],
        policy_length = policy_length,
        number_simulation_steps = num_iterations,
        number_simulations = 1,
        float_type = Float32,
    )
    
    # Reinitialize everything
    model = make_model(CONFIG)

    make_A(model, CONFIG);
    make_B(model, CONFIG);

    PREFERENCES_matrix = [
        -1.000  -1.000  -1.000  -1.000  -1.000  -1.000  -1.000  -1.000;
        -1.000  -0.433  -0.178   0.000   0.051   0.000  -0.178  -1.000;
        -1.000  -1.000  -1.000  -0.178  -1.000  -1.000   0.106  -1.000;
        -1.000  -1.000   0.293   0.568   0.684  -1.000   0.293  -1.000;
        -1.000  -1.000   0.568  -1.000   1.000   0.684   0.368  -1.000;
        -1.000  -1.000   0.293  -1.000  -1.000  -1.000   0.293  -1.000;
        -1.000  -0.178   0.106   0.293   0.568   0.293   0.106  -1.000;
        -1.000  -0.433  -1.000  -1.000  -1.000  -1.000  -1.000  -1.000
    ]

    model.preferences.loc_pref.C[:] =  vec(PREFERENCES_matrix)
    start_idx = start_id = findfirst(x -> x == CONFIG.start_cell, model.states.loc.labels)
    model.states.loc.D[start_idx] = 1
    parameters = AI.get_parameters()
    parameters = @set parameters.gamma = 16.0
    model = make_policies(model, CONFIG)
    
    settings = AI.get_settings()
    settings = @set settings.EFE_over = [:policies,  :actions][2]
    settings = @set settings.policy_postprocessing_method = [:G_prob,  :G_prob_q_pi][1] 
    settings = @set settings.policy_inference_method = [:standard,  :sophisticated][2]
    settings = @set settings.graph = [:none, :explicit, :implicit][2]
    settings = @set settings.EFE_reduction = [:sum, :min_max, :custom][2]
    
    settings = @set settings.use_param_info_gain = false
    settings = @set settings.SI_observation_prune_threshold = 1/16  
    settings = @set settings.SI_policy_prune_threshold = .2
    settings = @set settings.verbose = false
    settings = @set settings.SI_use_pymdp_methods = false
    settings = @set settings.action_selection = :stochastic
     
    
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
            
            action, policy, G, q_pi = custom_sample_action!(agent, t)
            observation = step_env!(env, action)

            push!(agent.history.policy, policy)
            push!(agent.history.G, G)
            push!(agent.history.q_pi, q_pi)
            push!(agent.history.observation, observation)

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
        Symbol("folder") => repeat([folder], num_iterations),
        Symbol("n_sim_steps") => repeat([num_iterations], num_iterations),
        Symbol("horizon") => repeat([horizon], num_iterations),
        Symbol("sim_time") => repeat([sim_time], num_iterations),
        
        # for graph experiments only
        Symbol("graph_initial_node_count") => agent.history.graph_initial_node_count,  
        Symbol("graph_final_node_count") => agent.history.graph_final_node_count,  
        Symbol("graph_min_level") => agent.history.graph_min_level, 
        Symbol("graph_max_level") => agent.history.graph_max_level, 

        Symbol("action") => [x.move for x in agent.history.action],
        Symbol("observation") => [x.loc_obs for x in agent.history.observation],
        Symbol("G") => agent.history.G,
        Symbol("q_pi") => agent.history.q_pi,
        Symbol("policy") => [values(x) for x in agent.history.policy]
        
    )
    #CSV.write(joinpath(folder, format("results_horz_{}.csv", horizon)), df_total)

    printfmtln("\nTime: {}", sim_time)
    printfmtln("Last Observation= {}\n", last_obs)
    
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
    for horizon in 5:5
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
