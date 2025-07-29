
module siMaze


import Plots
import Random
import Setfield: @set
#import StaticArrays as SA
import Statistics

using Format
using Infiltrator
using Revise

import ActiveInference.ActiveInferenceFactorized as AI

include("./lib/make_model.jl")
include("./environment.jl")

include("./lib/make_A.jl")
include("./lib/make_B.jl")
include("./lib/simulate.jl")
include("./lib/make_policies.jl")


Random.seed!(51233) # Set random seed for reproducibility
Plots.scalefontsizes()
Plots.scalefontsizes(0.8)

# @infiltrate; @assert false



####################################################################################################

function run_example()
    
    experiment_id = 1
    #@infiltrate; @assert false

    #=
    Location matrix is in column major order. All cell location, e.g., (9,3) are:  (Y -- column, 
    X -- row) from upper left corner of grid.
    =#

    if experiment_id == 1
        experiment = format("siMaze{}", experiment_id)
        grid_dims = (9,9)
        grid_size = prod(grid_dims)
        grid = reshape(1:grid_size, (grid_dims[2], grid_dims[1]))
        cells = [[i, j] for j in 1:grid_dims[2] for i in 1:grid_dims[1] ]
        
        global CONFIG = (
            experiment = experiment,
            grid_dims = grid_dims,
            grid = grid,
            cells = cells,
            start_cell = [9,2], 
            policy_length = 3,
            number_simulation_steps = 10,
            number_simulations = 1,
        )
    end

    # get_settings and modify as needed
    settings = AI.get_settings()
    settings = @set settings.EFE_over = [:policies, :actions][2]
    settings = @set settings.graph_postprocessing_method = [:G_prob, :G_prob_q_pi][2]
    settings = @set settings.EFE_reduction = :sum
    settings = @set settings.policy_inference_method = [:sophisticated  :standard][2]  
    settings = @set settings.graph = [:explicit  :none][2]
    settings = @set settings.use_param_info_gain = false
    settings = @set settings.SI_use_pymdp_methods = false
    settings = @set settings.verbose = true

    # to run standard inference with an explicit graph, use:
    #settings = @set settings.policy_inference_method = :standard 
    #settings = @set settings.graph = :explicit

    # SI thresholds can be set here
    #settings = @set settings.SI_observation_prune_threshold = 1/16
    #settings = @set settings.SI_policy_prune_threshold = 1/16
    
    # get_parameters and modify as needed
    parameters = AI.get_parameters()
    parameters = @set parameters.gamma = 1.0
    
    model = make_model(CONFIG)
    model = make_policies(model, CONFIG)
    
    make_A(model, CONFIG)
    make_B(model, CONFIG)

    env = init_env(model, CONFIG.start_cell)
    
    # no preferences for loc or walls
    model.preferences.safe_pref.C[:] = [1.0, -1.0] # Preference for safe Locations

    # prior for starting location is deterministic
    start_idx = start_id = findfirst(x -> x == CONFIG.start_cell, model.states.loc.labels)
    model.states.loc.D[start_idx] = 1
    
    # how many simulations?
    if settings.action_selection == "deterministic"
        number_simulations = 1
    else
        # stochastic
        number_simulations = CONFIG.number_simulations
    end

    history = []
    #agent = nothing

    to_label = [(start = CONFIG.start_cell,)]  # for graphing    
    
    #@infiltrate; @assert false
    for simulation_number in 1:number_simulations
        global CONFIG
        agent = AI.create_agent(
            model,
            settings, 
        )
        println("\n\n")

        results = simulate(model, agent, env, CONFIG, to_label, simulation_number)
        push!(history, results)
    end
    
    @infiltrate; @assert false

end


end  # -- module