
module GridRL


import Plots
import Random
import Setfield: @set
#import StaticArrays as SA
import Statistics

using Format
using Infiltrator
using Revise

import ActiveInference.ActiveInferenceFactorized as AI

include("./lib/structs.jl")
include("./lib/make_A.jl")
include("./lib/make_B.jl")
include("./lib/make_plots.jl")
include("./lib/make_env.jl")
include("./lib/simulate.jl")
include("./lib/make_model.jl")

Random.seed!(51233) # Set random seed for reproducibility
Plots.scalefontsizes()
Plots.scalefontsizes(0.8)

# @infiltrate; @assert false



####################################################################################################

function run_example()
    
   

    
    if !ispath("./pngs")
        # make folders
        mkdir("./pngs")
        #mkdir("./gifs")
    end

    experiment_id = 2
    #@infiltrate; @assert false

    #=
    All cell location, e.g., (9,3) are:  (nY -- column height, nX -- row width) from upper left 
    corner of grid.
    =#
    
    if experiment_id == 1
        experiment = format("RL{}", experiment_id)
        grid_dims = (5,5)
        grid_size = prod(grid_dims)
        grid = permutedims(reshape(1:grid_size, (grid_dims[2], grid_dims[1])), [2,1])
        cells = [[i, j] for i in 1:grid_dims[1] for j in 1:grid_dims[2]]
        
        global CONFIG = (
            # simple 1-D grid, no clues, only one reward
            experiment = experiment,
            grid_dims = grid_dims,
            grid = grid,
            cells = cells,
            start_cell = [5,1], 
            n_actions = 1,
            
            #=
            walls = [
                # wall A
                [([8, i], :DOWN) for i in 1:6],  # i.e., lower edge of cell (8,1)
                
                # wall B & D
                [([i, 7], :LEFT) for i in 7:8],
                [([i, 7], :RIGHT) for i in 7:8],
                
                # wall C
                [([7,7], :UP)],

                # wall E
                [([9, i], :UP) for i in 8:9],

                # wall F
                [([4, i], :DOWN) for i in 3:5],

                # wall G
                [([i, 5], :RIGHT) for i in 3:4],
                
                # walls HIJK
                [([3,5], :DOWN)],
                [([2,5], :DOWN)],
                [([2,4], :RIGHT)],
                [([1,4], :DOWN)],
            ],
            =#
            walls = [],
            stop_cells = [],
            
            B_true = missing,
            policy_length = 2,
            number_simulation_steps = 500,
            number_simulations = 1,
            use_filtering = false,
        )
    end


    if experiment_id == 2
        experiment = format("RL{}", experiment_id)
        grid_dims = (5,5)
        grid_size = prod(grid_dims)
        grid = permutedims(reshape(1:grid_size, (grid_dims[2], grid_dims[1])), [2,1])
        cells = [[i, j] for i in 1:grid_dims[1] for j in 1:grid_dims[2]]
        
        global CONFIG = (
            # simple 1-D grid, no clues, only one reward
            experiment = experiment,
            grid_dims = grid_dims,
            grid = grid,
            cells = cells,
            start_cell = [5,1], 
            n_actions = 2,
            
            #=
            walls = [
                # wall A
                [([8, i], :DOWN) for i in 1:6],  # i.e., lower edge of cell (8,1)
                
                # wall B & D
                [([i, 7], :LEFT) for i in 7:8],
                [([i, 7], :RIGHT) for i in 7:8],
                
                # wall C
                [([7,7], :UP)],

                # wall E
                [([9, i], :UP) for i in 8:9],

                # wall F
                [([4, i], :DOWN) for i in 3:5],

                # wall G
                [([i, 5], :RIGHT) for i in 3:4],
                
                # walls HIJK
                [([3,5], :DOWN)],
                [([2,5], :DOWN)],
                [([2,4], :RIGHT)],
                [([1,4], :DOWN)],
            ],
            =#
            walls = [],
            stop_cells = [],
            
            B_true = missing,
            policy_length = 2,
            number_simulation_steps = 500,
            number_simulations = 1,
            use_filtering = false,
            float_type = Float32,
        )
    end
    
    
    # get_settings and modify as needed
    settings = AI.get_settings()
    settings = @set settings.EFE_over = [:policies,  :actions][1]
    settings = @set settings.policy_postprocessing_method = [:G_prob,  :G_prob_q_pi][1] 
    settings = @set settings.policy_inference_method = [:standard,  :sophisticated][1]
    settings = @set settings.graph = [:none, :explicit, :implicit][1]
    
    settings = @set settings.use_param_info_gain = false
    settings = @set settings.SI_observation_prune_threshold = 0.039 #1/16  
    settings = @set settings.SI_policy_prune_threshold = 1/16
    settings = @set settings.verbose = true
    settings = @set settings.SI_use_pymdp_methods = false
    settings = @set settings.action_selection = :deterministic

    model = make_model(CONFIG)

    start_cell = CONFIG.start_cell 
    env = RLEnv(model, start_cell)

    model = make_policies(model, CONFIG, env)

    make_A(model, CONFIG)
    model = make_B(model, CONFIG)

    # no preferences for model.obs.loc_obs.C
    
    # prior for starting location is deterministic
    D = model.states.loc.D
    D[findfirst(x -> x == CONFIG.start_cell, model.states.loc.labels)] = 1
    model = @set model.states.loc.D = D
    
    ### Initialize agent
    if true
        pB = deepcopy(model.states.loc.B)
        scale_concentration_parameter = 2.0
        pB .*= scale_concentration_parameter
        model = @set model.states.loc.pB = pB
    end
    #@infiltrate; @assert false
    # run
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
            CONFIG.float_type 
        )
        println("\n\n")

        # give env the same A,B etc. model as agent
        #@infiltrate; @assert false
        start_cell = CONFIG.start_cell 
        env = RLEnv(model, start_cell)
        
        add_walls(env, CONFIG.walls)
        CONFIG = make_B_true(model, env, CONFIG)
        
        #Assess.assess_results(CONFIG, agent, model, to_label)
        #@infiltrate; @assert false
        
        results = simulate(model, agent, env, CONFIG, to_label, simulation_number)
        push!(history, results)
    end
    
    @infiltrate; @assert false

    
end


end  # -- module