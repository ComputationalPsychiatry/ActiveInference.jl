
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

function run_grid_example()
    
   

    
    if !ispath("./pngs")
        # make folders
        mkdir("./pngs")
        #mkdir("./gifs")
    end

    experiment_id = 1
    #@infiltrate; @assert false

    #=
    All cell location, e.g., (9,3) are:  (nY -- column height, nX -- row width) from upper left 
    corner of grid.
    =#
    
    if experiment_id == 1
        experiment = format("RL{}", experiment_id)
        grid_dims = (3,3)
        grid_size = prod(grid_dims)
        grid = permutedims(reshape(1:grid_size, (grid_dims[2], grid_dims[1])), [2,1])
        cells = [[i, j] for i in 1:grid_dims[1] for j in 1:grid_dims[2]]
        
        global CONFIG = (
            # simple 1-D grid, no clues, only one reward
            experiment = experiment,
            grid_dims = grid_dims,
            grid = grid,
            cells = cells,
            start_cell = [3,1], 
            
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
            policy_length = 3,
            number_simulation_steps = 400,
            number_simulations = 1,
            use_filtering = false,
        )
    end

    # get_settings and modify as needed
    settings = AI.get_settings()
    settings = @set settings.EFE_over = :policies
    settings = @set settings.graph_postprocessing_method = :G_prob_q_pi
    settings = @set settings.policy_inference_method = :standard
    settings = @set settings.graph = :none

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
    pB = deepcopy(model.states.loc.B)
    scale_concentration_parameter = 2.0
    pB .*= scale_concentration_parameter
    model = @set model.states.loc.pB = pB
    
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
    
    #=
    # for plotting corr coef when calculating r for RL2 etc.
    locations = [model.cells[idx] for idx in history[1][:loc_id]]
    plot_visited(CONFIG, to_label, nothing, 1, 400, CONFIG[:walls], locations)
    Plots.plot(history[1][:r], title=format("{}, Corr Coef, Emp Over Policies vs. G", CONFIG[:experiment]), ylabel="Corr Coef", xlabel="Iteration", label=nothing)
    Plots.savefig(format("./pngs/{}_corr_coef.png", CONFIG[:experiment]))
    Plots.histogram(history[1][:r], title=format("{}, Corr Coef, Emp Over Policies vs. G", CONFIG[:experiment]), xlabel="Corr Coef", ylabel="Count", label=nothing)
    Plots.savefig(format("./pngs/{}_corr_coef_hist.png", CONFIG[:experiment]))
    printfmtln("\nExperiment {}, r mean = {}\n", CONFIG[:experiment], Statistics.mean(history[1][:r]))
    =#
    # Experiment RL3, r mean = 0.021582477108425802
    # Experiment RL4, r mean = -0.07849980596469375


    #Serialization.serialize(format("{}_history.ser", CONFIG.experiment), history)
    #Serialization.serialize(format("{}_B.ser", CONFIG.experiment), model.B_true)
    #Serialization.serialize(format("{}_model.ser", CONFIG.experiment), model)
    #Serialization.serialize(format("{}_config.ser", CONFIG.experiment), CONFIG)

    @infiltrate; @assert false

    Assess.assess_results(CONFIG, agent, model)
    @infiltrate; @assert false
    
    
    
    @infiltrate; @assert false

    # save history
    dfs_efe = []
    dfs_metrics = []
    for simulation_number in 1:CONFIG.number_simulations
        
        @infiltrate; @assert false
        results = Metrics.calc_metrics(history[simulation_number], model)
        
        # record EFE and decomposition products over simulation steps, based on chosen actions
        df = DFS.DataFrame(permutedims(hcat(history[simulation_number][:EFE]...)), 
            [:info_gain, :utility, :risk, :ambiguity, :EFE])
        printfmtln("\nEFE, sim {}= \n{}", simulation_number, df)
        
        CSV.write(format("./sim_results/EFE_{}_sim{}.csv", CONFIG.experiment, simulation_number), df)
        push!(dfs_efe, df)

        # summarize metrics over a simulation
        metrics = Union{Missing, Float64}[]
        push!(metrics, vcat(sum.(eachcol(df)), model.policies.number_policies, CONFIG.gamma)...)
        fields = vcat(names(df), ["n_policies", "gamma"])
        
        for obsID in 1:length(results)
            obs_name = fieldnames(Observations)[obsID]
            labels = [x * String(obs_name)[1:end-4] for x in ["TE_", "TE_reverse_", "empowerment_"]]
            push!(metrics, results[obsID]...)
            push!(fields, labels...)
        end
        
        df2 = DFS.DataFrame(Metric=fields, Value=metrics)
        printfmtln("\nMetrics, sim {}= \n{}", simulation_number, df2)
        CSV.write(format("./sim_results/metrics_{}_sim{}.csv", CONFIG.experiment, simulation_number), df2)
        push!(dfs_metrics, df2)
    end


    @infiltrate; @assert false
    
    
end


end  # -- module