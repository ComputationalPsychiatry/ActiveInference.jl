# https://pymdp-rtd.readthedocs.io/en/latest/notebooks/clue_chaining_demo.html
# Active Inference Demo: Epistemic Chaining

# include("./grid.jl")
#show(stdout, "text/plain", x)
# @infiltrate; @assert false



include("./structs.jl")
#include("./get_matrix_dims.jl")
include("./make_policies.jl")

#using Format
#using Infiltrator
#using Revise


####################################################################################################

function make_model(CONFIG)
    
    grid_size = prod(CONFIG.grid_dims)

    # experiments with one or two actions
    if CONFIG.n_actions == 2
        B_dim_names = (:loc, :loc, :move_vert, :move_horz)
        B_dims = (grid_size, grid_size, 3, 3)

        actions = (
            move_vert = (
                name = :move_vert, 
                values = collect(1:3),
                labels = [:UP, :DOWN, :STAY],
                null_action = :STAY,
                extra = nothing,
            ),
            move_horz = (
                name = :move_horz, 
                values = collect(1:3),
                labels = [:LEFT, :RIGHT, :STAY],
                null_action = :STAY,
                extra = nothing,
            ),
        )
    
    else
        # experiments with one action
        B_dim_names = (:loc, :loc, :move)
        B_dims = (grid_size, grid_size, 5)

        actions = (
            move = (
                name = :move, 
                values = collect(1:5),
                labels = [:UP, :DOWN, :LEFT, :RIGHT, :STAY],
                null_action = :STAY,
                extra = nothing,
            ),
        )
    end


    model = (
        states = (
            loc = (
                name = :loc,
                values = 1:grid_size,
                labels = CONFIG.cells,
                B = missing,
                B_dim_names = B_dim_names,
                B_dims = B_dims,
                D = missing,
                pB = missing,
                pD = nothing,
                extra = (
                    grid_dims = CONFIG.grid_dims,
                    stop_cells = CONFIG.stop_cells,
                ),
            ),
        ),

        obs = (
            loc_obs = (
                name = :loc_obs, 
                values = 1:grid_size,
                labels = CONFIG.cells,
                A = missing,
                A_dim_names = (:loc, :loc),
                A_dims = (grid_size, grid_size),
                pA = nothing,
                extra = nothing,
            ),
        ),

        actions = actions,

        preferences = (
            loc_pref = (
                name = :loc_pref,
                C = missing,
                C_dim_names = (:loc_obs,),
                C_dims = (grid_size,),
                extra = nothing,
            ),
        ),

        policies = (
            policy_iterator = missing,
            action_iterator = missing,
            policy_length = CONFIG.policy_length,
            n_policies = missing,
            policy_tests = (policy, model) -> true,  # unused for now
            action_tests = (qs_pi, model) -> true,
            earlystop_tests = (qs, model) -> true,
            utility_reduction_fx = nothing,  # user-supplied function for EFE reduction that allows missings
            info_gain_reduction_fx = nothing,  # user-supplied function for EFE reduction that allows missings
            E_policies = missing,
            E_actions = missing,
            extra = nothing,
        )
    )
    
    # initialize some matrices
    model = @set model.states.loc.B = zeros(model.states.loc.B_dims)
    model = @set model.obs.loc_obs.A = zeros(model.obs.loc_obs.A_dims)
    model = @set model.preferences.loc_pref.C = zeros(model.preferences.loc_pref.C_dims)
    model = @set model.states.loc.D = zeros(model.states.loc.B_dims[1]) 
    

    println("\nA matrix sizes:")
    for (ii, x) in enumerate(model.obs)
        printfmtln("    {}:  {}, {}", ii, x.A_dims, x.name)
    end

    println("\nB matrix sizes:")
    for (ii, x) in enumerate(model.states)
        printfmtln("    {}:  {}, {}", ii, x.B_dims, x.name)
    end

    println("\naction sizes:")
    for (ii, x) in enumerate(model.actions)
        printfmtln("    {}:  {}, {}", ii, x.labels, x.name)
    end
    println("\n")

    #@infiltrate; @assert false

    return model

end