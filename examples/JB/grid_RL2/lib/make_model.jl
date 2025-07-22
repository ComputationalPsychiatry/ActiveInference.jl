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
    model = (
        states = (
            loc = (
                name = :loc,
                values = 1:grid_size,
                labels = CONFIG.cells,
                B = missing,
                B_dim_names = (:loc, :loc, :move_vert, :move_horz),
                B_dims = (grid_size, grid_size, 3, 3),
                D = missing,
                is_B_learned = true,
                pB = missing,
                is_D_learned = false,
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
                is_learned = false,
                pA = nothing,
                extra = nothing,
            ),
        ),

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
        ),

        preferences = (
            loc_pref = (
                name = :loc_pref,
                state_dependencies = nothing,
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
            policy_tests = missing,
            action_tests = missing,
            earlystop_tests = missing,
            EFE_reduction = nothing,  # user-supplied function for EFE reduction that allows missings
            E = missing,
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