# https://pymdp-rtd.readthedocs.io/en/latest/notebooks/clue_chaining_demo.html
# Active Inference Demo: Epistemic Chaining

# include("./grid.jl")
#show(stdout, "text/plain", x)
# @infiltrate; @assert false


#using Format
#using Infiltrator
#using Revise


####################################################################################################

function make_model(CONFIG)
    
    grid_size = prod(CONFIG.grid_dims)


    MAZE  = [
        1 1 1 1 1 1 1 1 1;
        1 1 1 1 1 1 1 0 1;
        1 1 1 1 1 0 1 0 1;
        1 1 1 0 1 0 1 0 1;
        1 0 1 0 1 0 1 0 1;
        1 0 1 0 1 0 1 0 1;
        1 0 1 0 1 0 1 0 1;
        1 0 0 0 0 0 0 0 1;
        1 0 1 1 1 1 1 1 1
    ]

    LOCATIONS = reshape(collect(1:grid_size), CONFIG.grid_dims...)

    PREFERENCES = [
        0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000;
        0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000  0.000;
        0.000  0.000  0.000  0.000  0.000  0.700  0.000  0.825  0.000;
        0.000  0.000  0.000  0.500  0.000  0.600  0.000  0.700  0.000;
        0.000  0.300  0.000  0.400  0.000  0.525  0.000  0.625  0.000;
        0.000  0.200  0.000  0.300  0.000  0.425  0.000  0.575  0.000;
        0.000  0.150  0.000  0.200  0.000  0.325  0.000  0.500  0.000;
        0.000  0.100  0.125  0.150  0.175  0.225  0.300  0.400  0.000;
        0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000
    ]

    model = (
        states = (
            loc = (
                name = :loc,
                values = 1:grid_size,
                labels = CONFIG.cells,
                B = missing,
                B_dim_names = (:loc, :loc, :move),
                B_dims = (grid_size, grid_size, 2),
                D = missing,
                pB = nothing,
                pD = nothing,
                extra = (
                    MAZE = MAZE,
                ),
            
            ),
        ),

        obs = (
            # location    
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

            # walls: open/closed
            wall_obs = (
                name = :wall_obs, 
                values = 1:2,
                labels = (:open, :closed),
                A = missing,
                A_dim_names = (:open_obs, :loc),  # no associated state, use obs name
                A_dims = (2, grid_size),
                pA = nothing,
                extra = nothing,
            ),

            # safe/adverse
            safe_obs = (
                name = :safe_obs, 
                values = 1:2,
                labels = (:safe, :adverse),
                A = missing,
                A_dim_names = (:safe_obs, :loc),  # no associated state, use obs name
                A_dims = (2, grid_size),
                pA = nothing,
                extra = (
                    PREFERENCES = PREFERENCES,
                ),
            ),
        ),

        actions = (
            move = (
                name = :move, 
                values = collect(1:2),
                labels = (:UP, :DOWN),
                null_action = nothing,
                extra = nothing,
            ),
        ),

        preferences = (
            loc_pref = (
                name = :loc_pref,
                C = missing,
                C_dim_names = (:loc_obs,),
                C_dims = (grid_size,),
                extra = nothing,
            ),

            wall_pref = (
                name = :wall_pref,
                C = missing,
                C_dim_names = (:wall_obs,),
                C_dims = (2,),
                extra = nothing,
            ),

            safe_pref = (
                name = :safe_pref,
                C = missing,
                C_dim_names = (:safe_obs,),
                C_dims = (2,),
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
            E = missing,
            extra = nothing,
        )
    )
    
    # initialize some matrices
    model = @set model.states.loc.B = zeros(model.states.loc.B_dims)

    model = @set model.states.loc.D = zeros(model.states.loc.B_dims[1]) 
    
    model = @set model.obs.loc_obs.A = zeros(model.obs.loc_obs.A_dims)
    model = @set model.obs.wall_obs.A = zeros(model.obs.wall_obs.A_dims)
    model = @set model.obs.safe_obs.A = zeros(model.obs.safe_obs.A_dims)
    
    model = @set model.preferences.loc_pref.C = zeros(model.preferences.loc_pref.C_dims)
    model = @set model.preferences.wall_pref.C = zeros(model.preferences.wall_pref.C_dims)
    model = @set model.preferences.safe_pref.C = zeros(model.preferences.safe_pref.C_dims)

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