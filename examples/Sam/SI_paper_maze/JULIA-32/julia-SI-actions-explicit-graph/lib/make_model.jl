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

    MAZE = [
        1 1 1 1 1 1 1 1;
        1 0 0 0 0 0 0 1;
        1 1 1 0 1 1 0 1;
        1 1 0 0 0 1 0 1;
        1 1 0 1 0 0 0 1;
        1 1 0 1 1 1 0 1;
        1 0 0 0 0 0 0 1;
        1 0 1 1 1 1 1 1
    ]

    PREFERENCES_matrix = [
        0.434315  0.5       0.552786  0.587689  0.6      0.587689  0.552786  0.5;
        0.5       0.575736  0.639445  0.683772  0.7      0.683772  0.639445  0.575736;
        0.552786  0.639445  0.717157  0.776393  0.8      0.776393  0.717157  0.639445;
        0.587689  0.683772  0.776393  0.858579  0.9      0.858579  0.776393  0.683772;
        0.6       0.7       0.8       0.9       1.0      0.9       0.8       0.7;
        0.587689  0.683772  0.776393  0.858579  0.9      0.858579  0.776393  0.683772;
        0.552786  0.639445  0.717157  0.776393  0.78819  0.776393  0.717157  0.639445;
        0.5       0.575736  0.639445  0.683772  0.7      0.683772  0.639445  0.575736
    ]

    LOCATIONS = reshape(collect(1:grid_size), CONFIG.grid_dims...)

    model = (
        states = (
            loc = (
                name = :loc,
                values = 1:grid_size,
                labels = CONFIG.cells,
                B = missing,
                B_dim_names = (:loc, :loc, :move),
                B_dims = (grid_size, grid_size, 5),
                D = missing,
                pB = nothing,
                pD = nothing,
                extra = (
                    MAZE = MAZE,
                    vec_maze = vec(MAZE),
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
                HA = missing,
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
                HA = missing,
                A_dim_names = (:open_obs, :loc),  # no associated state, use obs name
                A_dims = (2, grid_size),
                pA = nothing,
                extra = nothing,
            ),

            # preferences: prefered/shock
            valence_obs = (
                name = :valence_obs, 
                values = 1:2,
                labels = (:safe, :aversive),
                A = missing,
                HA = missing,
                A_dim_names = (:valence_obs, :loc), 
                A_dims = (2, grid_size),
                pA = nothing,
                extra = (
                    PREFERENCES_matrix = PREFERENCES_matrix,
                )
            ),

        ),

        actions = (
            move = (
                name = :move, 
                values = collect(1:5),
                labels = (:UP, :DOWN, :LEFT, :RIGHT, :STAY),
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

            valence_pref = (
                name = :valence_pref,
                C = missing,
                C_dim_names = (:valence_obs,),
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
            action_tests = (qs_pi_next, qs_pi_last, model) -> true,
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

    model = @set model.states.loc.D = zeros(model.states.loc.B_dims[1]) 
    
    model = @set model.obs.loc_obs.A = zeros(model.obs.loc_obs.A_dims)
    model = @set model.obs.wall_obs.A = zeros(model.obs.wall_obs.A_dims)
    model = @set model.obs.valence_obs.A = zeros(model.obs.valence_obs.A_dims)

    model = @set model.preferences.loc_pref.C = zeros(model.preferences.loc_pref.C_dims)
    model = @set model.preferences.wall_pref.C = zeros(model.preferences.wall_pref.C_dims)
    model = @set model.preferences.valence_pref.C = zeros(model.preferences.valence_pref.C_dims)

    #@infiltrate; @assert false

    return model

end