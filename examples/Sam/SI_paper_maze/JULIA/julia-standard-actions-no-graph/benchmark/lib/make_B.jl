
# @infiltrate; @assert false



####################################################################################################



function make_B(model, CONFIG)
    ### B matrix
        
   ######################## Transitions ########################
    # [UP, DOWN, LEFT, RIGHT, STAY]
    model.states.loc.B[:] = construct_B_matrices(model.states.loc.extra.MAZE, 5)

    return
end


function construct_B_matrices(MAZE::Matrix{Int}, nactions::Int=5)
    nrows, ncols = size(MAZE)
    nstates = nrows * ncols

    B = zeros(Float64, nstates, nstates, nactions)

    for row in 1:nrows, col in 1:ncols
        s = (col - 1) * nrows + row 

        if MAZE[row, col] == 1
            for a in 1:nactions
                B[s, s, a] = 1.0
            end
            continue
        end

        # --- UP ---
        if row > 1 && MAZE[row - 1, col] == 0
            s2 = (col - 1) * nrows + (row - 1)
            B[s2, s, 1] = 1.0
        else
            B[s, s, 1] = 1.0
        end

        # --- DOWN ---
        if row < nrows && MAZE[row + 1, col] == 0
            s2 = (col - 1) * nrows + (row + 1)
            B[s2, s, 2] = 1.0
        else
            B[s, s, 2] = 1.0
        end

        # --- LEFT ---
        if col > 1 && MAZE[row, col - 1] == 0
            s2 = (col - 2) * nrows + row
            B[s2, s, 3] = 1.0
        else
            B[s, s, 3] = 1.0
        end

        # --- RIGHT ---
        if col < ncols && MAZE[row, col + 1] == 0
            s2 = (col) * nrows + row
            B[s2, s, 4] = 1.0
        else
            B[s, s, 4] = 1.0
        end

        # --- STAY ---
        B[s, s, 5] = 1.0

    end

    return B
end
