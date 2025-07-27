function construct_B_matrices(MAZE::Matrix{Int}, nactions::Int=2)
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


        # --- RIGHT ---
        if col < ncols && MAZE[row, col + 1] == 0
            s2 = (col) * nrows + row
            B[s2, s, 2] = 1.0
        else
            B[s, s, 2] = 1.0
        end

    end

    return B
end