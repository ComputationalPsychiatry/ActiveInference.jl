using Plots

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
    -1.000  -1.000  -1.000  -1.000  -1.000  -1.000  -1.000  -1.000;
    -1.000  -0.433  -0.178   0.000   0.051   0.000  -0.178  -1.000;
    -1.000  -1.000  -1.000  -0.178  -1.000  -1.000   0.106  -1.000;
    -1.000  -1.000   0.293   0.568   0.684  -1.000   0.293  -1.000;
    -1.000  -1.000   0.568  -1.000   1.000   0.684   0.368  -1.000;
    -1.000  -1.000   0.293  -1.000  -1.000  -1.000   0.293  -1.000;
    -1.000  -0.178   0.106   0.293   0.568   0.293   0.106  -1.000;
    -1.000  -0.433  -1.000  -1.000  -1.000  -1.000  -1.000  -1.000
]



function animate!(history, horizon=1)

    function linear_to_coords(index, rows)
        col = div(index - 1, rows) + 1
        row = index - (col - 1) * rows
        return row, col
    end

    path_indices = [step.loc_obs for step in history]
    rows, cols = size(PREFERENCES_matrix)

    path_coords = [linear_to_coords(idx, rows) for idx in path_indices]

    anim = @animate for i in 1:length(path_indices)
        hm = heatmap(
            reverse(PREFERENCES_matrix, dims=1);
            aspect_ratio=1,
            legend=false,
            color=cgrad(:Reds, rev=true),
            axis=false,
            frame=false,
            size=(800, 600),
            title="Horizon: $horizon | Timestep: $i"
        )
        rows, cols = size(MAZE)

        for r in 1:rows
            for c in 1:cols
                index = (c-1) * rows + r
                annotate!(c, rows-r+1, text(string(index), :center, 8, :grey2))
            end
        end
        
        if i > 1
            for j in 1:(i-1)
                r1, c1 = path_coords[j]
                r2, c2 = path_coords[j+1]
                plot!([c1, c2], [rows-r1+1, rows-r2+1], color=:green, linewidth=3)
            end
        end
        
        current_row, current_col = path_coords[i]
        scatter!([current_col], [rows-current_row+1], color=:green, markersize=10, markershape=:circle)
    end

    gif(anim, joinpath(folder, "agent_path_$(horizon).gif"), fps=2)
    
end


