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
    0.434315  0.5       0.552786  0.587689  0.6      0.587689  0.552786  0.5;
    0.5       0.575736  0.639445  0.683772  0.7      0.683772  0.639445  0.575736;
    0.552786  0.639445  0.717157  0.776393  0.8      0.776393  0.717157  0.639445;
    0.587689  0.683772  0.776393  0.858579  0.9      0.858579  0.776393  0.683772;
    0.6       0.7       0.8       0.9       1.0      0.9       0.8       0.7;
    0.587689  0.683772  0.776393  0.858579  0.9      0.858579  0.776393  0.683772;
    0.552786  0.639445  0.717157  0.776393  0.78819  0.776393  0.717157  0.639445;
    0.5       0.575736  0.639445  0.683772  0.7      0.683772  0.639445  0.575736
]

for i in 1:size(MAZE, 1)
    for j in 1:size(MAZE, 2)
        if MAZE[i, j] == 1
            PREFERENCES_matrix[i, j] = 0.4
        else
            PREFERENCES_matrix[i, j] = PREFERENCES_matrix[i, j]
        end
    end
end

hm = heatmap(
    reverse(PREFERENCES_matrix, dims=1);
    aspect_ratio=1,
    legend=false,
    color=cgrad(:Reds, rev=true),
    axis=false,
    frame=false,
    size=(800, 600),
    title=""
)
rows, cols = size(MAZE)

for r in 1:rows
    for c in 1:cols
        index = (c-1) * rows + r
        annotate!(c, rows-r+1, text(string(index), :center, 8, :grey2))
    end
end

hm


function animate!(history, horizon=1)

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

    # changing preferences on place of wall just for plotting
    for i in 1:size(MAZE, 1)
        for j in 1:size(MAZE, 2)
            if MAZE[i, j] == 1
                PREFERENCES_matrix[i, j] = 0.4
            else
                PREFERENCES_matrix[i, j] = PREFERENCES_matrix[i, j]
            end
        end
    end

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
    gif(anim, "test_$(horizon).gif", fps=2)
    
    #gif(anim, joinpath(folder, "agent_path_$(horizon).gif"), fps=2)
    
end


