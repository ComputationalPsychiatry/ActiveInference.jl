# https://pymdp-rtd.readthedocs.io/en/latest/notebooks/clue_chaining_demo.html
# Active Inference Demo: Epistemic Chaining

# include("./grid.jl")
#show(stdout, "text/plain", x)
# @infiltrate; @assert false



import Plots

using Format
using Infiltrator
using Revise

Plots.scalefontsizes()
Plots.scalefontsizes(0.8)

####################################################################################################


# --------------------------------------------------------------------------------------------------
function getIdx(obj, label, value=nothing)
    # get index of label or value
    if isnothing(value)
        return findfirst(obj.labels .== label)
    else
        return findfirst(obj.value[findfirst(obj.labels .== label)] .== value)
    end
end

# --------------------------------------------------------------------------------------------------
function plot_sq_error(CONFIG, sim_i, step_i, history_of_sq_error)
    title = format("{}, Square Error of B Matrix", CONFIG[:experiment])
    plt = Plots.plot(1:length(history_of_sq_error), history_of_sq_error, title=title, 
        xlabel="Iteration", ylabel="Square Error", legend=nothing
    )
    Plots.savefig(plt, format("./pngs/{}_sq_error_sim{}.png", CONFIG[:experiment], sim_i))
    #@infiltrate; @assert false
end


# --------------------------------------------------------------------------------------------------
function plot_visited(CONFIG, to_label, plot_title, sim_i, step_i, walls, locations)
    
    nX, nY = CONFIG[:grid_dims]
    data = zeros((nX, nY))
    
    locations = StatsBase.countmap(locations)
    for (loc, cnt) in locations
        data[loc...] = cnt
    end
    
    
    # make a figure + axes
    data = reverse(data, dims=1)
    mypal   = [:white, :red]
    
    ex = range(0, nY)
    ey = range(0, nX)
    
    title = format("{}, Heatmap of Visited Cells", CONFIG[:experiment])
    plt = Plots.heatmap(ex, ey, data, legend=nothing, colorbar = true, title=title, color=mypal, 
        alpha=.3, aspect_ratio=1, 
        xlim=(0,nY), ylim=(0,nX), ticks = false
    )  
    Plots.vline!(plt, ex, c=:black)
    Plots.hline!(plt, ey, c=:black)
    
    for walls_ in walls
        for (cell, edge) in walls_
            cell = reverse(cell)
            cell[2] = nY - cell[2] + 1
            # now cell is in (x,y) matrix axes, where (1,1) is in lower left corner of plot
                
            if edge == :DOWN
                Plots.plot!(plt, [cell[1]-1, cell[1]], [cell[2]-1, cell[2]-1], linewidth=10, c=:green)
            
            elseif edge == :UP
                Plots.plot!(plt, [cell[1]-1, cell[1]], [cell[2], cell[2]], linewidth=10, c=:green)
                
            elseif edge == :LEFT
                Plots.plot!(plt, [cell[1]-1, cell[1]-1], [cell[2]-1, cell[2]], linewidth=10, c=:green)
            
            else edge == :RIGHT
                Plots.plot!(plt, [cell[1], cell[1]], [cell[2]-1, cell[2]], linewidth=10, c=:green)
            
            #@infiltrate; @assert false        
            end
        end
    end
    title = format("{}_heatmap_sim{}", CONFIG[:experiment], sim_i)
    Plots.savefig(plt, format("./pngs/{}.png", title))

    #@infiltrate; @assert false
end


# --------------------------------------------------------------------------------------------------
function plot_empowerment(CONFIG, to_label, walls, E)
    
    nX, nY = CONFIG[:grid_dims]
    data = zeros((nX, nY))
    
       
    
    # make a figure + axes
    #data = reverse(data, dims=1)
    data = reverse(E, dims=1)
    mypal   = [:navyblue, :green, :yellow]
    
    ex = range(0, nY)
    ey = range(0, nX)
    
    title = format("{}, True Empowerment", CONFIG[:experiment])
    plt = Plots.heatmap(ex, ey, data, legend=nothing, colorbar = true, title=title, color=mypal, 
        alpha=.3, aspect_ratio=1, 
        xlim=(0,nY), ylim=(0,nX), ticks = false
    )  
    Plots.vline!(plt, ex, c=:black)
    Plots.hline!(plt, ey, c=:black)
    
    for walls_ in walls
        for (cell, edge) in walls_
            cell = reverse(cell)
            cell[2] = nY - cell[2] + 1
            # now cell is in (x,y) matrix axes, where (1,1) is in lower left corner of plot
                
            if edge == :DOWN
                Plots.plot!(plt, [cell[1]-1, cell[1]], [cell[2]-1, cell[2]-1], linewidth=10, c=:green)
            
            elseif edge == :UP
                Plots.plot!(plt, [cell[1]-1, cell[1]], [cell[2], cell[2]], linewidth=10, c=:green)
                
            elseif edge == :LEFT
                Plots.plot!(plt, [cell[1]-1, cell[1]-1], [cell[2]-1, cell[2]], linewidth=10, c=:green)
            
            else edge == :RIGHT
                Plots.plot!(plt, [cell[1], cell[1]], [cell[2]-1, cell[2]], linewidth=10, c=:green)
            
            #@infiltrate; @assert false        
            end
        end
    end
    fn = format("{}_heatmap_empowerment", CONFIG[:experiment])
    Plots.savefig(plt, format("./pngs/{}.png", fn))

    #@infiltrate; @assert false
end



# --------------------------------------------------------------------------------------------------
function plot_grid(CONFIG, to_label, plot_title, sim_i, step_i, walls; locations::Union{Vector, Nothing}=nothing)
    
    nX, nY = CONFIG[:grid_dims]

    data = zeros((nX, nY))
    
    # make a figure + axes
    data = reverse(data, dims=1)
    mypal   = [:white, :red]
    
    ex = range(0, nY)
    ey = range(0, nX)
    
    plt = Plots.heatmap(ex, ey, data, legend=nothing, title=plot_title, color=mypal, 
        alpha=.3, aspect_ratio=1, 
        xlim=(0,nY), ylim=(0,nX), ticks = false
    )  
    Plots.vline!(plt, ex, c=:black)
    Plots.hline!(plt, ey, c=:black)
    
    for walls_ in walls
        for (cell, edge) in walls_
            cell = reverse(cell)
            cell[2] = nY - cell[2] + 1
            # now cell is in (x,y) matrix axes, where (1,1) is in lower left corner of plot
                
            if edge == :DOWN
                Plots.plot!(plt, [cell[1]-1, cell[1]], [cell[2]-1, cell[2]-1], linewidth=10, c=:green)
            
            elseif edge == :UP
                Plots.plot!(plt, [cell[1]-1, cell[1]], [cell[2], cell[2]], linewidth=10, c=:green)
                
            elseif edge == :LEFT
                Plots.plot!(plt, [cell[1]-1, cell[1]-1], [cell[2]-1, cell[2]], linewidth=10, c=:green)
            
            else edge == :RIGHT
                Plots.plot!(plt, [cell[1], cell[1]], [cell[2]-1, cell[2]], linewidth=10, c=:green)
            
            #@infiltrate; @assert false        
            end
        end
    end
    #@infiltrate; @assert false

    punish_label = [:prize]
    
     nX, nY = (nY, nX)
    for obj in to_label
        if keys(obj)[1] == :start
            loc = obj.start
            column_coord, row_coord = loc
            column_coord = nY - column_coord
            c = :black
            Plots.annotate!(plt, row_coord -.5, column_coord +.5, Plots.text("Start", c, 10))
        end

        # add other grid cells to label here
    end
    
    if !isnothing(locations)

        T = length(locations)
        locations = stack(locations, dims=1)
        #@infiltrate; @assert false
        locations .+= -1
        locations[:,1] = nY - 1 .- locations[:,1]
        
        c = reshape( range(Plots.colorant"white", stop=Plots.colorant"green",length=T), 1, T )[1,:]
        s = 20
        Plots.plot!(plt, locations[:,2] .+ 0.5, locations[:,1] .+ 0.5) 
        Plots.scatter!(plt, locations[:,2] .+ 0.5, locations[:,1] .+ 0.5, markercolor=c, markersize=s, alpha=.6)  # 
    end

    # turn off the axis labels
    #ax.axis("off")
    if sim_i == 1 && step_i == 0
        plt2 = deepcopy(plt)
        title = format("{}_sim{}", CONFIG[:experiment], sim_i)
        Plots.title!(plt2, title)
        Plots.savefig(plt2, format("./pngs/{}.png", title))
    end
    
    #@infiltrate; @assert false

    return plt
end



