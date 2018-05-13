import DataFrames: DataFrame, groupby
import Colors: distinguishable_colors
using PGFPlotsX

"""
    compare(rlsetupcreators::Dict, N; callbackid = 1, verbose = false)

Run different setups in dictionary `rlsetupcreators` `N` times. The dictionary
has elements "name" => createrlsetup, where createrlsetup is a function that
has a single integer argument (id of the comparison; useful for saving 
intermediate results). For each run, `getvalue(rlsetup.callbacks[callbackid])`
gets entered as result in a DataFrame with columns "name", "result", "seed".
"""
function compare(rlsetupcreators::Dict, N; callbackid = 1, verbose = false)
    res = @parallel (hcat) for t in 1:N
        seed = rand(1:typemax(UInt64)-1)
        tmp = []
        for (name, setupcreator) in rlsetupcreators
            if verbose
                info("$(now()) \tStarting comparison $t, setup $name with seed $seed.")
            end
            srand(seed)
            rlsetup = setupcreator(t)
            learn!(rlsetup)
            push!(tmp, [name, getvalue(rlsetup.callbacks[callbackid]), seed])
        end
        hcat(tmp...)
    end
    DataFrame(name = res[1,:], result = res[2,:], seed = res[3,:])
end
export compare

"""
    plotcomparison(df; nmaxpergroup = 20, colors = [])

Plots results obtained with [`compare`](@ref).
"""
function plotcomparison(df; nmaxpergroup = 20, colors = [])
    groups = groupby(df, :name)
    colors = colors == [] ? distinguishable_colors(length(groups)) : colors
    plots = []
    legendentries = []
    isnumber = typeof(df[:result][1]) <: Number
    for (i, g) in enumerate(groups)
        if isnumber
            push!(plots, @pgf Plot({boxplot}, Table({y_index = 0}, 
                                                    Dict("res" => g[:result]))))
            push!(legendentries, g[:name][1])
        else
            m = mean(g[:result])
            push!(plots, @pgf Plot({thick, color = colors[i]}, 
                                   Coordinates(1:length(m), m)))
            push!(legendentries, g[:name][1])
            ma = g[:result][indmax(map(mean, g[:result]))]
            push!(plots, @pgf Plot({thick,dashed, color = colors[i]}, 
                                   Coordinates(1:length(ma), ma)))
            push!(legendentries, "")
            for k in 1:min(nmaxpergroup, length(g[:result]))
                push!(plots, @pgf Plot({very_thin, color = colors[i], opacity = .3},
                                       Coordinates(1:length(g[:result][k]),
                                                   g[:result][k])))
                push!(legendentries, "")
            end
        end
    end
    if isnumber
        @pgf Axis({"boxplot/draw direction=y", 
                   xticklabels = legendentries, 
                   xtick = collect(1:length(legendentries))}, plots...)
    else
        @pgf Axis({no_markers}, plots..., Legend(legendentries))
    end
end
export plotcomparison
