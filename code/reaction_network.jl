using Catalyst
using OrdinaryDiffEq
using Plots
using LaTeXStrings
using Dierckx
using Latexify
using CSV
using Tables

function cc_directory(path::AbstractString)
    if !isdir(path)
        try
            mkdir(path)
            println("Directory created: $path")
        catch err
            error("Failed to create directory: $path\n$error")
        end
    end
end

if Sys.iswindows()
    const plot_dump = pwd()
    cc_directory(plot_dump)
else
    # Set to headless plotting
    ENV["GKSwstype"] = 100
    const plot_dump = pwd()
end

function os_display(p)
    """
    OS safe plot display, prevents issues with plotting when on a cluster
    and not on an X11 forwarding
    """
    if Sys.iswindows()
        display(p)
    end
end

function best_time_units(time; f_print=false)
    prefix_name = ["mili-", "micro-", "nano-", "pico-", "femto-", "atto-"]
    prefix_unit = ["m", L"\mu", "n", "p", "f", "a"]
    prefix_amount = [1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12, 1.0e-15, 1.0e-18]

    # Convert to array and convert time from AUT to SI
    time = Array(time) .* au_time

    # Compare
    max_time = maximum(time)
    dif = @. abs(log10(max_time) - log10(prefix_amount))

    # Pick best range
    min_loc = argmin(dif)

    prefix = prefix_unit[min_loc]
    name = prefix_name[min_loc]
    # Convert the new time
    new_time = time ./ prefix_amount[min_loc]

    if f_print
        println("Best prefix: $(prefix), $(name)seconds")
    end
    return new_time, prefix, name
end

function rnd(x; n=3)
    """
    Rounds a number to n digits (3)
    """
    return round(x; sigdigits=n)
end

function reaction_unwrap(sol)
    u = sol.u
    t = sol.t
    nt = length(t)
    n = length(u[end])

    # unwrap
    arr = zeros(nt, n)
    for i = 1:nt
        for j = 1:n
            arr[i, j] = abs.(u[i][j])
        end
    end
    return arr
end

function calculate_trace(arr)
    return sum(arr, dims=2)
end

function fix_trace(arr)
    trace = calculate_trace(arr)
    arr ./= trace
    return arr
end

function plot_trace(sol; f_units="SI", f_trace=false)
    time_sim = sol.t ./ au_time
    if f_units == "SI"
        time_sim, prefix, _ = best_time_units(time_sim)
        tlab = latexstring("\\mathrm{Time}, t, [$(prefix)s]")
    else
        tlab = latexstring("\\mathrm{Time}, t, [AUT]")
    end

    # unwrap to get an array
    arr = reaction_unwrap(sol)
    # Fix the trace
    if f_trace
        arr = fix_trace(arr)
    end
    # Calculate the trace
    trace = calculate_trace(arr)
    println("Final trace = $(rnd(trace[end]))")
    p = plot(time_sim, 1.0 .- trace; lw=2, xlabel=tlab, ylabel="Trace", legend=nothing)
    os_display(p)
    return p
end

function plot_reactions(sol, labs; idx=nothing, f_units="SI", yscale=:log10, f_trace=false, name="plot_reaction_", leg_loc=:bottomright)
    time_sim = sol.t ./ au_time
    if f_units == "SI"
        time_sim, prefix, _ = best_time_units(time_sim)
        tlab = latexstring("\\mathrm{Time}, t, [$(prefix)s]")
    else
        tlab = latexstring("\\mathrm{Time}, t, [AUT]")
    end
    ylab = "Population"

    # unwrap to get an array
    arr = reaction_unwrap(sol)
    # Fix the trace
    if f_trace
        arr = fix_trace(arr)
    end

    if idx === nothing
        n_plots = length(labs)
        idx = range(1, n_plots)
        f_name = name * ".pdf"
    else
        n_plots = length(idx)
        f_name = name * string(idx) * ".pdf"
    end
    # Fix zero data
    if yscale == :log10
        arr[1, 2:end] .= abs(minimum(arr[2:end, :]))
    end

    p = plot()
    for i = 1:n_plots
        p = plot!(time_sim[1:end], abs.(arr[1:end, idx[i]]), lw=2, labels=labs[idx[i]])
    end
    p = plot!(p, xlabel=tlab, ylabel=ylab, yscale=yscale, legend=leg_loc, legendfontsize=10)
    os_display(p)

    savefig(p, joinpath(plot_dump, f_name))
end

function calculate_properties(sol, f_trace=true)
    # unwrap to get an array
    arr = reaction_unwrap(sol)
    # Fix the trace
    if f_trace
        arr = fix_trace(arr)
    end
    vals_max = maximum(arr, dims=1)
    println("Final values = $(arr[end,:])")
    println("Maximum values = $(vals_max)")
    println("tautomeric ratio = $(rnd(arr[end,end]/arr[end,2]))")
    return
end


function load_reaction_data(file_name)
    # load the data using CSV.jl
    file = CSV.File(file_name; delim=',', header=false, skipto=2, missingstring="NA")
    data = file |> CSV.Tables.matrix
    return data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6]
end


# a function that saves the solution to a file
function save_solution(sol, labs, file_name)
    u = sol.u
    n = length(labs)

    # Convert time to SI
    time_sim = sol.t ./ au_time
    nt = length(time_sim)
    time_sim, prefix, _ = best_time_units(time_sim)
    tlab = latexstring("$(prefix)s")

    # Convert headers to strings
    labs = [string(labs[i]) for i = 1:n]
    # unwrap to get an array
    arr = zeros(nt, n)
    for i = 1:nt
        for j = 1:n
            arr[i, j] = u[i][j]
        end
    end

    # save the data
    CSV.write(file_name, Tables.table(arr); header=labs)
    # Save the time
    CSV.write("time_" * file_name, Tables.table(time_sim); header=[string(tlab)])
    return
end


rs_simple = @reaction_network begin
    k_hel, R1 --> P1 + P2         # Helicase separation
    (k1_f, k1_r), R1 <--> R2      # Proton transfer reaction
    k_hel, R2 --> P1_t + P2_t     # Helicase separation
end

rs_extended = @reaction_network begin
    k_hel, R1 --> P1 + P2         # Helicase separation
    (k1_f, k1_r), R1 <--> R2      # 1st Proton transfer reaction
    k_hel, R2 --> P1_t + P2_t     # Helicase separation
    (k2_f, k2_r), R2 <--> R3      # 2nd Proton transfer reaction
    k_hel, R3 --> P1_tt + P2_tt   # Helicase separation
end

rs_simple_m = @reaction_network begin
    k_hel, R1 --> P1              # Helicase separation
    (k1_f, k1_r), R1 <--> R2      # Proton transfer reaction
    k_hel, R2 --> P2              # Helicase separation
end

rs_extended_m = @reaction_network begin
    k_hel, R1 --> P1              # Helicase separation
    (k1_f, k1_r), R1 <--> R2      # 1st Proton transfer reaction
    k_hel, R2 --> P2              # Helicase separation
    (k2_f, k2_r), R2 <--> R3      # 2nd Proton transfer reaction
    k_hel, R3 --> P3              # Helicase separation
end

tol = 1e-18
algo = Tsit5()
k_helicase = 1.2 / 1e-12 # 1/ps
au_time = 2.4188843265857e-17
f_choice = "GC"

if f_choice == "GC"
    dist, time_vals, rate_f, rate_r, rate_f_neo, rate_r_neo = load_reaction_data("data/gc_rates.csv")
    k_f = rate_f_neo[1]
    k_r = rate_r_neo[1]
    labs = LaTeXString[L"$G\mathrm{-}C$" L"$G\mathrm{+}C$" L"$G^\mathrm{*}\mathrm{-}C^\mathrm{*}$" L"$G^\mathrm{*}\mathrm{+}C^\mathrm{*}$"]
    network = rs_simple_m
    tc = 6.0
    params = (:k1_f => k_f, :k1_r => k_r, :k_hel => k_helicase)
    u0 = [:R1 => 1.0, :R2 => 0.0, :P1 => 0.0, :P2 => 0.0]

elseif f_choice == "mGC"
    dist, time_vals, rate_f, rate_r, rate_f_neo, rate_r_neo = load_reaction_data("data/m_gc_rates.csv")
    k_f = rate_f_neo[1]
    k_r = rate_r_neo[1]
    labs = LaTeXString[L"$mG\mathrm{-}C$" L"$mG\mathrm{+}C$" L"$mG^\mathrm{*}\mathrm{-}C^\mathrm{*}$" L"$mG^\mathrm{*}\mathrm{+}C^\mathrm{*}$"]
    network = rs_simple_m
    tc = 6.0
    params = (:k1_f => k_f, :k1_r => k_r, :k_hel => k_helicase)
    u0 = [:R1 => 1.0, :R2 => 0.0, :P1 => 0.0, :P2 => 0.0]
end

# # plot the data vs time
# p = plot(time_vals, rate_f, lw=2, label="k_f")
# p = plot!(p, time_vals, rate_r, lw=2, label="k_r")
# p = plot!(p, time_vals, rate_f_neo, lw=2, label="k_f_neo")
# p = plot!(p, time_vals, rate_r_neo, lw=2, label="k_r_neo")
# p = plot!(p, time_vals, guess, lw=2, label="Guess")
# # Make the scale log
# p = plot!(p, yscale=:log10, xlabel="Time [ps]", ylabel="Rate [1/s]", legend=:bottomright)
# os_display(p)

println("K_eq = k_f/k_r = $(rnd(k_f/k_r))")
println("k_helicase = $(rnd(k_helicase))")
println("labs = $(labs)")

t0 = 0.0
t1 = tc / k_helicase # Integration time
nt = 1000
saveat = range(t0, stop=t1, length=nt)
prob = ODEProblem(network, u0, (t0, t1), params)
sol = solve(prob, algo, reltol=tol, abstol=tol, maxiters=1e14, progress=true, saveat=saveat)
calculate_properties(sol)

tlab = latexstring("\\mathrm{Time}, t, [s]")
ylab = "Population"

plot_trace(sol)

if f_choice in ["GC"]
    plot_reactions(sol, labs; name="plot_reaction_all_log", leg_loc=:right)
    plot_reactions(sol, labs; name="plot_reaction_all", yscale=:identity, leg_loc=:right)
    plot_reactions(sol, labs; idx=[3, 4], yscale=:identity, leg_loc=:right)
    save_solution(sol, labs, "reaction_data_gc.csv")
end

if f_choice in ["mGC"]
    plot_reactions(sol, labs; name="plot_reaction_all_log", leg_loc=:right)
    plot_reactions(sol, labs; name="plot_reaction_all", yscale=:identity, leg_loc=:right)
    plot_reactions(sol, labs; idx=[3, 4], yscale=:identity, leg_loc=:right)
    save_solution(sol, labs, "reaction_data_mgc.csv")
end


