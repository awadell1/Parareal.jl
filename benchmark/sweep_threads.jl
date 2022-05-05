using Parareal
using BenchmarkTools
using PkgJogger
using CairoMakie
using Statistics

include("../test/problems.jl")

function setup_mgrit(n)
    GC.gc()
    prob = heat(n)
    dt = prob.p
    N = prob.tspan[end] / dt
    levels = floor(Int, log2(log2(N)))
    return init(prob, MGRIT(Tsit5()); dt, levels)
end

function setup_parareal(n)
    GC.gc()
    prob = heat(n)
    dt = prob.p
    return init(prob, PararealAlgo(Threads.nthreads(), Tsit5(), Tsit5()); dt, dt_coarse=2*dt)
end

function setup_serial(n)
    GC.gc()
    prob = heat(n)
    return init(prob, Tsit5(); dt=prob.p, adaptive=false)
end

function run_benchmark()
    suite = BenchmarkGroup()
    suite["MGRIT"] = sm = BenchmarkGroup()
    suite["Parareal"] = sp = BenchmarkGroup()
    suite["Tsit5"] = st = BenchmarkGroup()
    for n = [2^12, 2^14, 2^16, 2^18, 2^20]
        sm[n] = @benchmarkable solve!(integrator) setup=(integrator=setup_mgrit($n))
        st[n] = @benchmarkable solve!(integrator) setup=(integrator=setup_serial($n))
        sp[n] = @benchmarkable solve!(integrator) setup=(integrator=setup_parareal($n))
    end

    # Adjust benchmarking settings
    for (_, v) in leaves(suite)
        v.params.evals = 1
        v.params.samples = 4
    end

    # Warmup, run and save the suite
    warmup(suite)
    r = run(suite; verbose=true)
    filename = joinpath(@__DIR__, "trial", "sweep", "threads_$(Threads.nthreads()).json.gz")
    mkpath(dirname(filename))
    PkgJogger.save_benchmarks(filename, r)

    return r
end

function cli()
    isempty(ARGS) && return nothing
    run_benchmark()
    exit()
end

function run_case(t)
    # Run benchmark in seperate julia process (So we can change the thread count)
    cmd = Cmd(String[
        joinpath(Sys.BINDIR, "julia"), "--startup-file=no", "--project", "--threads", "$t", "-O3",
        "--", @__FILE__, "run"
    ] |> Cmd,
    env=Dict(
        "JULIA_EXCLUSIVE"=>"1",
        "JULIA_LOAD_PATH"=>"benchmark:"),
    dir=dirname(@__DIR__),
    )
    run(cmd)
end

function sweep(t = [1, 2, 4, 8, 16, 32, 64, 128])
    for t in t
        @info "Benchmarking with $t threads"
        @time run_case(t)
        GC.gc()
    end
    return nothing
end

function plot_case!(ax, speedup, case)
    x = sort(collect(keys(speedup)))
    traces = get_case_trace(speedup, case)
    for name in sort(collect(keys(traces)))
        lines!(ax, x, traces[name]; label=L"2^{%$(Int(log2(name)))}")
    end
    return nothing
end

function plot_benchmark(speedup)
    resolution = 400 .* (3., 1.5)
    f = Figure(resolution=resolution)
    for (i, case) in enumerate(["MGRIT", "Parareal"])
        ax = Axis(f[1, i], xscale=log2, yscale=log2, title=case,
            xlabel="Number of Threads",
            ylabel="Speedup",
            xticks=LogTicks(LinearTicks(4)),
        )
        xlims!(ax; low=1)
        plot_case!(ax, speedup, case)
    end
    f[2,:] = Legend(f, current_axis(), L"N_s",
        orientation=:horizontal,
        tellwidth=false,
        titleposition=:left
    )
    return f
end

function get_case_trace(speedup, case)
    traces = Dict{Int, Vector{Float64}}()
    for t in sort(collect(keys(speedup)))
        case_data = speedup[t][case]
        for (k, v) in case_data
            N = parse(Int, k)
            if N in keys(traces)
                append!(traces[N], v.time)
            else
                traces[N] = [v.time]
            end
        end
    end
    traces
end

function load_results()
    results = Dict{Int, BenchmarkGroup}()
    for f in readdir(joinpath(@__DIR__, "trial", "sweep"); join=true)
        threads = parse(Int, match(r"[0-9]+", basename(f)).match)
        b = PkgJogger.load_benchmarks(f)["benchmarks"]
        results[threads] = b
    end
    return results
end

function compute_speedup_serial(results; basis=Statistics.median)
    # Normalize by Serial runtimes
    speedup = Dict{Int, BenchmarkGroup}()
    ref = basis(results[1])
    for t in keys(results)
        speedup[t] = ratio(ref, basis(results[t]))
    end
    return speedup
end

function compute_speedup_rel(results; case="Tsit5", basis=Statistics.median)
    # Normalize by Serial runtimes
    speedup = Dict{Int, BenchmarkGroup}()
    ref = basis(results[1][case])
    for t in keys(results)
        s = speedup[t] = BenchmarkGroup()
        for case in filter(!=(case), keys(results[t]))
            s[case] = ratio(ref, basis(results[t][case]))
        end
    end
    return speedup
end


cli()

