
struct PararealAlgo{FineIntegrator, CoarseIntegrator} <: OrdinaryDiffEq.OrdinaryDiffEqAlgorithm
    intervals::Int
    fine::FineIntegrator
    coarse::CoarseIntegrator
end

struct PararealIntegrator{IIP, uType, tType, FineIntegrator, CoarseIntegrator, O} <: DiffEqBase.AbstractODEIntegrator{PararealAlgo,IIP,uType,tType}
    iteration::Ref{UInt}        # Iteration count
    u::Vector{uType}            # State at start/stop of each interval using the coarse integrator
    u_update::Vector{uType}     # Correction terms from the fine integrators
    t::Vector{tType}            # Start/stop time for each of the fine integrators
    fine::Vector{FineIntegrator}    # Fine Integrators for each interval
    coarse::CoarseIntegrator        # Coarse Integrator
    opts::O
end
function PararealIntegrator{IIP, uType, tType}(u, t, fine::Vector{F}, coarse::C, opts::O) where {IIP, uType, tType, F, C, O}
    u_update = similar(u, length(u)-1)
    for i = 1:length(u_update)
        u_update[i] = zero(u[1])
    end
    PararealIntegrator{IIP, uType, tType, F, C, O}(0, u, u_update, t, fine, coarse, opts)
end

function DiffEqBase.__init(prob::ODEProblem{uType, tType, IIP}, alg::PararealAlgo;
    coarse_steps = 1, coarse_maxsteps = 20, dt_coarse=nothing,
    save_start=true, save_end=true, save_intervals=false,
    abstol=1e-6, reltol=1e-3,
    kwargs...
) where {uType, tType, IIP}

    # Initialize the time, state and update vectors
    u = Vector{uType}(undef, alg.intervals+1)
    u[1] = prob.u0
    t = range(first(prob.tspan), last(prob.tspan); length=alg.intervals+1)

    # Setup coarse integrator
    coarse_step_size = minimum(diff(t))
    if dt_coarse === nothing
        dt_coarse = coarse_step_size / coarse_steps;
        dtmin_coarse = coarse_step_size / coarse_maxsteps;
    else
        dtmin_coarse = dt_coarse * coarse_steps / coarse_maxsteps
    end
    coarse = init(prob, alg.coarse;
        dt=dtmin_coarse,
        dtmin=dtmin_coarse,
        force_dtmin=true,
        tstops=(),
        dense=false,
        saveat=(),
        advance_to_tstop=true,
        save_everystep=false,
        save_start=false,
        calck = false,

    )

    # Setup fine integrators
    fine = map(1:alg.intervals) do i
        prob_interval = remake(prob; tspan=(t[i], t[i+1]))
        init(prob_interval, alg.fine;
            save_start= i==1 ? save_start : save_intervals,
            save_end = i == alg.intervals ? save_end : save_intervals,
            save_everystep=false, dense=false,
            adaptive=false,
            kwargs...
        )
    end

    # Setup options
    opts = (; dt_coarse, dtmin_coarse, abstol, reltol, kwargs...)

    PararealIntegrator{IIP, uType, eltype(t)}(u, t, fine, coarse, opts)
end

function accumulate_destats!(r, x)
    for p in propertynames(r)
        n = getproperty(r, p) + getproperty(x, p)
        setproperty!(r, p, n)
    end
    return r
end

function DiffEqBase.__solve(prob::ODEProblem, alg::PararealAlgo; kwargs...)

    intervals = alg.intervals
    integrator = init(prob, alg; kwargs...)
    solve!(integrator)

    # Collect results
    destats = deepcopy(integrator.coarse.destats)
    @timeit_debug to "Collect Results" begin
        ntsteps = sum(x -> length(x.sol), integrator.fine)
        ts = similar(integrator.t, ntsteps)
        trace = similar(integrator.u, ntsteps)
        tdx = 1
        for i in 1:intervals
            n = length(integrator.fine[i].sol)
            ts[tdx:tdx+n-1] .= integrator.fine[i].sol.t
            trace[tdx:tdx+n-1] .= integrator.fine[i].sol.u
            tdx += n

            # Accumulate destats
            accumulate_destats!(destats, integrator.fine[i].destats)
        end
    end

    sol = DiffEqBase.build_solution(prob,alg,ts,trace; destats)

    return sol
end

function update_coarse!(integrator::PararealIntegrator)
    # Get local copies
    coarse = integrator.coarse
    u = integrator.u
    tstops = integrator.t
    u_updates = integrator.u_update

    # Start from the end of the first fine integrator
    reinit!(coarse, first(u); t0=first(tstops))
    n = length(integrator.u) -1
    iter = integrator.iteration[] += 1

    # Update coarse solution using fine solutions
    for i in 1:n
        DiffEqBase.add_tstop!(coarse, integrator.t[i+1])
        step!(coarse)
        @assert coarse.t == integrator.t[i+1]
        u[i+1] = coarse.u + u_updates[i]
        set_u!!(coarse, u[i+1])
    end

    return nothing
end


function DiffEqBase.solve!(integrator::PararealIntegrator)
    # Initial coarse solution
    update_coarse!(integrator)

    # Perform up to P iterations
    n_intervals = length(integrator.fine)
    for k in 1:n_intervals
        # Update fine integrators
        @timeit_debug to "Update fine" begin
            Threads.@threads for i in 1:n_intervals
                @inbounds F = integrator.fine[i]
                @inbounds u0 = integrator.u[i]
                @inbounds t0 = integrator.t[i]
                @inbounds tf = integrator.t[i+1]

                # Update fine integrator
                reinit!(F, u0; t0, tf, erase_sol=true, reset_dt=false)
                solve!(F)

                # Compute Update
                @inbounds @. integrator.u_update[i] = F.u - integrator.u[i+1]
            end
        end

        # Update coarse integrator
        @timeit_debug to "Update coarse" begin
            update_coarse!(integrator)
        end

        # Check for convergence
        u_update = integrator.u_update
        e = sum(x -> sum(x.^2), u_update)
        (; reltol, abstol) = integrator.opts
        isapprox(e, 0; rtol=reltol, atol=abstol) && break
    end

    return nothing
end
