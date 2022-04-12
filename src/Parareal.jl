module Parareal

export PararealAlgo, PararealIntegrator

using OrdinaryDiffEq
using DiffEqBase

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
    dt_coarse = 0.1, dtmin_coarse = 0.1, save_start=true, save_end=true, save_intervals=true, kwargs...) where {uType, tType, IIP}

    # Initialize the time, state and update vectors
    u = Vector{uType}(undef, alg.intervals+1)
    u[1] = prob.u0
    t = range(first(prob.tspan), last(prob.tspan); length=alg.intervals+1)

    # Setup coarse integrator
    coarse = init(prob, alg.coarse;
        dt=dt_coarse,
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
            kwargs...
        )
    end

    # Setup options
    opts = (; dt_coarse, dtmin_coarse, kwargs...)

    PararealIntegrator{IIP, uType, eltype(t)}(u, t, fine, coarse, opts)
end

function DiffEqBase.__solve(prob::ODEProblem, alg::PararealAlgo; kwargs...)

    intervals = alg.intervals
    integrator = init(prob, alg; kwargs...)
    solve!(integrator)

    # Collect results
    ntsteps = sum(x -> length(x.sol), integrator.fine)
    ts = similar(integrator.t, ntsteps)
    trace = similar(integrator.u, ntsteps)
    tdx = 1
    for i in 1:intervals
        n = length(integrator.fine[i].sol)
        ts[tdx:tdx+n-1] .= integrator.fine[i].sol.t
        trace[tdx:tdx+n-1] .= integrator.fine[i].sol.u
        tdx += n
    end

    sol = DiffEqBase.build_solution(prob,alg,ts,trace)

    return sol
end

function update_coarse!(integrator::PararealIntegrator)
    # Get local copies
    coarse = integrator.coarse
    u = integrator.u
    tstops = integrator.t
    u_updates = integrator.u_update

    # Start from the end of the first fine integrator
    DiffEqBase.set_ut!(coarse, first(u), first(tstops))
    n = length(integrator.u) -1
    iter = integrator.iteration[] += 1

    # Update coarse solution using fine solutions
    for i in 1:n
        DiffEqBase.add_tstop!(coarse, integrator.t[i+1])
        step!(coarse)
        @assert coarse.t == integrator.t[i+1]
        u[i+1] = coarse.u + u_updates[i]
        set_u!(coarse, u[i+1])
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
        for i in 1:n_intervals
            F = integrator.fine[i]
            u = integrator.u
            t = integrator.t
            reinit!(F, u[i]; t0=t[i])
            solve!(F)
            integrator.u_update[i] = F.u - u[i+1]
        end

        # Update coarse integrator
        update_coarse!(integrator)

        # Check for convergence
        u_update = integrator.u_update
        u_zero = zero(first(u_update))
        all(x -> isapprox(x, u_zero; rtol=1e-3, atol=1e-6), u_update) && break
    end

    return nothing
end

end
