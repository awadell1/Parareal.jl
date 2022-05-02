
struct MGRIT{Integrator <: OrdinaryDiffEq.OrdinaryDiffEqAlgorithm} <: OrdinaryDiffEq.OrdinaryDiffEqAlgorithm
    integrator::Integrator
end

const SteppedTimeRange{T} = StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int} where {T}

struct ThreadedIntegrator{ILP, uType, tType, Integrator, O} <: DiffEqBase.AbstractODEIntegrator{MGRIT,ILP,uType,tType}
    u::Vector{Vector{uType}}    # State at each time step for the level
    g::Vector{Vector{uType}}    # Right hand side of FAS
    v::Vector{Vector{uType}}    # Restricted Unknowns
    t::SteppedTimeRange{tType}  # Times at each time step for the level
    m::Int                      # Temporal Coarsening Factor
    pool::Vector{Integrator}    # Pool of integrators for each threads
    opts::O                     # Integrator Options
end

ThreadedIntegrator{ILP, uType, tType}(u, g, v, t, m, pool::Vector{Integrator}, opts::O) where {ILP, uType, tType, Integrator, O} =
    ThreadedIntegrator{ILP, uType, tType, Integrator, O}(u, g, v, t, m, pool, opts)

function DiffEqBase.__init(prob::ODEProblem{uType, tType, ILP}, alg::MGRIT;
    dt = 0.1,
    abstol=1e-6, reltol=1e-3,
    maxiters=missing,
    m=2, levels=typemax(Int),
    threads=Threads.nthreads()
) where {uType, tType, ILP}

    # Initialize solution vectors and integrators
    t = range(first(prob.tspan), last(prob.tspan); step=dt)

    # Allocate the cache for storing the solution state
    nt = length(t)
    u0 = prob.u0
    u = Vector{Vector{uType}}()
    g = Vector{Vector{uType}}()
    v = Vector{Vector{uType}}()
    for i = 1:levels
        # Fill each level with the initial condition
        u_lvl = map(_ -> deepcopy(u0), 1:nt)
        push!(u, u_lvl)

        # Fill each level with zeros for g and vectors
        ng = i == 1 ? nt-1 : nt
        push!(g, map(_ -> zero(u0), 1:ng))
        push!(v, map(_ -> zero(u0), 1:ng))

        # Get the size of the next level and break if we are at the last level
        nt = floor(Int, nt/m)
        if nt <= 1
            levels = i
            break
        end
    end

    # Initialize an integrator for each thread
    integrator_pool = map(1:threads) do _
        init(prob, alg.integrator;
            dt=dt,
            tstops=(),
            dense=false,
            saveat=(),
            advance_to_tstop=true,
            save_everystep=false,
            save_start=false,
            calck = false,
            adaptive=false,
            force_dtmin=true,
        )
    end

    # Set Integrator Options
    opts = (;
        abstol, reltol, threads, levels, internalnorm=DiffEqBase.ODE_DEFAULT_NORM,
        maxiters=ismissing(maxiters) ? length(t) : maxiters,
    )

    return ThreadedIntegrator{ILP, uType, eltype(t)}(u, g, v, t, m, integrator_pool, opts)
end

function DiffEqBase.solve!(integrator::ThreadedIntegrator)
    level = 1
    iteration = 0
    converged = false
    (; maxiters) = integrator.opts

    # Iterate until convergence or maxiters
    while !converged && iteration <= maxiters
        perform_cycle!(integrator, level, iteration)
        iteration += 1
        converged = residual(integrator) < 1
    end
    return nothing
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem, alg::MGRIT, args...; kwargs...)
    integrator = DiffEqBase.__init(prob, alg, args...; kwargs...)
    solve!(integrator)
    DiffEqBase.build_solution(prob, alg, integrator.t, integrator.u[1])
end

@inline function get_state(integrator, level, i)
    if level == 1 || i == 1
        return integrator.u[1][i]
    else
        return integrator.u[level][i-1]
    end
end

function perform_cycle!(integrator, level, iteration)
    # At the coarsest level -> Solve forward
    max_level = integrator.opts[:levels]
    if level == max_level
        forward_solve!(integrator)
        return nothing
    end

    # At and intermediate level -> Perform FCF Cycle
    if level > 1 || iteration == 1
        f_relax!(integrator, level)
    end
    c_relax!(integrator, level)
    f_relax!(integrator, level)

    # Inject the current solution on to the next level
    # then solve that level, and use it' to update this
    # level's solution
    inject!(integrator, level)
    perform_cycle!(integrator, level + 1, iteration)
    refine!(integrator, level)

    f_relax!(integrator, level)

    return nothing
end

function f_relax!(integrator, level)
    m = integrator.m
    integrator.t
    nt = length(integrator.u[level])
    nt += level == 1 ? 0 : 1

    # Parallelize over regions of F-points
    n_c_regions = floor(Int, nt/m)
    Threads.@threads for cdx in 1:n_c_regions
        # Re-initialize the integrator for integrating from the `id` c-point to the next c-point
        Φ = integrator.pool[Threads.threadid()]

        # Index of c-point to integrate from
        i = (cdx-1)*m + 1
        u0 = get_state(integrator, level, i)
        t = timespan(integrator, level, cdx)
        set_ut!!(Φ, u0, first(t))
        DiffEqBase.set_proposed_dt!(Φ, step(t))

        # Index of the F-points to update
        fdx = level == 1 ? range(i+1, i+m-1) : range(i, i+m-2)
        gdx = i:i+m-2

        ## Update all F-points in current segment
        u = @view integrator.u[level][fdx]
        g = @view integrator.g[level][gdx]
        @inbounds for i in 1:m-1
            dt = t[i+1] - Φ.t
            step!(Φ, dt, true)
            if level == 1
                copy!(g[i], Φ.u - u[i])
                copy!(u[i], Φ.u)
            else
                copy!(u[i], Φ.u + g[i])
                set_u!!(Φ, u[i])
            end
        end
    end
    @debug "f-relax" level integrator.u[level] integrator.g[level]
    return nothing
end

function set_u!!(Φ::OrdinaryDiffEq.ODEIntegrator, u)
    copy!(Φ.u, u)
    DiffEqBase.u_modified!(Φ, true)
    return nothing
end

"""
    set_ut!!(Φ::OrdinaryDiffEq.ODEIntegrator, u, t)

Reset the integrator `Φ` to the given state `u` and time `t`
Effectively, a `reinit!` call, but with zero allocations, at the cost of only supporting
a subset of integrator algorithms. Does support Euler, RK4, Tsit5
"""
function set_ut!!(Φ::OrdinaryDiffEq.ODEIntegrator, u, t)
    set_u!!(Φ, u)
    DiffEqBase.set_t!(Φ, t)
    terminate!(Φ)
    return nothing
end

function c_relax!(integrator, level)
    m = integrator.m
    integrator.t
    nt = length(integrator.u[level])

    # Parallelize over regions of F-points
    n_c_regions = floor(Int, nt/m)
    Threads.@threads for cdx in 1:n_c_regions
        # Re-initialize the integrator for integrating from the `id` c-point to the next c-point
        Φ = integrator.pool[Threads.threadid()]

        # Index of the last f-point in the current segment
        fdx = level == 1 ? cdx*m : cdx*m-1
        u0 = integrator.u[level][fdx]
        t = timespan(integrator, level, cdx)[end-1:end]
        set_ut!!(Φ, u0, first(t))
        dt = step(t)
        DiffEqBase.set_proposed_dt!(Φ, dt)

        # Update c-point with the last f-point
        step!(Φ, dt, true)
        u = integrator.u[level][fdx+1]
        g = integrator.g[level][fdx]
        if level == 1
            copy!(g, Φ.u - u)
            copy!(u, Φ.u)
        else
            copy!(u, Φ.u + g)
        end
    end
    @debug "c-relax" level integrator.u[level] integrator.g[level]
    return nothing
end

function forward_solve!(integrator::ThreadedIntegrator)
    # Get u0 and a view into the last level
    u0 = @inbounds integrator.u[1][1]
    max_level = integrator.opts[:levels]
    u = @view integrator.u[max_level][:]

    # Get timesteps for the last level
    m = integrator.m
    nt = length(u)
    t0 = first(integrator.t)
    dt = step(integrator.t) * m^(max_level-1)
    t = range(t0; length=nt+1, step=dt)

    # Reinit the integrator for the last level
    Φ = first(integrator.pool)
    set_ut!!(Φ, u0, t0)
    DiffEqBase.set_proposed_dt!(Φ, dt)

    # Forward solve over the entire domain
    g = @view integrator.g[max_level][:]
    @inbounds for i in 1:nt
        dt = t[i+1] - Φ.t
        step!(Φ, dt, true)
        copy!(u[i], Φ.u + g[i])
        set_u!!(Φ, u[i])
    end
    return nothing
end

"""
    t = timespan(integrator, level, cdx)

Return the timesteps for the between c-point `cdx` and `cdx+1` on `level`
"""
function timespan(integrator::ThreadedIntegrator, level::Integer, cdx::Integer)
    # Get the state and check that level and cdx are valid
    u = integrator.u
    m = integrator.m
    @assert level <= length(u)

    # Compute the dt of the current level
    t = integrator.t
    m_dt = m^(level-1)

    # Compute the start and stop indices of the current segment
    m_lvl = m_dt * m
    sdx = m_lvl*(cdx-1) + 1
    @assert sdx < length(t)
    edx = min(length(t), sdx + m_lvl)

    # Construct range from start and stop indices
    @inbounds t[sdx:m_dt:edx]
end

function inject!(integrator::ThreadedIntegrator, level)
    m = integrator.m
    integrator.t
    nt = length(integrator.u[level])

    # Update the next level's v
    u = integrator.u
    sdx = level == 1 ? m+1 : m
    nc = length(u[level+1])
    u_lvl = @view u[level][range(sdx; length=nc, step=m)]
    g_lvl = @view integrator.g[level][range(m, length=nc, step=m)]
    u_next = @view integrator.u[level+1][:]
    v_next = @view integrator.v[level+1][:]
    g_next = @view integrator.g[level+1][:]

    # Update the next level's v and u
    @inbounds Threads.@threads for i in 1:nc
        copy!(v_next[i], u_lvl[i])
        copy!(u_next[i], u_lvl[i])
    end

    # Parallelize over regions of F-points
    n_c_regions = floor(Int, nt/m)
    Threads.@threads for i in 1:n_c_regions
        # Get this thread's integrator
        Φ = integrator.pool[Threads.threadid()]

        # Compute the residual on the current level
        fdx = level == 1 ? i*m : i*m-1
        u0 = integrator.u[level][fdx]
        t = timespan(integrator, level, i)
        set_ut!!(Φ, u0, t[end-1])
        dt = step(t)
        DiffEqBase.set_proposed_dt!(Φ, dt)
        step!(Φ, dt, true)
        g_next[i] = Φ.u - u_lvl[i]

        # Compute the residual on the next level
        u0 = get_state(integrator, level+1, i)
        t0 = first(t)
        tf = last(t)
        set_ut!!(Φ, u0, t0)
        dt = tf - t0
        DiffEqBase.set_proposed_dt!(Φ, dt)
        step!(Φ, dt, true)
        g_next[i] += v_next[i] - Φ.u

        # Add this level's g to the next level's g
        if level > 1
            g_next[i] += g_lvl[i]
        end

    end
    return nothing
end

function refine!(integrator::ThreadedIntegrator, level::Integer)
    # Get the state and check that level is valid
    u = integrator.u
    m = integrator.m
    @assert level+1 <= length(u)

    # Create a view for the c-points of the current level
    sdx = level == 1 ? m+1 : m
    nc = length(u[level+1])
    cdx = range(sdx; length=nc, step=m)
    u_lvl = @view u[level][cdx]

    # Create a view of the residuals of the next level
    u_next = @view u[level+1][:]
    v_next = @view integrator.v[level+1][:]

    # Refine this level using the residuals the next level
    @inbounds @batch minbatch=32 for i in 1:nc
        @. u_lvl[i] += u_next[i] - v_next[i]
    end

    return nothing
end

function residual(integrator::ThreadedIntegrator{ILP, uType}) where {ILP, uType}
    (; abstol, reltol, internalnorm) = integrator.opts
    t = integrator.t

    # Get the residual and state at the base level and check for a no-op
    @inbounds g = integrator.g[1]
    @inbounds u = integrator.u[1]
    T = eltype(uType)
    r = Threads.Atomic{T}(zero(T))
    isempty(g) && return r[]

    # Loop over timesteps and compute the scaled error of the residual
    # See: https://diffeq.sciml.ai/stable/basics/common_solver_opts/#Basic-Stepsize-Control
    @batch minbatch=64 for i in 1:length(g)
        @inbounds tc = t[i+1]   # Time of the current f/c-point
        @inbounds uc = u[i+1]   # Solution at the current f/c-point
        @inbounds gc = g[i]     # Residual at the current f/c-point
        u_norm = internalnorm(uc, tc)   # Norm of the state used for reltol
        e = internalnorm(gc, tc)        # Norm of the residual
        es = e / (abstol + u_norm*reltol)   # Scaled error at time step
        Threads.atomic_max!(r, es)
    end
    return r[]
end

