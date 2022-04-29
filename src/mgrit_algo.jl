
struct MGRIT{Integrator <: OrdinaryDiffEq.OrdinaryDiffEqAlgorithm} <: OrdinaryDiffEq.OrdinaryDiffEqAlgorithm
    integrator::Integrator
end

const SteppedTimeRange{T} = StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int} where {T}

struct ThreadedIntegrator{ILP, uType, tType, Integrator, O} <: DiffEqBase.AbstractODEIntegrator{MGRIT,ILP,uType,tType}
    u::Vector{Vector{uType}}    # State at each time step for the level
    g::Vector{Vector{uType}}    # Residual at each time step for the level
    t::SteppedTimeRange{tType}  # Times at each time step for the level
    m::Int                      # Temporal Coarsening Factor
    pool::Vector{Integrator}    # Pool of integrators for each threads
    opts::O                     # Integrator Options
end

ThreadedIntegrator{ILP, uType, tType}(u, g, t, m, pool::Vector{Integrator}, opts::O) where {ILP, uType, tType, Integrator, O} =
    ThreadedIntegrator{ILP, uType, tType, Integrator, O}(u, g, t, m, pool, opts)

function DiffEqBase.__init(prob::ODEProblem{uType, tType, ILP}, alg::MGRIT;
    dt = 0.1,
    abstol=1e-6,
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
    for i = 1:levels
        u_lvl = map(_ -> deepcopy(u0), 1:nt)
        push!(u, u_lvl)
        if i == 1
            push!(g, deepcopy(u_lvl[1:end-1]))
        else
            push!(g, deepcopy(u_lvl))
        end
        nt = floor(Int, nt/m)
        if nt == 0
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
        )
    end
    opts = (; abstol, threads, levels)

    return ThreadedIntegrator{ILP, uType, eltype(t)}(u, g, t, m, integrator_pool, opts)
end

function DiffEqBase.solve(integrator::ThreadedIntegrator)
    opts = integrator.opts
    level = 1
    iteration = 0
    !converged = false
    while !converged && iteration < 100
        f_relax!(integrator, level)
        perform_cycle!(integrator, level, iteration)
        residual = residual(integrator)
        @debug "Residual: $residual after iteration $iteration"
        converged = residual < opts.abstol
    end

    return integrator.t, integrator.u
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
    level > 1 && f_relax!(integrator, level)
    c_relax!(integrator, level)
    f_relax!(integrator, level)

    # Inject the current solution on to the next level
    # then solve that level, and use it' to update this
    # level's solution
    #inject!(integrator, level)
    perform_cycle!(integrator, level + 1, iteration)
    #refine!(integrator, level)

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
    for cdx in 1:n_c_regions
        # Re-initialize the integrator for integrating from the `id` c-point to the next c-point
        Φ = integrator.pool[Threads.threadid()]

        # Index of c-point to integrate from
        i = (cdx-1)*m + 1
        u0 = get_state(integrator, level, i)
        t = timespan(integrator, level, cdx)
        DiffEqBase.reinit!(Φ, u0; t0=first(t), tf=last(t))
        DiffEqBase.set_proposed_dt!(Φ, step(t))

        # Index of the F-points to update
        fdx = level == 1 ? range(i+1, i+m-1) : range(i, i+m-2)
        gdx = i:i+m-2

        ## Update all F-points in current segment
        u = @view integrator.u[level][fdx]
        g = @view integrator.g[level][gdx]
        for i in 1:m-1
            dt = t[i+1] - Φ.t
            step!(Φ, dt, true)
            @. g[i] .= u[i] - Φ.u  # Update residual
            @. u[i] .= Φ.u    # Update solution
        end
    end
    @info "f-relax" level integrator.u[level]
    return nothing
end

function c_relax!(integrator, level)
    m = integrator.m
    integrator.t
    nt = length(integrator.u[level])

    # Parallelize over regions of F-points
    n_c_regions = floor(Int, nt/m)
    for cdx in 1:n_c_regions
        # Re-initialize the integrator for integrating from the `id` c-point to the next c-point
        Φ = integrator.pool[Threads.threadid()]

        # Index of the last f-point in the current segment
        fdx = level == 1 ? cdx*m : cdx*m-1
        u0 = integrator.u[level][fdx]
        t = timespan(integrator, level, cdx)[end-1:end]
        DiffEqBase.reinit!(Φ, u0; t0=first(t), tf=last(t))
        dt = step(t)
        DiffEqBase.set_proposed_dt!(Φ, dt)

        # Update c-point with the last f-point
        step!(Φ, dt, true)
        u = integrator.u[level][fdx+1]
        g = integrator.g[level][fdx]
        @. g .= u - Φ.u  # Update residual
        @. u .= Φ.u    # Update solution
    end
    @info "c-relax" level integrator.u[level]
    return nothing
end

function forward_solve!(integrator::ThreadedIntegrator)
    Φ = first(integrator.pool)
    u0 = integrator.u[1][1]
    m = integrator.m
    max_level = integrator.opts[:levels]
    t = timespan(integrator, max_level, 1)
    DiffEqBase.reinit!(Φ, u0; t0=first(t), tf=last(t))
    DiffEqBase.set_proposed_dt!(Φ, step(t))

    # Forward solve over the entire domain
    u_seg = @view integrator.u[max_level][:]
    for i in 1:m-1
        dt = t[i+1] - Φ.t
        step!(Φ, dt, true)
        u_seg[i] .= Φ.u
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
    nt = length(u[level])
    nt -= level == 1 ? 1 : 0
    @assert 1 <= cdx && cdx <= ceil(Int, nt/m)

    # Compute the effect coarsening of the current level
    m_eff = m^level

    # Compute the start and stop indices of the current segment
    t = integrator.t
    sdx = m_eff*(cdx-1) + 1
    @assert sdx < length(t)
    edx = min(length(t), sdx + m_eff)

    # Scale dt by the effective coarsening
    dt = step(t)
    dt *= level == 1 ? 1 : m * (level-1)

    # Construct range from start and stop indices
    t0 = t[sdx]
    tf = t[edx]
    range(t0, tf; step=dt)
end

function inject!(integrator::ThreadedIntegrator, level::Integer)
    # Get the state and check that level is valid
    u = integrator.u
    m = integrator.m
    @assert level+1 <= length(u)

    # Inject the state into the next
    u_next = u[level+1]
    sdx = level == 1 ? m+1 : m
    u_idx = range(sdx; length=length(u_next), step=m)
    u_next .= view(u[level], u_idx)

    # Inject residual into the next level
    g_idx = range(m, length=length(u_next), step=m)
    integrator.g[level+1] .= @view integrator.g[level][g_idx]

    return nothing
end

function refine!(integrator::ThreadedIntegrator, level::Integer)
    # Get the state and check that level is valid
    u = integrator.u
    g = integrator.g
    m = integrator.m
    @assert level+1 <= length(u)

    # Refine the state using the next level
    u_lvl = u[level]
    u_next = u[level+1]
    sdx = level == 1 ? m+1 : m
    cdx = range(sdx; length=length(u_next), step=m)
    u[level][cdx] .+= g[level+1]
    return nothing
end

