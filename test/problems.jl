using OrdinaryDiffEq
using DiffEqOperators
using LinearAlgebra

"""
Linear decay from initial displacement
"""
function ode_linear_problem()
    function linear!(du, u, p, t)
        du[1] = p*u[1]
        return nothing
    end
    linear_analytic = (u0,p,t) -> u0*exp(p*t)
    f = ODEFunction(linear!, analytic=linear_analytic)
    return ODEProblem(f, [1/2], (0.0,1.0), 1.01)
end

"""
    heat(n=100)

1D Head Diffusion problem: dT/dt = α dT²/dx²
"""
function heat(n=100, N=256)
    h = 1.0 / (n-1)
    x = range(h, step=h, length=n)
    Δ = CenteredDifference(2, 2, h, n)
    bc = Dirichlet0BC(Float64)
    A = Δ*bc
    function dudt!(du, u, p, t)
        mul!(du, A, u)
        return nothing
    end

    # Estimate the dt required for stability
    dt = 0.5*h^2    # Baseline stability criteria
    dt /= log2(N)   # Refine so log2(N) coarsening levels
    dt = 10^round(log10(dt), RoundDown) # Round for neatness / safety factor
    tf = N * dt

    u_analytic(x, t) = sin(2*π*x) * exp(-t*(2*π)^2)
    f = ODEFunction(dudt!, analytic=(u0,p,t) -> u_analytic.(x, t))
    return ODEProblem(f, u_analytic.(x, 0), (0.0, tf), dt)
end

TEST_PROBLEMS = Dict(
    "ode linear" => ode_linear_problem(),
    "heat" => heat(10),
)
