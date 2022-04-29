using OrdinaryDiffEq
using DiffEqOperators

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
function heat(n=100)
    h = 1.0 / (n-1)
    x = range(h, step=h, length=n)
    Δ = CenteredDifference(2, 2, h, n)
    bc = Dirichlet0BC(Float64)
    dudt = (u,p,t) -> Δ*bc*u

    u_analytic(x, t) = sin(2*π*x) * exp(-t*(2*π)^2)
    f = ODEFunction(dudt, analytic=(u0,p,t) -> u_analytic.(x, t))
    return ODEProblem(f, u_analytic.(x, 0), (0.0, 0.03))
end

