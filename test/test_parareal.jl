using Test
using Parareal
using OrdinaryDiffEq
using DiffEqDevTools

include("problems.jl")

@testset "smoke" begin
    alg = PararealAlgo(4, RK4(), RK4())
    @test alg isa PararealAlgo

    # Simulate
    saveat = 0.0:0.1:1.0
    sol = solve(ode_linear_problem(), alg; reltol=1e-3, abstol=1e-6, saveat)
    @test sol isa ODESolution
    @test sol.t[1] == 0.0
    @test sol.t[end] == 1.0
    @test sol.t == collect(saveat)

    # Check that reference solution is resonable
    sol_ref = solve(ode_linear_problem(), RK4(); saveat)
    sol = DiffEqDevTools.appxtrue(sol, sol_ref)
    @test sol.errors[:final] <= 1e-4
end


@testset "heat" begin
    sol = solve(heat(10), PararealAlgo(32, Euler(), Euler()); dt=1e-4, dt_coarse=1e-2)
    sol_ref = solve(heat(10), Euler(); dt=1e-4)
    sol = DiffEqDevTools.appxtrue(sol, sol_ref)
    @test sol.errors[:l2] <= 1e-4
    @test sol.errors[:l∞] <= 1e-4
    @test sol.errors[:final] <= 1e-4
end


