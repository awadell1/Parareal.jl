using Test
using Parareal
using OrdinaryDiffEq
using DiffEqProblemLibrary.ODEProblemLibrary
ODEProblemLibrary.importodeproblems()

@testset "smoke" begin
    alg = PararealAlgo(4, RK4(), RK4())
    @test alg isa PararealAlgo

    # Simulate
    saveat = 0.0:0.1:1.0
    sol = solve(prob_ode_linear, alg; save_intervals=false, reltol=1e-3, abstol=1e-6, saveat)
    @test sol isa ODESolution
    @test sol.t[1] == 0.0
    @test sol.t[end] == 1.0
    @test sol.t == collect(saveat)
    @test sol.errors[:l2] <= 1e-6

    # Check that reference solution is resonable
    sol_ref = solve(prob_ode_linear, RK4(); saveat)
    @test 10 * sol_ref.errors[:l2] >= sol.errors[:l2]
    @test sol_ref.t == collect(saveat)

    # Check that the reference solution is similar
    e = @.((sol.u - sol_ref.u)^2)
    l2 = sum(e) / length(e)
    @test l2 <= 1e-6
end





