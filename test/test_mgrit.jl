using Test
using Parareal
using OrdinaryDiffEq

include("problems.jl")

@testset "Multi-Grid Indexing" begin
    prob = ode_linear_problem()
    integrator = init(prob, MGRIT(Euler()))
    m = integrator.m
    @test m isa Int
    @test m > 1
    @test m == 2

    @testset "construct" begin
        u = integrator.u
        nt = length(integrator.t)
        @test length(u) == 3  # 4 Levels in total
        @test length(u[1]) == nt == 11  # 11 time steps at the finest level
        @test length(u[2]) == 5 # C-points at the first level + first c-point (constant)
        @test length(u[3]) == 2 # C-points at the second level + first c-point (constant)
    end

    @testset "timespan" begin
        t_ref = integrator.t
        @test t_ref isa StepRangeLen{Float64}
        @test first(t_ref) == first(prob.tspan)
        @test last(t_ref) == last(prob.tspan)

        # Check the 1st level
        @testset "1st level" begin
            ts = first(t_ref)
            for i in 1:5
                t = Parareal.timespan(integrator, 1, i)
                @test t isa StepRangeLen{Float64}
                @test first(t) ≈ ts
                ts += 2*step(t)
                @test last(t) ≈ ts
            end
            @test ts ≈ last(t_ref)
        end

        # Check the 2nd level
        @testset "2nd Level" begin
            t = Parareal.timespan(integrator, 2, 1)
            @test t isa StepRangeLen{Float64}
            @test first(t) == t_ref[1]
            @test last(t) == t_ref[5]
            t = Parareal.timespan(integrator, 2, 2)
            @test t isa StepRangeLen{Float64}
            @test first(t) == t_ref[5]
            @test last(t) == t_ref[9]
            #@test_throws AssertionError Parareal.timespan(integrator, 2, 3)
        end

        # Check the 3rd level
        t = Parareal.timespan(integrator, 3, 1)
        @test t isa StepRangeLen{Float64}
        @test first(t) ≈ first(t_ref)
        @test last(t) == t_ref[9]
        #@test_throws AssertionError Parareal.timespan(integrator, 3, 2)

        # No more levels
        #@test_throws AssertionError Parareal.timespan(integrator, 4, 1)
    end
end

@testset "relaxation" begin
    integrator = init(ode_linear_problem(), MGRIT(Euler()); dt=0.1)
    Parareal.f_relax!(integrator, 1)
    c_ref = [[0.5]]
    f_ref = deepcopy([integrator.u[1][2]])
    @test all(integrator.u[1][1:2:end] .≈ c_ref)   # C-points are unchanged
    @test all(integrator.u[1][2:2:end] .≈ f_ref)   # F-points are identical

    Parareal.c_relax!(integrator, 1)
    @test integrator.u[1][1] == c_ref[1]
    c_ref = deepcopy([integrator.u[1][3]])
    @test all(integrator.u[1][3:2:end] .≈ c_ref)   # C-points are identical after first
    @test all(integrator.u[1][2:2:end] .≈ f_ref)   # F-points are unchanged

    # Now finish the FCF cycle, inject 1 level down, and FCF there
    Parareal.f_relax!(integrator, 1) # One more f-relaxation

    # Inject to next level and perform FCF cycle
    Parareal.inject!(integrator, 1)
    Parareal.f_relax!(integrator, 2)
    Parareal.c_relax!(integrator, 2)
    Parareal.f_relax!(integrator, 2)

    u_lvl = deepcopy(integrator.u[1])
    u_next = deepcopy(integrator.u[2])
    Parareal.refine!(integrator, 1)
    @test integrator.u[2] ≈ u_next # The next level is unchanged
    @test all(integrator.u[1][3:2:end] .≈ u_next) # This level matches the next level
    @test all(integrator.u[1][3:2:end] .!== u_next) # But it not the same as the previous level
end

# Repeated application of f and c relaxation should match a serial solution
@testset "serial" begin
    prob = ode_linear_problem()
    integrator = init(prob, MGRIT(Euler()); dt=0.1)

    @testset "level-$lvl" for lvl in 1:3
        nt = length(integrator.u[lvl])
        for i = 1:5
            Parareal.f_relax!(integrator, lvl)
            Parareal.c_relax!(integrator, lvl)
        end

        # Get reference solution and compare to serial
        dt = 0.1 * 2^(lvl-1)
        sol = solve(prob, Euler(); dt)
        u_ref = lvl == 1 ? sol.u : sol.u[2:nt+1]
        @test integrator.u[lvl] ≈ u_ref
    end
end

@testset "solve - linear" begin
    prob = ode_linear_problem()
    @testset "$alg" for alg in [Euler(), RK4(), Tsit5()]
        @testset "dt = $dt" for dt in [1e-1, 1e-2, 1e-3]
            sol = solve(prob, MGRIT(alg); dt)
            sol_ref = solve(prob, alg; dt, adaptive=false)
            @test isapprox(sol.u, sol_ref.u; rtol=1e-3, atol=1e-6)
        end
    end
end

@testset "solve - heat" begin
    prob = heat(10)
    @testset "$alg" for alg in [RK4(), Tsit5()]
        dt = 1e-3
        sol = solve(prob, MGRIT(alg); dt)
        sol_ref = solve(prob, alg; dt, adaptive=false)
        @test isapprox(sol.u, sol_ref.u; rtol=1e-3, atol=1e-6)
    end
end
