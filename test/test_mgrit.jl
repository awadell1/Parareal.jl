using Test
using Parareal
using OrdinaryDiffEq

include("problems.jl")


@testset "Multi-Grid Indexing" begin
    prob = ode_linear_problem()
    integrator = init(prob, MGRIT(Euler()); m=2, levels=typemax(Int))
    m = integrator.m
    @test m isa Int
    @test m > 1
    @test m == 2

    @testset "construct" begin
        u = integrator.u
        nt = length(integrator.t)
        @test length(u) == 4  # 4 Levels in total
        @test length(u[1]) == nt == 11  # 11 time steps at the finest level
        @test length(u[2]) == 5 # C-points at the first level + first c-point (constant)
        @test length(u[3]) == 2 # C-points at the second level + first c-point (constant)
        @test length(u[4]) == 1 # C-points at the third level + first c-point (constant)
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
            @test_throws AssertionError Parareal.timespan(integrator, 2, 3)
        end

        # Check the 3rd level
        t = Parareal.timespan(integrator, 3, 1)
        @test t isa StepRangeLen{Float64}
        @test first(t) ≈ first(t_ref)
        @test last(t) == t_ref[9]
        @test_throws AssertionError Parareal.timespan(integrator, 3, 2)

        # No more levels
        @test_throws AssertionError Parareal.timespan(integrator, 4, 1)
    end
end

@testset "relaxation" begin
    integrator = init(ode_linear_problem(), MGRIT(Euler()); m=2, levels=typemax(Int), dt=0.1)
    Parareal.f_relax!(integrator, 1)
    c_ref = [[0.5]]
    f_ref = deepcopy([integrator.u[1][2]])
    @test all(integrator.u[1][1:2:end] .== c_ref)   # C-points are unchanged
    @test all(integrator.u[1][2:2:end] .== f_ref)   # F-points are identical

    Parareal.c_relax!(integrator, 1)
    @test integrator.u[1][1] == c_ref[1]
    c_ref = deepcopy([integrator.u[1][3]])
    @test all(integrator.u[1][3:2:end] .== c_ref)   # C-points are identical after first
    @test all(integrator.u[1][2:2:end] .== f_ref)   # F-points are unchanged
end