using Test
using Parareal

include("problems.jl")

function test_distinct(u)
    for i in 1:length(u)
        test_distinct(u[i], u[i]; diag_ok=true) || return false
        for j in i+1:length(u)
            u[i] === u[j] && return false
            test_distinct(u[i], u[j]) || return false
        end
    end
    return true
end

function test_distinct(u, v; diag_ok=false)
    for i in 1:length(u)
        for j in 1:length(v)
            i == j && diag_ok && continue
            u[i] === v[j] && return false
        end
    end
    return true
end

@testset "All Distinct" begin
    integrator = init(ode_linear_problem(), MGRIT(Euler()))
    @testset "init" begin
        @test test_distinct(integrator.u)
        @test test_distinct(integrator.g)
        @test test_distinct(integrator.v)
    end
    @testset "f_relax!" begin
        Parareal.f_relax!(integrator, 1)
        @test test_distinct(integrator.u)
        @test test_distinct(integrator.g)
        @test test_distinct(integrator.v)
        Parareal.f_relax!(integrator, 2)
        @test test_distinct(integrator.u)
        @test test_distinct(integrator.g)
        @test test_distinct(integrator.v)
    end
    @testset "c_relax!" begin
        Parareal.c_relax!(integrator, 1)
        @test test_distinct(integrator.u)
        @test test_distinct(integrator.g)
        @test test_distinct(integrator.v)
        Parareal.c_relax!(integrator, 2)
        @test test_distinct(integrator.u)
        @test test_distinct(integrator.g)
        @test test_distinct(integrator.v)
    end
    @testset "inject!" begin
        Parareal.inject!(integrator, 1)
        @test test_distinct(integrator.u)
        @test test_distinct(integrator.g)
        @test test_distinct(integrator.v)
        Parareal.inject!(integrator, 2)
        @test test_distinct(integrator.u)
        @test test_distinct(integrator.g)
        @test test_distinct(integrator.v)
    end
    @testset "refine!" begin
        Parareal.refine!(integrator, 1)
        @test test_distinct(integrator.u)
        @test test_distinct(integrator.g)
        @test test_distinct(integrator.v)
        Parareal.refine!(integrator, 2)
        @test test_distinct(integrator.u)
        @test test_distinct(integrator.g)
        @test test_distinct(integrator.v)
    end
end
