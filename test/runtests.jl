using Test
using SafeTestsets
using PkgJogger

@testset "Parareal" begin
    @safetestset "Parareal" begin include("test_parareal.jl") end
    @safetestset "Benchmarks" begin include("test_benchmarks.jl") end
    @testset "MGRIT" begin
        @safetestset "Distinct" begin include("test_distinct.jl") end
        @safetestset "Functional" begin include("test_mgrit.jl") end
    end
end
