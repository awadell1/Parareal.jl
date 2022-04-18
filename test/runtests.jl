using Test
using SafeTestsets
using PkgJogger

@testset "Parareal" begin
    @safetestset "Parareal" begin include("test_parareal.jl") end
    @safetestset "Benchmarks" begin include("test_benchmarks.jl") end
end
