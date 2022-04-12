using Test
using SafeTestsets

@testset "Parareal" begin
    @safetestset "Parareal" begin include("test_parareal.jl") end
end
