using BenchmarkTools
using Parareal

include("../test/problems.jl")

const suite = BenchmarkGroup()

intergrators = Dict(
    "RK4" => RK4(),
    "Parareal-4-RK4" => PararealAlgo(4, RK4(), RK4()),
)

for (k,v) in intergrators
    s = suite[k] = BenchmarkGroup()
    i = init(ode_linear_problem(), v)
    s[1] = @benchmarkable solve(ode_linear_problem(), $v; dtmax=1e-2)
end
