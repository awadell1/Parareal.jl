using BenchmarkTools
using Parareal

include("../test/problems.jl")

const suite = BenchmarkGroup()

function setup(dt=1e-2)
    init(ode_linear_problem(), MGRIT(Euler()); m=2, dt, levels=typemax(Int))
end

i = setup(1e-2)
suite["timespan"] = @benchmarkable Parareal.timespan($i, 1, 1)

suite["f_relax!"] = BenchmarkGroup()
suite["c_relax!"] = BenchmarkGroup()
for dt in [1e-1, 1e-2, 5e-3]
    suite["f_relax!"][dt] = @benchmarkable Parareal.f_relax!($(setup(dt)), 1)
    suite["c_relax!"][dt] = @benchmarkable Parareal.c_relax!($(setup(dt)), 1)
end