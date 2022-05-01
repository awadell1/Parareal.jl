using BenchmarkTools
using Parareal

include("../test/problems.jl")

const suite = BenchmarkGroup()

function setup(dt=1e-2)
    init(ode_linear_problem(), MGRIT(Euler()); dt)
end

i = setup(1e-2)
suite["timespan"] = @benchmarkable Parareal.timespan($i, 1, 1)
suite["f_relax!"] = BenchmarkGroup()
suite["c_relax!"] = BenchmarkGroup()
suite["inject!"] = BenchmarkGroup()
suite["refine!"] = BenchmarkGroup()
suite["forward_solve!"] = BenchmarkGroup()
suite["perform_cycle!"] = BenchmarkGroup()
suite["residual"] = BenchmarkGroup()
for dt in [1e-1, 1e-2, 5e-3]
    suite["f_relax!"][dt] = @benchmarkable Parareal.f_relax!($(setup(dt)), 1)
    suite["c_relax!"][dt] = @benchmarkable Parareal.c_relax!($(setup(dt)), 1)
    suite["inject!"][dt] = @benchmarkable Parareal.inject!($(setup(dt)), 1)
    suite["refine!"][dt] = @benchmarkable Parareal.refine!($(setup(dt)), 1)
    suite["forward_solve!"] = @benchmarkable Parareal.forward_solve!($(setup(dt)))
    suite["perform_cycle!"][dt] = @benchmarkable Parareal.perform_cycle!($(setup(dt)), 1, 1)
    suite["residual"][dt] = @benchmarkable Parareal.residual($(setup(dt)))
end
