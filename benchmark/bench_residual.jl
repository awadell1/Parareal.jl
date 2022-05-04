using BenchmarkTools
using Parareal

include("../test/problems.jl")

const suite = BenchmarkGroup()

"""
    scale_results(b::BenchmarkGroup)

Scale benchmarking results by workload size
"""
function scale_results(b)
    for (k,v) in b
        @info "$k" round(median(v.times)/k; sigdigits=2)
    end
end

function setup(n=10)
    init(ode_linear_problem(), MGRIT(Euler()); dt=1/n)
end

for n in [1e1, 1e2, 1e3, 1e4]
    suite[n] = @benchmarkable Parareal.residual($(setup(n)))
end
