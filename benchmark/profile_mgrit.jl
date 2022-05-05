using Parareal
using Profile
using PProf

include(joinpath(@__DIR__, "..", "test", "problems.jl"))

function setup_mgrit(n)
    GC.gc()
    prob = heat(n)
    dt = prob.p
    N = prob.tspan[end] / dt
    levels = floor(Int, log2(log2(N)))
    return init(prob, MGRIT(Tsit5()); dt, levels)
end

function profile(n=2^16)
    Profile.clear()
    integrator = setup_mgrit(n)
    @pprof solve!(integrator)
end
