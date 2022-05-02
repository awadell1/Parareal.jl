module Parareal

using OrdinaryDiffEq
using DiffEqBase
using LoopVectorization

export PararealAlgo, PararealIntegrator, MGRIT

# Enable Timing
using TimerOutputs
const to = TimerOutput()

include("parareal_algo.jl")
include("mgrit_algo.jl")

end
