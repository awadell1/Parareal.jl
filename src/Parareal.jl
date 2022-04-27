module Parareal

using OrdinaryDiffEq
using DiffEqBase

export PararealAlgo, PararealIntegrator

# Enable Timing
using TimerOutputs
const to = TimerOutput()

include("parareal_algo.jl")
include("mgrit_algo.jl")

end
