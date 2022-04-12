# Parareal.jl

A [Julia] implementation of the [Multi-Grid-Reduction-In-Time (MGRIT)][MGRIT] parallel-in-time solver for differential equations.
Despite the importance of solving differential equations, [most solver suites lack any form of parallelization](https://www.stochasticlifestyle.com/comparison-differential-equation-solver-suites-matlab-r-julia-python-c-fortran/).
When parallelism is present, it is often parallelism within a single time step (i.e. via spatial decomposition).
This is highly applicable when solving massive Partial Differential Equations (PDEs), but less so for smaller ODEs computed over long time intervals.
This project aims to provide a MGRIT solver for a shared memory parallel environment using [Julia's threading facilities][julia-multithreading].


## Background

MGRIT works by solving iteratively solving the system on a increasingly finer temporal grid, in parallel.
The fine solution is then used to update the coarse solution, sweeping back up the levels of temporal resolution.
This process is repeated until the desired level of accuracy is achieved.
Effectively, MGRIT is a generalization of the [Parareal Method][Parareal] from a two-level temporal hierarchy (Parareal) to at most `log T` levels for `T` time.
This hierarchy of temporal grids enables near optimal and work-efficient parallel decomposition of the problem.

![Two Iterations of MGRIT](docs/assets/pymgrit_iteration.png)
> Two iterations of MGRIT for 2D Heat Conduction  Source: [Algorithm 1016: PyMGRIT: A Python Package for the
Parallel-in-time Method MGRIT][PyMGRIT]

For a two level hierarchy this results in the following pseudoscope, which is effectively the same as the [Parareal Method][Parareal].

```julia
"""
dt = Minimum time step step of the fine grid
m = Number of coarse grid points per fine grid point
T = Total time of the problem
y0 = Initial condition
"""
function solve(y0, T, m, dt)
    # Assume an initially constant solution
    y = repeat(y0, T/dt)

    while !converged
        # Using the coarse solution, update the fine solution in parallel
        @threads for t = 1:m:T/dt
            y[t+1:t+m] = solve_fine(y[t])
        end

        # Using the fine solution, update the coarse solution in parallel
        # This is taking the final solution from the fine solution (y[t+m*dt]) and
        # using it to update the next coarse point (y[t+2*m*dt])
        @threads for t = m:m:T/dt
            y[t+m] = solve_coarse(y[t])
        end
    end
end
```

Recursively decomposing the above problem via `solve_fine(y) = solve(y0, m*dt, m*2, dt/2)` gives the MGRIT method.
As such, the algorithm has there distinct axes of parallelism:

1. Updating the fine temporal grid in parallel
2. Updating the coarse temporal grid in parallel
3. Within the differential: `dy = f(y)`

## Challenges

MGRIT has been shown to provide near optimal scaling, there are some challenges that I hope to explore over the course of the project:

1. While the above code shows two sequential parallel loops, a fork-and-join approach is also possible.
2. Exploiting parallelism within the differential, may be advantageous for some levels of the temporal multigrid.
3. Viable parallelism is likely problem dependent. For example, the quality of the coarse-grid solution for a low-order equations (i.e. [projectile motion](https://en.wikipedia.org/wiki/Projectile_motion)) is likely to be much better than a chaotic system (ie. [strange attractors](https://en.wikipedia.org/wiki/Attractor#Strange_attractor)). Achiving good speed-up will require balancing parallelism-in-time with in differential parallelism.

## Resources

For this project I plan on building on Julia's already fantastic ecosystem, notably:

1. Various solvers (But not MGRIT) for Differential Equations will be provided by [DifferentialEquations.jl][DiffEq.jl]. With the ultimate goal of providing a MGRIT solver that is compatible with [DifferentialEquations.jl][DiffEq.jl].
2. Julia's builtin [threading primitives][julia-multithreading] and other threading libraries such as [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl) and [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl).

Additionally, I wil be referring extensively to both the [MGRIT paper][MGRIT] and the [PyMGRIT package][PyMGRIT] implementation for the technical details of the algorithm.

### Computing Resources

As [Julia] runs on a range of hardware, I plan on doing most of my development work on my laptop (Mac M1 Processor).
For testing at large thread counts, I plan on either leverage the [Bridges2 RM-machines](https://www.psc.edu/resources/bridges-2/) or the latest node on [Arjuna] featuring 2 AMD EPYC 7713 64-Core Processor (Pending Advisor Approval).

## Deliverables

- 75% Goal: Functional Parareal solver if generalizing to multi-grid provides poor scaling
- 100% Goal: Functional MGRIT solver
- 125% Goal: Functional MGRIT solver with dynamic trade-off between in-time and in-differential parallelism

## Platform Choice

[Julia] is fantastic language for scientific computing with a [fantastic ecosystem](https://www.stochasticlifestyle.com/comparison-differential-equation-solver-suites-matlab-r-julia-python-c-fortran/) for solving differential equations.
In my research (Modeling Battery Dynamics with Scientific Machine Learning), it is my primary language of development due to it's speed, dynamism and multiple dispatch abilities.
Thus from a learning perspective applying the concept from [class](https://www.cs.cmu.edu/afs/cs/academic/class/15418-s22/www/index.html) to [Julia] is key.

Machine-wise, running code faster on my personal machine has an immediate impact on my productivity, while testing on more powerful machines (i.e. [Arjuna]) be useful for benchmarking (It's a brand new machine) and my research goals.

## References

- [A Multigrid-in-Time Algorithm for Solving Evolving Equation in Parallel][MGRIT]
- [A "parareal" in time discretication of PDE's][Parareal]
- [PyMGRIT: A Python Package for the Parallel-in-time Method MGRIT][PyMGRIT]
- [DifferentialEquations.jl][DiffEq.jl]
- [Julia]
- [Arjuna] - A Multi-Departmental Computer Cluster that I help administer

[Julia]: https://julialang.org/
[MGRIT]: https://www.osti.gov/servlets/purl/1073108
[Parareal]: https://doi.org/10.1016/S0764-4442(00)01793-6
[PyMGRIT]: https://dl.acm.org/doi/pdf/10.1145/3446979
[DiffEq.jl]: https://diffeq.sciml.ai/stable/
[julia-multithreading]: https://docs.julialang.org/en/v1/manual/multi-threading/
[Arjuna]: https://arjunacluster.github.io/ArjunaUsers/
