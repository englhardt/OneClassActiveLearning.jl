module Optimization

using Evolutionary

import BlackBoxOptim

import ..QuerySynthesisStrategies:
    check_epsilon,
    data_boundaries

include("optimization_base.jl")
include("BlackBoxOptimization.jl")
include("EvolutionaryOptimization.jl")
include("ParticleSwarmOptimization.jl")

export
    QuerySynthesisOptimizer,
    BlackBoxOptimization,
    EvolutionaryOptimization,
    ParticleSwarmOptimization,

    query_synthesis_optimize
end
