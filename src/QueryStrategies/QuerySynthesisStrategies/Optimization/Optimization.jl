module Optimization

using BlackBoxOptim
using Evolutionary

import ..QuerySynthesisStrategies: check_epsilon, data_boundaries

abstract type QuerySynthesisOptimizer end

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
