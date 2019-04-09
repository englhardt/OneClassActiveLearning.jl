module QuerySynthesisStrategies

using MLKernels
using MLLabelUtils
using NearestNeighbors
using Statistics
using LinearAlgebra
using InteractiveUtils
using LIBSVM
using SVDD
using JuMP
using Distances
using Memento
using Reexport

import ...QueryStrategies: QueryStrategy, qs_score, get_query_object, MissingLabelTypeException, HybridPQs, initialize_qs

# Query Synthesis Strategies
abstract type QuerySynthesisStrategy <: QueryStrategy end

abstract type DataBasedQss <: QuerySynthesisStrategy end
abstract type ModelBasedQss <: QuerySynthesisStrategy end
abstract type HybridQss <: QuerySynthesisStrategy end

include("query_synthesis_utils.jl")

include("Optimization/Optimization.jl")
@reexport using .Optimization


include("query_synthesis_utils.jl")
include("TestQss.jl")
include("RandomQss.jl")
include("RandomOutlierQss.jl")
include("DecisionBoundaryQss.jl")
include("NaiveExplorativeMarginQss.jl")
include("ExplorativeMarginQss.jl")
include("HybridQuerySynthesisPQs.jl")

function get_query_object(qs::QuerySynthesisStrategy, query_data::Array{T, 2}, pools::Vector{Symbol}, history::Vector{Array{T, 2}})::Array{T, 2} where T <: Real
    return query_synthesis_optimize(qs_score_function(qs, query_data, labelmap(pools)), qs.optimizer, query_data, pools)
end


export
    QuerySynthesisStrategy,
    DataBasedQss, ModelBasedQss, HybridQss,
    # BaselineQss:
    TestQss, RandomQss,
    # DataBasedQss:
    # ModelBasedQss:
    DecisionBoundaryQss, RandomOutlierQss,
    # HybridQss:
    ExplorativeMarginQss, NaiveExplorativeMarginQss, HybridQuerySynthesisPQs,

    get_query_object,
    estimate_margin_epsilon, estimate_limit_epsilon,
    check_epsilon, check_limits,
    extrema_arrays,
    rand_in_hypercube,
    estimate_boundary_shift_epsilon, data_limit_epsilon, data_boundaries
end
