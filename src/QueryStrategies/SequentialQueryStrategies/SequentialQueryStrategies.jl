module SequentialQueryStrategies
import ...QueryStrategies: PoolQs, qs_score, get_query_object, MissingLabelTypeException, multi_kde, leave_out_one_cv_kde

abstract type SequentialPQs <: PoolQs end

abstract type DataBasedPQs <: SequentialPQs end
abstract type ModelBasedPQs <: SequentialPQs end
abstract type HybridPQs <: SequentialPQs end

using MLKernels
using MLLabelUtils
using NearestNeighbors
using Statistics
using LinearAlgebra
using InteractiveUtils
using PyCall
using SVDD

include("seq_qs_utils.jl")
include("TestPQs.jl")
include("RandomPQs.jl")
include("RandomOutlierPQs.jl")
include("MinimumMarginPQs.jl")
include("ExpectedMinimumMarginPQs.jl")
include("ExpectedMaximumEntropyPQs.jl")
include("MinimumLossPQs.jl")
include("HighConfidencePQs.jl")
include("DecisionBoundaryPQs.jl")
include("NeighborhoodBasedPQs.jl")
include("BoundaryNeighborCombinationPQs.jl")

export
    SequentialPQs,
    DataBasedPQs,
    ModelBasedPQs, HybridPQs,

    # data-based query strategies
    TestPQs, RandomPQs, MinimumMarginPQs, ExpectedMinimumMarginPQs, ExpectedMaximumEntropyPQs,
    MinimumLossPQs,
    # model-based query strategies
    RandomOutlierPQs, HighConfidencePQs, DecisionBoundaryPQs,
    # hybrid query strategies
    NeighborhoodBasedPQs, BoundaryNeighborCombinationPQs,

    filter_array,
    multi_kde, KDEException, MissingLabelTypeException
end
