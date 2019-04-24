module SequentialQueryStrategies

import SVDD
import MLLabelUtils
import Statistics: mean

import ..QueryStrategies:
    QueryStrategy,
    MissingLabelTypeException,

    get_query_objects,
    knn_indices,
    knn_mean_dist,
    leave_out_one_cv_kde,
    multi_kde,
    qs_score


include("sequential_qs_base.jl")

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
    PoolQs,
    SequentialPQs,
    DataBasedPQs,
    ModelBasedPQs,
    HybridPQs,

    # data-based query strategies
    TestPQs, RandomPQs, MinimumMarginPQs, ExpectedMinimumMarginPQs, ExpectedMaximumEntropyPQs,
    MinimumLossPQs,
    # model-based query strategies
    RandomOutlierPQs, HighConfidencePQs, DecisionBoundaryPQs,
    # hybrid query strategies
    NeighborhoodBasedPQs, BoundaryNeighborCombinationPQs
end
