module QueryStrategies

abstract type QueryStrategy end

abstract type DataBasedQs <: QueryStrategy end
abstract type ModelBasedQs <: QueryStrategy end
abstract type NeighborBasedQs <: QueryStrategy end

using SVDD

include("qs_utils.jl")

include("TestQs.jl")
include("RandomQs.jl")
include("RandomOutlierQs.jl")
include("MinimumMarginQs.jl")
include("ExpectedMinimumMarginQs.jl")
include("ExpectedMaximumEntropyQs.jl")
include("MinimumLossQs.jl")
include("HighConfidenceQs.jl")
include("DecisionBoundaryQs.jl")
include("NeighborhoodBasedQs.jl")
include("BoundaryNeighborCombinationQs.jl")

export
    QueryStrategy,
    DataBasedQs,
    ModelBasedQs, NeighborBasedQs,

    # data-based query strategies
    TestQs, RandomQs, MinimumMarginQs, ExpectedMinimumMarginQs, ExpectedMaximumEntropyQs,
    MinimumLossQs,
    # model-based query strategies
    RandomOutlierQs, HighConfidenceQs, DecisionBoundaryQs,
    # hybrid query strategies
    NeighborhoodBasedQs, BoundaryNeighborCombinationQs,

    qs_score,
    initialize_qs,
    filter_array,
    multi_kde, KDEException, MissingLabelTypeException

end
