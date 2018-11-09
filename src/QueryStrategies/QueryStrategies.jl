module QueryStrategies

using PyCall

const gaussian_kde = PyNULL()

function __init__()
    copy!(gaussian_kde, pyimport_conda("scipy.stats", "scipy")[:gaussian_kde])
end

include("qs_base.jl")
include("qs_utils.jl")

include("pool/TestPQs.jl")
include("pool/RandomPQs.jl")
include("pool/RandomOutlierPQs.jl")
include("pool/MinimumMarginPQs.jl")
include("pool/ExpectedMinimumMarginPQs.jl")
include("pool/ExpectedMaximumEntropyPQs.jl")
include("pool/MinimumLossPQs.jl")
include("pool/HighConfidencePQs.jl")
include("pool/DecisionBoundaryPQs.jl")
include("pool/NeighborhoodBasedPQs.jl")
include("pool/BoundaryNeighborCombinationPQs.jl")

export
    QueryStrategy,
    DataBasedPQs,
    ModelBasedPQs, HybridPQs,

    # data-based query strategies
    TestPQs, RandomPQs, MinimumMarginPQs, ExpectedMinimumMarginPQs, ExpectedMaximumEntropyPQs,
    MinimumLossPQs,
    # model-based query strategies
    RandomOutlierPQs, HighConfidencePQs, DecisionBoundaryPQs,
    # hybrid query strategies
    NeighborhoodBasedPQs, BoundaryNeighborCombinationPQs,

    get_query_object,
    qs_score,
    initialize_qs,
    filter_array,
    multi_kde, KDEException, MissingLabelTypeException

end
