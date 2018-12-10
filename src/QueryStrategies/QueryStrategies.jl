module QueryStrategies

using PyCall, Memento

const gaussian_kde = PyNULL()

function __init__()
    copy!(gaussian_kde, pyimport_conda("scipy.stats", "scipy")[:gaussian_kde])
end

include("qs_base.jl")
include("qs_utils.jl")
include("qs_subspace.jl")

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

include("query_synthesis/query_synthesis_base.jl")
include("query_synthesis/query_synthesis_utils.jl")
include("query_synthesis/TestQss.jl")
include("query_synthesis/RandomQss.jl")
include("query_synthesis/RandomOutlierQss.jl")
include("query_synthesis/DecisionBoundaryQss.jl")
include("query_synthesis/ExplorativeMarginQss.jl")

export
    QueryStrategy,
    PoolQs,
    SubspaceQs,
    DataBasedPQs, ModelBasedPQs, HybridPQs,
    QuerySynthesisStrategy,
    SubspaceQueryStrategy,
    DataBasedQss, ModelBasedQss, HybridQss,

    # pool based query strategies
    # data-based query strategies
    TestPQs, RandomPQs, MinimumMarginPQs, ExpectedMinimumMarginPQs, ExpectedMaximumEntropyPQs,
    MinimumLossPQs,
    # model-based query strategies
    RandomOutlierPQs, HighConfidencePQs, DecisionBoundaryPQs,
    # hybrid query strategies
    NeighborhoodBasedPQs, BoundaryNeighborCombinationPQs,

    # query synthesis query query strategies
    TestQss, RandomQss, RandomOutlierQss,
    DecisionBoundaryQss, ExplorativeMarginQss,


    # query synthesis optimizers
    QuerySynthesisOptimizer,
    ParticleSwarmOptimization,

    get_query_object,
    qs_score,
    initialize_qs,
    filter_array,
    multi_kde, KDEException, MissingLabelTypeException,
    estimate_margin_epsilon, estimate_limit_epsilon

end
