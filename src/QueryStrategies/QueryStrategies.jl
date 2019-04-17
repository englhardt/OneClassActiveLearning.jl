module QueryStrategies

using Reexport
using SVDD

import PyCall
import LinearAlgebra:
    det
import Statistics:
    mean,
    cov
import NearestNeighbors

const gaussian_kde = PyCall.PyNULL()
function __init__()
    copy!(gaussian_kde, PyCall.pyimport_conda("scipy.stats", "scipy").gaussian_kde)
end


include("qs_base.jl")
include("qs_utils.jl")

include("SequentialQueryStrategies/SequentialQueryStrategies.jl")
@reexport using .SequentialQueryStrategies

include("QuerySynthesisStrategies/QuerySynthesisStrategies.jl")
@reexport using .QuerySynthesisStrategies

include("HybridQuerySynthesisPQs.jl")

include("SubspaceQueryStrategies/SubspaceQueryStrategies.jl")
@reexport using .SubspaceQueryStrategies


export
    QueryStrategy,
    PoolQs,
    HybridQuerySynthesisPQs,

    qs_score,
    initialize_qs,
    get_query_object
end
