module QueryStrategies

using Reexport
using Distances

import MLKernels
import MLLabelUtils
import PyCall
import LinearAlgebra:
    det
import Statistics:
    mean,
    cov
import SVDD
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
    HybridQuerySynthesisPQs,
    KDEException, MissingLabelTypeException,

    get_query_object,
    initialize_qs,
    qs_score,
    multi_kde,
    filter_array
end
