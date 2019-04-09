module SubspaceQueryStrategies

import ...QueryStrategies: QueryStrategy, qs_score, get_query_object

abstract type SubspaceQueryStrategy <: QueryStrategy end

import ...QueryStrategies: initialize_qs

include("subspace_util.jl")
include("SubspaceQs.jl")

export
    SubspaceQueryStrategy,
    SubspaceQs
end
