module SubspaceQueryStrategies

import ..QueryStrategies:
    QueryStrategy,

    get_query_object,
    initialize_qs,
    qs_score

include("subspace_qs_base.jl")
include("SubspaceQs.jl")

export
    SubspaceQueryStrategy,
    SubspaceQs
end
