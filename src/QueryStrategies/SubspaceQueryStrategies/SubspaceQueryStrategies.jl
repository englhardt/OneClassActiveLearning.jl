module SubspaceQueryStrategies

using Memento

import MLLabelUtils
import ..QueryStrategies:
    QueryStrategy,

    get_query_objects,
    initialize_qs,
    qs_score

include("subspace_qs_base.jl")

include("SubspaceQs.jl")

export
    SubspaceQueryStrategy,
    SubspaceQs
end
