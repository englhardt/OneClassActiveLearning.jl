module QuerySynthesisStrategies

using Memento
using Reexport
using MLKernels

import JuMP
import MLLabelUtils
import LIBSVM
import SVDD

import ..QueryStrategies:
    HybridPQs,
    MissingLabelTypeException,
    QueryStrategy,

    get_query_object,
    initialize_qs,
    qs_score

include("query_synthesis_base.jl")
include("query_synthesis_utils.jl")

include("Optimization/Optimization.jl")
@reexport using .Optimization

include("TestQss.jl")
include("RandomQss.jl")
include("RandomOutlierQss.jl")
include("DecisionBoundaryQss.jl")
include("NaiveExplorativeMarginQss.jl")
include("ExplorativeMarginQss.jl")

export
    QuerySynthesisStrategy,
    DataBasedQss, ModelBasedQss, HybridQss,
    # BaselineQss:
    TestQss, RandomQss,
    # DataBasedQss:
    # ModelBasedQss:
    DecisionBoundaryQss, RandomOutlierQss,
    # HybridQss:
    ExplorativeMarginQss, NaiveExplorativeMarginQss,

    get_query_object,
    estimate_margin_epsilon, estimate_limit_epsilon,
    check_epsilon, check_limits,
    extrema_arrays,
    rand_in_hypercube,
    estimate_boundary_shift_epsilon, data_limit_epsilon, data_boundaries
end
