module Oracles

using SVDD
using LIBSVM
using MLLabelUtils
using MLKernels
using MLBase: StratifiedKfold
using Distances
using Distributions
using Serialization
import GaussianMixtures: GMM
using Memento

import ..OneClassActiveLearning: convert_labels_to_learning, cohens_kappa, ConfusionMatrix, load_data, matthews_corr

abstract type Oracle end

include("oracle_util.jl")
include("PoolOracle.jl")
include("QuerySynthesisFunctionOracle.jl")
include("QuerySynthesisGMMOracle.jl")
include("QuerySynthesisKNNOracle.jl")
include("QuerySynthesisOCCOracle.jl")
include("QuerySynthesisSVMOracle.jl")
include("QuerySynthesisCVWrapperOracle.jl")

export
    Oracle,
    PoolOracle,
    QuerySynthesisFunctionOracle,
    QuerySynthesisGMMOracle,
    QuerySynthesisKNNOracle,
    QuerySynthesisOCCOracle,
    QuerySynthesisSVMOracle,
    QuerySynthesisCVWrapperOracle,

    ask_oracle,
    initialize_oracle
end
