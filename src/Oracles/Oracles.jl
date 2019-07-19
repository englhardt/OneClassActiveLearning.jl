module Oracles

using Memento
using Distances
using Statistics
using Serialization

import SVDD
import Distributions
import LIBSVM
import MLKernels

import GaussianMixtures: GMM
import MLBase: StratifiedKfold

import ..OneClassActiveLearning:
    ConfusionMatrix,

    cohens_kappa,
    convert_labels_to_learning,
    load_data,
    matthews_corr

include("oracle_base.jl")

include("PoolOracle.jl")
include("NoisyOracle.jl")
include("QuerySynthesisFunctionOracle.jl")
include("QuerySynthesisGMMOracle.jl")
include("QuerySynthesisKNNOracle.jl")
include("QuerySynthesisOCCOracle.jl")
include("QuerySynthesisSVMOracle.jl")
include("QuerySynthesisCVWrapperOracle.jl")

export
    Oracle,
    PoolOracle,
    NoisyOracle,
    QuerySynthesisFunctionOracle,
    QuerySynthesisGMMOracle,
    QuerySynthesisKNNOracle,
    QuerySynthesisOCCOracle,
    QuerySynthesisSVMOracle,
    QuerySynthesisCVWrapperOracle,

    ask_oracle,
    initialize_oracle
end
