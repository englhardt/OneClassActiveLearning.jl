module OneClassActiveLearning

using Reexport
using SVDD

using MLLabelUtils, MLKernels, MLDataUtils
using ROCAnalysis
using ValueHistories
using JSON, Unmarshal
using DataStructures
using Printf
using DelimitedFiles
using Statistics
using Random
using LinearAlgebra
using Distances
using Dates
using Pkg
using JuMP
using LIBSVM
using Distributions
using Serialization
import StatsBase: countmap
import Base.show

using Formatting
using Memento

include("data_util.jl")
include("evaluate.jl")

include("QueryStrategies/QueryStrategies.jl")
@reexport using .QueryStrategies

include("Oracles/Oracles.jl")
@reexport using .Oracles

include("result.jl")
include("serialize.jl")
include("run.jl")
include("log_util.jl")

const LOGGER = getlogger(@__MODULE__)

function __init__()
    Memento.register(LOGGER)
    # Memento.config!(LOGGER, "debug"; fmt="[{level} | {name}]: {msg}")
end

export
    load_data,
    SplitStrategy,
    FullSplitStrat, UnlabeledSplitStrat, UnlabeledAndLabeledInlierSplitStrat,
    LabeledSplitStrat, LabeledInlierSplitStrat, LabeledOutlierSplitStrat,
    DataSplits,
    get_train, get_test, get_query, calc_mask,
    get_splits_and_init_pools, get_initial_pools,
    Oracle, PoolOracle,
    ask_oracle,

    ConfusionMatrix,
    cohens_kappa,
    matthews_corr,
    roc_auc,
    get_positive,
    get_negative,
    tpr,
    fpr,
    tnr,
    sensitivity,
    specificity,
    recall,
    get_n,
    f1_score,
    qs_score,

    DataStats, Result
end
