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
using LinearAlgebra
using Dates
using Pkg
using JuMP
import StatsBase: countmap

using Formatting
using Memento

include("QueryStrategies/QueryStrategies.jl")
@reexport using .QueryStrategies

include("data_util.jl")
include("evaluate.jl")
include("result.jl")
include("serialize.jl")
include("oracle.jl")
include("run.jl")
include("log_util.jl")

const LOGGER = getlogger(@__MODULE__)

function __init__()
    Memento.register(LOGGER)
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

    DataStats, Result
end
