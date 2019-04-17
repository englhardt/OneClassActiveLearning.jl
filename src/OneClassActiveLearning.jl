module OneClassActiveLearning

using Memento
using Reexport
using Statistics

import Dates
import DelimitedFiles
import Formatting
import JSON
import JuMP
import MLDataUtils
import MLKernels
import MLLabelUtils
import Printf
import Random
import ROCAnalysis
import SVDD
import Unmarshal
import ValueHistories

import Base.show
import StatsBase: countmap

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
end

export
    load_data,
    SplitStrategy,
    FullSplitStrat, UnlabeledSplitStrat, UnlabeledAndLabeledInlierSplitStrat,
    LabeledSplitStrat, LabeledInlierSplitStrat, LabeledOutlierSplitStrat,
    DataSplits,
    get_train, get_test, get_query, calc_mask,
    get_splits_and_init_pools, get_initial_pools,

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
