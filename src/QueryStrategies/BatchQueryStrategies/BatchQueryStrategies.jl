module BatchQueryStrategies

using Memento
using Random

import SVDD
import MLLabelUtils
import MLKernels
import JuMP
import Distances
import LinearAlgebra

import ..QueryStrategies:
    MissingLabelTypeException,
    PoolQs,
    get_query_objects,
    multi_kde
import ..SequentialQueryStrategies:
    SequentialPQs,
    qs_score
import IterTools: subsets
import Clustering: kmedoids
import StatsBase: sample

include("batch_qs_base.jl")

include("TopKBatchQs.jl")
include("RandomBestBatchQs.jl")
include("AllRandomBatchQs.jl")
include("IterativeBatchQs.jl")
include("IterativeNRBatchQs.jl")
include("EnumHierarchicalBatchQs.jl")
include("GreedyHierarchicalBatchQs.jl")
include("KMedoidsBatchQs.jl")
include("EnumerativeBatchQs.jl")
include("EnsembleBatchQs.jl")
include("ClusterBatchQs.jl")

export
    BatchPQs,
    ExtendingBatchQs,
    TopKBatchQs, RandomBestBatchQs,
    AllRandomBatchQs,
    KMedoidsBatchQs,

    MultiObjectiveBatchQs,
    IterativeBatchQs,
    IterativeNRBatchQs,
    GreedyHierarchicalBatchQs,
    EnumHierarchicalBatchQs,
    EnumerativeBatchQs,

    EnsembleBatchQs,
    ClusterBatchQs,

    select_batch
end
