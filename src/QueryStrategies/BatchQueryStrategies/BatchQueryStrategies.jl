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
include("GappedTopkBatchQs.jl")
include("RandomBatchQs.jl")
include("IterativeBatchQs.jl")
include("EnumFilterHierarchicalBatchQs.jl")
include("FilterHierarchicalBatchQs.jl")
include("ClusterBatchQs.jl")
include("EnumerativeBatchQs.jl")
include("EnsembleBatchQs.jl")
include("ClusterTopKBatchQs.jl")
include("FilterSimilarBatchQs.jl")

export
    BatchPQs,
    ExtendingBatchQs,
    TopKBatchQs, GappedTopkBatchQs,
    RandomBatchQs,
    ClusterBatchQs,

    MultiObjectiveBatchQs,
    IterativeBatchQs,
    FilterHierarchicalBatchQs,
    EnumFilterHierarchicalBatchQs,
    EnumerativeBatchQs,

    EnsembleBatchQs,
    ClusterTopKBatchQs,

    FilterSimilarBatchQs,

    select_batch,
    angle_batch_diversity,
    euclidean_batch_diversity
end
