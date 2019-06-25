abstract type BatchPQs <: PoolQs end
abstract type ExtendingBatchQs <: BatchPQs end
abstract type MultiObjectiveBatchQs <: BatchPQs end

"""
    select_batch(batch_query_strategy, query_data, label_map, candidate_indices)
Applies batch_query_strategy to select the most useful unlabeled observations
from query_data.

pools is used to find out which observations in query_data are still unlabeled,
candidate_indices is used to find out which observations in query_data are valid
candidates for the query.

Returns list of indices of most useful observations.
"""
function select_batch end

function get_query_objects(qs::BatchPQs,
                           query_data::Array{T, 2} where T <: Real,
                           pools::Vector{Symbol},
                           global_indices::Vector{Int},
                           history::Vector{Vector{Int}})::Vector{Int}
    pool_map = MLLabelUtils.labelmap(pools)
    haskey(pool_map, :U) || throw(ArgumentError("No more points that are unlabeled."))
    all_history_values = collect(Iterators.flatten(history))
    candidate_indices = [i for i in pool_map[:U] if global_indices[i] ∉ all_history_values]
    debug(getlogger(@__MODULE__), "[QS] Selecting best batch of $(qs.k) from $(length(candidate_indices)) candidates.")
    local_query_indices = select_batch(qs, query_data, pool_map, candidate_indices)
    return global_indices[local_query_indices]
end

"""
rep_measure computes representativeness of observations.
Only KDE is supported so far.
"""
function get_rep_measure(name::Symbol)::Function
    if (name == :KDE)
        return (model::SVDD.OCClassifier, data::Array{T, 2} where T <: Real, labels::Dict{Symbol, Vector{Int}}, batch::Vector{Int}) -> begin
            γ = MLKernels.getvalue(model.kernel_fct.alpha)
            return multi_kde(data, γ)(data[:, batch])
        end
    else
        return throw(ArgumentError("Invalid representativeness measure $(name) specified."))
    end
end

"""
div_measure computes diversity of all candidates incrementally
Incremental computation: only diversity of the candidates to the last sample that
has been added to the batch (new_in_batch_idx) must be computed,
old_scores saves aggregated result, i.e., the diversity to the rest of the batch

currently two measures are implemented:
:AngleDiversity - Batch diversity is minimal angle between two batch samples in kernel space
:EuclideanDistance - Batch diversity is minimal euclidean distance between two batch samples in feature space
"""
function get_incremental_div_measure(name::Symbol)::Function
    if (name == :AngleDiversity)
        return (model::SVDD.OCClassifier, data::Array{T, 2} where T <: Real, new_in_batch_idx::Int, candidates::Vector{Int}, old_scores::Vector{Float64}) -> begin
            if model.data == data
                # reuse model kernel matrix
                K = SVDD.is_K_adjusted(model) ? model.K_adjusted : model.K
            else
                # compute a new kernel matrix because query data differs from model data
                K = MLKernels.kernelmatrix(Val(:col), SVDD.get_kernel(model), data)
            end
            div_scores = [-abs(K[i, new_in_batch_idx]) / (sqrt(K[i, i]) * sqrt(K[new_in_batch_idx, new_in_batch_idx])) for i in candidates]
            if !isempty(old_scores)
                div_scores = min.(div_scores, old_scores)
            end
            return div_scores
        end
    elseif (name == :EuclideanDistance)
        return (model::SVDD.OCClassifier, data::Array{T, 2} where T <: Real, new_in_batch_idx::Int, candidates::Vector{Int}, old_scores::Vector{Float64}) -> begin
            div_scores = Distances.colwise(Distances.Euclidean(), data[:, candidates], data[:, new_in_batch_idx])
            if !isempty(old_scores)
                div_scores = min.(div_scores, old_scores)
            end
            return div_scores
        end
    else
        return throw(ArgumentError("Invalid diversity measure $(name) specified."))
    end
end

function angle_batch_diversity(kernel::MLKernels.Kernel, data::Array{T, 2}) where T <: Real
    K = MLKernels.kernelmatrix(Val(:col), kernel, data)
    return angle_batch_diversity(K)
end

"""
The angle batch diversity is minimal angle between all pairs in the batch and
is calculated in the kernel space
"""
function angle_batch_diversity(K::Array{T, 2}) where T <: Real
    min_div = Inf
    for i in 1:size(K, 1)
        for j in i+1:size(K, 2)
            @inbounds div = -abs(K[i, j]) / (sqrt(K[i, i]) * sqrt(K[j, j]))
            min_div = min(div, min_div)
        end
    end
    return min_div
end

"""
The euclidean batch diversity is the minimal euclidean distance between all pairs
in the batch and is calculated in the feature space
"""
function euclidean_batch_diversity(data::Array{T, 2}) where T <: Real
    distances = Distances.pairwise(Distances.Euclidean(), data, dims=2)
    distances[LinearAlgebra.diagind(distances)] .= Inf
    return minimum(distances)
end

"""
div_measure computes diversity of the samples in a batch
enumerative computation: compute diversity for the whole batch from scratch

currently two measures are implemented:
:AngleDiversity - Batch diversity is minimal angle between two batch samples in kernel space
:EuclideanDistance - Batch diversity is minimal euclidean distance between two batch samples in feature space
"""
function get_div_measure(name::Symbol)::Function
    if (name == :AngleDiversity)
        return (model::SVDD.OCClassifier, data::Array{T, 2} where T <: Real, batch::Vector{Int}) -> begin
            if model.data == data
                # reuse model kernel matrix
                K = SVDD.is_K_adjusted(model) ? model.K_adjusted[batch, batch] : model.K[batch, batch]
                return angle_batch_diversity(K)
            else
                # compute a new kernel matrix because query data differs from model data
                return angle_batch_diversity(SVDD.get_kernel(model), data[:, batch])
            end
        end
    elseif (name == :EuclideanDistance)
        return (model::SVDD.OCClassifier, data::Array{T, 2} where T <: Real, batch::Vector{Int}) -> begin
            return euclidean_batch_diversity(data[:, batch])
        end
    else
        throw(ArgumentError("Invalid diversity measure $(name) specified."))
    end
end

function min_max_normalization(x::Vector{Float64})::Vector{Float64}
    return (x .- minimum(x)) ./ (maximum(x) - minimum(x))
end
