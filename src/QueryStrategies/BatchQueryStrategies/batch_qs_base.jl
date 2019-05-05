abstract type BatchPQs <: PoolQs end
abstract type ExtendingBatchQs <: BatchPQs end
abstract type MultiObjectiveBatchQs <: BatchPQs end

"""
    select_batch(batch_query_strategy, query_data, label_map, candidate_indices)
Applies batch_query_strategy to select the most useful unlabeled observations
from query_data.

label_map is used to find out which observations in query_data are still unlabeled,
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
    all_history_values = (length(history) > 0) && (collect(Iterators.flatten(history)))
    candidate_indices = [i for i in pool_map[:U] if global_indices[i] ∉ all_history_values]
    debug(getlogger(@__MODULE__), "[QS] Selecting best batch of $(qs.k) from $(length(candidate_indices)) candidates.")
    local_query_indices = select_batch(qs, query_data, pool_map, candidate_indices)
    return global_indices[local_query_indices]
end

"""
rep_measure computes representativeness of observations.
Only KDE is supported so far.
"""
function set_rep_measure!(strategy::MultiObjectiveBatchQs, name::Symbol)::Function
    if (name == :KDE)
        strategy.rep_measure = (data::Array{T, 2} where T <: Real, labels::Dict{Symbol, Vector{Int}}, candidate_indices::Vector{Int}) -> begin
            γ = MLKernels.getvalue(strategy.model.kernel_fct.alpha)
            return multi_kde(data, γ)(data[:,candidate_indices])
        end
    else
        return throw(ArgumentError("Invalid representativeness measure $(name) specified."))
    end
end

"""
div_measure computes diversity of all samples to one specific sample
Iterative computation: only value for added sample needs to be computed, old_scores saves aggregated result
"""
function set_iterative_div_measure!(strategy::MultiObjectiveBatchQs, name::Symbol)::Function
    if (name == :AngleDiversity)
        strategy.div_measure = (model::SVDD.OCClassifier, candidate_indices::Vector{Int}, j::Int, old_scores::Vector{Float64}) -> begin
            K = SVDD.is_K_adjusted(model) ? model.K_adjusted : model.K
            div_scores = [-abs(K[i,j]) / (sqrt(K[i,i]) * sqrt(K[j,j])) for i in candidate_indices]
            if (length(old_scores) > 0)
                div_scores = min.(div_scores, old_scores)
            end
            return div_scores
        end
    elseif (name == :EuclideanDistance)
        strategy.div_measure = (model::SVDD.OCClassifier, candidate_indices::Vector{Int}, j::Int, old_scores::Vector{Float64}) -> begin
            data = model.data
            # vector containing norms of columns
            # use vec() as otherwise a row vector is returned as opposed to a column vector
            div_scores = Distances.colwise(Distances.Euclidean(), data[:,candidate_indices], data[:,j])
            if (length(old_scores) > 0)
                div_scores = min.(div_scores, old_scores)
            end
            return div_scores
        end
    else
        return throw(ArgumentError("Invalid diversity measure $(name) specified."))
    end
end

"""
div_measure computes diversity of the samples in a batch
enumerative computation: compute diversity for the whole batch each time
currently two measures are implemented:
:AngleDiversity - Batch diversity is minimal angle between two batch samples in kernel space
:EuclideanDistance - Batch diversity is minimal euclidean distance between two batch samples in feature space
"""
function set_enumerative_div_measure!(strategy::MultiObjectiveBatchQs, name::Symbol)::Function
    if (name == :AngleDiversity)
        strategy.div_measure = (model::SVDD.OCClassifier, data::Array{T, 2} where T <: Real, batch::Vector{Int}) -> begin
            K = SVDD.is_K_adjusted(model) ? model.K_adjusted : model.K
            min_div = Inf
            batch_size = length(batch)
            for i in 1:batch_size
                for j in i+1:batch_size
                    @inbounds div = -abs(K[i,j]) / (sqrt(K[i,i]) * sqrt(K[j,j]))
                    min_div = min(div, min_div)
                end
            end
            return min_div
        end
    elseif (name == :EuclideanDistance)
        strategy.div_measure = (model::SVDD.OCClassifier, data::Array{T, 2} where T <: Real, batch::Vector{Int}) -> begin
            # Compute pairwise distances, save only upper diagonal matrix
            distances = LinearAlgebra.UpperTriangular(Distances.pairwise(Distances.Euclidean(), data, data))
            # Values on diagonal are always 0
            # ignore them for minimum computation by setting them to Inf
            distances[LinearAlgebra.diagind(distances)] .= Inf

            return minimum(distances)
        end
    else
        throw(ArgumentError("Invalid diversity measure $(name) specified."))
    end
end

function min_max_normalization(x::Vector{Float64})::Vector{Float64}
    return (x .- min(x...)) ./ (max(x...) - min(x...))
end
